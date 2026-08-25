#!/usr/bin/env python3
"""Validate cached static radix data against the ordinary full-scene path."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import gaussian_set_from_hugsim_model
from decouplegs.rasterizer import (
    CompactRenderInstance,
    StaticBackgroundRasterCache,
    rasterize_compact_scene,
)
from decouplegs.registration import RegistrationConfig
from gaussian_renderer import GaussianModel
from tools.benchmark_decouplegs_scalability import instance_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("asset", type=Path)
    parser.add_argument("--agent-counts", type=int, nargs="+", default=(1, 10, 20))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--motion-m", type=float, default=0.031)
    parser.add_argument("--asset-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def comparison(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | bool]:
    difference = (candidate - reference).abs()
    mse = float(difference.square().mean())
    return {
        "bit_exact": bool(torch.equal(candidate, reference)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "mse": mse,
        "psnr_db": 300.0 if mse == 0.0 else -10.0 * math.log10(mse),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("static-cache validation requires CUDA")
    if any(count < 0 for count in args.agent_counts):
        raise ValueError("agent counts must be non-negative")

    scene_config_path = args.scene_dir / "cfg.yaml"
    scene_config = (
        yaml.safe_load(scene_config_path.read_text())
        if scene_config_path.is_file()
        else {}
    ) or {}
    degree = int(scene_config.get("model", {}).get("sh_degree", 3))
    background_model = GaussianModel(
        degree,
        affine=bool(scene_config.get("affine", True)),
    )
    state, iteration = torch.load(
        args.scene_dir / "scene.pth", map_location="cuda", weights_only=False
    )
    background_model.restore(state, None)
    background = gaussian_set_from_hugsim_model(background_model, include_ground=True)
    asset = CompactGaussianAsset.load(args.asset).to("cuda")
    frames = json.loads((args.scene_dir / "meta_data.json").read_text())["frames"]
    frame = frames[args.frame_index]
    c2w = torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda")
    viewmats = torch.linalg.inv(c2w)[None]
    intrinsics = torch.as_tensor(
        np.asarray(frame["intrinsics"]), dtype=torch.float32, device="cuda"
    )[None, :3, :3]
    width, height = int(frame["width"]), int(frame["height"])
    black = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    registration = RegistrationConfig(
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        forward_axis=0,
        up_axis=1,
        up_sign=-1,
    )
    cache = StaticBackgroundRasterCache(1)

    def make_instances(count: int, motion: float) -> list[CompactRenderInstance]:
        poses = instance_poses(
            count,
            frame,
            registration,
            lanes=5,
            near_distance=12.0,
            row_spacing=4.2,
            lane_spacing=3.0,
            camera_height=1.5,
        )
        for pose in poses:
            pose[:3, 3] += c2w[:3, 2] * motion
        return [
            CompactRenderInstance(f"agent-{index:03d}", asset, pose)
            for index, pose in enumerate(poses)
        ]

    def render(
        instances: list[CompactRenderInstance],
        static_cache: StaticBackgroundRasterCache | None,
        cache_key: object | None,
    ):
        return rasterize_compact_scene(
            background,
            instances,
            viewmats,
            intrinsics,
            width,
            height,
            sh_degree=background_model.active_sh_degree,
            backgrounds=black,
            asset_batch_size=args.asset_batch_size,
            background_cache=static_cache,
            background_cache_key=cache_key,
        )

    rows = []
    with torch.inference_mode():
        # Populate from an ordinary full-scene radix result, then move every
        # agent so the checked frame cannot reuse dynamic projection or order.
        fill = render(
            make_instances(args.agent_counts[0], 0.0),
            cache,
            ("frame", args.frame_index),
        )
        for count in args.agent_counts:
            moved = make_instances(count, args.motion_m)
            candidate = render(moved, cache, ("frame", args.frame_index))
            reference = render(moved, None, None)
            rgb = comparison(candidate.render, reference.render)
            alpha = comparison(candidate.alpha, reference.alpha)
            if not rgb["bit_exact"] or not alpha["bit_exact"]:
                raise RuntimeError(
                    f"cached render diverged for {count} agents: rgb={rgb}, alpha={alpha}"
                )
            rows.append(
                {
                    "agents": count,
                    "sort_mode": candidate.info["intersection_sort_mode"],
                    "merge_backend": candidate.info["intersection_merge_backend"],
                    "visible_gaussians": candidate.info["visible_gaussians"],
                    "intersections": candidate.info["intersections"],
                    "rgb": rgb,
                    "alpha": alpha,
                }
            )

        # A distinct camera key must miss and evict the one-entry LRU. This is
        # a cheap explicit regression check for camera-cache invalidation.
        invalidation = render([], cache, ("frame", args.frame_index + 1))
        if invalidation.info["background_cache_hit"]:
            raise RuntimeError("a changed camera cache key incorrectly hit")

    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_static_background_cache_equivalence",
        "scene": str(args.scene_dir.resolve()),
        "asset": str(args.asset.resolve()),
        "scene_iteration": int(iteration),
        "resolution": [width, height],
        "frame_index": args.frame_index,
        "motion_m": args.motion_m,
        "cache_fill_mode": fill.info["intersection_sort_mode"],
        "cache_invalidation_mode": invalidation.info["intersection_sort_mode"],
        "cache_invalidation_hit": invalidation.info["background_cache_hit"],
        "cache": {
            "entries": cache.entries,
            "hits": cache.hits,
            "misses": cache.misses,
            "evictions": cache.evictions,
            "memory_mib": cache.memory_bytes / 2**20,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
