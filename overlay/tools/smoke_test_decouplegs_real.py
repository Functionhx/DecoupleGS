#!/usr/bin/env python3
"""Render one real HUGSIM scene with one canonical DecoupleGS vehicle asset."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.registration import RegistrationConfig
from decouplegs.runtime import AssetLibrary, DecoupleRuntime
from gaussian_renderer import GaussianModel, render
from scene.obj_model import ObjModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-checkpoint smoke test for unified DecoupleGS rasterization",
    )
    parser.add_argument("scene_dir", type=Path, help="Extracted HUGSIM scene containing scene.pth")
    parser.add_argument("asset", type=Path, help="3DRealCar gs.pth or compressed .dgs asset")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.5, help="Render resolution multiplier")
    parser.add_argument("--distance", type=float, default=20.0, help="Placement distance along camera heading")
    parser.add_argument("--lateral", type=float, default=0.0, help="World-X placement offset in metres")
    parser.add_argument("--camera-height", type=float, default=1.5, help="Initial camera-to-road +Y offset")
    parser.add_argument("--no-grounding", action="store_true")
    parser.add_argument("--no-rotate-sh", action="store_true")
    parser.add_argument("--contact-shadows", action="store_true")
    parser.add_argument("--adaptive-relighting", action="store_true")
    parser.add_argument("--packed", action="store_true", help="Use gsplat sparse packed projection")
    parser.add_argument("--rgb-only", action="store_true", help="Skip depth/semantic rasterization")
    parser.add_argument("--warmup", type=int, default=0, help="Untimed warm-up frames")
    parser.add_argument("--repeats", type=int, default=1, help="Number of measured frames")
    parser.add_argument("--output", type=Path, default=None, help="Optional rendered PNG")
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": percentile(0.95),
    }


def load_scene_config(scene_dir: Path) -> dict:
    path = scene_dir / "cfg.yaml"
    if not path.is_file():
        return {"affine": True, "model": {"sh_degree": 3}}
    with path.open() as stream:
        return yaml.safe_load(stream) or {}


def canonical_pose(frame: dict, args: argparse.Namespace, config: RegistrationConfig) -> torch.Tensor:
    c2w = torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda")
    camera_forward = c2w[:3, 2].clone()
    camera_forward[config.vertical_axis] = 0.0
    camera_forward = torch.nn.functional.normalize(camera_forward, dim=0)
    yaw = torch.atan2(
        camera_forward[config.horizontal_axes[1]],
        camera_forward[config.horizontal_axes[0]],
    )
    from decouplegs.registration import rotation_from_yaw_and_normal

    normal = torch.zeros(3, device="cuda")
    normal[config.vertical_axis] = config.vertical_sign
    rotation = rotation_from_yaw_and_normal(yaw, normal, config)
    translation = c2w[:3, 3] + camera_forward * args.distance
    translation[0] += args.lateral
    translation[config.vertical_axis] += args.camera_height
    pose = torch.eye(4, dtype=torch.float32, device="cuda")
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("HUGSIM_splat real-checkpoint smoke testing requires CUDA")
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0, 1]")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be non-negative and --repeats must be positive")

    scene_config = load_scene_config(args.scene_dir)
    sh_degree = int(scene_config.get("model", {}).get("sh_degree", 3))
    affine = bool(scene_config.get("affine", True))
    with (args.scene_dir / "meta_data.json").open() as stream:
        frames = json.load(stream)["frames"]
    if not -len(frames) <= args.frame_index < len(frames):
        raise IndexError(f"frame index {args.frame_index} is outside a {len(frames)}-frame scene")
    frame = frames[args.frame_index]

    background = GaussianModel(sh_degree, affine=affine)
    background_state, scene_iteration = torch.load(
        args.scene_dir / "scene.pth",
        map_location="cuda",
        weights_only=False,
    )
    background.restore(background_state, None)

    registration = RegistrationConfig(
        column_radius=0.35,
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        forward_axis=0,
        up_axis=1,
        up_sign=-1,
    )
    runtime = DecoupleRuntime(
        AssetLibrary(),
        registration_config=registration,
        relighting=args.adaptive_relighting,
        adaptive_relighting=args.adaptive_relighting,
        rotate_sh=not args.no_rotate_sh,
        contact_shadows=args.contact_shadows,
        opacity_grounding=not args.no_grounding,
        frustum_culling=True,
    )
    bridge = HUGSIMRuntimeBridge(runtime)
    asset_id = "decouple-smoke-car"
    dynamic_models: dict[str, ObjModel] = {}
    asset_iteration: int | None = None
    if args.asset.suffix == ".dgs":
        compact = CompactGaussianAsset.load(args.asset)
        runtime.library.add(asset_id, compact)
        asset_primitives = len(compact)
    else:
        asset_model = ObjModel(sh_degree, feat_mutable=False)
        asset_state = torch.load(args.asset, map_location="cuda", weights_only=False)
        if isinstance(asset_state, (tuple, list)) and len(asset_state) == 2:
            asset_state, asset_iteration = asset_state
        asset_model.restore(list(asset_state), None)
        bridge.register_model(asset_id, asset_model)
        dynamic_models[asset_id] = asset_model
        asset_primitives = int(asset_model.get_xyz.shape[0])

    intrinsics = torch.as_tensor(np.array(frame["intrinsics"]), dtype=torch.float32, device="cuda")
    intrinsics[0, :3] *= args.scale
    intrinsics[1, :3] *= args.scale
    width = max(1, round(int(frame["width"]) * args.scale))
    height = max(1, round(int(frame["height"]) * args.scale))
    pose = canonical_pose(frame, args, registration)
    viewpoint = SimpleNamespace(
        c2w=torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda"),
        K=intrinsics,
        width=width,
        height=height,
        timestamp=float(frame.get("timestamp", -1.0)),
        dynamics={asset_id: pose},
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = render(
                viewpoint,
                None,
                background,
                dynamic_models,
                {},
                torch.zeros(3, dtype=torch.float32, device="cuda"),
                decouple_bridge=bridge,
                rasterizer_packed=args.packed,
                rgb_only=args.rgb_only,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starting_memory = torch.cuda.memory_allocated()
        elapsed_frames: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            output = render(
                viewpoint,
                None,
                background,
                dynamic_models,
                {},
                torch.zeros(3, dtype=torch.float32, device="cuda"),
                decouple_bridge=bridge,
                rasterizer_packed=args.packed,
                rgb_only=args.rgb_only,
            )
            torch.cuda.synchronize()
            elapsed_frames.append(time.perf_counter() - started)
    elapsed = sum(elapsed_frames)
    latency = distribution(elapsed_frames)
    image = output["render"]
    alpha = output["alphas"]
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pixels = (image.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        Image.fromarray(pixels).save(args.output)

    print(
        json.dumps(
            {
                "scene_iteration": int(scene_iteration),
                "asset_iteration": None if asset_iteration is None else int(asset_iteration),
                "background_gaussians": int(background.get_full_xyz.shape[0]),
                "asset_gaussians": asset_primitives,
                "merged_gaussians": int(background.get_full_xyz.shape[0]) + asset_primitives,
                "resolution": [width, height],
                "warmup_frames": args.warmup,
                "measured_frames": args.repeats,
                "elapsed_seconds": elapsed,
                "fps": args.repeats / max(elapsed, 1e-12),
                "latency_ms": {key: value * 1000.0 for key, value in latency.items()},
                "rgb_finite": bool(torch.isfinite(image).all()),
                "alpha_finite": bool(torch.isfinite(alpha).all()),
                "alpha_max": float(alpha.max()),
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "incremental_peak_memory_mib": (
                    torch.cuda.max_memory_allocated() - starting_memory
                )
                / 2**20,
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved() / 2**20,
                "output": None if args.output is None else str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
