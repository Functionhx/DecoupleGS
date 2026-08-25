#!/usr/bin/env python3
"""Benchmark the VQ-indexed DecoupleGS renderer on real HUGSIM checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import HUGSIMRuntimeBridge, gaussian_set_from_hugsim_model
from decouplegs.rasterizer import (
    CompactRenderInstance,
    StaticBackgroundRasterCache,
    rasterize_compact_scene,
)
from decouplegs.registration import RegistrationConfig, rotation_from_yaw_and_normal
from decouplegs.relighting import RelightingConfig
from decouplegs.runtime import AssetLibrary, DecoupleRuntime
from gaussian_renderer import GaussianModel, render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-scene DecoupleGS scalability benchmark")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("asset", type=Path, help="Compressed .dgs asset")
    parser.add_argument("--agent-counts", type=int, nargs="+", default=(1, 5, 10, 20, 50))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=5)
    parser.add_argument("--near-distance", type=float, default=12.0)
    parser.add_argument("--row-spacing", type=float, default=4.2)
    parser.add_argument("--lane-spacing", type=float, default=3.0)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--lod-radius-clip", type=float, default=None)
    parser.add_argument("--lod-start-distance", type=float, default=20.0)
    parser.add_argument("--asset-batch-size", type=int, default=8)
    parser.add_argument(
        "--static-background-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache static projection/radix data and incrementally merge moving assets",
    )
    parser.add_argument("--static-background-cache-entries", type=int, default=1)
    parser.add_argument("--static-background-cache-min-observations", type=int, default=2)
    parser.add_argument("--incremental-merge-max-dynamic-ratio", type=float, default=1.25)
    parser.add_argument("--auxiliary", action="store_true", help="Also render depth and semantics")
    parser.add_argument(
        "--full-runtime",
        action="store_true",
        help="Include grounding, local probes, relighting, shadows, and appearance affine",
    )
    parser.add_argument(
        "--runtime-state",
        choices=("moving", "cached"),
        default="moving",
        help="Whether each measured frame invalidates the scene preparation cache",
    )
    parser.add_argument(
        "--shadow-mask-epsilon",
        type=float,
        default=1e-5,
        help="Finite shadow-mask error bound for full-runtime measurements",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
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


def repository_state() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_scene_config(scene_dir: Path) -> dict:
    path = scene_dir / "cfg.yaml"
    if not path.is_file():
        return {"affine": True, "model": {"sh_degree": 3}}
    with path.open() as stream:
        return yaml.safe_load(stream) or {}


def instance_poses(
    count: int,
    frame: dict,
    config: RegistrationConfig,
    *,
    lanes: int,
    near_distance: float,
    row_spacing: float,
    lane_spacing: float,
    camera_height: float,
) -> list[torch.Tensor]:
    c2w = torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda")
    forward = c2w[:3, 2].clone()
    forward[config.vertical_axis] = 0.0
    forward = torch.nn.functional.normalize(forward, dim=0)
    right = c2w[:3, 0].clone()
    right[config.vertical_axis] = 0.0
    right = torch.nn.functional.normalize(right, dim=0)
    yaw = torch.atan2(forward[config.horizontal_axes[1]], forward[config.horizontal_axes[0]])
    normal = torch.zeros(3, dtype=torch.float32, device="cuda")
    normal[config.vertical_axis] = config.vertical_sign
    rotation = rotation_from_yaw_and_normal(yaw, normal, config)
    poses = []
    for index in range(count):
        row, lane = divmod(index, lanes)
        agents_in_row = min(lanes, count - row * lanes)
        lateral = (lane - 0.5 * (agents_in_row - 1)) * lane_spacing
        translation = (
            c2w[:3, 3]
            + forward * (near_distance + row * row_spacing)
            + right * lateral
        )
        translation[config.vertical_axis] += camera_height
        pose = torch.eye(4, dtype=torch.float32, device="cuda")
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        poses.append(pose)
    return poses


def affine_parameters(background: GaussianModel, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not background.affine:
        return None
    camera_position, camera_direction = c2w[:3, 3], c2w[:3, 2]
    encoded_position = background.pos_enc(camera_position[None] / 60)
    encoded_direction = background.dir_enc(camera_direction[None])
    appearance = background.appearance_model(
        torch.cat((encoded_position, encoded_direction), dim=1)
    ) * 1e-1
    weight = appearance[:, :9].view(3, 3) + torch.eye(3, device=appearance.device)
    return weight, appearance[:, -3:]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the scalability benchmark requires CUDA")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if any(count < 0 for count in args.agent_counts):
        raise ValueError("agent counts must be non-negative")
    if not 0 < args.scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    if args.static_background_cache_entries <= 0:
        raise ValueError("static-background-cache-entries must be positive")
    if args.static_background_cache_min_observations <= 0:
        raise ValueError("static-background-cache-min-observations must be positive")
    if args.incremental_merge_max_dynamic_ratio < 0:
        raise ValueError("incremental-merge-max-dynamic-ratio must be non-negative")

    scene_config = load_scene_config(args.scene_dir)
    sh_degree = int(scene_config.get("model", {}).get("sh_degree", 3))
    background_model = GaussianModel(sh_degree, affine=bool(scene_config.get("affine", True)))
    background_state, scene_iteration = torch.load(
        args.scene_dir / "scene.pth", map_location="cuda", weights_only=False
    )
    background_model.restore(background_state, None)
    background = gaussian_set_from_hugsim_model(background_model, include_ground=True)
    compact = CompactGaussianAsset.load(args.asset).to("cuda")
    with (args.scene_dir / "meta_data.json").open() as stream:
        frames = json.load(stream)["frames"]
    frame = frames[args.frame_index]
    c2w = torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda")
    viewmats = torch.linalg.inv(c2w)[None]
    intrinsics = torch.as_tensor(np.array(frame["intrinsics"]), dtype=torch.float32, device="cuda")
    intrinsics[0, :3] *= args.scale
    intrinsics[1, :3] *= args.scale
    intrinsics = intrinsics[None, :3, :3]
    width = max(1, round(int(frame["width"]) * args.scale))
    height = max(1, round(int(frame["height"]) * args.scale))
    registration = RegistrationConfig(
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        forward_axis=0,
        up_axis=1,
        up_sign=-1,
    )
    affine = affine_parameters(background_model, c2w)
    rgb_background = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    results = []
    runtime_bridge = None
    static_raster_cache = (
        StaticBackgroundRasterCache(args.static_background_cache_entries)
        if args.static_background_cache
        else None
    )
    runtime_index_setup_seconds = None
    if args.full_runtime:
        runtime = DecoupleRuntime(
            AssetLibrary(),
            registration_config=registration,
            relighting_config=RelightingConfig(
                probe_sigma=3.0,
                probe_radius=9.0,
                shadow_mask_epsilon=args.shadow_mask_epsilon,
            ),
            rotate_sh=True,
            relighting=True,
            adaptive_relighting=True,
            contact_shadows=True,
            opacity_grounding=True,
            frustum_culling=True,
        )
        runtime_bridge = HUGSIMRuntimeBridge(
            runtime,
            compact_asset_batch_size=args.asset_batch_size,
            compact_radius_clip=args.radius_clip,
            compact_lod_radius_clip=args.lod_radius_clip,
            compact_lod_start_distance=args.lod_start_distance,
            static_background_cache=args.static_background_cache,
            static_background_cache_entries=args.static_background_cache_entries,
            static_background_cache_min_observations=(
                args.static_background_cache_min_observations
            ),
            incremental_merge_max_dynamic_ratio=args.incremental_merge_max_dynamic_ratio,
        )
        for index in range(max(args.agent_counts, default=0)):
            runtime.library.add(f"agent-{index:03d}", compact)
        # Spatial structures and projected road index are scene setup, not a
        # per-frame renderer cost. Build both explicitly and report the time.
        setup_started = time.perf_counter()
        cached_background = runtime_bridge._background(background_model)
        ground_mask = (
            None
            if cached_background.semantics is None
            else cached_background.semantics.argmax(dim=-1) <= 1
        )
        grounding_background = runtime_bridge._ground_background(
            background_model, cached_background, ground_mask
        )
        spatial_index = runtime_bridge._background_spatial_index(
            background_model, grounding_background
        )
        spatial_index.query_radius(
            grounding_background.means[:1, list(registration.horizontal_axes)],
            registration.column_radius,
            axes=registration.horizontal_axes,
        )
        runtime_index_setup_seconds = time.perf_counter() - setup_started

    with torch.inference_mode():
        for count in args.agent_counts:
            poses = instance_poses(
                count,
                frame,
                registration,
                lanes=args.lanes,
                near_distance=args.near_distance,
                row_spacing=args.row_spacing,
                lane_spacing=args.lane_spacing,
                camera_height=args.camera_height,
            )
            if args.full_runtime:
                assert runtime_bridge is not None
                viewpoint = SimpleNamespace(
                    c2w=c2w,
                    K=intrinsics[0],
                    width=width,
                    height=height,
                    timestamp=float(frame.get("timestamp", 0.0)),
                    dynamics={},
                )
                base_dynamics = {
                    f"agent-{index:03d}": pose for index, pose in enumerate(poses)
                }
                runtime_frame = 0

                def render_once():
                    nonlocal runtime_frame
                    if args.runtime_state == "moving":
                        runtime_frame += 1
                        phase = ((runtime_frame % 5) - 2) * 0.01
                        dynamics = {}
                        for track_id, pose in base_dynamics.items():
                            moved = pose.clone()
                            moved[:3, 3] += c2w[:3, 2] * phase
                            dynamics[track_id] = moved
                        viewpoint.timestamp = float(frame.get("timestamp", 0.0)) + runtime_frame * 0.5
                        viewpoint.dynamics = dynamics
                    else:
                        viewpoint.dynamics = base_dynamics
                    output = render(
                        viewpoint,
                        None,
                        background_model,
                        {},
                        {},
                        rgb_background[0],
                        decouple_bridge=runtime_bridge,
                        rgb_only=not args.auxiliary,
                    )
                    return output, output["render"]

            else:
                instances = []
                for index, pose in enumerate(poses):
                    instance_radius_clip = None
                    distance = float(torch.linalg.vector_norm(pose[:3, 3] - c2w[:3, 3]).item())
                    if args.lod_radius_clip is not None and distance >= args.lod_start_distance:
                        instance_radius_clip = args.lod_radius_clip
                    instances.append(
                        CompactRenderInstance(
                            f"agent-{index:03d}",
                            compact,
                            pose,
                            radius_clip=instance_radius_clip,
                        )
                    )

                def render_once():
                    output = rasterize_compact_scene(
                        background,
                        instances,
                        viewmats,
                        intrinsics,
                        width,
                        height,
                        sh_degree=background_model.active_sh_degree,
                        backgrounds=rgb_background,
                        auxiliary=args.auxiliary,
                        radius_clip=args.radius_clip,
                        asset_batch_size=args.asset_batch_size,
                        background_cache=static_raster_cache,
                        background_cache_key=("fixed-benchmark-camera", args.frame_index),
                        incremental_merge_max_dynamic_ratio=args.incremental_merge_max_dynamic_ratio,
                    )
                    image = output.render
                    if affine is not None:
                        weight, bias = affine
                        image = (image.reshape(-1, 3) @ weight + bias).clip(0, 1).reshape_as(image)
                    return output, image

            for _ in range(args.warmup):
                output, image = render_once()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            starting_allocated = torch.cuda.memory_allocated()
            starting_reserved = torch.cuda.memory_reserved()
            latencies = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                output, image = render_once()
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)
            elapsed = sum(latencies)
            latency = distribution(latencies)
            info = output["info"] if args.full_runtime else output.info
            alpha = output["alphas"] if args.full_runtime else output.alpha
            result = {
                "agents": count,
                "fps": args.repeats / elapsed,
                "latency_ms": {key: value * 1000.0 for key, value in latency.items()},
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "incremental_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() - starting_allocated
                )
                / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "incremental_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved() - starting_reserved
                )
                / 2**20,
                "visible_gaussians": info["visible_gaussians"],
                "tile_intersections": info["intersections"],
                "intersection_sort_mode": info.get("intersection_sort_mode"),
                "intersection_merge_backend": info.get("intersection_merge_backend"),
                "background_cache_hit": info.get("background_cache_hit"),
                "background_cache_memory_mib": info.get(
                    "background_cache_memory_bytes", 0
                )
                / 2**20,
                "rgb_finite": bool(torch.isfinite(image).all()),
                "alpha_finite": bool(torch.isfinite(alpha).all()),
            }
            if args.full_runtime and runtime_bridge._prepared_compact is not None:
                prepared = runtime_bridge._prepared_compact
                result["grounded_agents"] = len(prepared.ground_planes)
                result["relit_agents"] = len(prepared.descriptors)
                result["contact_shadow_semantic_candidates"] = prepared.background.metadata.get(
                    "contact_shadow_semantic_candidates"
                )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_compact_scalability",
        "renderer": (
            "vq_indexed_full_runtime_v1"
            if args.full_runtime
            else "vq_indexed_split_projection_v1"
        ),
        "scene": str(args.scene_dir.resolve()),
        "asset": str(args.asset.resolve()),
        "scene_iteration": int(scene_iteration),
        "resolution": [width, height],
        "auxiliary": args.auxiliary,
        "warmup_frames": args.warmup,
        "measured_frames": args.repeats,
        "full_runtime": args.full_runtime,
        "runtime_state": args.runtime_state if args.full_runtime else None,
        "static_background_cache": args.static_background_cache,
        "static_background_cache_entries": args.static_background_cache_entries,
        "static_background_cache_min_observations": (
            args.static_background_cache_min_observations
        ),
        "incremental_merge_max_dynamic_ratio": args.incremental_merge_max_dynamic_ratio,
        "runtime_index_setup_seconds": runtime_index_setup_seconds,
        "background_gaussians": len(background),
        "asset_gaussians": len(compact),
        "background_tensor_mib": background.memory_bytes / 2**20,
        "compact_asset_mib": compact.memory_bytes / 2**20,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "repository": repository_state(),
        "placement": {
            "lanes": args.lanes,
            "near_distance": args.near_distance,
            "row_spacing": args.row_spacing,
            "lane_spacing": args.lane_spacing,
            "camera_height": args.camera_height,
            "radius_clip": args.radius_clip,
            "lod_radius_clip": args.lod_radius_clip,
            "lod_start_distance": args.lod_start_distance,
            "asset_batch_size": args.asset_batch_size,
        },
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
