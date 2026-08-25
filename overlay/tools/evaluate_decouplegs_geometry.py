#!/usr/bin/env python3
"""Evaluate DecoupleGS map registration and opacity grounding on nuScenes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import yaml
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import (
    gaussian_set_from_hugsim_checkpoint,
    gaussian_set_from_hugsim_model,
)
from decouplegs.metrics import trajectory_ade
from decouplegs.registration import (
    RegistrationConfig,
    apply_se2,
    bottom_anchors,
    ground_asset_from_pose,
    opacity_accumulated_heights,
    project_trajectory_to_polyline,
    register_trajectory_to_lanes,
    rotation_from_yaw_and_normal,
)
from decouplegs.spatial import BackgroundSpatialIndex
from gaussian_renderer import GaussianModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute trajectory ADE and ground penetration on a released HUGSIM scene"
    )
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("nuscenes_root", type=Path)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--map-radius", type=float, default=100.0)
    parser.add_argument("--map-resolution", type=float, default=0.1)
    parser.add_argument("--heading-weight", type=float, default=2.5)
    parser.add_argument("--asset", type=Path, default=None)
    parser.add_argument("--ground-tolerance", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _front_dynamic_frames(metadata: dict) -> list[dict]:
    frames = [frame for frame in metadata["frames"] if "/CAM_FRONT/" in frame["rgb_path"]]
    if not frames:
        # Some exporters store paths without a leading './'. Avoid matching
        # CAM_FRONT_LEFT/RIGHT by checking path components exactly.
        frames = [
            frame
            for frame in metadata["frames"]
            if "CAM_FRONT" in Path(frame["rgb_path"]).parts
        ]
    if not frames:
        raise ValueError("metadata contains no CAM_FRONT frames")
    return frames


def _select_track(frames: list[dict], requested: str | None) -> tuple[str, list[dict]]:
    counts = Counter(track_id for frame in frames for track_id in frame.get("dynamics", {}))
    if not counts:
        raise ValueError("scene has no dynamic tracks")
    track_id = requested if requested is not None else counts.most_common(1)[0][0]
    selected = [frame for frame in frames if track_id in frame.get("dynamics", {})]
    if len(selected) < 2:
        raise ValueError(f"track {track_id!r} has fewer than two poses")
    return track_id, selected


def _scene_location(nuscenes: NuScenes, scene_name: str) -> str:
    scene_tokens = nuscenes.field2token("scene", "name", scene_name)
    if len(scene_tokens) != 1:
        raise ValueError(f"expected exactly one nuScenes scene named {scene_name!r}")
    scene = nuscenes.get("scene", scene_tokens[0])
    return str(nuscenes.get("log", scene["log_token"])["location"])


def _bfs_route(map_api: NuScenesMap, start: str, end: str, max_hops: int = 32) -> list[str]:
    queue: deque[list[str]] = deque([[start]])
    seen = {start}
    while queue:
        route = queue.popleft()
        if route[-1] == end:
            return route
        if len(route) >= max_hops:
            continue
        for token in map_api.get_outgoing_lane_ids(route[-1]):
            if token not in seen:
                seen.add(token)
                queue.append(route + [token])
    raise ValueError(f"no outgoing lane route connects {start} to {end}")


def _join_route(
    map_api: NuScenesMap,
    tokens: list[str],
    resolution: float,
    trajectory: torch.Tensor,
) -> torch.Tensor:
    discretized = map_api.discretize_lanes(tokens, resolution)
    parts = []
    for index, token in enumerate(tokens):
        points = np.asarray(discretized[token], dtype=np.float64)[:, :2]
        parts.append(points if index == 0 else points[1:])
    route = torch.as_tensor(np.concatenate(parts), dtype=trajectory.dtype)
    first_index = int(torch.linalg.vector_norm(route - trajectory[0], dim=-1).argmin())
    last_index = int(torch.linalg.vector_norm(route - trajectory[-1], dim=-1).argmin())
    if last_index < first_index:
        route = route.flip(0)
        first_index = int(torch.linalg.vector_norm(route - trajectory[0], dim=-1).argmin())
        last_index = int(torch.linalg.vector_norm(route - trajectory[-1], dim=-1).argmin())
    if last_index <= first_index:
        raise ValueError("topological route does not follow the trajectory direction")
    return route[first_index : last_index + 1]


def _nearest_distances(points: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    origin = points.mean(dim=0, keepdim=True)
    return torch.cdist(points - origin, references.to(points) - origin).min(dim=-1).values


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _load_asset(path: Path, device: torch.device):
    if path.suffix == ".dgs":
        return CompactGaussianAsset.load(path).to(device)
    return gaussian_set_from_hugsim_checkpoint(path, map_location=device)


def _load_background(scene_dir: Path, device: torch.device):
    with (scene_dir / "cfg.yaml").open() as stream:
        config = yaml.safe_load(stream) or {}
    degree = int(config.get("model", {}).get("sh_degree", 3))
    model = GaussianModel(degree, affine=bool(config.get("affine", True)))
    state, iteration = torch.load(
        scene_dir / "scene.pth", map_location=device, weights_only=False
    )
    model.restore(state, None)
    return model, gaussian_set_from_hugsim_model(model, include_ground=True), int(iteration)


def _world_anchors(asset, transforms: torch.Tensor, config: RegistrationConfig) -> torch.Tensor:
    anchors = bottom_anchors(asset, config)
    return (
        torch.einsum("tij,aj->tai", transforms[:, :3, :3], anchors)
        + transforms[:, None, :3, 3]
    )


def _penetration_summary(
    anchors: torch.Tensor,
    ground_means: torch.Tensor,
    ground_opacities: torch.Tensor,
    index: BackgroundSpatialIndex,
    config: RegistrationConfig,
    tolerance: float,
) -> dict[str, float | int]:
    flat = anchors.reshape(-1, 3)
    heights, valid = opacity_accumulated_heights(
        flat[:, list(config.horizontal_axes)],
        ground_means,
        ground_opacities,
        config,
        reference_height=flat[:, config.vertical_axis],
        spatial_index=index,
    )
    clearance = (
        flat[:, config.vertical_axis] - heights
    ) * config.vertical_sign
    evaluated = clearance[valid]
    if evaluated.numel() == 0:
        raise ValueError("no valid ground anchor queries")
    return {
        "ground_penetration_rate_strict": float((evaluated < 0).float().mean()),
        "ground_penetration_rate_tolerance": float(
            (evaluated < -tolerance).float().mean()
        ),
        "valid_anchors": int(valid.sum()),
        "total_anchors": int(valid.numel()),
        "clearance_mean_m": float(evaluated.mean()),
        "clearance_min_m": float(evaluated.min()),
        "clearance_max_m": float(evaluated.max()),
    }


def _plane_penetration_summary(
    anchors: torch.Tensor,
    planes: list,
    tolerance: float,
) -> dict[str, float | int]:
    if len(anchors) != len(planes):
        raise ValueError("one fitted ground plane is required per anchor set")
    clearances = torch.cat(
        [plane.signed_distance(frame_anchors) for frame_anchors, plane in zip(anchors, planes)]
    )
    return {
        "ground_penetration_rate_strict": float((clearances < 0).float().mean()),
        "ground_penetration_rate_tolerance": float(
            (clearances < -tolerance).float().mean()
        ),
        "valid_anchors": int(clearances.numel()),
        "total_anchors": int(clearances.numel()),
        "clearance_mean_m": float(clearances.mean()),
        "clearance_min_m": float(clearances.min()),
        "clearance_max_m": float(clearances.max()),
    }


def _grounding_metrics(
    scene_dir: Path,
    asset_path: Path,
    raw_poses: np.ndarray,
    corrected_global: torch.Tensor,
    global_heights: torch.Tensor,
    inverse_origin_pose: torch.Tensor,
    device: torch.device,
    tolerance: float,
) -> dict:
    started = time.perf_counter()
    _, background, iteration = _load_background(scene_dir, device)
    asset = _load_asset(asset_path, device)
    config = RegistrationConfig(
        heading_weight=2.5,
        map_resolution=0.1,
        column_radius=0.35,
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        forward_axis=0,
        up_axis=1,
        up_sign=-1,
    )
    ground_mask = (
        torch.ones(len(background), dtype=torch.bool, device=device)
        if background.semantics is None
        else background.semantics.argmax(dim=-1) <= 1
    )
    ground = background.select(ground_mask)
    index_started = time.perf_counter()
    spatial_index = BackgroundSpatialIndex(ground.means)
    index_seconds = time.perf_counter() - index_started

    raw = torch.as_tensor(raw_poses, dtype=ground.dtype, device=device)
    raw_anchors = _world_anchors(asset, raw, config)
    raw_column_diagnostic = _penetration_summary(
        raw_anchors,
        ground.means,
        ground.opacities,
        spatial_index,
        config,
        tolerance,
    )

    global_xyz = torch.cat((corrected_global, global_heights[:, None]), dim=-1).to(
        device=device, dtype=ground.dtype
    )
    inverse_origin_pose = inverse_origin_pose.to(device=device, dtype=ground.dtype)
    scene_xyz = (
        global_xyz @ inverse_origin_pose[:3, :3].transpose(0, 1)
        + inverse_origin_pose[:3, 3]
    )
    delta = torch.empty_like(corrected_global)
    delta[0] = corrected_global[1] - corrected_global[0]
    delta[-1] = corrected_global[-1] - corrected_global[-2]
    if len(delta) > 2:
        delta[1:-1] = corrected_global[2:] - corrected_global[:-2]
    global_direction = torch.cat(
        (delta, torch.zeros((len(delta), 1), dtype=delta.dtype)), dim=-1
    ).to(device=device, dtype=ground.dtype)
    scene_direction = global_direction @ inverse_origin_pose[:3, :3].transpose(0, 1)
    yaws = torch.atan2(scene_direction[:, 2], scene_direction[:, 0])
    flat_normal = torch.tensor((0.0, -1.0, 0.0), device=device, dtype=ground.dtype)

    initial = raw.clone()
    initial[:, :3, 3] = scene_xyz
    for frame_index, yaw in enumerate(yaws):
        initial[frame_index, :3, :3] = rotation_from_yaw_and_normal(
            yaw, flat_normal, config
        )

    raw_planes = []
    for pose in raw:
        try:
            raw_planes.append(
                ground_asset_from_pose(
                    asset,
                    pose,
                    ground.means,
                    ground.opacities,
                    config,
                    spatial_index=spatial_index,
                ).plane
            )
        except ValueError:
            pass
    raw_plane_anchors = raw_anchors[: len(raw_planes)]
    raw_metrics = _plane_penetration_summary(raw_plane_anchors, raw_planes, tolerance)

    grounded_transforms = []
    grounded_planes = []
    failed = 0
    for pose in initial:
        try:
            result = ground_asset_from_pose(
                asset,
                pose,
                ground.means,
                ground.opacities,
                config,
                spatial_index=spatial_index,
            )
            grounded_transforms.append(result.transform)
            grounded_planes.append(result.plane)
        except ValueError:
            failed += 1
    if not grounded_transforms:
        raise ValueError("opacity grounding failed for every pose")
    grounded = torch.stack(grounded_transforms)
    grounded_anchors = _world_anchors(asset, grounded, config)
    grounded_column_diagnostic = _penetration_summary(
        grounded_anchors,
        ground.means,
        ground.opacities,
        spatial_index,
        config,
        tolerance,
    )
    grounded_metrics = _plane_penetration_summary(
        grounded_anchors,
        grounded_planes,
        tolerance,
    )
    vertical_shift = (
        grounded[:, config.vertical_axis, 3]
        - initial[: len(grounded), config.vertical_axis, 3]
    )
    return {
        "protocol": {
            "hypothesis": "H-GEO-02 semantic-ground gating",
            "column_radius_m": config.column_radius,
            "ground_semantic_rule": "argmax(class) <= 1",
            "penetration_tolerance_m": tolerance,
        },
        "scene_iteration": iteration,
        "background_gaussians": len(background),
        "ground_candidate_gaussians": len(ground),
        "asset_primitives": len(asset),
        "raw_pose": raw_metrics,
        "opacity_grounded": grounded_metrics,
        "column_sample_diagnostic": {
            "note": (
                "Direct per-anchor column samples are noisy; paper GPR is evaluated against "
                "the opacity-accumulated fitted local ground plane."
            ),
            "raw_pose": raw_column_diagnostic,
            "opacity_grounded": grounded_column_diagnostic,
        },
        "failed_grounding_frames": failed,
        "vertical_shift_m": _distribution(vertical_shift),
        "spatial_index_setup_seconds": index_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.map_radius <= 0 or args.map_resolution <= 0:
        raise ValueError("map radius and resolution must be positive")
    metadata = json.loads((args.scene_dir / "meta_data.json").read_text())
    track_id, frames = _select_track(_front_dynamic_frames(metadata), args.track_id)
    raw_poses = np.asarray([frame["dynamics"][track_id] for frame in frames])
    inverse_origin = np.asarray(metadata["inv_pose"], dtype=np.float64)
    origin = np.linalg.inv(inverse_origin)
    scene_centers = raw_poses[:, :3, 3]
    global_centers = scene_centers @ origin[:3, :3].T + origin[:3, 3]
    trajectory = torch.as_tensor(global_centers[:, :2], dtype=torch.float64)

    scene_name = args.scene_name or args.scene_dir.name
    nuscenes = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=False)
    location = _scene_location(nuscenes, scene_name)
    map_api = NuScenesMap(dataroot=str(args.nuscenes_root), map_name=location)
    center = global_centers.mean(axis=0)
    records = map_api.get_records_in_radius(
        float(center[0]),
        float(center[1]),
        args.map_radius,
        ("lane", "lane_connector"),
    )
    lane_tokens = list(records["lane"]) + list(records["lane_connector"])
    discretized = map_api.discretize_lanes(lane_tokens, args.map_resolution)
    all_lane_points = torch.as_tensor(
        np.concatenate(
            [np.asarray(discretized[token], dtype=np.float64)[:, :2] for token in lane_tokens]
        ),
        dtype=torch.float64,
    )
    start_lane = map_api.get_closest_lane(*global_centers[0, :2], radius=10.0)
    end_lane = map_api.get_closest_lane(*global_centers[-1, :2], radius=10.0)
    route_tokens = _bfs_route(map_api, start_lane, end_lane)
    route = _join_route(map_api, route_tokens, args.map_resolution, trajectory)

    registration = register_trajectory_to_lanes(
        trajectory,
        [route],
        RegistrationConfig(
            heading_weight=args.heading_weight,
            map_resolution=None,
        ),
    )
    se2_trajectory = apply_se2(trajectory, registration.transform)
    projection = project_trajectory_to_polyline(se2_trajectory, route)
    projected_trajectory = projection.points

    trajectory_results = {}
    for name, points in (
        ("raw_no_registration", trajectory),
        ("paper_literal_se2", se2_trajectory),
        ("rd_h_geo_01_frenet_residual", projected_trajectory),
    ):
        distances = _nearest_distances(points, all_lane_points)
        trajectory_results[name] = {
            "trajectory_ade_m": float(trajectory_ade(points, all_lane_points)),
            **{f"distance_{key}_m": value for key, value in _distribution(distances).items()},
        }

    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_geometry_registration",
        "scene": str(args.scene_dir.resolve()),
        "nuscenes_root": str(args.nuscenes_root.resolve()),
        "nuscenes_version": args.version,
        "scene_name": scene_name,
        "map_location": location,
        "track_id": track_id,
        "trajectory_samples": len(trajectory),
        "map_radius_m": args.map_radius,
        "map_resolution_m": args.map_resolution,
        "nearby_lane_records": len(lane_tokens),
        "route_tokens": route_tokens,
        "route_points": len(route),
        "protocol": {
            "paper_literal": "topological route + constrained DTW + global orthogonal Procrustes SE(2)",
            "rd_h_geo_01": (
                "topological route stitching followed by monotone continuous Frenet residual projection"
            ),
            "numeric_precision": "float64 centred cdist for global map coordinates",
        },
        "registration": {
            "heading_weight": args.heading_weight,
            "normalized_dtw_cost": registration.normalized_cost,
            "se2_transform": registration.transform.tolist(),
            "mean_frenet_residual_m": float(
                torch.linalg.vector_norm(projected_trajectory - se2_trajectory, dim=-1).mean()
            ),
        },
        "trajectory": trajectory_results,
    }
    if args.asset is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for opacity grounding but is unavailable")
        payload["grounding"] = _grounding_metrics(
            args.scene_dir,
            args.asset,
            raw_poses,
            projected_trajectory,
            torch.as_tensor(global_centers[:, 2], dtype=torch.float64),
            torch.as_tensor(inverse_origin, dtype=torch.float64, device=device),
            device,
            args.ground_tolerance,
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
