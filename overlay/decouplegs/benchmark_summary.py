"""Strict aggregation for matched multi-scene DecoupleGS benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ALL_IMAGE_METRICS = (
    "psnr_all_channel_mse_db",
    "ssim_all",
    "lpips_all_alex",
)
VEHICLE_IMAGE_METRICS = (
    "psnr_vehicle_channel_mse_db",
    "psnr_vehicle_pixel_l2_db",
    "peak_intensity_error",
    "peak_angular_error_deg",
)


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _weighted_mean(rows: Iterable[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in rows:
        if value is None or weight <= 0:
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    return None if denominator == 0 else numerator / denominator


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _fidelity(metrics: dict[str, Any]) -> dict[str, Any]:
    mean = metrics["aggregate"]["mean_per_image"]
    vehicle_images = int(metrics["aggregate"]["vehicle_mask_images"])
    return {
        "pair_count": int(metrics["pair_count"]),
        "vehicle_mask_images": vehicle_images,
        **{key: mean.get(key) for key in (*ALL_IMAGE_METRICS, *VEHICLE_IMAGE_METRICS)},
    }


def _rendering(pairs: dict[str, Any]) -> dict[str, Any]:
    runtime = pairs["runtime"]
    pair_count = int(pairs["pair_count"])
    milliseconds = float(runtime["mean_ms_per_camera"])
    return {
        "pair_count": pair_count,
        "mean_ms_per_camera": milliseconds,
        "camera_fps": 1000.0 / milliseconds,
        "peak_allocated_mib": float(runtime["peak_allocated_mib"]),
        "gpu": str(runtime["gpu"]),
    }


def _behavior(paired: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for row in paired["variants"]:
        variants[str(row["label"])] = {
            "frames": int(row["frames"]),
            "mADE_to_reference_plan_m": float(row["mADE_to_reference_plan_m"]),
            "mFDE_to_reference_plan_m": float(row["mFDE_to_reference_plan_m"]),
            "mADE_to_logged_gt_m": float(row["mADE_to_logged_gt_m"]),
            "minTTC_paper_literal_center_s": _optional_float(
                row["minTTC_paper_literal_center_s"]
            ),
        }
    reference = paired["reference"]
    return {
        "reference": {
            "mADE_to_logged_gt_m": float(reference["mADE_to_logged_gt_m"]),
            "minTTC_paper_literal_center_s": _optional_float(
                reference["minTTC_paper_literal_center_s"]
            ),
        },
        "variants": variants,
    }


def load_scene_result(name: str, root: Path) -> dict[str, Any]:
    """Load one scene using the strict keyframe directory contract."""

    raw_pairs = _load(root / "raw/pairs.json")
    compact_pairs = _load(root / "compact-base/pairs.json")
    if int(raw_pairs["pair_count"]) != int(compact_pairs["pair_count"]):
        raise ValueError(f"{name}: raw and compact pair counts differ")
    if raw_pairs.get("nuscenes_manifest") != compact_pairs.get("nuscenes_manifest"):
        raise ValueError(f"{name}: raw and compact manifests differ")
    return {
        "scene": name,
        "root": str(root.resolve()),
        "frame_protocol": raw_pairs["frame_protocol"],
        "rendering": {
            "raw_3dgs": _rendering(raw_pairs),
            "compact_base": _rendering(compact_pairs),
        },
        "fidelity": {
            "raw_3dgs": _fidelity(_load(root / "raw/metrics-sam-vit-h.json")),
            "compact_base": _fidelity(
                _load(root / "compact-base/metrics-sam-vit-h.json")
            ),
        },
        "behavior": {
            planner: _behavior(_load(root / f"{planner}-paired-behavior.json"))
            for planner in ("uniad", "vad")
        },
    }


def _aggregate_variant_rendering(
    scenes: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    rows = [scene["rendering"][variant] for scene in scenes]
    total_pairs = sum(row["pair_count"] for row in rows)
    total_seconds = sum(
        row["pair_count"] * row["mean_ms_per_camera"] / 1000.0 for row in rows
    )
    return {
        "pair_count": total_pairs,
        "mean_ms_per_camera": total_seconds * 1000.0 / total_pairs,
        "camera_fps": total_pairs / total_seconds,
        "peak_allocated_mib_max": max(row["peak_allocated_mib"] for row in rows),
        "gpu_set": sorted({row["gpu"] for row in rows}),
    }


def _aggregate_variant_fidelity(
    scenes: list[dict[str, Any]], variant: str
) -> dict[str, Any]:
    rows = [scene["fidelity"][variant] for scene in scenes]
    result = {
        "pair_count": sum(row["pair_count"] for row in rows),
        "vehicle_mask_images": sum(row["vehicle_mask_images"] for row in rows),
    }
    for key in ALL_IMAGE_METRICS:
        result[key] = _weighted_mean((row.get(key), row["pair_count"]) for row in rows)
    for key in VEHICLE_IMAGE_METRICS:
        result[key] = _weighted_mean(
            (row.get(key), row["vehicle_mask_images"]) for row in rows
        )
    return result


def _aggregate_behavior(
    scenes: list[dict[str, Any]], planner: str
) -> dict[str, Any]:
    planner_rows = [scene["behavior"][planner] for scene in scenes]
    variant_names = set(planner_rows[0]["variants"])
    if any(set(row["variants"]) != variant_names for row in planner_rows[1:]):
        raise ValueError(f"{planner}: variant sets differ across scenes")
    reference_frames = [
        max(value["frames"] for value in row["variants"].values())
        for row in planner_rows
    ]
    result = {
        "reference": {
            key: _weighted_mean(
                (row["reference"][key], frames)
                for row, frames in zip(planner_rows, reference_frames, strict=True)
            )
            for key in ("mADE_to_logged_gt_m", "minTTC_paper_literal_center_s")
        },
        "variants": {},
    }
    for variant in sorted(variant_names):
        rows = [row["variants"][variant] for row in planner_rows]
        result["variants"][variant] = {
            "frames": sum(row["frames"] for row in rows),
            **{
                key: _weighted_mean((row[key], row["frames"]) for row in rows)
                for key in (
                    "mADE_to_reference_plan_m",
                    "mFDE_to_reference_plan_m",
                    "mADE_to_logged_gt_m",
                    "minTTC_paper_literal_center_s",
                )
            },
        }
    return result


def aggregate_scene_results(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    if not scenes:
        raise ValueError("at least one scene is required")
    rendering = {
        variant: _aggregate_variant_rendering(scenes, variant)
        for variant in ("raw_3dgs", "compact_base")
    }
    rendering["compact_vs_raw_speedup"] = (
        rendering["compact_base"]["camera_fps"]
        / rendering["raw_3dgs"]["camera_fps"]
    )
    return {
        "schema_version": 1,
        "benchmark": "decouplegs_strict_multiscene_open_loop",
        "aggregation": {
            "rendering": "total rendered camera views / summed timed CUDA wall time",
            "all_image_fidelity": "pair-count weighted mean of per-image metrics",
            "vehicle_fidelity": "vehicle-mask-image-count weighted mean",
            "behavior": "planner-frame-count weighted mean",
            "warning": "logged-GT mADE is scene-distribution dependent; paired mADE isolates visual-domain plan drift",
        },
        "scene_count": len(scenes),
        "scene_names": [scene["scene"] for scene in scenes],
        "aggregate": {
            "rendering": rendering,
            "fidelity": {
                variant: _aggregate_variant_fidelity(scenes, variant)
                for variant in ("raw_3dgs", "compact_base")
            },
            "behavior": {
                planner: _aggregate_behavior(scenes, planner)
                for planner in ("uniad", "vad")
            },
        },
        "scenes": scenes,
    }
