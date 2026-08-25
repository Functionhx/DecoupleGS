#!/usr/bin/env python3
"""Evaluate DecoupleGS rendering fidelity with explicit paper metric variants."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.metrics import (
    channel_mse_psnr,
    masked_channel_mse_psnr,
    masked_psnr,
    peak_angular_error,
    peak_intensity_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="JSON containing a list or {'pairs': list} of pred/target/mask paths",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mask-index",
        type=Path,
        help="Optional JSON {'masks': {pair_id: path}} overriding pair masks",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help="Skip full-frame PSNR/SSIM/LPIPS and compute only masked paper metrics",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the paper metric summary")
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=1,
        help="Minimum nonzero mask area required for paper metrics (default: 1)",
    )
    parser.add_argument(
        "--pae-support-threshold",
        type=float,
        default=1.0 / 255.0,
        help=(
            "RGB L2 norm threshold for the separately reported supported-PAE "
            "sensitivity result; literal paper PAE always retains all mask pixels"
        ),
    )
    return parser.parse_args()


def _resolve(path: str, base: Path) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else base / result


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).to(device)


def load_mask(path: Path, expected_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        value = np.load(path)
    else:
        with Image.open(path) as image:
            value = np.asarray(image.convert("L"))
    value = np.asarray(value).squeeze()
    if value.shape != expected_shape:
        raise ValueError(f"mask {path} has shape {value.shape}, expected {expected_shape}")
    return torch.from_numpy((value > 0).copy()).to(device)


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _finite_summary(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _global_psnr(sum_squared_error: float, count: int) -> float | None:
    if count <= 0:
        return None
    mse = max(sum_squared_error / count, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def supported_peak_angular_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> tuple[float | None, int, float]:
    """Audit the paper's undefined zero/near-zero RGB-vector corner case."""

    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    support = mask & (pred_norm >= threshold) & (target_norm >= threshold)
    mask_pixels = int(mask.sum())
    supported_pixels = int(support.sum())
    unsupported_fraction = (
        0.0 if mask_pixels == 0 else 1.0 - supported_pixels / mask_pixels
    )
    if not supported_pixels:
        return None, 0, unsupported_fraction
    return (
        float(peak_angular_error(pred, target, support)),
        supported_pixels,
        unsupported_fraction,
    )


def masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pae_support_threshold: float,
) -> dict[str, float | int | None]:
    squared_error = (pred - target).square()[mask]
    supported_pae, supported_pixels, unsupported_fraction = (
        supported_peak_angular_error(
            pred, target, mask, pae_support_threshold
        )
    )
    return {
        "mask_pixels": int(mask.sum()),
        "masked_rgb_sse": float(squared_error.sum()),
        "psnr_vehicle_channel_mse_db": float(
            masked_channel_mse_psnr(pred, target, mask)
        ),
        "psnr_vehicle_pixel_l2_db": float(masked_psnr(pred, target, mask)),
        "peak_intensity_error": float(peak_intensity_error(pred, target, mask)),
        "peak_angular_error_deg": float(peak_angular_error(pred, target, mask)),
        "peak_angular_error_supported_deg": supported_pae,
        "pae_supported_pixels": supported_pixels,
        "pae_unsupported_fraction": unsupported_fraction,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = (
        "psnr_vehicle_channel_mse_db",
        "psnr_vehicle_pixel_l2_db",
        "peak_intensity_error",
        "peak_angular_error_deg",
        "peak_angular_error_supported_deg",
        "pae_unsupported_fraction",
    )
    valid = [row for row in rows if row.get("psnr_vehicle_pixel_l2_db") is not None]
    pixel_count = sum(int(row["mask_pixels"]) for row in valid)
    rgb_sse = sum(float(row["masked_rgb_sse"]) for row in valid)
    return {
        "images": len(rows),
        "masked_images": len(valid),
        "mask_pixels": pixel_count,
        "per_image": {
            key: _finite_summary([row.get(key) for row in valid])
            for key in metric_keys
        },
        "pooled_mask": {
            "psnr_vehicle_channel_mse_db": _global_psnr(rgb_sse, pixel_count * 3),
            "psnr_vehicle_pixel_l2_db": _global_psnr(rgb_sse, pixel_count),
        },
        "dataset_peaks": {
            key: (
                max(values)
                if (values := [
                    float(row[key])
                    for row in valid
                    if row.get(key) is not None
                ])
                else None
            )
            for key in (
                "peak_intensity_error",
                "peak_angular_error_deg",
                "peak_angular_error_supported_deg",
            )
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    with args.pairs.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs JSON must contain a non-empty list")

    device = torch.device(args.device)
    if args.min_mask_pixels <= 0:
        raise ValueError("--min-mask-pixels must be positive")
    if args.pae_support_threshold < 0:
        raise ValueError("--pae-support-threshold must be non-negative")
    mask_override = None
    instance_mask_override: dict[str, dict[str, str]] | None = None
    mask_protocol = None
    mask_base = base = args.pairs.parent
    if args.mask_index is not None:
        mask_payload = json.loads(args.mask_index.read_text(encoding="utf-8"))
        mask_override = mask_payload["masks"]
        instance_mask_override = mask_payload.get("instance_masks")
        mask_protocol = mask_payload.get("protocol")
        mask_base = args.mask_index.parent
    ssim_metric = (
        None
        if args.paper_only
        else StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    )
    lpips_metric = None
    if not args.no_lpips and not args.paper_only:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    all_sse = 0.0
    all_channel_count = 0
    vehicle_sse = 0.0
    vehicle_channel_count = 0

    for index, pair in enumerate(pairs):
        pred_path = _resolve(str(pair["pred"]), base)
        target_path = _resolve(str(pair["target"]), base)
        mask_value = (
            mask_override.get(str(pair.get("id", index)))
            if mask_override is not None
            else pair.get("mask")
        )
        mask_path = _resolve(str(mask_value), mask_base if mask_override is not None else base) if mask_value else None
        pred = load_rgb(pred_path, device)
        target = load_rgb(target_path, device)
        if pred.shape != target.shape:
            raise ValueError(
                f"image shape mismatch for pair {index}: {tuple(pred.shape)} != {tuple(target.shape)}"
            )
        if mask_path is None:
            mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = load_mask(mask_path, tuple(pred.shape[:2]), device)

        pred_nchw = pred.permute(2, 0, 1).unsqueeze(0)
        target_nchw = target.permute(2, 0, 1).unsqueeze(0)
        squared_error = (pred - target).square()
        masked_error = squared_error[mask]
        all_sse += float(squared_error.sum())
        all_channel_count += squared_error.numel()

        mask_pixels = int(mask.sum())
        if mask_pixels >= args.min_mask_pixels:
            vehicle_sse += float(masked_error.sum())
            vehicle_channel_count += masked_error.numel()
        row = {
            "id": pair.get("id", str(index)),
            "split": pair.get("split"),
            "camera": pair.get("camera"),
            "scene": pair.get("scene", payload.get("scene_id") if isinstance(payload, dict) else None),
            "track_ids": pair.get("track_ids", []),
            "pred": str(pred_path),
            "target": str(target_path),
            "mask": None if mask_path is None else str(mask_path),
            "height": int(pred.shape[0]),
            "width": int(pred.shape[1]),
            "mask_pixels": mask_pixels,
            "psnr_all_channel_mse_db": (
                None if args.paper_only else float(channel_mse_psnr(pred, target))
            ),
            "ssim_all": (
                None
                if ssim_metric is None
                else float(ssim_metric(pred_nchw, target_nchw))
            ),
        }
        if mask_pixels >= args.min_mask_pixels:
            row.update(
                masked_metrics(
                    pred,
                    target,
                    mask,
                    pae_support_threshold=args.pae_support_threshold,
                )
            )
        else:
            row.update(
                masked_rgb_sse=None,
                psnr_vehicle_channel_mse_db=None,
                psnr_vehicle_pixel_l2_db=None,
                peak_intensity_error=None,
                peak_angular_error_deg=None,
                peak_angular_error_supported_deg=None,
                pae_supported_pixels=0,
                pae_unsupported_fraction=None,
            )
        if lpips_metric is not None:
            row["lpips_all_alex"] = float(lpips_metric(pred_nchw, target_nchw))
        rows.append(row)

        if instance_mask_override is not None:
            for track_id, instance_mask_value in sorted(
                instance_mask_override.get(str(pair.get("id", index)), {}).items()
            ):
                instance_path = _resolve(str(instance_mask_value), mask_base)
                instance_mask = load_mask(
                    instance_path, tuple(pred.shape[:2]), device
                )
                instance_pixels = int(instance_mask.sum())
                instance_row: dict[str, Any] = {
                    "id": pair.get("id", str(index)),
                    "scene": row["scene"],
                    "camera": row["camera"],
                    "track_id": track_id,
                    "mask": str(instance_path),
                    "mask_pixels": instance_pixels,
                }
                if instance_pixels >= args.min_mask_pixels:
                    instance_row.update(
                        masked_metrics(
                            pred,
                            target,
                            instance_mask,
                            pae_support_threshold=args.pae_support_threshold,
                        )
                    )
                else:
                    instance_row.update(
                        masked_rgb_sse=None,
                        psnr_vehicle_channel_mse_db=None,
                        psnr_vehicle_pixel_l2_db=None,
                        peak_intensity_error=None,
                        peak_angular_error_deg=None,
                        peak_angular_error_supported_deg=None,
                        pae_supported_pixels=0,
                        pae_unsupported_fraction=None,
                    )
                instance_rows.append(instance_row)

    mean_keys = [
        "psnr_vehicle_channel_mse_db",
        "psnr_vehicle_pixel_l2_db",
        "peak_intensity_error",
        "peak_angular_error_deg",
        "peak_angular_error_supported_deg",
        "pae_unsupported_fraction",
    ]
    if not args.paper_only:
        mean_keys.extend(("psnr_all_channel_mse_db", "ssim_all"))
    if lpips_metric is not None:
        mean_keys.append("lpips_all_alex")
    mean_per_image = {
        key: _finite_mean(
            [None if row[key] is None else float(row[key]) for row in rows]
        )
        for key in mean_keys
    }
    median_per_image = {
        key: _finite_summary(
            [None if row[key] is None else float(row[key]) for row in rows]
        )["median"]
        for key in mean_keys
    }

    per_scene: dict[str, Any] = {}
    for scene_id in sorted({str(row["scene"]) for row in rows if row.get("scene")}):
        per_scene[scene_id] = aggregate_rows(
            [row for row in rows if row.get("scene") == scene_id]
        )
    per_track: dict[str, Any] = {}
    for track_id in sorted({str(row["track_id"]) for row in instance_rows}):
        per_track[track_id] = aggregate_rows(
            [row for row in instance_rows if row["track_id"] == track_id]
        )

    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {
            "decode": "Pillow RGB uint8 scaled to [0,1]",
            "paper_only": args.paper_only,
            "aggregation_primary": "arithmetic mean of per-image metrics",
            "aggregation_disclosure": (
                "paper does not disclose cross-image aggregation; arithmetic mean, "
                "median, pooled masked PSNR, and dataset peaks are all emitted"
            ),
            "psnr_all": "MSE averaged over pixels and RGB channels",
            "psnr_vehicle_channel_mse": "masked MSE averaged over pixels and RGB channels",
            "psnr_vehicle_pixel_l2": "supplement equation: masked mean of per-pixel squared RGB L2",
            "psnr_variant_offset_db": 10.0 * math.log10(3.0),
            "ssim": "torchmetrics StructuralSimilarityIndexMeasure(data_range=1.0)",
            "lpips": None if lpips_metric is None else "torchmetrics LPIPS AlexNet, normalize=True",
            "pie_pae_aggregation": "mean of per-image masked peaks; dataset maximum also reported",
            "pae_literal_zero_norm": "RGB norm product clamped to 1e-12",
            "pae_supported_sensitivity": (
                "exclude pixels where either RGB norm is below "
                f"{args.pae_support_threshold:.12g}"
            ),
            "minimum_mask_pixels": args.min_mask_pixels,
            "mask_index": None if args.mask_index is None else str(args.mask_index.resolve()),
            "mask_protocol": mask_protocol,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "pair_count": len(rows),
        "aggregate": {
            "mean_per_image": mean_per_image,
            "median_per_image": median_per_image,
            "global_channel_mse": {
                "psnr_all_channel_mse_db": _global_psnr(all_sse, all_channel_count),
                "psnr_vehicle_channel_mse_db": _global_psnr(
                    vehicle_sse, vehicle_channel_count
                ),
                "psnr_vehicle_pixel_l2_db": _global_psnr(
                    vehicle_sse, vehicle_channel_count // 3
                ),
            },
            "dataset_peaks": {
                key: (
                    max(values)
                    if (values := [row[key] for row in rows if row[key] is not None])
                    else None
                )
                for key in (
                    "peak_intensity_error",
                    "peak_angular_error_deg",
                    "peak_angular_error_supported_deg",
                )
            },
            "vehicle_mask_images": sum(row["mask_pixels"] > 0 for row in rows),
            "vehicle_metric_images": sum(
                row["mask_pixels"] >= args.min_mask_pixels for row in rows
            ),
            "paper_mask_metrics": aggregate_rows(rows),
            "per_scene": per_scene,
            "per_track": per_track,
        },
        "pairs": rows,
        "instances": instance_rows,
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if args.quiet:
        mean = result["aggregate"]["mean_per_image"]
        print(
            json.dumps(
                {
                    "pair_count": result["pair_count"],
                    "vehicle_metric_images": result["aggregate"]["vehicle_metric_images"],
                    "masked_psnr_db": mean["psnr_vehicle_pixel_l2_db"],
                    "pie": mean["peak_intensity_error"],
                    "pae_deg": mean["peak_angular_error_deg"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
