#!/usr/bin/env python3
"""Measure compact-asset fidelity against its raw 3DGS over proxy views."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import gaussian_set_from_hugsim_checkpoint
from decouplegs.metrics import channel_mse_psnr, masked_channel_mse_psnr
from decouplegs.visibility import OrbitVisibilityConfig, orbit_view_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("compact", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth-views", type=int, default=24)
    parser.add_argument("--elevations", type=float, nargs="+", default=(0.0, 10.0, 20.0))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--mask-alpha", type=float, default=0.01)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def render_views(gaussians, viewmats, intrinsics, width: int, height: int, background: torch.Tensor):
    from gsplat.rendering import rasterization

    images, alphas = [], []
    for viewmat, intrinsic in zip(viewmats, intrinsics, strict=True):
        rendered, alpha, _ = rasterization(
            means=gaussians.means,
            quats=gaussians.quats,
            scales=gaussians.scales,
            opacities=gaussians.opacities,
            colors=gaussians.sh,
            viewmats=viewmat[None],
            Ks=intrinsic[None],
            width=width,
            height=height,
            sh_degree=gaussians.sh_degree,
            render_mode="RGB",
            near_plane=0.01,
            far_plane=100.0,
            packed=True,
            backgrounds=background[None],
        )
        # Sensor images are quantized from display-range RGB. gsplat can
        # overshoot that range slightly through SH evaluation, which also
        # violates LPIPS' normalized-input contract.
        images.append(rendered[0].detach().clamp(0.0, 1.0).cpu())
        alphas.append(alpha[0, ..., 0].detach().clamp(0.0, 1.0).cpu())
    return torch.stack(images), torch.stack(alphas)


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return statistics.fmean(values) if values else math.nan


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("asset compression evaluation requires CUDA")
    if not 0 <= args.mask_alpha <= 1:
        raise ValueError("--mask-alpha must be in [0,1]")
    raw = gaussian_set_from_hugsim_checkpoint(args.raw).to("cuda")
    orbit = OrbitVisibilityConfig(
        azimuth_views=args.azimuth_views,
        elevations_degrees=tuple(args.elevations),
        image_width=args.width,
        image_height=args.height,
    )
    viewmats, intrinsics = orbit_view_matrices(raw.physical_bounds, orbit)
    background = torch.ones(3, device="cuda") if args.background == "white" else torch.zeros(3, device="cuda")
    with torch.inference_mode():
        reference, reference_alpha = render_views(
            raw, viewmats, intrinsics, args.width, args.height, background
        )
    masks = reference_alpha >= args.mask_alpha
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips = None
    if not args.no_lpips:
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to("cuda")

    rows = []
    for path in args.compact:
        compact = CompactGaussianAsset.load(path)
        decoded = compact.decode(device="cuda")
        with torch.inference_mode():
            predicted, predicted_alpha = render_views(
                decoded, viewmats, intrinsics, args.width, args.height, background
            )
        per_view = []
        for index, (prediction_cpu, target_cpu, mask_cpu) in enumerate(
            zip(predicted, reference, masks, strict=True)
        ):
            prediction = prediction_cpu.to("cuda")
            target = target_cpu.to("cuda")
            mask = mask_cpu.to("cuda")
            prediction_nchw = prediction.permute(2, 0, 1)[None]
            target_nchw = target.permute(2, 0, 1)[None]
            metrics = {
                "view": index,
                "psnr_all_db": float(channel_mse_psnr(prediction, target)),
                "psnr_vehicle_db": float(
                    masked_channel_mse_psnr(prediction, target, mask)
                ),
                "ssim": float(ssim(prediction_nchw, target_nchw)),
                "alpha_psnr_db": float(
                    channel_mse_psnr(predicted_alpha[index], reference_alpha[index])
                ),
            }
            if lpips is not None:
                metrics["lpips_alex"] = float(lpips(prediction_nchw, target_nchw))
            per_view.append(metrics)
        keys = ["psnr_all_db", "psnr_vehicle_db", "ssim", "alpha_psnr_db"]
        if lpips is not None:
            keys.append("lpips_alex")
        rows.append(
            {
                "asset": str(path.resolve()),
                "primitives": len(compact),
                "retained_fraction": len(compact) / len(raw),
                "compact_bytes": compact.memory_bytes,
                "aggregate": {
                    key: finite_mean([float(view[key]) for view in per_view])
                    for key in keys
                },
                "per_view": per_view,
            }
        )
        del decoded, predicted, predicted_alpha
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "benchmark": "decouplegs_asset_compression_proxy_fidelity",
        "raw": str(args.raw.resolve()),
        "raw_primitives": len(raw),
        "protocol": {
            "views": orbit.as_dict(),
            "background": args.background,
            "vehicle_mask": f"raw alpha >= {args.mask_alpha}",
            "aggregation": "arithmetic mean of per-view metrics",
            "reference": "raw 3DRealCar gs.pth rendered with gsplat",
            "candidate": "decoded compact asset rendered with identical gsplat path",
            "lpips": None if lpips is None else "torchmetrics AlexNet normalize=True",
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: row[key] for key in ("asset", "primitives", "retained_fraction", "aggregate")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
