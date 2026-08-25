#!/usr/bin/env python3
"""Estimate DecoupleGS training-view visibility for released canonical assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.hugsim import gaussian_set_from_hugsim_checkpoint
from decouplegs.visibility import OrbitVisibilityConfig, estimate_opacity_contribution_visibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover per-Gaussian expected opacity contribution from proxy orbit views",
    )
    parser.add_argument("asset", type=Path, help="HUGSIM 3DRealCar gs.pth")
    parser.add_argument("output", type=Path, help="Output .pt importance-statistics sidecar")
    parser.add_argument("--azimuth-views", type=int, default=24)
    parser.add_argument("--elevations", type=float, nargs="+", default=(0.0, 10.0, 20.0))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--axis-convention",
        choices=("3drealcar", "hugsim_scene"),
        default="3drealcar",
        help=(
            "Canonical asset axes. 3drealcar uses X-forward/-Y-up; native "
            "HUGSIM scene tracks use Y-forward/+Z-up."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visibility estimation")
    if args.axis_convention == "3drealcar":
        vertical_axis, vertical_sign, horizontal_axes = 1, -1, (0, 2)
    else:
        vertical_axis, vertical_sign, horizontal_axes = 2, 1, (0, 1)
    config = OrbitVisibilityConfig(
        azimuth_views=args.azimuth_views,
        elevations_degrees=tuple(args.elevations),
        image_width=args.resolution,
        image_height=args.resolution,
        horizontal_fov_degrees=args.fov,
        vertical_axis=vertical_axis,
        vertical_sign=vertical_sign,
        horizontal_axes=horizontal_axes,
        batch_size=args.batch_size,
        packed=args.packed,
    )
    asset = gaussian_set_from_hugsim_checkpoint(args.asset).to(args.device)
    torch.cuda.reset_peak_memory_stats()
    visibility = estimate_opacity_contribution_visibility(asset, config)
    quantile_points = visibility.new_tensor((0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0))
    quantiles = torch.quantile(visibility, quantile_points)
    payload = {
        "visibility": visibility.cpu(),
        "estimator": "orbit_autodiff_opacity_contribution_v1",
        "axis_convention": args.axis_convention,
        "config": config.as_dict(),
        "asset": str(args.asset.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "asset_primitives": len(asset),
                "views": config.azimuth_views * len(config.elevations_degrees),
                "visibility_nonzero": int((visibility > 0).sum()),
                "visibility_at_least_0_005": int((visibility >= 0.005).sum()),
                "visibility_quantiles": dict(zip(("min", "p01", "p10", "p50", "p90", "p99", "max"), quantiles.tolist())),
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
