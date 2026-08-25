#!/usr/bin/env python3
"""Fit the DecoupleGS affine SH relighting operator by closed-form OLS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.relighting import RelightingCalibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OLS calibration from synthetic HDRI renders")
    parser.add_argument(
        "samples",
        type=Path,
        help=".pt mapping: descriptors [N,27], canonical [M,C,3] or [N,M,C,3], targets [N,M,C,3]",
    )
    parser.add_argument("output", type=Path, help="Output relighting.pt")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--ridge-prior", choices=("zero", "identity"), default="zero")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = torch.load(args.samples, map_location="cpu", weights_only=True)
    required = {"descriptors", "canonical", "targets"}
    if not isinstance(samples, dict) or not required.issubset(samples):
        raise ValueError(f"sample file must contain {sorted(required)}")
    descriptors = samples["descriptors"]
    canonical = samples["canonical"]
    targets = samples["targets"]
    if canonical.ndim not in (3, 4) or targets.ndim != 4:
        raise ValueError("canonical/targets must use SH layouts [M,C,3] or [N,M,C,3]")
    canonical_shape = canonical.shape
    target_shape = targets.shape
    canonical_flat = canonical.reshape(canonical_shape[:-2] + (-1,))
    target_flat = targets.reshape(target_shape[:-2] + (-1,))
    calibration = RelightingCalibration.fit_ols(
        descriptors,
        canonical_flat,
        target_flat,
        ridge=args.ridge,
        ridge_prior=args.ridge_prior,
    )
    calibration.metadata.update({
        "source": str(args.samples.resolve()),
        "sh_coefficients": target_shape[-2],
        "channels": target_shape[-1],
    })
    calibration.save(args.output)
    print(json.dumps({
        "attribute_dim": calibration.attribute_dim,
        "descriptor_dim": calibration.descriptor_dim,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
