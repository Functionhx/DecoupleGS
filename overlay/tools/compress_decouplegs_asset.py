#!/usr/bin/env python3
"""Prune and VQ-compress a canonical HUGSIM/3DRealCar Gaussian asset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompressionConfig, compress_asset
from decouplegs.hugsim import gaussian_set_from_hugsim_checkpoint


def _load_terms(path: Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, torch.Tensor):
        return {"visibility": state}
    if not isinstance(state, dict):
        raise TypeError("importance statistics must be a tensor or mapping")
    allowed = {"visibility", "color_contrast", "texture_entropy"}
    output = {key: value for key, value in state.items() if key in allowed}
    if not output:
        raise ValueError(f"{path} contains none of {sorted(allowed)}")
    if not all(isinstance(value, torch.Tensor) for value in output.values()):
        raise TypeError("all importance statistics must be tensors")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DecoupleGS semantic-aware pruning and covariance/SH EMA-VQ",
    )
    parser.add_argument("input", type=Path, help="HUGSIM canonical-asset gs.pth")
    parser.add_argument("output", type=Path, help="Output .dgs compact asset")
    parser.add_argument(
        "--importance-stats",
        type=Path,
        default=None,
        help="Optional .pt tensor/dict with visibility, color_contrast, texture_entropy",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prune-threshold", type=float, default=0.005)
    parser.add_argument("--visibility-weight", type=float, default=0.5)
    parser.add_argument("--color-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.2)
    parser.add_argument("--color-codebook", type=int, default=1024)
    parser.add_argument("--shape-codebook", type=int, default=512)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--ema-iterations", type=int, default=5000)
    parser.add_argument("--ema-batch-size", type=int, default=16384)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument(
        "--visibility-gate-saliency",
        action="store_true",
        help="R&D H-COMP-01: treat D/H as expected visible contributions",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sh-frame",
        choices=("local", "world"),
        default="local",
        help="Coordinate frame in which the source SH lobes are stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    asset = gaussian_set_from_hugsim_checkpoint(args.input).to(device=device)
    # H-APP-01: the paper's canonical 3DRealCar assets use local-frame SH,
    # while native HUGSIM scene tracks preserve world-anchored SH.  Persist
    # the representation contract so mixed asset libraries transform safely.
    asset.metadata["sh_frame"] = args.sh_frame
    terms = {
        key: value.to(device=device, dtype=asset.dtype)
        for key, value in _load_terms(args.importance_stats).items()
    }
    config = CompressionConfig(
        prune_threshold=args.prune_threshold,
        weight_visibility=args.visibility_weight,
        weight_color=args.color_weight,
        weight_entropy=args.entropy_weight,
        color_codebook_size=args.color_codebook,
        shape_codebook_size=args.shape_codebook,
        ema_momentum=args.ema_momentum,
        ema_iterations=args.ema_iterations,
        ema_batch_size=args.ema_batch_size,
        kmeans_iterations=args.kmeans_iterations,
        visibility_gate_saliency=args.visibility_gate_saliency,
        seed=args.seed,
    )
    compact, report = compress_asset(asset, config, **terms)
    compact.save(args.output)
    payload = asdict(report)
    payload.update(
        retained_fraction=report.retained_fraction,
        compression_ratio=report.compression_ratio,
        output=str(args.output.resolve()),
        sh_frame=args.sh_frame,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
