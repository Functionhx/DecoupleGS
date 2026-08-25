#!/usr/bin/env python3
"""Run DecoupleGS DTW -> SE(2) Orthogonal Procrustes map registration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.registration import (
    RegistrationConfig,
    apply_se2,
    register_trajectory_to_lanes,
)


def _load_lanes(archive: np.lib.npyio.NpzFile) -> list[torch.Tensor]:
    if "lanes" in archive:
        lanes = archive["lanes"]
        if lanes.dtype == object:
            return [torch.as_tensor(lane, dtype=torch.float32) for lane in lanes.tolist()]
        if lanes.ndim == 3:
            return [torch.as_tensor(lane, dtype=torch.float32) for lane in lanes]
    keys = sorted(key for key in archive.files if key.startswith("lane_"))
    if not keys:
        raise ValueError("lane archive needs `lanes` or one or more `lane_*` arrays")
    return [torch.as_tensor(archive[key], dtype=torch.float32) for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map-guided trajectory registration")
    parser.add_argument("trajectory", type=Path, help=".npy [N,2] or .npz containing `trajectory`")
    parser.add_argument("lanes", type=Path, help=".npz containing lane centerlines")
    parser.add_argument("output", type=Path, help="Output .npz")
    parser.add_argument("--heading-weight", type=float, default=2.5)
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=0.1,
        help="Lane polyline sampling spacing in meters (Supplementary: 0.1)",
    )
    parser.add_argument("--dtw-window", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = np.load(args.trajectory, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        trajectory_np = loaded["trajectory"]
        loaded.close()
    else:
        trajectory_np = loaded
    with np.load(args.lanes, allow_pickle=True) as lane_archive:
        lanes = _load_lanes(lane_archive)
    trajectory = torch.as_tensor(trajectory_np, dtype=torch.float32)
    config = RegistrationConfig(
        heading_weight=args.heading_weight,
        map_resolution=args.map_resolution,
        dtw_window=args.dtw_window,
    )
    result = register_trajectory_to_lanes(trajectory, lanes, config)
    corrected = apply_se2(trajectory, result.transform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        corrected_trajectory=corrected.numpy(),
        transform=result.transform.numpy(),
        lane_index=np.int64(result.lane_index),
        trajectory_indices=result.trajectory_indices.numpy(),
        lane_indices=result.lane_indices.numpy(),
        normalized_cost=np.float64(result.normalized_cost),
    )
    print(json.dumps({
        "lane_index": result.lane_index,
        "normalized_cost": result.normalized_cost,
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
