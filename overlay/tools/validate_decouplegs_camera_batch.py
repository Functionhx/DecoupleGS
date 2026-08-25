#!/usr/bin/env python3
"""Compare six-camera batched rendering with the sequential HUGSIM path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import gymnasium
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIM_ROOT = ROOT / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import hugsim_env  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--decouple", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configuration(args: argparse.Namespace, batch: bool):
    config = OmegaConf.merge(
        {"scenario": OmegaConf.load(args.scenario)},
        {"base": OmegaConf.load(args.base)},
        {"camera": OmegaConf.load(args.camera)},
        {"kinematic": OmegaConf.load(args.kinematic)},
        OmegaConf.load(args.decouple),
    )
    model_path = Path(config.base.model_base) / config.scenario.scene_name
    config.update(OmegaConf.load(model_path / "cfg.yaml"))
    config.model_path = str(model_path)
    config.decouplegs.batch_camera_sensor = batch
    return config


def render_reset(args: argparse.Namespace, batch: bool) -> dict[str, np.ndarray]:
    output = args.work_dir / ("batch" if batch else "sequential")
    output.mkdir(parents=True, exist_ok=True)
    environment = gymnasium.make(
        "hugsim_env/HUGSim-v0",
        cfg=configuration(args, batch),
        output=str(output),
    )
    try:
        observation, _ = environment.reset()
        torch.cuda.synchronize()
        return {name: image.copy() for name, image in observation["rgb"].items()}
    finally:
        environment.close()
        del environment
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    sequential = render_reset(args, False)
    batched = render_reset(args, True)
    rows = {}
    total_squared = 0.0
    total_values = 0
    for name in sequential:
        first = sequential[name].astype(np.float64) / 255.0
        second = batched[name].astype(np.float64) / 255.0
        difference = np.abs(first - second)
        squared = (first - second) ** 2
        mse = float(squared.mean())
        total_squared += float(squared.sum())
        total_values += squared.size
        rows[name] = {
            "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "changed_channel_fraction": float((difference > 0).mean()),
        }
    global_mse = total_squared / total_values
    result = {
        "schema_version": 1,
        "benchmark": "decouplegs_six_camera_batch_equivalence",
        "protocol": "same reset state; uint8 HUGSIM observations; sequential vs one multi-camera rasterization",
        "global_psnr_db": 10.0 * math.log10(1.0 / max(global_mse, 1e-12)),
        "cameras": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
