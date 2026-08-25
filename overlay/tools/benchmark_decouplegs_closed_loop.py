#!/usr/bin/env python3
"""Measure the real six-camera HUGSIM loop without an external E2E client."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import gymnasium
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIM_ROOT = ROOT / "sim"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

import hugsim_env  # noqa: E402,F401 - registers the Gym environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DecoupleGS six-camera closed-loop benchmark")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--kinematic", type=Path, required=True)
    parser.add_argument("--decouple", type=Path, default=ROOT / "configs/decouplegs.yaml")
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--same-state-repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    position = 0.95 * (len(ordered) - 1)
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    p95 = ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
    mean = statistics.fmean(seconds)
    return {
        "frames": len(seconds),
        "mean_ms": mean * 1000.0,
        "median_ms": statistics.median(seconds) * 1000.0,
        "p95_ms": p95 * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
        "control_fps": 1.0 / mean,
        "sensor_image_fps": 6.0 / mean,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("closed-loop benchmarking requires CUDA")
    if args.warmup < 0 or args.same_state_repeats <= 0 or args.steps <= 0:
        raise ValueError("warmup must be non-negative and repeat/step counts must be positive")
    scenario = OmegaConf.load(args.scenario)
    base = OmegaConf.load(args.base)
    camera = OmegaConf.load(args.camera)
    kinematic = OmegaConf.load(args.kinematic)
    decouple = OmegaConf.load(args.decouple)
    config = OmegaConf.merge(
        {"scenario": scenario},
        {"base": base},
        {"camera": camera},
        {"kinematic": kinematic},
        decouple,
    )
    model_path = Path(config.base.model_base) / config.scenario.scene_name
    config.update(OmegaConf.load(model_path / "cfg.yaml"))
    config.model_path = str(model_path)
    args.environment_output.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    environment = gymnasium.make(
        "hugsim_env/HUGSim-v0",
        cfg=config,
        output=str(args.environment_output),
    )
    initialization_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        observation, _ = environment.reset()
        torch.cuda.synchronize()
        cold_reset_seconds = time.perf_counter() - started
        unwrapped = environment.unwrapped
        for _ in range(args.warmup):
            unwrapped._get_obs()
        torch.cuda.synchronize()

        same_state = []
        for _ in range(args.same_state_repeats):
            started = time.perf_counter()
            unwrapped._get_obs()
            torch.cuda.synchronize()
            same_state.append(time.perf_counter() - started)

        step_latencies = []
        terminated = truncated = False
        for _ in range(args.steps):
            started = time.perf_counter()
            observation, _, terminated, truncated, _ = environment.step(
                {"acc": 0.0, "steer_rate": 0.0}
            )
            torch.cuda.synchronize()
            step_latencies.append(time.perf_counter() - started)
            if terminated or truncated:
                break
        bridge = unwrapped.render_kwargs["decouple_bridge"]
        payload = {
            "schema_version": 1,
            "benchmark": "decouplegs_hugsim_six_camera_loop",
            "scenario": str(args.scenario.resolve()),
            "scene": str(model_path.resolve()),
            "initialization_seconds": initialization_seconds,
            "cold_reset_seconds": cold_reset_seconds,
            "same_state": summary(same_state),
            "new_state_step": summary(step_latencies),
            "terminated": terminated,
            "truncated": truncated,
            "camera_names": list(observation["rgb"]),
            "camera_shapes": {
                name: list(image.shape) for name, image in observation["rgb"].items()
            },
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "gpu": torch.cuda.get_device_name(),
            "compact_assets": {
                asset_id: type(asset).__name__
                for asset_id, asset in bridge.runtime.library.assets.items()
            },
            "raw_dynamic_models": list(unwrapped.render_kwargs["dynamic_gaussians"]),
            "rgb_only_sensor": bool(unwrapped.render_kwargs.get("rgb_only", False)),
        }
    finally:
        environment.close()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
