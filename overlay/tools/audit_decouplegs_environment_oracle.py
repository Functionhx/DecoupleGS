#!/usr/bin/env python3
"""Run the route oracle through the full HUGSim environment.

Unlike ``audit_decouplegs_closed_loop_control.py``, this audit includes the
released scene Gaussian collision volume, ground-height lookup, Gym action
clipping, and the exact environment termination checks.  Learned planners are
excluded and scenario agents are cleared by default, so a failure isolates the
simulator/controller contract rather than perception or traffic interaction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hugsim_env  # noqa: F401  # registers hugsim_env/HUGSim-v0
from sim.utils.sim_utils import traj2control
from tools.audit_decouplegs_closed_loop_control import Route, oracle_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_paths", nargs="+", type=Path)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/benchmark/local_nuscenes_8scene.yaml",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=ROOT / "configs/sim/nuscenes_camera.yaml",
    )
    parser.add_argument(
        "--kinematic-config",
        type=Path,
        default=ROOT / "configs/sim/decouplegs_kinematic.yaml",
    )
    parser.add_argument(
        "--decouple-config",
        type=Path,
        default=ROOT / "configs/decouplegs.yaml",
    )
    parser.add_argument("--target-speed", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--keep-agents",
        action="store_true",
        help="Retain scenario traffic; default clears it to isolate background collision.",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def make_config(args: argparse.Namespace, scenario_path: Path):
    scenario = OmegaConf.load(scenario_path)
    if not args.keep_agents:
        scenario.plan_list = []
    cfg = OmegaConf.merge(
        {"scenario": scenario},
        {"base": OmegaConf.load(args.base_config)},
        {"camera": OmegaConf.load(args.camera_config)},
        {"kinematic": OmegaConf.load(args.kinematic_config)},
        OmegaConf.load(args.decouple_config),
    )
    model_path = Path(cfg.base.model_base) / cfg.scenario.scene_name
    cfg.update(OmegaConf.load(model_path / "cfg.yaml"))
    cfg.model_path = str(model_path)
    return cfg


def run_scenario(args: argparse.Namespace, scenario_path: Path) -> dict[str, Any]:
    cfg = make_config(args, scenario_path)
    artifact_dir = args.artifact_root / f"{cfg.scenario.scene_name}_{cfg.scenario.mode}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    environment = gymnasium.make("hugsim_env/HUGSim-v0", cfg=cfg, output=str(artifact_dir))
    try:
        _, info = environment.reset()
        native = environment.unwrapped
        route = Route(np.asarray(native._route_xy, dtype=np.float64))
        records = []
        termination_reason = None
        for step in range(args.steps):
            position = np.asarray(info["ego_box"][:2], dtype=np.float64)
            yaw = float(info["ego_box"][6])
            plan = oracle_plan(
                route,
                position,
                yaw,
                args.target_speed,
                float(cfg.kinematic.dt),
            )
            acceleration, steer_rate = traj2control(plan, info)
            _, reward, terminated, truncated, next_info = environment.step(
                {"acc": acceleration, "steer_rate": steer_rate}
            )
            records.append(
                {
                    "step": step,
                    "command": info.get("command"),
                    "ego_box": info["ego_box"],
                    "route_completion": next_info.get("rc_distance"),
                    "route_lateral_error_m": next_info.get("route_lateral_error"),
                    "requested_action": {"acc": acceleration, "steer_rate": steer_rate},
                    "applied_action": (next_info.get("control_contract") or {}).get("applied"),
                    "background_collision_points": (
                        next_info.get("collision_geometry") or {}
                    ).get("background_point_count"),
                    "collision": next_info.get("collision"),
                    "termination_reason": next_info.get("termination_reason"),
                    "reward": reward,
                }
            )
            info = next_info
            if terminated or truncated:
                termination_reason = next_info.get("termination_reason")
                break
        return {
            "scene": str(cfg.scenario.scene_name),
            "mode": str(cfg.scenario.mode),
            "scenario_path": str(scenario_path.resolve()),
            "agents_cleared": not args.keep_agents,
            "steps": len(records),
            "termination_reason": termination_reason or "step_limit",
            "route_completion": info.get("rc_distance"),
            "final_lateral_error_m": info.get("route_lateral_error"),
            "max_lateral_error_m": max(
                (float(record["route_lateral_error_m"]) for record in records),
                default=None,
            ),
            "max_background_collision_points": max(
                (int(record["background_collision_points"]) for record in records),
                default=None,
            ),
            "records": records,
        }
    finally:
        environment.close()
        del environment
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.target_speed <= 0 or args.steps <= 0:
        raise ValueError("target speed and steps must be positive")
    scenes = []
    for scenario_path in args.scenario_paths:
        result = run_scenario(args, scenario_path)
        scenes.append(result)
        print(
            f"{result['scene']}: RC={result['route_completion']:.3f}, "
            f"max lateral={result['max_lateral_error_m']:.3f}m, "
            f"max background points={result['max_background_collision_points']}, "
            f"reason={result['termination_reason']}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_full_environment_route_oracle_audit",
        "hypothesis": "H-CONTROL-03 full environment isolates collision and termination contracts",
        "target_speed_mps": args.target_speed,
        "maximum_steps": args.steps,
        "agents_cleared": not args.keep_agents,
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
