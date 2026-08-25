#!/usr/bin/env python3
"""Audit the closed-loop iLQR/kinematic contract with a route oracle.

This deliberately excludes rendering and learned planners.  A six-point
oracle trajectory is sampled from each released HUGSIM route, converted to the
planner's ``[right, forward]`` convention, tracked by the production iLQR
controller, and integrated by the same bicycle equations as HUGSimEnv.  The
counterfactual inverted-steering boundary makes coordinate-sign bugs directly
falsifiable.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.utils.sim_utils import dense_cam_poses, traj2control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--kinematic",
        type=Path,
        default=ROOT / "configs/sim/decouplegs_kinematic.yaml",
    )
    parser.add_argument("--target-speed", type=float, default=6.0)
    parser.add_argument("--initial-speed", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class Route:
    def __init__(self, xy: np.ndarray):
        self.xy = np.asarray(xy, dtype=np.float64)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.xy) < 2:
            raise ValueError("route must have shape [N, 2], N >= 2")
        self.delta = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=-1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])

    def project(self, point: np.ndarray) -> tuple[float, float]:
        point = np.asarray(point, dtype=np.float64)
        denominator = np.maximum(self.lengths**2, 1e-12)
        ratio = np.clip(
            np.sum((point - self.xy[:-1]) * self.delta, axis=-1) / denominator,
            0.0,
            1.0,
        )
        projections = self.xy[:-1] + ratio[:, None] * self.delta
        distance = np.linalg.norm(projections - point, axis=-1)
        index = int(np.argmin(distance))
        arc = self.cumulative[index] + ratio[index] * self.lengths[index]
        return float(arc), float(distance[index])

    def sample(self, arc: float) -> np.ndarray:
        arc = float(np.clip(arc, 0.0, self.length))
        index = min(
            int(np.searchsorted(self.cumulative, arc, side="right") - 1),
            len(self.lengths) - 1,
        )
        ratio = (arc - self.cumulative[index]) / max(self.lengths[index], 1e-12)
        return self.xy[index] + ratio * self.delta[index]


def load_route(scene_dir: Path) -> Route:
    with (scene_dir / "ground_param.pkl").open("rb") as stream:
        camera_poses, _, commands = pickle.load(stream)
    camera_poses, _ = dense_cam_poses(camera_poses, commands)
    # Same IMU-plane conversion used by HUGSimEnv.metric_route_completion.
    route_xy = np.stack((camera_poses[:, 2, 3], -camera_poses[:, 0, 3]), axis=-1)
    return Route(route_xy)


def oracle_plan(
    route: Route,
    position: np.ndarray,
    abstract_yaw: float,
    target_speed: float,
    dt: float,
) -> np.ndarray:
    current_arc, _ = route.project(position)
    horizons = np.arange(1, 7, dtype=np.float64) * dt
    targets = np.stack(
        [route.sample(current_arc + target_speed * horizon) for horizon in horizons]
    )
    delta = targets - position
    # Global route coordinates use +Y to the vehicle's left.  The planners
    # and iLQR use +Y/right, hence the right basis below.
    forward = np.asarray((math.cos(abstract_yaw), math.sin(abstract_yaw)))
    right = np.asarray((math.sin(abstract_yaw), -math.cos(abstract_yaw)))
    return np.stack((delta @ right, delta @ forward), axis=-1)


def simulate(
    route: Route,
    kinematic: dict[str, Any],
    *,
    target_speed: float,
    initial_speed: float,
    steps: int,
    invert_steering_boundary: bool,
) -> dict[str, Any]:
    dt = float(kinematic["dt"])
    wheelbase = float(kinematic["Lr"] + kinematic["Lf"])
    min_steer_rate = -math.radians(float(kinematic["min_steer"]))
    max_steer_rate = math.radians(float(kinematic["max_steer"]))
    position = np.zeros(2, dtype=np.float64)
    abstract_yaw = 0.0
    velocity = float(initial_speed)
    environment_steer = 0.0
    start_arc, _ = route.project(position)
    max_arc = start_arc
    records = []

    for step in range(steps):
        plan = oracle_plan(route, position, abstract_yaw, target_speed, dt)
        controller_steer = (
            -environment_steer if invert_steering_boundary else environment_steer
        )
        acceleration, steer_rate = traj2control(
            plan,
            {"ego_velo": velocity, "ego_steer": controller_steer},
        )
        if invert_steering_boundary:
            steer_rate = -steer_rate
        acceleration = float(
            np.clip(acceleration, kinematic["min_acc"], kinematic["max_acc"])
        )
        steer_rate = float(np.clip(steer_rate, min_steer_rate, max_steer_rate))

        velocity += acceleration * dt
        environment_steer += steer_rate * dt
        # HUGSimEnv stores theta with the opposite sign from its exposed
        # ego-box yaw and maps planar state as ego_xy = [b, -a].
        theta = -abstract_yaw
        a = -position[1] + velocity * math.sin(theta) * dt
        b = position[0] + velocity * math.cos(theta) * dt
        theta += velocity * math.tan(environment_steer) / wheelbase * dt
        position = np.asarray((b, -a))
        abstract_yaw = -theta

        arc, lateral_error = route.project(position)
        max_arc = max(max_arc, arc)
        records.append(
            {
                "step": step,
                "position": position.tolist(),
                "velocity_mps": velocity,
                "environment_steer_rad": environment_steer,
                "acceleration_mps2": acceleration,
                "steer_rate_radps": steer_rate,
                "route_arc_m": arc,
                "route_lateral_error_m": lateral_error,
            }
        )

    planned = max(route.length - start_arc, 1e-9)
    travelled = max(0.0, max_arc - start_arc)
    return {
        "route_completion": min(travelled / planned, 1.0),
        "route_distance_traveled_m": travelled,
        "route_distance_planned_m": planned,
        "final_lateral_error_m": records[-1]["route_lateral_error_m"],
        "max_lateral_error_m": max(
            record["route_lateral_error_m"] for record in records
        ),
        "final_velocity_mps": velocity,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.target_speed <= 0 or args.initial_speed < 0 or args.steps <= 0:
        raise ValueError("target speed and steps must be positive; initial speed non-negative")
    kinematic = OmegaConf.to_container(OmegaConf.load(args.kinematic), resolve=True)
    scenes = []
    for scene_dir in args.scene_dirs:
        route = load_route(scene_dir)
        current = simulate(
            route,
            kinematic,
            target_speed=args.target_speed,
            initial_speed=args.initial_speed,
            steps=args.steps,
            invert_steering_boundary=False,
        )
        inverted = simulate(
            route,
            kinematic,
            target_speed=args.target_speed,
            initial_speed=args.initial_speed,
            steps=args.steps,
            invert_steering_boundary=True,
        )
        scenes.append(
            {
                "scene": scene_dir.name,
                "scene_dir": str(scene_dir.resolve()),
                "current_contract": current,
                "counterfactual_inverted_steering_boundary": inverted,
            }
        )
        print(
            f"{scene_dir.name}: current RC={current['route_completion']:.3f}, "
            f"max lateral={current['max_lateral_error_m']:.3f}m; "
            f"inverted RC={inverted['route_completion']:.3f}, "
            f"max lateral={inverted['max_lateral_error_m']:.3f}m",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_closed_loop_controller_oracle_audit",
        "hypothesis": "H-CONTROL-02 route oracle isolates controller/kinematic coordinates",
        "target_speed_mps": args.target_speed,
        "initial_speed_mps": args.initial_speed,
        "steps": args.steps,
        "control_dt_seconds": float(kinematic["dt"]),
        "scenes": scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
