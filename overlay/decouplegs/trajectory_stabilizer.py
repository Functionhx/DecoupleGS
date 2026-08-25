"""Auditable route-corridor stabilization for closed-loop planner outputs.

This is an explicit reproduction R&D component, not a claimed DecoupleGS
paper module.  It is disabled in the strict baseline.  When enabled, it uses
the simulator's already-known evaluation route to keep a learned trajectory
inside a bounded corridor and to supply a configurable cruise-speed floor.
The learned planner can retain a bounded lateral residual, while nearby lead
agents cap the target speed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TrajectoryStabilizerConfig:
    enabled: bool = False
    cruise_speed_floor_mps: float = 6.0
    cruise_speed_cap_mps: float = 8.0
    learned_lateral_weight: float = 0.25
    maximum_lateral_residual_m: float = 1.0
    stable_current_lateral_threshold_m: float = 0.75
    stable_input_lateral_threshold_m: float = 3.5
    stable_learned_lateral_weight: float = 1.0
    stable_maximum_lateral_residual_m: float = 3.0
    lead_corridor_half_width_m: float = 4.0
    lead_lookahead_m: float = 35.0
    lead_minimum_gap_m: float = 5.0
    lead_time_headway_s: float = 2.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        if value is None:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        result = cls(**kwargs)
        if result.cruise_speed_floor_mps < 0:
            raise ValueError("cruise_speed_floor_mps must be non-negative")
        if result.cruise_speed_cap_mps < result.cruise_speed_floor_mps:
            raise ValueError("cruise_speed_cap_mps must be >= cruise_speed_floor_mps")
        for name in ("learned_lateral_weight", "stable_learned_lateral_weight"):
            if not 0.0 <= float(getattr(result, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "maximum_lateral_residual_m",
            "stable_current_lateral_threshold_m",
            "stable_input_lateral_threshold_m",
            "stable_maximum_lateral_residual_m",
            "lead_corridor_half_width_m",
            "lead_lookahead_m",
            "lead_minimum_gap_m",
            "lead_time_headway_s",
        ):
            if float(getattr(result, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        return result


class RouteGeometry:
    def __init__(self, route_xy: np.ndarray):
        self.xy = np.asarray(route_xy, dtype=np.float64)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or len(self.xy) < 2:
            raise ValueError("route_xy must have shape [N, 2], N >= 2")
        self.delta = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=-1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))

    @property
    def length(self) -> float:
        return float(self.cumulative[-1])

    def project(self, point: Sequence[float]) -> tuple[float, float, float]:
        point = np.asarray(point, dtype=np.float64)
        denominator = np.maximum(self.lengths**2, 1e-12)
        ratio = np.clip(
            np.sum((point - self.xy[:-1]) * self.delta, axis=-1) / denominator,
            0.0,
            1.0,
        )
        projections = self.xy[:-1] + ratio[:, None] * self.delta
        distances = np.linalg.norm(projections - point, axis=-1)
        index = int(np.argmin(distances))
        arc = self.cumulative[index] + ratio[index] * self.lengths[index]
        tangent = self.delta[index] / max(self.lengths[index], 1e-12)
        offset = point - projections[index]
        signed_left = tangent[0] * offset[1] - tangent[1] * offset[0]
        return float(arc), float(distances[index]), float(signed_left)

    def sample(self, arc: float) -> np.ndarray:
        arc = float(np.clip(arc, 0.0, self.length))
        index = min(
            int(np.searchsorted(self.cumulative, arc, side="right") - 1),
            len(self.lengths) - 1,
        )
        ratio = (arc - self.cumulative[index]) / max(self.lengths[index], 1e-12)
        return self.xy[index] + ratio * self.delta[index]


class RouteTrajectoryStabilizer:
    """Convert learned plans into bounded route-conditioned references."""

    def __init__(
        self,
        route_xy: np.ndarray,
        control_dt_seconds: float,
        config: TrajectoryStabilizerConfig,
    ):
        if control_dt_seconds <= 0:
            raise ValueError("control_dt_seconds must be positive")
        self.route = RouteGeometry(route_xy)
        self.dt = float(control_dt_seconds)
        self.config = config

    def contract(self) -> dict[str, Any]:
        return {
            "description": (
                "R&D route-corridor safety filter; disabled for strict paper baseline"
            ),
            "paper_claim": False,
            "configuration": asdict(self.config),
        }

    @staticmethod
    def _local_basis(yaw: float) -> tuple[np.ndarray, np.ndarray]:
        forward = np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64)
        right = np.asarray((math.sin(yaw), -math.cos(yaw)), dtype=np.float64)
        return forward, right

    def _lead_speed_cap(
        self,
        info: Mapping[str, Any],
        current_arc: float,
    ) -> tuple[float | None, float | None, str | None, str | None]:
        ego_box = np.asarray(info["ego_box"], dtype=np.float64)
        boxes = list(info.get("obj_boxes") or [])
        velocities = list(info.get("obj_velocities") or [])
        behavior_states = list(info.get("obj_behavior_states") or [])
        object_ids = list(info.get("obj_ids") or [])
        best_gap = None
        best_cap = None
        best_id = None
        best_reason = None
        for index, box_value in enumerate(boxes):
            box = np.asarray(box_value, dtype=np.float64)
            object_arc, object_lateral, _ = self.route.project(box[:2])
            behavior = (
                behavior_states[index]
                if index < len(behavior_states) and behavior_states[index] is not None
                else {}
            )
            targets_ego_lane = bool(behavior.get("targets_ego_lane", False))
            if (
                object_lateral > self.config.lead_corridor_half_width_m
                and not targets_ego_lane
            ):
                continue
            bumper_gap = (
                object_arc
                - current_arc
                - 0.5 * float(ego_box[4])
                - 0.5 * float(box[4])
            )
            if not 0.0 < bumper_gap <= self.config.lead_lookahead_m:
                continue
            if "speed_mps" in behavior:
                object_speed = float(behavior["speed_mps"])
            elif index < len(velocities):
                object_speed = float(np.linalg.norm(np.asarray(velocities[index])[:2]))
            else:
                object_speed = 0.0
            cap = object_speed + max(
                0.0, bumper_gap - self.config.lead_minimum_gap_m
            ) / self.config.lead_time_headway_s
            if best_gap is None or bumper_gap < best_gap:
                best_gap = float(bumper_gap)
                best_cap = float(max(0.0, cap))
                best_id = str(object_ids[index]) if index < len(object_ids) else str(index)
                best_reason = "adjacent_lane_cut_in" if targets_ego_lane else "route_corridor"
        return best_gap, best_cap, best_id, best_reason

    def stabilize(
        self,
        plan_traj: np.ndarray,
        info: Mapping[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        plan = np.asarray(plan_traj, dtype=np.float64)
        if plan.ndim != 2 or plan.shape[0] == 0 or plan.shape[1] < 2:
            raise ValueError(f"plan_traj must be [N, >=2], got {plan.shape}")
        if not self.config.enabled:
            return plan.copy(), {"enabled": False, "applied": False}

        plan_xy = plan[:, :2]
        path_steps = np.diff(
            np.concatenate((np.zeros((1, 2), dtype=np.float64), plan_xy), axis=0),
            axis=0,
        )
        horizon_seconds = len(plan_xy) * self.dt
        learned_speed = float(np.linalg.norm(path_steps, axis=-1).sum() / horizon_seconds)

        ego_box = np.asarray(info["ego_box"], dtype=np.float64)
        position = ego_box[:2]
        yaw = float(ego_box[6])
        current_arc, current_lateral, _ = self.route.project(position)
        forward, right = self._local_basis(yaw)

        predicted_global = (
            position[None]
            + plan_xy[:, 1:2] * forward[None]
            + plan_xy[:, 0:1] * right[None]
        )
        predicted_lateral = np.asarray(
            [self.route.project(point)[1] for point in predicted_global],
            dtype=np.float64,
        )

        target_speed = float(
            np.clip(
                max(learned_speed, self.config.cruise_speed_floor_mps),
                0.0,
                self.config.cruise_speed_cap_mps,
            )
        )
        lead_gap, lead_cap, lead_id, lead_reason = self._lead_speed_cap(info, current_arc)
        if lead_cap is not None:
            target_speed = min(target_speed, lead_cap)

        horizons = np.arange(1, len(plan_xy) + 1, dtype=np.float64) * self.dt
        route_global = np.stack(
            [self.route.sample(current_arc + target_speed * horizon) for horizon in horizons]
        )
        route_delta = route_global - position[None]
        route_plan = np.stack((route_delta @ right, route_delta @ forward), axis=-1)

        stable_corridor = bool(
            current_lateral <= self.config.stable_current_lateral_threshold_m
            and float(predicted_lateral.max())
            <= self.config.stable_input_lateral_threshold_m
        )
        lateral_weight = (
            self.config.stable_learned_lateral_weight
            if stable_corridor
            else self.config.learned_lateral_weight
        )
        maximum_residual = (
            self.config.stable_maximum_lateral_residual_m
            if stable_corridor
            else self.config.maximum_lateral_residual_m
        )
        lateral_residual = lateral_weight * (
            plan_xy[:, 0] - route_plan[:, 0]
        )
        lateral_residual = np.clip(
            lateral_residual,
            -maximum_residual,
            maximum_residual,
        )
        stabilized_xy = route_plan.copy()
        stabilized_xy[:, 0] += lateral_residual
        stabilized = plan.copy()
        stabilized[:, :2] = stabilized_xy

        output_global = (
            position[None]
            + stabilized_xy[:, 1:2] * forward[None]
            + stabilized_xy[:, 0:1] * right[None]
        )
        output_lateral = np.asarray(
            [self.route.project(point)[1] for point in output_global],
            dtype=np.float64,
        )
        diagnostics = {
            "enabled": True,
            "applied": True,
            "hypothesis": "H-PLAN-01 route-corridor stabilization",
            "learned_path_speed_mps": learned_speed,
            "target_speed_mps": target_speed,
            "current_route_arc_m": current_arc,
            "current_route_lateral_error_m": current_lateral,
            "input_max_route_lateral_error_m": float(predicted_lateral.max()),
            "output_max_route_lateral_error_m": float(output_lateral.max()),
            "stable_corridor_mode": stable_corridor,
            "applied_lateral_weight": lateral_weight,
            "applied_maximum_lateral_residual_m": maximum_residual,
            "lead_bumper_gap_m": lead_gap,
            "lead_speed_cap_mps": lead_cap,
            "lead_vehicle_id": lead_id,
            "lead_selection_reason": lead_reason,
        }
        return stabilized, diagnostics
