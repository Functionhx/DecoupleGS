from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_min(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else math.inf


def _box_support_radius(box: Sequence[float], direction: np.ndarray) -> float:
    """Radius of an oriented 2-D vehicle box along ``direction``."""

    width, length, yaw = float(box[3]), float(box[4]), float(box[6])
    heading = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    lateral = np.asarray([-heading[1], heading[0]], dtype=np.float64)
    return 0.5 * length * abs(float(direction @ heading)) + 0.5 * width * abs(
        float(direction @ lateral)
    )


def frame_ttc(frame: Mapping[str, Any], *, epsilon: float = 1e-6) -> dict[str, float]:
    """Evaluate all documented and diagnostic TTC variants for one state.

    The supplement writes ``distance / v_rel`` but does not define whether
    ``v_rel`` is the relative-speed norm or the projected closing speed, nor
    whether vehicle extents are subtracted.  We therefore retain the literal
    centre-distance equation and two physically useful diagnostics instead of
    silently choosing one interpretation.
    """

    ego_box = frame.get("ego_box")
    ego_velocity = frame.get("ego_velocity_xy")
    agent_boxes = frame.get("obj_boxes") or []
    agent_velocities = frame.get("obj_velocities") or []
    if ego_box is None or ego_velocity is None:
        return {
            "paper_literal_center": math.inf,
            "closing_center": math.inf,
            "closing_clearance": math.inf,
        }
    if len(agent_boxes) != len(agent_velocities):
        raise ValueError("obj_boxes and obj_velocities must have the same length")

    ego_position = np.asarray(ego_box[:2], dtype=np.float64)
    ego_velocity_array = np.asarray(ego_velocity, dtype=np.float64)
    literal_values: list[float] = []
    closing_values: list[float] = []
    clearance_values: list[float] = []
    for agent_box, agent_velocity in zip(agent_boxes, agent_velocities, strict=True):
        relative_position = np.asarray(agent_box[:2], dtype=np.float64) - ego_position
        relative_velocity = np.asarray(agent_velocity, dtype=np.float64) - ego_velocity_array
        distance = float(np.linalg.norm(relative_position))
        relative_speed = float(np.linalg.norm(relative_velocity))
        if relative_speed > epsilon:
            literal_values.append(distance / relative_speed)

        if distance <= epsilon:
            closing_values.append(0.0)
            clearance_values.append(0.0)
            continue
        direction = relative_position / distance
        closing_speed = -float(direction @ relative_velocity)
        if closing_speed <= epsilon:
            continue
        closing_values.append(distance / closing_speed)
        clearance = max(
            0.0,
            distance
            - _box_support_radius(ego_box, direction)
            - _box_support_radius(agent_box, -direction),
        )
        clearance_values.append(clearance / closing_speed)

    return {
        "paper_literal_center": _finite_min(literal_values),
        "closing_center": _finite_min(closing_values),
        "closing_clearance": _finite_min(clearance_values),
    }


def _termination_tokens(reason: Any) -> set[str]:
    if reason is None:
        return set()
    return {token for token in str(reason).split("+") if token}


def evaluate_episode(
    record: Mapping[str, Any],
    *,
    penalty_factors: Mapping[str, float] | None = None,
    success_completion: float = 0.99,
) -> dict[str, Any]:
    """Compute paper-facing closed-loop metrics from one saved episode."""

    frames = list(record.get("frames", []))
    episode = record.get("episode", {}) or {}
    final_info = episode.get("final_info", {}) or {}

    route_travelled = [float(frame.get("route_distance_traveled", 0.0)) for frame in frames]
    route_planned = [float(frame.get("route_distance_planned", 0.0)) for frame in frames]
    rc_distance = [float(frame.get("rc_distance", 0.0)) for frame in frames]
    rc_legacy = [float(frame.get("rc", 0.0)) for frame in frames]
    for frame in frames:
        transition = frame.get("transition", {}) or {}
        rc_distance.append(float(transition.get("rc_distance", 0.0)))
        rc_legacy.append(float(transition.get("rc", 0.0)))
    if final_info:
        route_travelled.append(float(final_info.get("route_distance_traveled", 0.0)))
        route_planned.append(float(final_info.get("route_distance_planned", 0.0)))
        rc_distance.append(float(final_info.get("rc_distance", 0.0)))
        rc_legacy.append(float(final_info.get("rc", 0.0)))

    distance_travelled = max(route_travelled, default=0.0)
    distance_planned = max(route_planned, default=0.0)
    if distance_planned > 0.0:
        paper_rc = min(max(distance_travelled / distance_planned, 0.0), 1.0)
    else:
        paper_rc = max(rc_distance, default=0.0)

    ttc_frames: list[Mapping[str, Any]] = list(frames)
    if final_info:
        ttc_frames.append(final_info)
    ttc_values = [frame_ttc(frame) for frame in ttc_frames]
    min_ttc = {
        key: _finite_min([value[key] for value in ttc_values])
        for key in ("paper_literal_center", "closing_center", "closing_clearance")
    }

    collision = any(bool(frame.get("collision", False)) for frame in frames)
    collision = collision or any(
        bool((frame.get("transition", {}) or {}).get("collision", False)) for frame in frames
    )
    collision = collision or bool(final_info.get("collision", False))
    reasons = _termination_tokens(episode.get("termination_reason"))
    for frame in frames:
        reasons |= _termination_tokens(
            (frame.get("transition", {}) or {}).get("termination_reason")
        )
    if final_info:
        reasons |= _termination_tokens(final_info.get("termination_reason"))

    infraction_counts = {
        # Episodes terminate on collision/off-route, so these are event counts,
        # not the number of frames for which a flag remains asserted.
        "collision": int(collision),
        "off_route": int("off_route" in reasons),
        "planner_failure": int("planner_failure" in reasons),
    }
    severe_failure = any(infraction_counts.values())
    strict_success = bool(paper_rc >= success_completion and not severe_failure)
    termination_reason = str(episode.get("termination_reason") or "")
    success_variants = {
        # H-METRIC-01 is the conservative primary definition used by this
        # reproduction.  The paper names SR but does not publish its rule.
        "strict_completion_h_metric_01": strict_success,
        # HUGSIM terminates after its legacy route-progress field reaches 1;
        # this can happen before arc-length RC reaches 0.99.
        "terminal_route_complete": bool(
            "route_complete" in _termination_tokens(termination_reason)
            and not severe_failure
        ),
        # Useful as a pure safety diagnostic, but deliberately not called
        # route success because a stationary agent could satisfy it.
        "safety_only": bool(not severe_failure),
    }

    driving_score = None
    if penalty_factors is not None:
        driving_score = paper_rc
        for name, count in infraction_counts.items():
            factor = float(penalty_factors.get(name, 1.0))
            if not 0.0 < factor <= 1.0:
                raise ValueError(f"penalty factor {name!r} must be in (0, 1]")
            driving_score *= factor**count

    timing_fields: dict[str, list[float]] = {}
    for frame in frames:
        for key, value in (frame.get("timing_ms", {}) or {}).items():
            timing_fields.setdefault(str(key), []).append(float(value))
    timing = {}
    for key, values in timing_fields.items():
        mean_ms = float(np.mean(values))
        throughput = 1000.0 / mean_ms if mean_ms > 0 else math.inf
        summary = {
            "mean_ms": float(np.mean(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples": len(values),
            "throughput_per_second": throughput,
            # Backward-compatible alias; sensor rendering is explicitly a
            # six-camera batch and must not be reported as per-image FPS.
            "fps": throughput,
        }
        if key == "simulator_sensor_render":
            # One simulator call renders the full six-camera nuScenes rig.
            # Keep batch and image throughput separate because the paper's
            # bare "FPS" label does not disclose this timing boundary.
            summary["camera_batches_per_second"] = throughput
            summary["sensor_images_per_second"] = throughput * 6.0
        timing[key] = summary

    return {
        "steps": int(episode.get("steps", len(frames))),
        "termination_reason": episode.get("termination_reason"),
        "route_completion": paper_rc,
        "route_completion_h_rc_01": max(rc_distance, default=0.0),
        "route_completion_legacy_hugsim": min(max(rc_legacy, default=0.0), 1.0),
        "distance_traveled_m": distance_travelled,
        "distance_planned_m": distance_planned,
        "success": strict_success,
        "success_definition": (
            f"RC >= {success_completion:g} and no collision/off-route/planner failure (H-METRIC-01)"
        ),
        "success_variants": success_variants,
        "min_ttc_seconds": min_ttc,
        "driving_score": driving_score,
        "driving_score_status": (
            "computed_from_user_supplied_penalty_factors"
            if penalty_factors is not None
            else "unresolved_paper_protocol_penalty_factors_not_disclosed"
        ),
        "penalty_factors": None if penalty_factors is None else dict(penalty_factors),
        "infraction_counts": infraction_counts,
        "timing": timing,
    }


def aggregate_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")

    def summarize(values: Sequence[float]) -> dict[str, float | int]:
        finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
        if len(finite) == 0:
            return {"mean": math.inf, "std": math.nan, "count": 0}
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)),
            "count": int(len(finite)),
        }

    driving_scores = [episode["driving_score"] for episode in episodes]
    success_variant_names = sorted(
        {
            name
            for episode in episodes
            for name in (episode.get("success_variants", {}) or {})
        }
    )
    timing_names = sorted(
        {name for episode in episodes for name in (episode.get("timing", {}) or {})}
    )
    return {
        "episodes": len(episodes),
        "driving_score": (
            None
            if any(value is None for value in driving_scores)
            else summarize([float(value) for value in driving_scores])
        ),
        "success_rate": summarize([float(episode["success"]) for episode in episodes]),
        "success_rate_variants": {
            name: summarize(
                [
                    float((episode.get("success_variants", {}) or {}).get(name, False))
                    for episode in episodes
                ]
            )
            for name in success_variant_names
        },
        "route_completion": summarize(
            [float(episode["route_completion"]) for episode in episodes]
        ),
        "min_ttc_seconds": {
            key: summarize(
                [float(episode["min_ttc_seconds"][key]) for episode in episodes]
            )
            for key in ("paper_literal_center", "closing_center", "closing_clearance")
        },
        "timing": {
            name: {
                field: summarize(
                    [
                        float(episode["timing"][name][field])
                        for episode in episodes
                        if name in (episode.get("timing", {}) or {})
                        and field in episode["timing"][name]
                    ]
                )
                for field in (
                    "mean_ms",
                    "p95_ms",
                    "throughput_per_second",
                    "fps",
                    "camera_batches_per_second",
                    "sensor_images_per_second",
                )
                if any(
                    name in (episode.get("timing", {}) or {})
                    and field in episode["timing"][name]
                    for episode in episodes
                )
            }
            for name in timing_names
        },
    }
