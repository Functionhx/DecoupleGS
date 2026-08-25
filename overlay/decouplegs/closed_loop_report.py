"""Statistical reporting for the DecoupleGS closed-loop benchmark.

The paper leaves the Driving Score penalty factors and Success Rate terminal
rule undisclosed.  This module therefore keeps those protocol gaps explicit
while computing the disclosed Route Completion and literal minTTC equations,
plus clearly named success diagnostics.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PLANNERS = ("uniad", "vad")
DIFFICULTIES = ("easy", "medium", "hard", "extreme")
PAPER_COLUMNS = ("driving_score", "success_rate", "route_completion", "min_ttc")


def _seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**63 - 1)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int = 20_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap 95% CI for a mean."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("bootstrap input must contain a finite value")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if len(array) == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binary rate."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half_width, center + half_width


def summarize_numeric(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("numeric summary requires at least one finite value")
    low, high = bootstrap_mean_interval(array, samples=samples, seed=seed)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "count": len(array),
        "ci95_low": low,
        "ci95_high": high,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_binary(values: Sequence[bool | float | int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError("binary summary requires one or more 0/1 values")
    successes = int(array.sum())
    low, high = wilson_interval(successes, len(array))
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "count": len(array),
        "successes": successes,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _numeric(
    episodes: Sequence[Mapping[str, Any]],
    accessor: Any,
    *,
    samples: int,
    base_seed: int,
    label: str,
) -> dict[str, float | int]:
    return summarize_numeric(
        [float(accessor(episode)) for episode in episodes],
        samples=samples,
        seed=_seed(base_seed, label),
    )


def _paper_cell(targets: Mapping[str, Any], planner: str, difficulty: str) -> dict[str, Any]:
    table = targets["table_3_difficulty_stress_test"]
    values = table[difficulty][planner.upper() if planner == "vad" else "UniAD"]
    return {
        name: {"mean": float(pair[0]), "std": float(pair[1])}
        for name, pair in zip(PAPER_COLUMNS, values, strict=True)
    }


def episode_key(episode: Mapping[str, Any]) -> tuple[int, int, int]:
    return int(episode["seed"]), int(episode["episode"]), int(episode["agents"])


def summarize_cell(
    episodes: Sequence[Mapping[str, Any]],
    *,
    planner: str,
    difficulty: str,
    paper_reference: Mapping[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(episodes) != 50:
        raise ValueError(f"{planner}/{difficulty} has {len(episodes)} episodes, expected 50")
    prefix = f"{planner}/{difficulty}"
    route_completion = _numeric(
        episodes,
        lambda row: row["route_completion"],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/route_completion",
    )
    min_ttc = _numeric(
        episodes,
        lambda row: row["min_ttc_seconds"]["paper_literal_center"],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/min_ttc",
    )
    sensor_batch_fps = _numeric(
        episodes,
        lambda row: row["timing"]["simulator_sensor_render"][
            "camera_batches_per_second"
        ],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/sensor_batch_fps",
    )
    sensor_image_fps = _numeric(
        episodes,
        lambda row: row["timing"]["simulator_sensor_render"][
            "sensor_images_per_second"
        ],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/sensor_image_fps",
    )
    planner_hz = _numeric(
        episodes,
        lambda row: row["timing"]["planner_and_ipc_wall"][
            "throughput_per_second"
        ],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/planner_hz",
    )
    steps = _numeric(
        episodes,
        lambda row: row["steps"],
        samples=bootstrap_samples,
        base_seed=bootstrap_seed,
        label=f"{prefix}/steps",
    )
    strict_success = summarize_binary([bool(row["success"]) for row in episodes])
    safety_only = summarize_binary(
        [bool(row["success_variants"]["safety_only"]) for row in episodes]
    )
    terminal_route_complete = summarize_binary(
        [
            bool(row["success_variants"]["terminal_route_complete"])
            for row in episodes
        ]
    )
    termination_reasons = [str(row["termination_reason"]) for row in episodes]
    terminations = dict(sorted(Counter(termination_reasons).items()))
    termination_tokens = dict(
        sorted(
            Counter(
                token
                for reason in termination_reasons
                for token in reason.split("+")
                if token
            ).items()
        )
    )
    infractions = {
        name: int(sum(int(row["infraction_counts"][name]) for row in episodes))
        for name in ("collision", "off_route", "planner_failure")
    }
    return {
        "episodes": len(episodes),
        "driving_score": None,
        "driving_score_status": "unresolved_paper_penalty_factors_not_disclosed",
        "success_rate_status": "paper_terminal_definition_not_disclosed",
        "success_rate_proxy_safety_only": safety_only,
        "success_rate_strict_h_metric_01": strict_success,
        "success_rate_terminal_route_complete": terminal_route_complete,
        "route_completion": route_completion,
        "min_ttc_paper_literal_center_seconds": min_ttc,
        "runtime": {
            "six_camera_observations_per_second": sensor_batch_fps,
            "sensor_images_per_second": sensor_image_fps,
            "planner_and_ipc_updates_per_second": planner_hz,
        },
        "steps": steps,
        "terminations": terminations,
        "termination_token_counts": termination_tokens,
        "composite_terminations": sum("+" in reason for reason in termination_reasons),
        "infractions": infractions,
        "paper_reference": dict(paper_reference),
        "delta_vs_paper": {
            "success_rate_proxy": safety_only["mean"]
            - paper_reference["success_rate"]["mean"],
            "route_completion": route_completion["mean"]
            - paper_reference["route_completion"]["mean"],
            "min_ttc": min_ttc["mean"] - paper_reference["min_ttc"]["mean"],
        },
    }


def _paired_summary(
    uniad: Sequence[Mapping[str, Any]],
    vad: Sequence[Mapping[str, Any]],
    *,
    difficulty: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    uniad_by_key = {episode_key(row): row for row in uniad}
    vad_by_key = {episode_key(row): row for row in vad}
    if len(uniad_by_key) != len(uniad) or len(vad_by_key) != len(vad):
        raise ValueError(f"duplicate paired scenario key in {difficulty}")
    if set(uniad_by_key) != set(vad_by_key):
        raise ValueError(f"UniAD/VAD scenarios are not paired in {difficulty}")
    keys = sorted(uniad_by_key)
    definitions = {
        "route_completion": lambda row: float(row["route_completion"]),
        "success_rate_proxy_safety_only": lambda row: float(
            row["success_variants"]["safety_only"]
        ),
        "min_ttc_paper_literal_center_seconds": lambda row: float(
            row["min_ttc_seconds"]["paper_literal_center"]
        ),
        "six_camera_observations_per_second": lambda row: float(
            row["timing"]["simulator_sensor_render"]["camera_batches_per_second"]
        ),
    }
    metrics: dict[str, Any] = {}
    for name, accessor in definitions.items():
        differences = [
            accessor(uniad_by_key[key]) - accessor(vad_by_key[key]) for key in keys
        ]
        summary = summarize_numeric(
            differences,
            samples=bootstrap_samples,
            seed=_seed(bootstrap_seed, f"paired/{difficulty}/{name}"),
        )
        array = np.asarray(differences, dtype=np.float64)
        summary["uniad_win_rate"] = float(np.mean(array > 0.0))
        summary["tie_rate"] = float(np.mean(array == 0.0))
        metrics[name] = summary
    return {"pairs": len(keys), "difference": "UniAD - VAD", "metrics": metrics}


def _overall(
    episodes: Sequence[Mapping[str, Any]],
    *,
    planner: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    prefix = f"{planner}/overall"
    return {
        "episodes": len(episodes),
        "driving_score": None,
        "success_rate_proxy_safety_only": summarize_binary(
            [bool(row["success_variants"]["safety_only"]) for row in episodes]
        ),
        "success_rate_strict_h_metric_01": summarize_binary(
            [bool(row["success"]) for row in episodes]
        ),
        "route_completion": _numeric(
            episodes,
            lambda row: row["route_completion"],
            samples=bootstrap_samples,
            base_seed=bootstrap_seed,
            label=f"{prefix}/route_completion",
        ),
        "min_ttc_paper_literal_center_seconds": _numeric(
            episodes,
            lambda row: row["min_ttc_seconds"]["paper_literal_center"],
            samples=bootstrap_samples,
            base_seed=bootstrap_seed,
            label=f"{prefix}/min_ttc",
        ),
        "six_camera_observations_per_second": _numeric(
            episodes,
            lambda row: row["timing"]["simulator_sensor_render"][
                "camera_batches_per_second"
            ],
            samples=bootstrap_samples,
            base_seed=bootstrap_seed,
            label=f"{prefix}/sensor_batch_fps",
        ),
        "sensor_images_per_second": _numeric(
            episodes,
            lambda row: row["timing"]["simulator_sensor_render"][
                "sensor_images_per_second"
            ],
            samples=bootstrap_samples,
            base_seed=bootstrap_seed,
            label=f"{prefix}/sensor_image_fps",
        ),
    }


def build_final_metrics(
    summary: Mapping[str, Any],
    paper_targets: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 260801761,
) -> dict[str, Any]:
    """Validate and summarize a complete 2-planner, 4-difficulty suite."""

    if summary.get("status") != "complete":
        raise ValueError(f"suite status is {summary.get('status')!r}, expected 'complete'")
    if int(summary.get("completed", -1)) != 400 or int(summary.get("expected", -1)) != 400:
        raise ValueError("formal report requires exactly 400/400 completed episodes")
    if summary.get("missing") or summary.get("invalid"):
        raise ValueError("formal report cannot include missing or invalid episodes")
    if len(manifest.get("scenarios", [])) != 200:
        raise ValueError("formal manifest must contain 200 scenarios")

    cells: dict[str, dict[str, Any]] = {planner: {} for planner in PLANNERS}
    all_by_planner: dict[str, list[Mapping[str, Any]]] = {planner: [] for planner in PLANNERS}
    for planner in PLANNERS:
        for difficulty in DIFFICULTIES:
            episodes = list(summary["episodes"][planner][difficulty])
            reference = _paper_cell(paper_targets, planner, difficulty)
            cells[planner][difficulty] = summarize_cell(
                episodes,
                planner=planner,
                difficulty=difficulty,
                paper_reference=reference,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            all_by_planner[planner].extend(episodes)

    paired = {
        difficulty: _paired_summary(
            summary["episodes"]["uniad"][difficulty],
            summary["episodes"]["vad"][difficulty],
            difficulty=difficulty,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for difficulty in DIFFICULTIES
    }
    overall = {
        planner: _overall(
            episodes,
            planner=planner,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        for planner, episodes in all_by_planner.items()
    }
    scenario_counts = Counter(str(row["scene"]) for row in manifest["scenarios"])
    difficulty_counts = Counter(str(row["difficulty"]) for row in manifest["scenarios"])
    seeds = [int(row["seed"]) for row in manifest["scenarios"]]
    return {
        "schema_version": 1,
        "benchmark": "decouplegs_closed_loop_final_metrics",
        "status": "complete",
        "source_suite": {
            "benchmark": summary["benchmark"],
            "completed": summary["completed"],
            "expected": summary["expected"],
        },
        "protocol_coverage": {
            "planners": list(PLANNERS),
            "difficulties": list(DIFFICULTIES),
            "episodes_per_planner_and_difficulty": 50,
            "scenario_count": len(manifest["scenarios"]),
            "planner_episode_count": 400,
            "episode_seconds": summary["paper_protocol"]["episode_seconds"],
            "control_frequency_hz": summary["paper_protocol"]["control_frequency_hz"],
            "maximum_steps": int(
                summary["paper_protocol"]["episode_seconds"]
                * summary["paper_protocol"]["control_frequency_hz"]
            ),
            "scene_counts": dict(sorted(scenario_counts.items())),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "asset_pool_size": len(manifest.get("asset_pool", [])),
            "unique_scenario_seeds": len(set(seeds)),
            "all_scenario_seeds_unique": len(set(seeds)) == len(seeds),
            "paired_scenarios_per_difficulty": 50,
        },
        "statistical_protocol": {
            "mean_std": "population mean and population std (ddof=0), matching the suite aggregator",
            "numeric_ci95": f"deterministic percentile bootstrap over episodes, {bootstrap_samples} resamples",
            "binary_ci95": "Wilson score interval",
            "bootstrap_seed": bootstrap_seed,
        },
        "metric_protocol": {
            "driving_score": "null: paper p_i penalty factors are not disclosed",
            "success_rate": "paper rule unresolved; safety-only proxy and strict H-METRIC-01 are separate",
            "route_completion": "D_traveled / D_planned",
            "min_ttc": "min_t ||x_ego(t)-x_agent(t)||_2 / ||v_agent(t)-v_ego(t)||_2",
            "fps": "six-camera observation batches/s and individual sensor images/s both retained",
        },
        "cells": cells,
        "overall": overall,
        "paired_uniad_minus_vad": paired,
        "paper_table_2_reference": paper_targets["table_2_sim_to_real_and_closed_loop"][
            "closed_loop"
        ],
        "known_non_identifiable_fields": [
            "Driving Score penalty factors p_i",
            "Success Rate terminal definition",
            "author scenario IDs and seeds",
            "FPS timing boundary and counting unit",
            "v_rel interpretation beyond the literal norm equation",
        ],
    }
