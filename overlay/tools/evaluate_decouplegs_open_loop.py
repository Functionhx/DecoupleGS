#!/usr/bin/env python3
"""Compare matched open-loop planner trajectories across visual inputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("benchmark") != "decouplegs_open_loop_behavior_consistency":
        raise ValueError(f"not an open-loop result: {path}")
    return value


def record_map(value: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(record["timestamp"]): record for record in value["records"]}


def summarize_candidate(real: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    real_records = record_map(real)
    candidate_records = record_map(candidate)
    timestamps = sorted(set(real_records) & set(candidate_records))
    if not timestamps:
        raise ValueError(f"{candidate['label']} has no timestamps in common with real input")
    frame_errors = []
    per_horizon: dict[int, list[float]] = {}
    for timestamp in timestamps:
        real_plan = np.asarray(real_records[timestamp]["plan_traj"], dtype=np.float64)
        candidate_plan = np.asarray(candidate_records[timestamp]["plan_traj"], dtype=np.float64)
        count = min(len(real_plan), len(candidate_plan))
        distance = np.linalg.norm(candidate_plan[:count, :2] - real_plan[:count, :2], axis=-1)
        frame_errors.append(float(distance.mean()))
        for index, value in enumerate(distance, start=1):
            per_horizon.setdefault(index, []).append(float(value))
    return {
        "label": candidate["label"],
        "matched_frames": len(timestamps),
        "mADE_to_real_plan_m": statistics.fmean(frame_errors),
        "mADE_to_logged_gt_m": candidate.get("mADE_to_logged_gt_m"),
        "delta_mADE_to_logged_gt_vs_real_m": (
            None
            if candidate.get("mADE_to_logged_gt_m") is None
            or real.get("mADE_to_logged_gt_m") is None
            else candidate["mADE_to_logged_gt_m"] - real["mADE_to_logged_gt_m"]
        ),
        "minTTC_paper_literal_center_s": candidate.get("minTTC_paper_literal_center_s"),
        "per_horizon_ADE_to_real_plan_m": {
            f"{index * 0.5:.1f}s": statistics.fmean(values)
            for index, values in per_horizon.items()
        },
    }


def main() -> None:
    args = parse_args()
    real = load(args.real)
    candidates = [load(path) for path in args.candidate]
    if any(candidate["planner"] != real["planner"] for candidate in candidates):
        raise ValueError("all results must use the same planner")
    result = {
        "schema_version": 1,
        "benchmark": "decouplegs_open_loop_sim_to_real_comparison",
        "planner": real["planner"],
        "reference": {
            "label": real["label"],
            "mADE_to_logged_gt_m": real.get("mADE_to_logged_gt_m"),
            "minTTC_paper_literal_center_s": real.get("minTTC_paper_literal_center_s"),
            "frames_fed": real["frames_fed"],
            "frames_with_future_gt": real["frames_with_future_gt"],
        },
        "candidates": [summarize_candidate(real, candidate) for candidate in candidates],
        "protocol": {
            "paper_table_interpretation": "mADE is planner-vs-logged-future error; real input is therefore non-zero",
            "supplement": "mADE_to_real_plan directly isolates the visual sim-to-real behavior gap",
            "unresolved": "paper does not disclose exact frames, CAN bus, or actor synchronization",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"reference": result["reference"], "candidates": result["candidates"]}, indent=2))


if __name__ == "__main__":
    main()
