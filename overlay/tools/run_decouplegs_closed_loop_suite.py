#!/usr/bin/env python3
"""Run, resume, validate, and aggregate the DecoupleGS 4x50 closed-loop suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.closed_loop_metrics import aggregate_episodes, evaluate_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/benchmark/nuscenes/stress_scene0383/manifest.json",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/benchmark/local_nuscenes_base.yaml",
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
    parser.add_argument(
        "--decouple-option",
        action="append",
        default=[],
        help="Repeatable OmegaConf dot-list override forwarded to closed_loop.py.",
    )
    parser.add_argument("--planners", nargs="+", choices=("uniad", "vad"), default=("uniad", "vad"))
    parser.add_argument(
        "--difficulties",
        nargs="+",
        choices=("easy", "medium", "hard", "extreme"),
        default=("easy", "medium", "hard", "extreme"),
    )
    parser.add_argument("--episode-start", type=int, default=0, help="Inclusive episode index")
    parser.add_argument("--episode-stop", type=int, default=50, help="Exclusive episode index")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--penalty",
        action="append",
        default=[],
        help="Optional explicit DS penalty NAME=FACTOR; omitted for paper-protocol null DS.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{path} contains {len(value)} records")
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain an episode mapping")
    return value


def validate_record(path: Path, expected_steps: int = 40) -> tuple[bool, str, dict[str, Any] | None]:
    if not path.is_file():
        return False, "missing_data", None
    try:
        record = load_record(path)
    except Exception as error:  # preserve failed artifact for diagnosis
        return False, f"unreadable_data:{error!r}", None
    # A failed forced rerun can leave the previous data.pkl intact.  Treat a
    # newer failure status as authoritative so aggregation never silently
    # resurrects that stale episode.
    status_path = path.parent / "suite-status.json"
    if status_path.is_file() and status_path.stat().st_mtime >= path.stat().st_mtime:
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            status = None
        if isinstance(status, dict) and status.get("state") == "failed":
            reason = status.get("validation", "failed_rerun")
            return False, f"newer_failed_status:{reason}", record
    protocol = record.get("protocol", {}) or {}
    maximum_steps = protocol.get("maximum_steps")
    frames = record.get("frames", []) or []
    episode = record.get("episode", {}) or {}
    if maximum_steps != expected_steps:
        return False, f"wrong_protocol_steps:{maximum_steps!r}", record
    contracts = protocol.get("implementation_contracts", {}) or {}
    required_contracts = {
        "H-E2E-03",
        "H-PHYSICS-01",
        "H-PHYSICS-02",
        "H-PHYSICS-04",
        "H-BEHAVIOR-03",
        "H-RC-01",
        "H-PLAN-01",
        "H-NAV-01",
        "H-CONTROL-01",
    }
    missing_contracts = required_contracts - set(contracts)
    if missing_contracts:
        return False, f"stale_implementation_contract:{sorted(missing_contracts)!r}", record
    if not frames and episode.get("termination_reason") != "planner_failure":
        return False, "empty_episode", record
    if "termination_reason" not in episode:
        return False, "missing_termination", record
    return True, "valid", record


def validate_fresh_record(
    path: Path,
    *,
    started_wall_time: float,
    expected_steps: int = 40,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Reject a stale artifact left behind when a forced rerun crashes early."""

    valid, reason, record = validate_record(path, expected_steps=expected_steps)
    if not path.is_file():
        return valid, reason, record
    # Leave a one-second tolerance for filesystems with coarse timestamp
    # resolution, while still rejecting artifacts from a previous run.
    if path.stat().st_mtime < started_wall_time - 1.0:
        return False, "stale_artifact_from_previous_run", record
    return valid, reason, record


def output_dir(base: dict[str, Any], planner: str, scenario: dict[str, Any]) -> Path:
    return Path(str(base["output_dir"]) + planner) / f"{scenario['scene_name']}_{scenario['mode']}"


def parse_penalties(values: list[str]) -> dict[str, float] | None:
    if not values:
        return None
    result: dict[str, float] = {}
    for value in values:
        name, separator, factor = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid penalty {value!r}; expected NAME=FACTOR")
        result[name] = float(factor)
    return result


def selected_rows(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed_difficulties = set(args.difficulties)
    output = []
    for row in manifest["scenarios"]:
        episode = int(Path(row["path"]).stem.rsplit("_", 1)[-1])
        if row["difficulty"] not in allowed_difficulties:
            continue
        if not args.episode_start <= episode < args.episode_stop:
            continue
        output.append({**row, "episode": episode})
    return output


def save_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_one(
    planner: str,
    row: dict[str, Any],
    scenario_path: Path,
    scenario: dict[str, Any],
    base: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    destination = output_dir(base, planner, scenario)
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / "data.pkl"
    valid, reason, _ = validate_record(data_path)
    if valid and not args.force:
        print(f"SKIP {planner:5s} {row['difficulty']:7s} {row['episode']:02d}: valid", flush=True)
        return {"state": "skipped", "reason": reason, "output": str(destination)}

    command = [
        sys.executable,
        str(ROOT / "closed_loop.py"),
        "--scenario_path",
        str(scenario_path),
        "--base_path",
        str(args.base_config),
        "--camera_path",
        str(args.camera_config),
        "--kinematic_path",
        str(args.kinematic_config),
        "--decouple_config",
        str(args.decouple_config),
        "--ad",
        planner,
        "--ad_cuda",
        args.cuda,
        "--skip_hugsim_score",
    ]
    for option in args.decouple_option:
        command.extend(("--decouple-option", option))
    print(f"RUN  {planner:5s} {row['difficulty']:7s} {row['episode']:02d}", flush=True)
    started_wall = time.perf_counter()
    started_epoch = time.time()
    started_at = utc_now()
    timed_out = False
    with (destination / "simulator-output.txt").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    elapsed = time.perf_counter() - started_wall
    valid, reason, record = validate_fresh_record(
        data_path,
        started_wall_time=started_epoch,
    )
    state = "complete" if valid else "failed"
    status = {
        "state": state,
        "validation": reason,
        "planner": planner,
        "difficulty": row["difficulty"],
        "episode": row["episode"],
        "seed": row["seed"],
        "agents": row["agents"],
        "scenario": str(scenario_path.resolve()),
        "output": str(destination.resolve()),
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "timed_out": timed_out,
        "episode_steps": None if record is None else (record.get("episode", {}) or {}).get("steps"),
        "termination_reason": None
        if record is None
        else (record.get("episode", {}) or {}).get("termination_reason"),
    }
    save_status(destination / "suite-status.json", status)
    print(
        f"{state.upper():8s} {planner:5s} {row['difficulty']:7s} {row['episode']:02d} "
        f"{elapsed:.1f}s steps={status['episode_steps']} reason={status['termination_reason']}",
        flush=True,
    )
    return status


def aggregate_suite(
    manifest: dict[str, Any], manifest_dir: Path, base: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    penalties = parse_penalties(args.penalty)
    selected = selected_rows(manifest, args)
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        planner: {difficulty: [] for difficulty in args.difficulties} for planner in args.planners
    }
    missing: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for planner in args.planners:
        for row in selected:
            scenario_path = manifest_dir / row["path"]
            scenario = yaml.safe_load(scenario_path.read_text())
            destination = output_dir(base, planner, scenario)
            valid, reason, record = validate_record(destination / "data.pkl")
            if not (destination / "data.pkl").exists():
                missing.append({"planner": planner, **row})
                continue
            if not valid or record is None:
                invalid.append({"planner": planner, "validation": reason, **row})
                continue
            metrics = evaluate_episode(record, penalty_factors=penalties)
            episode_result = {
                "source": str(destination),
                "seed": row["seed"],
                "agents": row["agents"],
                "episode": row["episode"],
                **metrics,
            }
            groups[planner][row["difficulty"]].append(episode_result)
            # Always refresh the episode-local metric artifact.  Otherwise a
            # valid rerun can coexist with a stale paper-metrics.json from an
            # earlier implementation contract and invite accidental mixing.
            episode_report = {
                "schema_version": 2,
                "protocol": {
                    "route_completion": "distance_traveled / distance_planned",
                    "driving_score": "null unless explicit penalty factors are supplied",
                    "success_rate": "paper definition unavailable; all H-METRIC-01 variants emitted",
                    "min_ttc": "literal centre and closing-speed diagnostics emitted separately",
                    "implementation_contracts": (record.get("protocol", {}) or {}).get(
                        "implementation_contracts", {}
                    ),
                },
                "aggregate": aggregate_episodes([episode_result]),
                "episode_results": [episode_result],
            }
            (destination / "paper-metrics.json").write_text(
                json.dumps(json_safe(episode_report), indent=2, allow_nan=False) + "\n"
            )

    aggregated: dict[str, Any] = {}
    for planner, difficulties in groups.items():
        aggregated[planner] = {}
        for difficulty, episodes in difficulties.items():
            aggregated[planner][difficulty] = (
                None if not episodes else aggregate_episodes(episodes)
            )
    result = {
        "schema_version": 1,
        "benchmark": "decouplegs_closed_loop_4x50",
        "status": "complete" if not missing and not invalid else "partial",
        "paper_protocol": manifest["paper_protocol"],
        "hypotheses": manifest.get("hypotheses", []),
        "planners": list(args.planners),
        "difficulties": list(args.difficulties),
        "episode_range": [args.episode_start, args.episode_stop],
        "driving_score_status": (
            "computed_from_explicit_penalties"
            if penalties is not None
            else "null_because_paper_penalty_factors_are_not_disclosed"
        ),
        "penalty_factors": penalties,
        "aggregate": aggregated,
        "completed": sum(len(values) for difficulties in groups.values() for values in difficulties.values()),
        "expected": len(selected) * len(args.planners),
        "missing": missing,
        "invalid": invalid,
        "episodes": groups,
    }
    suite_root = Path(base["output_dir"]).parent / "closed-loop-suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    output = suite_root / "summary.json"
    output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed": result["completed"],
                "expected": result["expected"],
                "summary": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main() -> None:
    args = parse_args()
    if args.episode_start < 0 or args.episode_stop <= args.episode_start:
        raise ValueError("episode range must satisfy 0 <= start < stop")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    manifest = json.loads(args.manifest.read_text())
    manifest_dir = args.manifest.parent
    base = yaml.safe_load(args.base_config.read_text())
    rows = selected_rows(manifest, args)

    attempted = 0
    if not args.aggregate_only:
        for planner in args.planners:
            for row in rows:
                scenario_path = manifest_dir / row["path"]
                scenario = yaml.safe_load(scenario_path.read_text())
                run_one(planner, row, scenario_path, scenario, base, args)
                attempted += 1
                if args.max_runs is not None and attempted >= args.max_runs:
                    aggregate_suite(manifest, manifest_dir, base, args)
                    return
    aggregate_suite(manifest, manifest_dir, base, args)


if __name__ == "__main__":
    main()
