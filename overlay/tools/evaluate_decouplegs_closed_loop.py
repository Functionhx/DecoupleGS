#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.closed_loop_metrics import aggregate_episodes, evaluate_episode


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_penalties(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    output: dict[str, float] = {}
    for value in values:
        name, separator, factor = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid penalty {value!r}; expected NAME=FACTOR")
        output[name] = float(factor)
    return output


def _load_record(path: Path) -> dict[str, Any]:
    data_path = path / "data.pkl" if path.is_dir() else path
    with data_path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"{data_path} contains {len(value)} records; expected one")
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError(f"{data_path} does not contain an episode dictionary")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate auditable DecoupleGS closed-loop paper metrics"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Episode directories or data.pkl files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--penalty",
        action="append",
        help=(
            "Explicit DS factor NAME=FACTOR. The paper omits these factors, so DS remains null "
            "unless this option is supplied."
        ),
    )
    parser.add_argument("--success-completion", type=float, default=0.99)
    args = parser.parse_args()

    penalties = _parse_penalties(args.penalty)
    episodes = []
    for path in args.inputs:
        metrics = evaluate_episode(
            _load_record(path),
            penalty_factors=penalties,
            success_completion=args.success_completion,
        )
        episodes.append({"source": str(path), **metrics})
    output = {
        "protocol": {
            "paper_literal": {
                "route_completion": "distance_traveled / distance_planned",
                "driving_score": "RC * product_i(p_i ** C_i)",
                "min_ttc": "min_t ||x_ego-x_agent||_2 / ||v_ego-v_agent||_2",
            },
            "unresolved": [
                "The paper does not publish DS penalty factors p_i.",
                "The paper does not define SR; H-METRIC-01 is reported explicitly.",
                "The paper does not disambiguate v_rel; literal and closing variants are all emitted.",
            ],
        },
        "aggregate": aggregate_episodes(episodes),
        "episode_results": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = _json_safe(output)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False, default=_json_default)
        handle.write("\n")
    print(json.dumps(output["aggregate"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
