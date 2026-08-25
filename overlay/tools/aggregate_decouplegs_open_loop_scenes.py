#!/usr/bin/env python3
"""Aggregate strict DecoupleGS keyframe benchmarks across matched scenes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.benchmark_summary import aggregate_scene_results, load_scene_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        action="append",
        required=True,
        metavar="NAME=RESULT_DIR",
        help="Repeat for each scene; directory contains raw/, compact-base/, and paired JSONs",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scene_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"invalid --scene {value!r}; expected NAME=RESULT_DIR")
    return name, Path(path)


def main() -> None:
    args = parse_args()
    scene_specs = [scene_argument(value) for value in args.scene]
    names = [name for name, _ in scene_specs]
    if len(names) != len(set(names)):
        raise ValueError("scene names must be unique")
    scenes = [load_scene_result(name, path) for name, path in scene_specs]
    result = aggregate_scene_results(scenes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
