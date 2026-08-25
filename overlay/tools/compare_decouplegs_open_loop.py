#!/usr/bin/env python3
"""Compare rendered-input planner trajectories against paired real-input plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="LABEL=OPEN_LOOP_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-leading-frames",
        type=int,
        default=0,
        help="Optional temporal warm-up exclusion; default reports every frame",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "decouplegs_open_loop_behavior_consistency":
        raise ValueError(f"not a DecoupleGS open-loop result: {path}")
    return payload


def parse_variant(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"--variant must be LABEL=PATH, got {value!r}")
    return label, Path(path)


def compare(
    reference: dict[str, Any],
    variant: dict[str, Any],
    *,
    exclude_leading_frames: int,
) -> dict[str, Any]:
    if reference["planner"] != variant["planner"]:
        raise ValueError("reference and variant planners differ")
    reference_records = reference["records"]
    variant_records = variant["records"]
    if len(reference_records) != len(variant_records):
        raise ValueError("reference and variant frame counts differ")
    rows = []
    for index, (real, simulated) in enumerate(
        zip(reference_records, variant_records, strict=True)
    ):
        if real.get("sample_token") != simulated.get("sample_token"):
            raise ValueError(f"sample-token mismatch at sequence index {index}")
        real_plan = np.asarray(real["plan_traj"], dtype=np.float64)
        simulated_plan = np.asarray(simulated["plan_traj"], dtype=np.float64)
        if real_plan.shape != simulated_plan.shape or real_plan.ndim != 2:
            raise ValueError(f"plan shape mismatch at sequence index {index}")
        displacement = np.linalg.norm(
            simulated_plan[:, :2] - real_plan[:, :2], axis=-1
        )
        rows.append(
            {
                "sequence_index": index,
                "sample_token": real.get("sample_token"),
                "timestamp": real["timestamp"],
                "ade_to_reference_plan_m": float(displacement.mean()),
                "fde_to_reference_plan_m": float(displacement[-1]),
            }
        )
    selected = rows[exclude_leading_frames:]
    if not selected:
        raise ValueError("warm-up exclusion removed every frame")
    return {
        "label": variant["label"],
        "result_file": variant.get("result_file"),
        "frames": len(selected),
        "mADE_to_reference_plan_m": float(
            np.mean([row["ade_to_reference_plan_m"] for row in selected])
        ),
        "mFDE_to_reference_plan_m": float(
            np.mean([row["fde_to_reference_plan_m"] for row in selected])
        ),
        "mADE_to_logged_gt_m": variant["mADE_to_logged_gt_m"],
        "minTTC_paper_literal_center_s": variant[
            "minTTC_paper_literal_center_s"
        ],
        "records": rows,
    }


def main() -> None:
    args = parse_args()
    if args.exclude_leading_frames < 0:
        raise ValueError("--exclude-leading-frames must be non-negative")
    reference = load_result(args.reference)
    variants = []
    for label, path in map(parse_variant, args.variant):
        result = load_result(path)
        result["result_file"] = str(path.resolve())
        comparison = compare(
            reference,
            result,
            exclude_leading_frames=args.exclude_leading_frames,
        )
        comparison["label"] = label
        variants.append(comparison)
    payload = {
        "schema_version": 1,
        "benchmark": "decouplegs_paired_open_loop_plan_consistency",
        "planner": reference["planner"],
        "reference": {
            "label": reference["label"],
            "result_file": str(args.reference.resolve()),
            "mADE_to_logged_gt_m": reference["mADE_to_logged_gt_m"],
            "minTTC_paper_literal_center_s": reference[
                "minTTC_paper_literal_center_s"
            ],
        },
        "protocol": {
            "pairing": "identical ordered sample tokens and planner state",
            "metric": "mean Euclidean distance between rendered-input and real-input planned waypoints",
            "exclude_leading_frames": args.exclude_leading_frames,
            "paper_gap": (
                "Table 2 reports plan-to-logged-GT mADE; this paired metric "
                "additionally isolates visual-domain behavior drift"
            ),
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "planner": payload["planner"],
                "reference": payload["reference"],
                "variants": [
                    {key: item[key] for key in (
                        "label",
                        "mADE_to_reference_plan_m",
                        "mFDE_to_reference_plan_m",
                        "mADE_to_logged_gt_m",
                    )}
                    for item in variants
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
