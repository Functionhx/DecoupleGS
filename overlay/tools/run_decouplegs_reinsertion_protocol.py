#!/usr/bin/env python3
"""Run the frozen DecoupleGS held-out vehicle reinsertion replacement protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_decouplegs_fidelity import aggregate_rows, evaluate as evaluate_metrics


VARIANTS: dict[str, dict[str, Any]] = {
    "raw-native": {
        "mode": "raw",
        "description": "uncompressed native HUGSIM dynamic Gaussian upper bound",
    },
    "compact-no-relight": {
        "mode": "compact",
        "description": "compressed canonical vehicle with SH rotation; no relighting",
    },
    "compact-ols": {
        "mode": "compact",
        "relighting": True,
        "description": "compressed vehicle plus public-HDRI global affine OLS",
    },
    "compact-ols-shadow": {
        "mode": "compact",
        "relighting": True,
        "shadow": True,
        "description": (
            "public-HDRI OLS plus contact shadow; logged pose preserved while "
            "the ground plane is estimated"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_reinsertion_public.yaml",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results/decouplegs/reinsertion-paper-protocol/evaluation-v1",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT
        / "results/decouplegs/reinsertion-paper-protocol/calibration/calibrations",
    )
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument(
        "--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS)
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-mask", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_is_current(path: Path, expected_pairs: int, variant: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    config = VARIANTS[variant]
    modules = payload.get("modules", {})
    return (
        payload.get("pair_count") == expected_pairs
        and modules.get("track_selection") == "manifest_vehicle_only"
        and modules.get("compact") == (config["mode"] == "compact")
        and modules.get("relighting") == bool(config.get("relighting", False))
        and modules.get("contact_shadows") == bool(config.get("shadow", False))
        and modules.get("preserve_logged_pose") == bool(config.get("shadow", False))
    )


def mask_is_current(path: Path, pairs: Path, expected_pairs: int, dilation: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("pair_count") == expected_pairs
        and payload.get("pairs_manifest") == str(pairs.resolve())
        and payload.get("protocol", {}).get("morphological_dilation_pixels") == dilation
    )


def eval_is_current(path: Path, mask_index: Path, expected_pairs: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 2
        and payload.get("pair_count") == expected_pairs
        and payload.get("protocol", {}).get("mask_index")
        == str(mask_index.resolve())
    )


def rendering_command(
    *,
    protocol_path: Path,
    paths: dict[str, str],
    calibration_root: Path,
    scene_id: str,
    variant: str,
    output: Path,
    limit: int | None,
) -> list[str]:
    config = VARIANTS[variant]
    command = [
        sys.executable,
        str(ROOT / "tools/render_decouplegs_holdout.py"),
        "--model-path",
        str(Path(paths["model_root"]) / scene_id),
        "--source-path",
        str(Path(paths["source_root"]) / scene_id),
        "--output",
        str(output),
        "--mode",
        config["mode"],
        "--split",
        "test",
        "--warmup",
        "2",
        "--mask-dilation",
        "0",
        "--track-manifest",
        str(protocol_path),
    ]
    if config["mode"] == "compact":
        command.extend(
            [
                "--compact-dir",
                str(Path(paths["compact_root"]) / scene_id),
            ]
        )
    if config.get("relighting"):
        command.extend(
            [
                "--relighting",
                "--calibration-dir",
                str(calibration_root / scene_id),
            ]
        )
    if config.get("shadow"):
        command.extend(
            [
                "--contact-shadows",
                "--opacity-grounding",
                "--preserve-logged-pose",
            ]
        )
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def aggregate_evaluation(
    *,
    protocol_path: Path,
    result_root: Path,
    scenes: list[str],
    variants: list[str],
    mask_kinds: dict[str, str],
    expected_pairs: int,
    limited: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "decouplegs_heldout_vehicle_reinsertion_public_replacement",
        "status": "smoke" if limited else "complete",
        "claim_scope": "same_metric_contract_but_not_author_vehicle_or_calibration_data",
        "paper_exact": False,
        "protocol": str(protocol_path.resolve()),
        "scenes": scenes,
        "variants": {name: VARIANTS[name] for name in variants},
        "expected_pairs_per_scene": expected_pairs,
        "mask_results": {},
        "unpublished_author_choices": [
            "held-out vehicle identities",
            "held-out frame sampling",
            "SAM checkpoint and prompting/post-processing",
            "cross-image peak aggregation",
            "zero-norm PAE handling",
            "HDRI/material/target-renderer calibration supervision",
        ],
    }
    for mask_name, metrics_filename in mask_kinds.items():
        variant_results: dict[str, Any] = {}
        reference_ids: dict[str, list[str]] = {}
        for variant in variants:
            rows: list[dict[str, Any]] = []
            instance_rows: list[dict[str, Any]] = []
            per_scene: dict[str, Any] = {}
            runtimes = []
            for scene_id in scenes:
                metrics_path = result_root / scene_id / variant / metrics_filename
                metrics = read_json(metrics_path)
                local_rows = metrics["pairs"]
                local_instances = metrics.get("instances", [])
                rows.extend(local_rows)
                instance_rows.extend(local_instances)
                per_scene[scene_id] = metrics["aggregate"]["paper_mask_metrics"]
                pairs_payload = read_json(result_root / scene_id / variant / "pairs.json")
                runtimes.append(pairs_payload["runtime"])
                ids = [str(row["id"]) for row in local_rows]
                if scene_id not in reference_ids:
                    reference_ids[scene_id] = ids
                elif reference_ids[scene_id] != ids:
                    raise ValueError(
                        f"unpaired frame IDs for {scene_id}/{variant}/{mask_name}"
                    )
            per_track = {
                track_id: aggregate_rows(
                    [row for row in instance_rows if row["track_id"] == track_id]
                )
                for track_id in sorted(
                    {str(row["track_id"]) for row in instance_rows}
                )
            }
            variant_results[variant] = {
                "metrics": aggregate_rows(rows),
                "per_scene": per_scene,
                "per_track": per_track,
                "runtime": {
                    "mean_camera_fps_across_scene_processes": sum(
                        float(entry["camera_fps"]) for entry in runtimes
                    )
                    / len(runtimes),
                    "max_peak_allocated_mib": max(
                        float(entry["peak_allocated_mib"]) for entry in runtimes
                    ),
                    "note": "offline single-camera renderer; excludes model loading",
                },
            }
        baseline = variant_results.get("compact-no-relight")
        if baseline is not None:
            baseline_mean = baseline["metrics"]["per_image"]
            for variant, entry in variant_results.items():
                current = entry["metrics"]["per_image"]
                entry["delta_vs_compact_no_relight"] = {
                    key: (
                        None
                        if current[key]["mean"] is None
                        or baseline_mean[key]["mean"] is None
                        else current[key]["mean"] - baseline_mean[key]["mean"]
                    )
                    for key in (
                        "psnr_vehicle_pixel_l2_db",
                        "peak_intensity_error",
                        "peak_angular_error_deg",
                    )
                }
        result["mask_results"][mask_name] = variant_results
    return result


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    scene_ids = list(protocol["scenes"])
    if args.scenes is not None:
        unknown = set(args.scenes) - set(scene_ids)
        if unknown:
            raise ValueError(f"unknown scenes: {sorted(unknown)}")
        scene_ids = list(args.scenes)
    variants = list(args.variants)
    if "compact-no-relight" not in variants:
        raise ValueError("compact-no-relight is required as the paired SAM prompt source")
    full_expected = int(protocol["frame_protocol"]["expected_views_per_scene"])
    expected_pairs = full_expected if args.limit is None else min(args.limit, full_expected)
    paths = protocol["paths"]
    checkpoint = Path(protocol["mask_protocol"]["checkpoint"])
    args.result_root.mkdir(parents=True, exist_ok=True)

    for scene_id in scene_ids:
        print(f"\n[{scene_id}] render paired variants", flush=True)
        for variant in variants:
            output = args.result_root / scene_id / variant
            pairs_path = output / "pairs.json"
            if args.force_render or not render_is_current(
                pairs_path, expected_pairs, variant
            ):
                run(
                    rendering_command(
                        protocol_path=args.protocol.resolve(),
                        paths=paths,
                        calibration_root=args.calibration_root.resolve(),
                        scene_id=scene_id,
                        variant=variant,
                        output=output,
                        limit=args.limit,
                    )
                )
            else:
                print(f"  reuse {variant}/pairs.json", flush=True)

        source_pairs = args.result_root / scene_id / "compact-no-relight/pairs.json"
        mask_root = args.result_root / scene_id / "sam-vit-h"
        primary_index = mask_root / "primary-undilated/mask-index.json"
        if args.force_mask or not mask_is_current(
            primary_index, source_pairs, expected_pairs, 0
        ):
            run(
                [
                    sys.executable,
                    str(ROOT / "tools/refine_decouplegs_masks_sam.py"),
                    "--pairs",
                    str(source_pairs),
                    "--checkpoint",
                    str(checkpoint),
                    "--model-type",
                    str(protocol["mask_protocol"]["model_type"]),
                    "--output-dir",
                    str(mask_root / "primary-undilated/masks"),
                    "--output-index",
                    str(primary_index),
                    "--dilation",
                    "0",
                    "--device",
                    args.sam_device,
                ]
            )
        else:
            print("  reuse undilated SAM masks", flush=True)

        dilated_index = mask_root / "sensitivity-dilated5/mask-index.json"
        if args.force_mask or not mask_is_current(
            dilated_index, source_pairs, expected_pairs, 5
        ):
            run(
                [
                    sys.executable,
                    str(ROOT / "tools/dilate_decouplegs_mask_index.py"),
                    "--input-index",
                    str(primary_index),
                    "--output-dir",
                    str(mask_root / "sensitivity-dilated5/masks"),
                    "--output-index",
                    str(dilated_index),
                    "--radius",
                    "5",
                ]
            )
        else:
            print("  reuse 5px SAM sensitivity masks", flush=True)

        mask_evaluations = {
            "metrics-sam-primary.json": primary_index,
            "metrics-sam-dilated5.json": dilated_index,
        }
        for variant in variants:
            pairs_path = args.result_root / scene_id / variant / "pairs.json"
            for filename, mask_index in mask_evaluations.items():
                output = args.result_root / scene_id / variant / filename
                if args.force_eval or not eval_is_current(
                    output, mask_index, expected_pairs
                ):
                    metrics = evaluate_metrics(
                        argparse.Namespace(
                            pairs=pairs_path,
                            output=output,
                            mask_index=mask_index,
                            device="cuda",
                            no_lpips=True,
                            paper_only=True,
                            quiet=True,
                            min_mask_pixels=1,
                            pae_support_threshold=1.0 / 255.0,
                        )
                    )
                    output.write_text(
                        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    mean = metrics["aggregate"]["mean_per_image"]
                    print(
                        f"  {variant}/{filename}: "
                        f"n={metrics['aggregate']['vehicle_metric_images']} "
                        f"PSNR={mean['psnr_vehicle_pixel_l2_db']!s} "
                        f"PIE={mean['peak_intensity_error']!s} "
                        f"PAE={mean['peak_angular_error_deg']!s}",
                        flush=True,
                    )
                else:
                    print(f"  reuse {variant}/{filename}", flush=True)

    summary = aggregate_evaluation(
        protocol_path=args.protocol,
        result_root=args.result_root,
        scenes=scene_ids,
        variants=variants,
        mask_kinds={
            "sam_primary_undilated": "metrics-sam-primary.json",
            "sam_sensitivity_dilated5": "metrics-sam-dilated5.json",
        },
        expected_pairs=expected_pairs,
        limited=args.limit is not None or len(scene_ids) != len(protocol["scenes"]),
    )
    summary_path = args.result_root / "protocol-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scenes": scene_ids,
                "variants": variants,
                "summary": str(summary_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
