#!/usr/bin/env python3
"""Calibrate and audit DecoupleGS relighting on the frozen 20-asset cohort.

This is a replacement protocol for unpublished author data, not a claim of
paper-heldout photometric reproduction. It uses disjoint HDRI train,
validation, and test panoramas; sweeps ridge only on validation; writes every
experimental calibration under ``results/``; and never modifies asset dirs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hdri_protocol import (
    HDRIEnvironment,
    TARGET_MODELS,
    TARGET_DEFINITION_VERSION,
    apply_probe_batch,
    build_probe_set,
    descriptor_domain,
    error_metrics,
    estimate_normals,
    improvement_gate,
    load_hdr,
    make_targets,
    sha256_file,
    tonemap,
    validation_score,
)
from decouplegs.relighting import RelightingNormalEquations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_assets_20.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_hdri_public.yaml",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ROOT / "data/decouplegs/3DRealCar",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results/decouplegs/relighting/public-hdri-cohort-v1",
    )
    parser.add_argument(
        "--target-models",
        nargs="+",
        choices=TARGET_MODELS,
        default=TARGET_MODELS,
    )
    parser.add_argument("--sample-primitives", type=int, default=None)
    parser.add_argument("--start-at")
    parser.add_argument("--stop-after")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    return parser.parse_args()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sampled_asset(
    compact: CompactGaussianAsset,
    count: int,
    seed: int,
) -> Any:
    decoded = compact.decode(device="cpu")
    if count <= 0:
        raise ValueError("sample_primitives must be positive")
    count = min(count, len(decoded))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(decoded), generator=generator)[:count]
    return decoded.select(indices)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def write_environment_contact_sheet(
    environments: Sequence[HDRIEnvironment],
    output: Path,
) -> None:
    """Make the exact input lighting set visible inside the workspace."""

    panel_width, image_height, label_height, columns = 384, 192, 42, 3
    rows = math.ceil(len(environments) / columns)
    canvas = Image.new(
        "RGB", (panel_width * columns, (image_height + label_height) * rows), "black"
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    split_colors = {
        "train": (130, 220, 150),
        "validation": (255, 210, 100),
        "test": (255, 135, 135),
    }
    for index, environment in enumerate(environments):
        image = load_hdr(environment.path, panel_width)
        mapped = tonemap(image, 0.8)
        preview = Image.fromarray(np.round(mapped * 255.0).astype(np.uint8))
        x = (index % columns) * panel_width
        y = (index // columns) * (image_height + label_height)
        canvas.paste(preview, (x, y))
        label = f"{environment.split}: {environment.identifier}"
        draw.rectangle((x, y + image_height, x + panel_width, y + image_height + label_height), fill="black")
        draw.text(
            (x + 8, y + image_height + 10),
            label,
            fill=split_colors[environment.split],
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "min": min(finite),
        "max": max(finite),
    }


def fit_model(
    *,
    asset_id: str,
    asset_path: Path,
    asset_sha256: str,
    canonical: torch.Tensor,
    normals: torch.Tensor,
    probes: Any,
    target_model: str,
    ridge_candidates: Sequence[float],
    ridge_prior: str,
    protocol_fingerprint: str,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    descriptors = probes.descriptors
    targets = make_targets(
        target_model,
        canonical,
        descriptors,
        normals=normals,
    )
    flattened = canonical.reshape(canonical.shape[0], -1)
    target_flattened = targets.reshape(targets.shape[0], targets.shape[1], -1)
    train_mask = probes.mask("train")
    validation_mask = probes.mask("validation")
    test_mask = probes.mask("test")

    train_statistics = RelightingNormalEquations.from_samples(
        descriptors[train_mask],
        flattened,
        target_flattened[train_mask],
    )
    validation_targets = targets[validation_mask]
    ridge_trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], float] | None = None
    for ridge in ridge_candidates:
        calibration = train_statistics.solve(
            ridge=float(ridge), ridge_prior=ridge_prior
        )
        prediction = apply_probe_batch(
            calibration, canonical, descriptors[validation_mask]
        )
        metrics = error_metrics(prediction, validation_targets)
        score = validation_score(metrics)
        ridge_trials.append(
            {"ridge": float(ridge), "score": list(score), "metrics": metrics}
        )
        candidate = (score, float(ridge))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    selected_ridge = best[1]

    fit_mask = train_mask | validation_mask
    final_statistics = RelightingNormalEquations.from_samples(
        descriptors[fit_mask],
        flattened,
        target_flattened[fit_mask],
    )
    calibration = final_statistics.solve(
        ridge=selected_ridge, ridge_prior=ridge_prior
    )
    test_targets = targets[test_mask]
    prediction = apply_probe_batch(calibration, canonical, descriptors[test_mask])
    baseline = canonical[None].expand_as(test_targets)
    baseline_metrics = error_metrics(baseline, test_targets)
    calibration_metrics = error_metrics(prediction, test_targets)
    aggregate_gate = improvement_gate(baseline_metrics, calibration_metrics)
    per_environment: dict[str, Any] = {}
    for environment in probes.environments:
        if environment["split"] != "test":
            continue
        environment_mask = torch.tensor(
            [
                entry["split"] == "test" and entry["hdri_id"] == environment["id"]
                for entry in probes.metadata
            ],
            dtype=torch.bool,
        )
        local_targets = targets[environment_mask]
        local_prediction = apply_probe_batch(
            calibration, canonical, descriptors[environment_mask]
        )
        local_baseline = canonical[None].expand_as(local_targets)
        local_baseline_metrics = error_metrics(local_baseline, local_targets)
        local_calibration_metrics = error_metrics(local_prediction, local_targets)
        per_environment[environment["id"]] = {
            "probes": int(environment_mask.sum()),
            "no_relighting": local_baseline_metrics,
            "ols_relighting": local_calibration_metrics,
            "improvement_gate": improvement_gate(
                local_baseline_metrics, local_calibration_metrics
            ),
        }
    gate = {
        "passed": bool(
            aggregate_gate["passed"]
            and all(
                entry["improvement_gate"]["passed"]
                for entry in per_environment.values()
            )
        ),
        "aggregate": aggregate_gate,
        "all_test_environments_passed": all(
            entry["improvement_gate"]["passed"]
            for entry in per_environment.values()
        ),
    }

    # An operator-oracle pass proves the implementation can recover Eq. (16),
    # but it is not meaningful physical supervision. Covariance diffuse is
    # experimentally usable only when it also clears every held-out gate.
    experimental_runtime_eligible = bool(
        target_model == "covariance_diffuse" and gate["passed"]
    )
    calibration.metadata.update(
        {
            "hypothesis": "H-LIGHT-03",
            "operator": "DecoupleGS Supplementary Eqs. (14)-(16) global affine OLS",
            "supervision_kind": target_model,
            "supervision_source": "public_HDRI_replacement_without_author_renderer",
            "paper_exact": False,
            "paper_exact_deployment_eligible": False,
            "experimental_runtime_eligible": experimental_runtime_eligible,
            "heldout_gate_passed": bool(gate["passed"]),
            "asset_id": asset_id,
            "asset": str(asset_path.resolve()),
            "asset_sha256": asset_sha256,
            "protocol_fingerprint": protocol_fingerprint,
            "descriptor_sha256": probes.descriptor_sha256,
            "descriptor_domain_fit": descriptor_domain(descriptors[fit_mask]),
            "splits": {
                split: int(probes.mask(split).sum()) for split in ("train", "validation", "test")
            },
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "experimental-calibration.pt"
    calibration.save(calibration_path)
    calibration_sha256 = sha256_file(calibration_path)

    report = {
        "schema_version": 1,
        "benchmark": "decouplegs_public_hdri_asset_relighting",
        "status": "public_proxy_not_paper_heldout_images",
        "hypothesis": "H-LIGHT-03",
        "asset_id": asset_id,
        "asset": str(asset_path.resolve()),
        "asset_sha256": asset_sha256,
        "sample_primitives": canonical.shape[0],
        "target_model": target_model,
        "target_interpretation": (
            "operator_oracle_exactly_representable_by_eq15"
            if target_model == "global_affine"
            else "covariance_normal_diffuse_stress_test_without_mesh_materials"
        ),
        "protocol_fingerprint": protocol_fingerprint,
        "descriptor_sha256": probes.descriptor_sha256,
        "ridge_prior": ridge_prior,
        "selected_ridge": selected_ridge,
        "ridge_trials": ridge_trials,
        "split_probes": {
            split: int(probes.mask(split).sum()) for split in ("train", "validation", "test")
        },
        "test": {
            "no_relighting": baseline_metrics,
            "ols_relighting": calibration_metrics,
            "improvement_gate": gate,
            "per_environment": per_environment,
        },
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": calibration_sha256,
        "paper_exact_deployment_eligible": False,
        "experimental_runtime_eligible": experimental_runtime_eligible,
        "limitation": (
            "Author HDRIs, vehicle meshes/materials, target renderer, and relit "
            "per-primitive colors are unpublished. Primitive proxy metrics are "
            "not the paper's masked image PAE/PSNR."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def report_is_current(
    path: Path,
    *,
    asset_sha256: str,
    protocol_fingerprint: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        report.get("asset_sha256") == asset_sha256
        and report.get("protocol_fingerprint") == protocol_fingerprint
        and Path(report.get("calibration", "")).is_file()
    )


def aggregate(
    *,
    asset_ids: Sequence[str],
    target_models: Sequence[str],
    result_root: Path,
    protocol_fingerprint: str,
    asset_manifest: Path,
    protocol_path: Path,
    probe_manifest_path: Path,
) -> dict[str, Any]:
    metric_keys = (
        "supported_angular_mean_deg",
        "supported_angular_p95_deg",
        "peak_intensity_error",
        "primitive_rgb_psnr_l2_db",
    )
    models: dict[str, Any] = {}
    for target_model in target_models:
        rows: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            path = result_root / "assets" / asset_id / target_model / "report.json"
            if not path.is_file():
                continue
            report = json.loads(path.read_text())
            if report.get("protocol_fingerprint") != protocol_fingerprint:
                continue
            baseline = report["test"]["no_relighting"]
            calibrated = report["test"]["ols_relighting"]
            rows.append(
                {
                    "asset_id": asset_id,
                    "report": str(path.resolve()),
                    "calibration": report["calibration"],
                    "selected_ridge": report["selected_ridge"],
                    "heldout_gate_passed": report["test"]["improvement_gate"]["passed"],
                    "experimental_runtime_eligible": report["experimental_runtime_eligible"],
                    "no_relighting": baseline,
                    "ols_relighting": calibrated,
                    "delta": {
                        key: calibrated[key] - baseline[key] for key in metric_keys
                    },
                }
            )
        models[target_model] = {
            "completed_assets": len(rows),
            "heldout_gate_passed_assets": sum(row["heldout_gate_passed"] for row in rows),
            "experimental_runtime_eligible_assets": [
                row["asset_id"] for row in rows if row["experimental_runtime_eligible"]
            ],
            "aggregate": {
                "no_relighting": {
                    key: finite_summary([row["no_relighting"][key] for row in rows])
                    for key in metric_keys
                },
                "ols_relighting": {
                    key: finite_summary([row["ols_relighting"][key] for row in rows])
                    for key in metric_keys
                },
                "delta_ols_minus_none": {
                    key: finite_summary([row["delta"][key] for row in rows])
                    for key in metric_keys
                },
            },
            "assets": rows,
        }
    complete = all(models[name]["completed_assets"] == len(asset_ids) for name in target_models)
    summary = {
        "schema_version": 1,
        "benchmark": "decouplegs_public_hdri_relighting_frozen_20_asset_cohort",
        "status": "complete" if complete else "partial",
        "hypothesis": "H-LIGHT-03",
        "claim_scope": "controlled_public_proxy_not_paper_heldout_images",
        "paper_exact_deployment_eligible": False,
        "expected_assets": len(asset_ids),
        "asset_manifest": str(asset_manifest.resolve()),
        "protocol": str(protocol_path.resolve()),
        "protocol_fingerprint": protocol_fingerprint,
        "probe_manifest": str(probe_manifest_path.resolve()),
        "models": models,
        "missing_author_artifacts": [
            "HDRI calibration probe set",
            "canonical vehicle mesh and materials used for relighting",
            "target relighting renderer and settings",
            "ground-truth relit per-primitive colors",
        ],
    }
    path = result_root / "cohort-summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    if args.sample_primitives is not None and args.sample_primitives <= 0:
        raise ValueError("--sample-primitives must be positive")
    asset_manifest = yaml.safe_load(args.asset_manifest.read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    asset_ids = list(asset_manifest["assets"])
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("asset manifest contains duplicate IDs")
    environments = [
        HDRIEnvironment(
            identifier=str(entry["id"]),
            path=Path(entry["path"]),
            split=str(entry["split"]),
        )
        for entry in protocol["environments"]
    ]
    settings = dict(protocol["protocol"])
    if int(settings.get("target_definition_version", -1)) != TARGET_DEFINITION_VERSION:
        raise ValueError(
            "protocol target_definition_version does not match the implementation "
            f"({settings.get('target_definition_version')} != {TARGET_DEFINITION_VERSION})"
        )
    sample_primitives = int(
        settings["sample_primitives"]
        if args.sample_primitives is None
        else args.sample_primitives
    )
    args.result_root.mkdir(parents=True, exist_ok=True)
    probes = build_probe_set(
        environments,
        environment_width=int(settings["environment_width"]),
        yaw_rotations=int(settings["yaw_rotations"]),
        exposures=[float(value) for value in settings["exposures"]],
        tints=settings["tints"],
    )
    protocol_identity = {
        "schema_version": protocol["schema_version"],
        "settings": settings,
        "sample_primitives_override": sample_primitives,
        "environments": probes.environments,
        "descriptor_sha256": probes.descriptor_sha256,
        "target_models": list(args.target_models),
    }
    protocol_fingerprint = digest_json(protocol_identity)
    probe_manifest = {
        "schema_version": 1,
        "benchmark": "decouplegs_public_hdri_probe_manifest",
        "status": protocol["status"],
        "hypothesis": protocol["hypothesis"],
        "note": protocol["note"],
        "protocol": settings,
        "sample_primitives": sample_primitives,
        "protocol_fingerprint": protocol_fingerprint,
        "descriptor_sha256": probes.descriptor_sha256,
        "probe_count": probes.descriptors.shape[0],
        "split_probes": {
            split: int(probes.mask(split).sum()) for split in ("train", "validation", "test")
        },
        "environments": probes.environments,
        "probe_metadata": probes.metadata,
        "claim_scope": "public replacement protocol, not author calibration data",
    }
    probe_manifest_path = args.result_root / "probe-manifest.json"
    probe_manifest_path.write_text(
        json.dumps(probe_manifest, indent=2, allow_nan=False) + "\n"
    )
    torch.save(
        {
            "descriptors": probes.descriptors,
            "metadata": probes.metadata,
            "protocol_fingerprint": protocol_fingerprint,
        },
        args.result_root / "probes.pt",
    )
    write_environment_contact_sheet(
        environments, args.result_root / "hdri-contact-sheet.jpg"
    )
    print(
        json.dumps(
            {
                "protocol_fingerprint": protocol_fingerprint,
                "probe_count": probes.descriptors.shape[0],
                "split_probes": probe_manifest["split_probes"],
                "contact_sheet": str((args.result_root / "hdri-contact-sheet.jpg").resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.probe_only:
        return

    selected_ids = asset_ids
    if args.start_at:
        selected_ids = selected_ids[selected_ids.index(args.start_at) :]
    if args.stop_after:
        selected_ids = selected_ids[: selected_ids.index(args.stop_after) + 1]
    ridge_candidates = [float(value) for value in settings["ridge_candidates"]]
    if not ridge_candidates or any(value < 0 for value in ridge_candidates):
        raise ValueError("ridge_candidates must be non-empty and non-negative")
    ridge_prior = str(settings["ridge_prior"])
    seed = int(settings["seed"])

    for index, asset_id in enumerate(selected_ids, start=1):
        asset_path = args.asset_root / asset_id / "decouplegs-visgate.dgs"
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        asset_sha256 = sha256_file(asset_path)
        pending = [
            target_model
            for target_model in args.target_models
            if args.force
            or not report_is_current(
                args.result_root
                / "assets"
                / asset_id
                / target_model
                / "report.json",
                asset_sha256=asset_sha256,
                protocol_fingerprint=protocol_fingerprint,
            )
        ]
        print(
            f"[{index}/{len(selected_ids)}] {asset_id}: "
            + (", ".join(pending) if pending else "current; skipped"),
            flush=True,
        )
        if not pending:
            continue
        compact = CompactGaussianAsset.load(asset_path)
        sample = sampled_asset(compact, sample_primitives, seed)
        canonical = sample.sh.to(torch.float32)
        normals = estimate_normals(sample).to(torch.float32)
        del compact, sample
        for target_model in pending:
            report = fit_model(
                asset_id=asset_id,
                asset_path=asset_path,
                asset_sha256=asset_sha256,
                canonical=canonical,
                normals=normals,
                probes=probes,
                target_model=target_model,
                ridge_candidates=ridge_candidates,
                ridge_prior=ridge_prior,
                protocol_fingerprint=protocol_fingerprint,
                output_dir=args.result_root / "assets" / asset_id / target_model,
            )
            test = report["test"]
            print(
                json.dumps(
                    {
                        "asset_id": asset_id,
                        "target_model": target_model,
                        "selected_ridge": report["selected_ridge"],
                        "gate_passed": test["improvement_gate"]["passed"],
                        "no_relighting": test["no_relighting"],
                        "ols_relighting": test["ols_relighting"],
                        "elapsed_seconds": report["elapsed_seconds"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            gc.collect()
        del canonical, normals
        gc.collect()

    summary = aggregate(
        asset_ids=asset_ids,
        target_models=args.target_models,
        result_root=args.result_root,
        protocol_fingerprint=protocol_fingerprint,
        asset_manifest=args.asset_manifest,
        protocol_path=args.protocol,
        probe_manifest_path=probe_manifest_path,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "expected_assets": summary["expected_assets"],
                "models": {
                    name: {
                        "completed_assets": value["completed_assets"],
                        "heldout_gate_passed_assets": value["heldout_gate_passed_assets"],
                    }
                    for name, value in summary["models"].items()
                },
                "summary": str((args.result_root / "cohort-summary.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
