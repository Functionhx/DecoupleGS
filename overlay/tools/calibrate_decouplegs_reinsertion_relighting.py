#!/usr/bin/env python3
"""Fit public-HDRI OLS sidecars for the frozen reinsertion vehicle cohort.

The author calibration images/renderer are not public. This tool deliberately
uses the already frozen disjoint public HDRI protocol and deploys only the
physically motivated covariance-diffuse model after its held-out HDRI gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hdri_protocol import (
    HDRIEnvironment,
    TARGET_DEFINITION_VERSION,
    build_probe_set,
    estimate_normals,
    sha256_file,
)
from tools.benchmark_decouplegs_relighting_cohort import (
    digest_json,
    fit_model,
    sampled_asset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reinsertion-protocol",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_reinsertion_public.yaml",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results/decouplegs/reinsertion-paper-protocol/calibration",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reinsertion = yaml.safe_load(args.reinsertion_protocol.read_text(encoding="utf-8"))
    hdri_path = Path(reinsertion["paths"]["public_hdri_protocol"])
    hdri = yaml.safe_load(hdri_path.read_text(encoding="utf-8"))
    settings = dict(hdri["protocol"])
    if int(settings.get("target_definition_version", -1)) != TARGET_DEFINITION_VERSION:
        raise ValueError("HDRI protocol target definition does not match implementation")
    environments = [
        HDRIEnvironment(
            identifier=str(entry["id"]),
            path=Path(entry["path"]),
            split=str(entry["split"]),
        )
        for entry in hdri["environments"]
    ]
    probes = build_probe_set(
        environments,
        environment_width=int(settings["environment_width"]),
        yaw_rotations=int(settings["yaw_rotations"]),
        exposures=[float(value) for value in settings["exposures"]],
        tints=settings["tints"],
    )
    protocol_identity = {
        "schema_version": hdri["schema_version"],
        "settings": settings,
        "environments": probes.environments,
        "descriptor_sha256": probes.descriptor_sha256,
        "target_models": ["covariance_diffuse"],
        "cohort": reinsertion["scenes"],
    }
    protocol_fingerprint = digest_json(protocol_identity)
    sample_primitives = int(settings["sample_primitives"])
    seed = int(settings["seed"])
    ridge_candidates = [float(value) for value in settings["ridge_candidates"]]
    ridge_prior = str(settings["ridge_prior"])
    compact_root = Path(reinsertion["paths"]["compact_root"])
    args.result_root.mkdir(parents=True, exist_ok=True)

    rows = []
    entries = [
        (scene_id, vehicle)
        for scene_id, scene in reinsertion["scenes"].items()
        for vehicle in scene["vehicles"]
    ]
    for index, (scene_id, vehicle) in enumerate(entries, start=1):
        track_id = str(vehicle["track_id"])
        asset_id = f"{scene_id}/{track_id}"
        asset_path = compact_root / scene_id / f"dynamic_{track_id}.dgs"
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        asset_sha256 = sha256_file(asset_path)
        output_dir = args.result_root / "assets" / scene_id / track_id
        report_path = output_dir / "report.json"
        current = False
        if not args.force and report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            current = (
                previous.get("asset_sha256") == asset_sha256
                and previous.get("protocol_fingerprint") == protocol_fingerprint
                and Path(previous.get("calibration", "")).is_file()
            )
        print(
            f"[{index}/{len(entries)}] {asset_id}: "
            + ("current; reuse" if current else "fit covariance_diffuse OLS"),
            flush=True,
        )
        if current:
            report = previous
        else:
            compact = CompactGaussianAsset.load(asset_path)
            sample = sampled_asset(compact, sample_primitives, seed)
            canonical = sample.sh.to(torch.float32)
            normals = estimate_normals(sample).to(torch.float32)
            report = fit_model(
                asset_id=asset_id,
                asset_path=asset_path,
                asset_sha256=asset_sha256,
                canonical=canonical,
                normals=normals,
                probes=probes,
                target_model="covariance_diffuse",
                ridge_candidates=ridge_candidates,
                ridge_prior=ridge_prior,
                protocol_fingerprint=protocol_fingerprint,
                output_dir=output_dir,
            )
            del compact, sample, canonical, normals
            gc.collect()

        deployed = None
        if report["experimental_runtime_eligible"]:
            deploy_dir = args.result_root / "calibrations" / scene_id
            deploy_dir.mkdir(parents=True, exist_ok=True)
            deploy_path = deploy_dir / f"dynamic_{track_id}.pt"
            shutil.copy2(report["calibration"], deploy_path)
            deployed = str(deploy_path.resolve())
        rows.append(
            {
                "scene": scene_id,
                "track_id": track_id,
                "category": str(vehicle["category"]),
                "asset": str(asset_path.resolve()),
                "asset_sha256": asset_sha256,
                "report": str(report_path.resolve()),
                "selected_ridge": report["selected_ridge"],
                "heldout_hdri_gate_passed": bool(
                    report["test"]["improvement_gate"]["passed"]
                ),
                "experimental_runtime_eligible": bool(
                    report["experimental_runtime_eligible"]
                ),
                "deployed_calibration": deployed,
                "deployed_sha256": None if deployed is None else sha256_file(Path(deployed)),
            }
        )

    summary = {
        "schema_version": 1,
        "benchmark": "decouplegs_public_reinsertion_vehicle_ols_calibration",
        "status": (
            "complete" if all(row["deployed_calibration"] for row in rows) else "gated"
        ),
        "claim_scope": "public_hdri_proxy_not_author_relighting_calibration",
        "paper_exact": False,
        "reinsertion_protocol": str(args.reinsertion_protocol.resolve()),
        "hdri_protocol": str(hdri_path.resolve()),
        "protocol_fingerprint": protocol_fingerprint,
        "descriptor_sha256": probes.descriptor_sha256,
        "probe_count": int(probes.descriptors.shape[0]),
        "vehicles": len(rows),
        "deployed": sum(row["deployed_calibration"] is not None for row in rows),
        "rows": rows,
        "missing_author_artifacts": [
            "author HDRI calibration set",
            "author target relighting renderer",
            "author per-primitive relit colors",
        ],
    }
    summary_path = args.result_root / "calibration-manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "vehicles": summary["vehicles"],
                "deployed": summary["deployed"],
                "manifest": str(summary_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
