#!/usr/bin/env python3
"""Run and aggregate the frozen DecoupleGS canonical-asset cohort.

The paper does not publish the identifiers of its 20 3DRealCar assets or the
implementation details behind the three saliency terms.  This runner makes
our replacement protocol (H-DATA-01 + H-COMP-01) explicit, resumable, and
auditable rather than hiding those choices in a shell history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import gaussian_set_from_hugsim_checkpoint


EXPECTED_COMPRESSION = {
    "prune_threshold": 0.005,
    "weight_visibility": 0.5,
    "weight_color": 0.3,
    "weight_entropy": 0.2,
    "color_codebook_size": 1024,
    "shape_codebook_size": 512,
    "ema_momentum": 0.99,
    "ema_iterations": 5000,
    "ema_batch_size": 16384,
    "visibility_gate_saliency": True,
    "seed": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_assets_20.yaml",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ROOT / "data/decouplegs/3DRealCar",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results/decouplegs/asset-cohort",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("visibility", "compression", "fidelity", "aggregate"),
        default=("visibility", "compression", "fidelity", "aggregate"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start-at", help="Skip manifest entries before this asset ID")
    parser.add_argument("--stop-after", help="Stop after this asset ID")
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path) -> None:
    """Run a child command with live output and retain an exact text log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return_code = process.wait()
    log_path.write_text("".join(lines))
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def compact_matches_protocol(path: Path) -> tuple[bool, CompactGaussianAsset]:
    compact = CompactGaussianAsset.load(path)
    config = compact.metadata.get("compression_config", {})
    matches = all(config.get(key) == value for key, value in EXPECTED_COMPRESSION.items())
    return matches, compact


def ensure_local_sh_contract(path: Path) -> None:
    """Migrate pre-H-APP-01 derived assets without changing their tensors."""

    compact = CompactGaussianAsset.load(path)
    if "sh_frame" not in compact.metadata:
        compact.metadata["sh_frame"] = "local"
        compact.metadata["sh_frame_hypothesis"] = "H-APP-01"
        compact.save(path)
        print(f"migrated SH-frame metadata: {path}", flush=True)
    if compact.metadata.get("sh_frame") != "local":
        raise ValueError(f"canonical 3DRealCar asset must use local SH: {path}")


def finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def summarize(values: list[float]) -> dict[str, float | int | None]:
    values = finite(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(
    asset_ids: list[str], asset_root: Path, result_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        asset_dir = asset_root / asset_id
        raw_path = asset_dir / "gs.pth"
        compact_path = asset_dir / "decouplegs-visgate.dgs"
        fidelity_path = result_root / "fidelity" / f"{asset_id}.json"
        visibility_path = asset_dir / "importance-orbit.pt"
        if not all(path.is_file() for path in (raw_path, compact_path, fidelity_path, visibility_path)):
            continue

        compact = CompactGaussianAsset.load(compact_path)
        fidelity = json.loads(fidelity_path.read_text())
        candidate = fidelity["results"][0]
        raw = gaussian_set_from_hugsim_checkpoint(raw_path)
        visibility_payload = torch.load(visibility_path, map_location="cpu", weights_only=True)
        visibility = visibility_payload["visibility"]
        row = {
            "asset_id": asset_id,
            "raw_sha256": sha256(raw_path),
            "compact_sha256": sha256(compact_path),
            "raw_file_bytes": raw_path.stat().st_size,
            "compact_file_bytes": compact_path.stat().st_size,
            "raw_memory_bytes": raw.memory_bytes,
            "compact_memory_bytes": compact.memory_bytes,
            "raw_primitives": len(raw),
            "compact_primitives": len(compact),
            "retained_fraction": len(compact) / max(len(raw), 1),
            "memory_compression_ratio": raw.memory_bytes / max(compact.memory_bytes, 1),
            "file_compression_ratio": raw_path.stat().st_size / max(compact_path.stat().st_size, 1),
            "visible_fraction": float((visibility > 0).float().mean()),
            "above_threshold_fraction": float((visibility >= 0.005).float().mean()),
            **candidate["aggregate"],
        }
        rows.append(row)
        del raw, compact, visibility_payload, visibility

    metric_keys = (
        "retained_fraction",
        "memory_compression_ratio",
        "file_compression_ratio",
        "visible_fraction",
        "above_threshold_fraction",
        "psnr_all_db",
        "psnr_vehicle_db",
        "ssim",
        "alpha_psnr_db",
        "lpips_alex",
    )
    aggregate_metrics = {
        key: summarize([float(row[key]) for row in rows if key in row]) for key in metric_keys
    }
    total_raw_memory = sum(row["raw_memory_bytes"] for row in rows)
    total_compact_memory = sum(row["compact_memory_bytes"] for row in rows)
    total_raw_primitives = sum(row["raw_primitives"] for row in rows)
    total_compact_primitives = sum(row["compact_primitives"] for row in rows)
    result = {
        "schema_version": 1,
        "benchmark": "decouplegs_frozen_asset_cohort",
        "status": "complete" if len(rows) == len(asset_ids) else "partial",
        "hypotheses": {
            "H-DATA-01": manifest.get("note"),
            "H-COMP-01": "D_i and H_i are gated by expected rendered visibility before Eq. (4) thresholding.",
            "H-APP-01": "Released canonical 3DRealCar assets store SH in their local frame.",
        },
        "manifest": str(Path(manifest["_path"]).resolve()),
        "asset_root": str(asset_root.resolve()),
        "expected_assets": len(asset_ids),
        "completed_assets": len(rows),
        "protocol": {
            "visibility": {
                "views": "24 azimuths x elevations [0,10,20] degrees",
                "resolution": "256x256",
                "estimator": "orbit_autodiff_opacity_contribution_v1",
            },
            "compression": EXPECTED_COMPRESSION,
            "fidelity": {
                "views": "same 72-view orbit",
                "reference": "raw released 3DRealCar checkpoint",
                "vehicle_mask": "raw alpha >= 0.01",
                "lpips": "AlexNet, normalize=True",
            },
        },
        "totals": {
            "raw_primitives": total_raw_primitives,
            "compact_primitives": total_compact_primitives,
            "retained_fraction": total_compact_primitives / max(total_raw_primitives, 1),
            "raw_memory_bytes": total_raw_memory,
            "compact_memory_bytes": total_compact_memory,
            "memory_compression_ratio": total_raw_memory / max(total_compact_memory, 1),
        },
        "aggregate": aggregate_metrics,
        "assets": rows,
    }
    result_root.mkdir(parents=True, exist_ok=True)
    output = result_root / "cohort-summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "expected_assets", "completed_assets", "totals", "aggregate")}, indent=2))
    return result


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    manifest["_path"] = str(args.manifest)
    asset_ids = list(manifest["assets"])
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("asset manifest contains duplicate IDs")
    if args.start_at:
        asset_ids = asset_ids[asset_ids.index(args.start_at) :]
    if args.stop_after:
        asset_ids = asset_ids[: asset_ids.index(args.stop_after) + 1]

    args.result_root.mkdir(parents=True, exist_ok=True)
    for index, asset_id in enumerate(asset_ids, start=1):
        asset_dir = args.asset_root / asset_id
        raw_path = asset_dir / "gs.pth"
        visibility_path = asset_dir / "importance-orbit.pt"
        compact_path = asset_dir / "decouplegs-visgate.dgs"
        fidelity_path = args.result_root / "fidelity" / f"{asset_id}.json"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        print(f"\n=== [{index}/{len(asset_ids)}] {asset_id} ===", flush=True)

        if "visibility" in args.stages and (args.force or not visibility_path.is_file()):
            run_logged(
                [
                    sys.executable,
                    str(ROOT / "tools/estimate_decouplegs_visibility.py"),
                    str(raw_path),
                    str(visibility_path),
                ],
                args.result_root / "logs" / f"{asset_id}-visibility.log",
            )

        needs_compression = args.force or not compact_path.is_file()
        if compact_path.is_file() and not args.force:
            matches, _ = compact_matches_protocol(compact_path)
            needs_compression = not matches
        if "compression" in args.stages and needs_compression:
            if not visibility_path.is_file():
                raise FileNotFoundError(visibility_path)
            run_logged(
                [
                    sys.executable,
                    str(ROOT / "tools/compress_decouplegs_asset.py"),
                    str(raw_path),
                    str(compact_path),
                    "--importance-stats",
                    str(visibility_path),
                    "--visibility-gate-saliency",
                    "--ema-iterations",
                    "5000",
                    "--ema-batch-size",
                    "16384",
                    "--seed",
                    "0",
                    "--sh-frame",
                    "local",
                ],
                args.result_root / "logs" / f"{asset_id}-compression.log",
            )
        if compact_path.is_file():
            ensure_local_sh_contract(compact_path)

        if "fidelity" in args.stages and (args.force or not fidelity_path.is_file()):
            if not compact_path.is_file():
                raise FileNotFoundError(compact_path)
            command = [
                sys.executable,
                str(ROOT / "tools/evaluate_decouplegs_asset_compression.py"),
                str(raw_path),
                str(compact_path),
                "--output",
                str(fidelity_path),
            ]
            if args.no_lpips:
                command.append("--no-lpips")
            run_logged(
                command,
                args.result_root / "logs" / f"{asset_id}-fidelity.log",
            )

    if "aggregate" in args.stages:
        # Aggregate the full manifest, including valid outputs from earlier
        # interrupted runs, regardless of --start-at/--stop-after.
        aggregate(list(manifest["assets"]), args.asset_root, args.result_root, manifest)


if __name__ == "__main__":
    main()
