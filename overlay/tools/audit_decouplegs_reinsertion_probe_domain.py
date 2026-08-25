#!/usr/bin/env python3
"""Audit real HUGSIM local-light probes against the public HDRI OLS fit domain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.hugsim import gaussian_set_from_hugsim_model
from decouplegs.relighting import RelightingCalibration, RelightingConfig, sample_local_probe
from gaussian_renderer import GaussianModel
from scene import Scene
from tools.render_decouplegs_holdout import filter_view_tracks, load_track_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs/benchmark/decouplegs_reinsertion_public.yaml",
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=ROOT
        / "results/decouplegs/reinsertion-paper-protocol/calibration/calibrations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/decouplegs/reinsertion-paper-protocol/evaluation-v1/probe-domain-audit.json",
    )
    return parser.parse_args()


def unique_positions(views, track_ids: set[str]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    seen: set[tuple[float, str]] = set()
    positions = []
    metadata = []
    for view in views:
        for track_id, transform in view.dynamics.items():
            if track_id not in track_ids:
                continue
            key = (round(float(view.timestamp), 9), track_id)
            if key in seen:
                continue
            seen.add(key)
            positions.append(transform[:3, 3])
            metadata.append(
                {
                    "timestamp": float(view.timestamp),
                    "track_id": track_id,
                }
            )
    if not positions:
        return torch.empty((0, 3), device="cuda"), []
    return torch.stack(positions), metadata


def batched_probe(
    positions: torch.Tensor,
    background,
    config: RelightingConfig,
    batch_size: int = 24,
) -> torch.Tensor:
    chunks = []
    for start in range(0, positions.shape[0], batch_size):
        chunks.append(
            sample_local_probe(
                positions[start : start + batch_size],
                background.means,
                background.sh,
                visibility=(
                    background.visibility
                    if background.visibility is not None
                    else background.opacities
                ),
                config=config,
            )
        )
    return torch.cat(chunks) if chunks else positions.new_empty((0, 27))


def domain_metrics(values: torch.Tensor, domain: dict[str, Any]) -> dict[str, Any]:
    minimum = values.new_tensor(domain["minimum"])
    maximum = values.new_tensor(domain["maximum"])
    mean = values.new_tensor(domain["mean"])
    std = values.new_tensor(domain["std"]).clamp_min(1e-6)
    outside = (values < minimum) | (values > maximum)
    z = (values - mean) / std
    return {
        "samples": int(values.shape[0]),
        "descriptor_dimensions": int(values.shape[1]),
        "element_outside_fit_range_fraction": float(outside.float().mean()),
        "samples_with_any_outside_dimension_fraction": float(outside.any(dim=1).float().mean()),
        "mean_absolute_z_score": float(z.abs().mean()),
        "maximum_absolute_z_score": float(z.abs().max()),
        "actual_minimum": values.amin(dim=0).tolist(),
        "actual_maximum": values.amax(dim=0).tolist(),
        "actual_mean": values.mean(dim=0).tolist(),
        "actual_std": values.std(dim=0, unbiased=False).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("probe audit requires CUDA")
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    paths = protocol["paths"]
    relighting = RelightingConfig(
        probe_sigma=3.0,
        probe_radius=9.0,
        shadow_strength=0.55,
        shadow_exponent=4.0,
        shadow_decay=2.0,
        shadow_ground_band=0.25,
        shadow_mask_epsilon=1e-5,
        adaptive_strength=0.65,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "hypothesis": "H-LIGHT-04",
        "audit": "real_scene_probe_domain_vs_public_hdri_ols_fit_domain",
        "protocol": str(args.protocol.resolve()),
        "scenes": {},
    }
    all_split: dict[str, list[torch.Tensor]] = {"train": [], "test": []}
    for scene_id in protocol["scenes"]:
        print(f"[{scene_id}] load scene and sample local probes", flush=True)
        model_path = Path(paths["model_root"]) / scene_id
        source_path = Path(paths["source_root"]) / scene_id
        cfg = OmegaConf.load(model_path / "cfg.yaml")
        cfg.model_path = str(model_path)
        cfg.source_path = str(source_path)
        gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
        with torch.inference_mode():
            scene = Scene(
                cfg,
                gaussians,
                load_iteration=int(cfg.get("iteration", 30000)),
                shuffle=False,
                data_type=cfg.data_type,
            )
        track_ids, categories = load_track_selection(args.protocol, scene_id)
        filter_view_tracks(
            [*scene.getTrainCameras(), *scene.getTestCameras()], track_ids
        )
        background = gaussian_set_from_hugsim_model(
            scene.gaussians, include_ground=True
        )
        split_payload: dict[str, Any] = {}
        values_by_split: dict[str, torch.Tensor] = {}
        metadata_by_split: dict[str, list[dict[str, Any]]] = {}
        for split, views in (
            ("train", scene.getTrainCameras()),
            ("test", scene.getTestCameras()),
        ):
            positions, metadata = unique_positions(views, track_ids)
            with torch.inference_mode():
                values = batched_probe(positions, background, relighting)
            values_by_split[split] = values.cpu()
            metadata_by_split[split] = metadata
            all_split[split].append(values.cpu())
            split_payload[split] = {
                "unique_track_timestamps": len(metadata),
                "descriptor_minimum": values.amin(dim=0).cpu().tolist(),
                "descriptor_maximum": values.amax(dim=0).cpu().tolist(),
            }

        tracks: dict[str, Any] = {}
        for track_id in sorted(track_ids):
            calibration_path = (
                args.calibration_root / scene_id / f"dynamic_{track_id}.pt"
            )
            calibration = RelightingCalibration.load(calibration_path)
            domain = calibration.metadata["descriptor_domain_fit"]
            track_payload: dict[str, Any] = {
                "category": categories[track_id],
                "calibration": str(calibration_path.resolve()),
                "fit_domain": domain,
            }
            for split in ("train", "test"):
                indices = [
                    index
                    for index, entry in enumerate(metadata_by_split[split])
                    if entry["track_id"] == track_id
                ]
                local = values_by_split[split][indices]
                track_payload[split] = domain_metrics(local, domain)
            tracks[track_id] = track_payload
        result["scenes"][scene_id] = {
            "background_gaussians": len(background),
            "splits": split_payload,
            "tracks": tracks,
        }
        del scene, gaussians, background
        torch.cuda.empty_cache()

    result["aggregate_observed_descriptor_domain"] = {
        split: {
            "samples": int((values := torch.cat(chunks)).shape[0]),
            "minimum": values.amin(dim=0).tolist(),
            "maximum": values.amax(dim=0).tolist(),
            "mean": values.mean(dim=0).tolist(),
            "std": values.std(dim=0, unbiased=False).tolist(),
        }
        for split, chunks in all_split.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    test_rows = [
        track["test"]
        for scene in result["scenes"].values()
        for track in scene["tracks"].values()
    ]
    print(
        json.dumps(
            {
                "tracks": len(test_rows),
                "mean_test_element_ood_fraction": sum(
                    row["element_outside_fit_range_fraction"] for row in test_rows
                )
                / len(test_rows),
                "mean_test_any_dimension_ood_fraction": sum(
                    row["samples_with_any_outside_dimension_fraction"]
                    for row in test_rows
                )
                / len(test_rows),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
