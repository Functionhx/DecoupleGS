#!/usr/bin/env python3
"""Render HUGSIM held-out views with raw or compact dynamic assets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.registration import RegistrationConfig
from decouplegs.relighting import RelightingConfig
from decouplegs.runtime import AssetLibrary, DecoupleRuntime
from gaussian_renderer import GaussianModel, render
from scene import Scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("raw", "compact"), required=True)
    parser.add_argument(
        "--compact-dir",
        type=Path,
        help="Directory containing dynamic_<track-id>.dgs files",
    )
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument(
        "--nuscenes-manifest",
        type=Path,
        help=(
            "Optional 12 Hz manifest produced by prepare_nuscenes_hugsim_gt.py. "
            "When supplied, render only official nuScenes 2 Hz keyframes."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mask-dilation", type=int, default=5)
    parser.add_argument(
        "--track-manifest",
        type=Path,
        help=(
            "Optional DecoupleGS reinsertion YAML. Only vehicle track IDs listed "
            "for source-path's scene are composed and used as mask prompts."
        ),
    )
    parser.add_argument("--relighting", action="store_true")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Directory containing dynamic_<track-id>.pt OLS relighting sidecars",
    )
    parser.add_argument(
        "--adaptive-relighting",
        action="store_true",
        help="Explicitly use the self-R&D exposure/tint fallback when OLS is absent",
    )
    parser.add_argument("--contact-shadows", action="store_true")
    parser.add_argument("--opacity-grounding", action="store_true")
    parser.add_argument(
        "--preserve-logged-pose",
        action="store_true",
        help=(
            "Estimate a ground plane for contact shadows without moving the "
            "dataset's logged vehicle transform"
        ),
    )
    parser.add_argument(
        "--rotate-sh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the paper's Wigner-D SH rotation (default: enabled)",
    )
    return parser.parse_args()


def projected_vehicle_masks(
    view,
    vertices_by_id: dict[str, np.ndarray],
    dilation: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Project one prompt per active vehicle and return it with their union."""

    masks: dict[str, np.ndarray] = {}
    world_to_camera = torch.linalg.inv(view.c2w).detach().cpu().numpy()
    intrinsic = view.K[:3, :3].detach().cpu().numpy()
    for track_id, transform_tensor in view.dynamics.items():
        vertices = vertices_by_id.get(track_id)
        if vertices is None:
            continue
        instance_mask = np.zeros((view.height, view.width), dtype=np.uint8)
        transform = transform_tensor.detach().cpu().numpy()
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        camera = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        visible = camera[:, 2] > 1e-3
        if visible.sum() < 3:
            continue
        projected = camera[visible] @ intrinsic.T
        xy = projected[:, :2] / projected[:, 2:3]
        xy[:, 0] = np.clip(xy[:, 0], 0, view.width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, view.height - 1)
        hull = cv2.convexHull(np.rint(xy).astype(np.int32))
        if hull.shape[0] >= 3:
            cv2.fillConvexPoly(instance_mask, hull, 255)
        if dilation > 0 and instance_mask.any():
            width = 2 * dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
            instance_mask = cv2.dilate(instance_mask, kernel)
        if instance_mask.any():
            masks[track_id] = instance_mask
    union = np.zeros((view.height, view.width), dtype=np.uint8)
    for instance_mask in masks.values():
        union = np.maximum(union, instance_mask)
    return union, masks


def projected_vehicle_mask(
    view,
    vertices_by_id: dict[str, np.ndarray],
    dilation: int,
) -> np.ndarray:
    """Backward-compatible union prompt used by older callers/tests."""

    mask, _ = projected_vehicle_masks(view, vertices_by_id, dilation)
    return mask


def load_track_selection(path: Path, scene_id: str) -> tuple[set[str], dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        entries = payload["scenes"][scene_id]["vehicles"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"track manifest has no vehicle cohort for {scene_id}") from error
    categories: dict[str, str] = {}
    for entry in entries:
        track_id = str(entry["track_id"])
        category = str(entry["category"])
        if not category.startswith("vehicle."):
            raise ValueError(
                f"track {track_id} is not a nuScenes vehicle category: {category}"
            )
        if track_id in categories:
            raise ValueError(f"duplicate track ID in manifest: {track_id}")
        categories[track_id] = category
    if not categories:
        raise ValueError(f"track manifest selects no vehicles for {scene_id}")
    return set(categories), categories


def filter_view_tracks(views, track_ids: set[str]) -> None:
    """Restrict composition to the frozen vehicle cohort in place."""

    for view in views:
        view.dynamics = {
            track_id: transform
            for track_id, transform in view.dynamics.items()
            if track_id in track_ids
        }


def make_bridge(
    scene: Scene,
    args: argparse.Namespace,
    track_ids: set[str] | None = None,
) -> HUGSIMRuntimeBridge:
    if args.compact_dir is None:
        raise ValueError("--compact-dir is required in compact mode")
    registration = RegistrationConfig(
        heading_weight=2.5,
        map_resolution=0.1,
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        # HUGSIM's reconstructed dynamic assets follow the nuScenes box
        # convention: local Y is vehicle-forward and local +Z is physical up.
        # This differs from the X-forward/-Y-up 3DRealCar canonical library
        # used by the interactive scenario generator.
        forward_axis=1,
        up_axis=2,
        up_sign=1,
    )
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
    runtime = DecoupleRuntime(
        AssetLibrary(),
        registration_config=registration,
        relighting_config=relighting,
        rotate_sh=args.rotate_sh,
        relighting=args.relighting,
        adaptive_relighting=args.adaptive_relighting,
        contact_shadows=args.contact_shadows,
        opacity_grounding=args.opacity_grounding,
        adjust_grounding_pose=not args.preserve_logged_pose,
        semantic_grounding=True,
        frustum_culling=True,
    )
    bridge = HUGSIMRuntimeBridge(runtime, compact_asset_batch_size=8)
    selected_ids = set(scene.dynamic_gaussians) if track_ids is None else track_ids
    for track_id in sorted(selected_ids):
        if track_id not in scene.dynamic_gaussians:
            raise KeyError(f"track {track_id} is absent from the HUGSIM checkpoint")
        compact_path = args.compact_dir / f"dynamic_{track_id}.dgs"
        if not compact_path.is_file():
            raise FileNotFoundError(compact_path)
        calibration_path = (
            None
            if args.calibration_dir is None
            else args.calibration_dir / f"dynamic_{track_id}.pt"
        )
        if calibration_path is not None and not calibration_path.is_file():
            calibration_path = None
        if args.relighting and calibration_path is None and not args.adaptive_relighting:
            raise FileNotFoundError(
                f"no OLS relighting sidecar for {track_id}; provide --calibration-dir "
                "or explicitly select --adaptive-relighting"
            )
        bridge.register_compact_asset(track_id, compact_path, calibration_path)
    return bridge


def select_views(scene: Scene, split: str):
    if split == "train":
        return list(scene.getTrainCameras())
    if split == "test":
        return list(scene.getTestCameras())
    return sorted(
        [*scene.getTrainCameras(), *scene.getTestCameras()],
        key=lambda view: (float(view.timestamp), view.image_name),
    )


def nuscenes_keyframe_ids(manifest_path: Path) -> tuple[set[str], dict[str, str]]:
    """Return HUGSIM image IDs and sample tokens for official 2 Hz frames."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    keyframe_ids: set[str] = set()
    sample_tokens: dict[str, str] = {}
    for mapping in payload.get("mappings", []):
        if not bool(mapping.get("is_key_frame", False)):
            continue
        image_id = f"{mapping['camera']}_{int(mapping['frame_index']):05d}"
        keyframe_ids.add(image_id)
        sample_tokens[image_id] = str(mapping["sample_token"])
    if not keyframe_ids:
        raise ValueError(f"manifest contains no keyframes: {manifest_path}")
    return keyframe_ids, sample_tokens


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("held-out rendering requires CUDA")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.mask_dilation < 0:
        raise ValueError("--mask-dilation must be non-negative")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.adaptive_relighting and not args.relighting:
        raise ValueError("--adaptive-relighting requires --relighting")
    if args.preserve_logged_pose and not args.opacity_grounding:
        raise ValueError("--preserve-logged-pose requires --opacity-grounding")

    cfg = OmegaConf.load(args.model_path / "cfg.yaml")
    cfg.model_path = str(args.model_path)
    cfg.source_path = str(args.source_path)
    gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)
    with torch.inference_mode():
        scene = Scene(
            cfg,
            gaussians,
            load_iteration=int(cfg.get("iteration", 30000)),
            shuffle=False,
            data_type=cfg.data_type,
        )
    scene_id = args.source_path.resolve().name
    selected_track_ids: set[str] | None = None
    selected_categories: dict[str, str] = {}
    if args.track_manifest is not None:
        selected_track_ids, selected_categories = load_track_selection(
            args.track_manifest, scene_id
        )
        missing_models = selected_track_ids - set(scene.dynamic_gaussians)
        missing_vertices = selected_track_ids - set(scene.dynamic_verts)
        if missing_models or missing_vertices:
            raise ValueError(
                "track manifest/checkpoint mismatch: "
                f"missing_models={sorted(missing_models)}, "
                f"missing_vertices={sorted(missing_vertices)}"
            )
        filter_view_tracks(
            [*scene.getTrainCameras(), *scene.getTestCameras()], selected_track_ids
        )
    views = select_views(scene, args.split)
    keyframe_sample_tokens: dict[str, str] = {}
    if args.nuscenes_manifest is not None:
        keyframe_ids, keyframe_sample_tokens = nuscenes_keyframe_ids(
            args.nuscenes_manifest
        )
        views = [view for view in views if view.image_name in keyframe_ids]
        missing = keyframe_ids - {view.image_name for view in views}
        if missing:
            preview = ", ".join(sorted(missing)[:6])
            raise ValueError(
                f"{len(missing)} manifest keyframe views are absent from split "
                f"{args.split!r}; first: {preview}. Use --split all for the paper protocol."
            )
    if args.limit is not None:
        views = views[: args.limit]
    bridge = (
        None
        if args.mode == "raw"
        else make_bridge(scene, args, selected_track_ids)
    )
    metadata = json.loads((args.source_path / "meta_data.json").read_text(encoding="utf-8"))
    target_by_name = {}
    for frame in metadata["frames"]:
        relative = Path(str(frame["rgb_path"]).removeprefix("./"))
        image_name = f"{relative.parent.name}_{relative.stem}"
        target_by_name[image_name] = args.source_path / relative
    background = torch.tensor(
        [1.0, 1.0, 1.0] if cfg.model.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    if views:
        for _ in range(args.warmup):
            with torch.inference_mode():
                render(
                    views[0],
                    views[0],
                    scene.gaussians,
                    scene.dynamic_gaussians,
                    None,
                    background,
                    render_optical=False,
                    decouple_bridge=bridge,
                    rasterizer_packed=True,
                    rgb_only=True,
                )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    render_dir = args.output / "render"
    mask_dir = args.output / "mask_geometry_prompt"
    instance_mask_dir = args.output / "mask_geometry_instances"
    render_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    instance_mask_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    seconds = []
    previous_by_camera: dict[str, object] = {}
    for index, view in enumerate(views):
        camera = view.image_name.rsplit("_", 1)[0]
        previous = previous_by_camera.get(camera, view)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            package = render(
                view,
                previous,
                scene.gaussians,
                scene.dynamic_gaussians,
                None,
                background,
                render_optical=False,
                decouple_bridge=bridge,
                rasterizer_packed=True,
                rgb_only=True,
            )
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        image = (
            package["render"]
            .detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        filename = f"{index:05d}_{view.image_name}.png"
        prediction_path = render_dir / filename
        Image.fromarray(image).save(prediction_path)
        mask, instance_masks = projected_vehicle_masks(
            view, scene.dynamic_verts, args.mask_dilation
        )
        mask_path = mask_dir / filename
        Image.fromarray(mask).save(mask_path)
        instance_prompts = []
        for track_id, instance_mask in sorted(instance_masks.items()):
            track_dir = instance_mask_dir / track_id
            track_dir.mkdir(parents=True, exist_ok=True)
            instance_path = track_dir / filename
            Image.fromarray(instance_mask).save(instance_path)
            instance_prompts.append(
                {
                    "track_id": track_id,
                    "category": selected_categories.get(track_id),
                    "mask": str(instance_path.resolve()),
                    "pixels": int(np.count_nonzero(instance_mask)),
                }
            )
        target_path = target_by_name.get(view.image_name)
        if target_path is None:
            raise KeyError(f"no source image found for {view.image_name}")
        pairs.append(
            {
                "id": view.image_name,
                "split": args.split,
                "camera": camera,
                "timestamp": float(view.timestamp),
                "sample_token": keyframe_sample_tokens.get(view.image_name),
                "pred": str(prediction_path.resolve()),
                "target": str(target_path.resolve()),
                "mask": str(mask_path.resolve()),
                "mask_source": (
                    "per-vehicle projected canonical 3D convex hull; "
                    f"elliptical dilation radius={args.mask_dilation}px"
                ),
                "track_ids": [entry["track_id"] for entry in instance_prompts],
                "instance_prompts": instance_prompts,
            }
        )
        previous_by_camera[camera] = view

    mean_seconds = float(np.mean(seconds))
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "split": args.split,
        "frame_protocol": (
            "nuscenes_2hz_keyframes"
            if args.nuscenes_manifest is not None
            else "hugsim_split"
        ),
        "nuscenes_manifest": (
            None
            if args.nuscenes_manifest is None
            else str(args.nuscenes_manifest.resolve())
        ),
        "model_path": str(args.model_path.resolve()),
        "source_path": str(args.source_path.resolve()),
        "scene_id": scene_id,
        "pair_count": len(pairs),
        "runtime": {
            "mean_ms_per_camera": mean_seconds * 1000.0,
            "camera_fps": 1.0 / mean_seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "gpu": torch.cuda.get_device_name(),
        },
        "modules": {
            "compact": args.mode == "compact",
            "relighting": args.relighting,
            "relighting_kind": (
                "disabled"
                if not args.relighting
                else (
                    "adaptive_self_rd"
                    if args.adaptive_relighting
                    else "paper_ols_sidecar"
                )
            ),
            "contact_shadows": args.contact_shadows,
            "opacity_grounding": args.opacity_grounding,
            "preserve_logged_pose": args.preserve_logged_pose,
            "rotate_sh": args.rotate_sh,
            "track_selection": (
                "all_hugsim_dynamics"
                if selected_track_ids is None
                else "manifest_vehicle_only"
            ),
        },
        "track_manifest": (
            None
            if args.track_manifest is None
            else str(args.track_manifest.resolve())
        ),
        "selected_tracks": [
            {"track_id": track_id, "category": selected_categories.get(track_id)}
            for track_id in sorted(
                scene.dynamic_gaussians if selected_track_ids is None else selected_track_ids
            )
        ],
        "pairs": pairs,
    }
    manifest = args.output / "pairs.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("mode", "pair_count", "runtime", "modules")}, indent=2))


if __name__ == "__main__":
    main()
