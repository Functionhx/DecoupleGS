#!/usr/bin/env python3
"""Render a real DecoupleGS effect ablation with zooms and difference maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.compression import CompactGaussianAsset
from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.registration import RegistrationConfig, rotation_from_yaw_and_normal
from decouplegs.relighting import RelightingCalibration, RelightingConfig
from decouplegs.runtime import AssetLibrary, DecoupleRuntime
from gaussian_renderer import GaussianModel, render
from scene.obj_model import ObjModel

VARIANTS: tuple[tuple[str, dict[str, bool]], ...] = (
    (
        "00_raw",
        {
            "rotate_sh": False,
            "opacity_grounding": False,
            "relighting": False,
            "contact_shadows": False,
        },
    ),
    (
        "01_sh_rotation",
        {
            "rotate_sh": True,
            "opacity_grounding": False,
            "relighting": False,
            "contact_shadows": False,
        },
    ),
    (
        "02_grounding",
        {
            "rotate_sh": True,
            "opacity_grounding": True,
            "relighting": False,
            "contact_shadows": False,
        },
    ),
    (
        "03_relighting",
        {
            "rotate_sh": True,
            "opacity_grounding": True,
            "relighting": True,
            "contact_shadows": False,
        },
    ),
    (
        "04_full_shadow",
        {
            "rotate_sh": True,
            "opacity_grounding": True,
            "relighting": True,
            "contact_shadows": True,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render cumulative DecoupleGS stages on a real HUGSIM checkpoint, then "
            "write full-frame, auto-cropped, and amplified-difference comparisons."
        ),
    )
    parser.add_argument("scene_dir", type=Path, help="Extracted HUGSIM scene containing scene.pth")
    parser.add_argument("asset", type=Path, help="3DRealCar gs.pth or compressed .dgs asset")
    parser.add_argument("output_dir", type=Path, help="Directory for PNGs and metrics.json")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Optional OLS sidecar; provenance is read from its metadata",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.5, help="Render resolution multiplier")
    parser.add_argument("--distance", type=float, default=14.0, help="Placement distance along camera heading")
    parser.add_argument("--lateral", type=float, default=0.0, help="World-X placement offset in metres")
    parser.add_argument("--camera-height", type=float, default=1.5, help="Initial camera-to-road +Y offset")
    parser.add_argument("--adaptive-strength", type=float, default=0.65)
    parser.add_argument("--shadow-strength", type=float, default=0.55)
    parser.add_argument("--crop-size", type=int, default=96, help="Minimum square source crop in pixels")
    parser.add_argument("--zoom-size", type=int, default=320, help="Displayed size of each crop panel")
    parser.add_argument("--difference-gain", type=float, default=20.0)
    return parser.parse_args()


def load_scene_config(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "cfg.yaml"
    if not path.is_file():
        return {"affine": True, "model": {"sh_degree": 3}}
    with path.open() as stream:
        return yaml.safe_load(stream) or {}


def canonical_pose(
    frame: dict[str, Any],
    args: argparse.Namespace,
    config: RegistrationConfig,
) -> torch.Tensor:
    c2w = torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda")
    camera_forward = c2w[:3, 2].clone()
    camera_forward[config.vertical_axis] = 0.0
    camera_forward = torch.nn.functional.normalize(camera_forward, dim=0)
    yaw = torch.atan2(
        camera_forward[config.horizontal_axes[1]],
        camera_forward[config.horizontal_axes[0]],
    )
    normal = torch.zeros(3, dtype=torch.float32, device="cuda")
    normal[config.vertical_axis] = config.vertical_sign
    rotation = rotation_from_yaw_and_normal(yaw, normal, config)
    translation = c2w[:3, 3] + camera_forward * args.distance
    translation[0] += args.lateral
    translation[config.vertical_axis] += args.camera_height
    pose = torch.eye(4, dtype=torch.float32, device="cuda")
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def annotate(image: Image.Image, label: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = load_font(max(15, round(min(output.size) * 0.045)))
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    pad = max(5, height // 3)
    draw.rectangle((0, 0, width + 2 * pad, height + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return output


def montage(
    panels: list[tuple[str, Image.Image]],
    path: Path,
    *,
    columns: int,
) -> None:
    if not panels:
        raise ValueError("a montage needs at least one panel")
    annotated = [annotate(image, label) for label, image in panels]
    width = max(image.width for image in annotated)
    height = max(image.height for image in annotated)
    rows = math.ceil(len(annotated) / columns)
    canvas = Image.new("RGB", (columns * width, rows * height), color=(0, 0, 0))
    for index, image in enumerate(annotated):
        x = (index % columns) * width + (width - image.width) // 2
        y = (index // columns) * height + (height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(path)


def square_crop_box(mask: np.ndarray, width: int, height: int, minimum_size: int) -> list[int]:
    rows, columns = np.nonzero(mask)
    if rows.size:
        x_center = 0.5 * (float(columns.min()) + float(columns.max()) + 1.0)
        y_center = 0.5 * (float(rows.min()) + float(rows.max()) + 1.0)
        content_size = max(
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        )
        side = max(minimum_size, math.ceil(content_size * 1.5))
    else:
        x_center, y_center = width / 2.0, height / 2.0
        side = minimum_size
    side = min(max(1, side), width, height)
    x0 = min(max(0, round(x_center - side / 2)), width - side)
    y0 = min(max(0, round(y_center - side / 2)), height - side)
    return [x0, y0, x0 + side, y0 + side]


def crop_and_resize(image: Image.Image, box: list[int], size: int) -> Image.Image:
    return image.crop(tuple(box)).resize((size, size), Image.Resampling.LANCZOS)


def difference_heatmap(before: np.ndarray, after: np.ndarray, gain: float) -> Image.Image:
    magnitude = np.abs(after - before).max(axis=-1)
    value = np.clip(magnitude * gain, 0.0, 1.0)
    # A compact black -> red -> yellow map leaves unchanged pixels truly black.
    red = np.sqrt(value)
    green = value**1.5
    blue = 0.15 * value * (1.0 - value)
    heat = np.stack((red, green, blue), axis=-1)
    return Image.fromarray(np.round(heat * 255.0).astype(np.uint8))


def resolve_calibration(args: argparse.Namespace) -> Path | None:
    if args.calibration is not None:
        if not args.calibration.is_file():
            raise FileNotFoundError(args.calibration)
        return args.calibration
    adjacent = args.asset.parent / "relighting.pt"
    return adjacent if adjacent.is_file() else None


def calibration_mode(calibration: RelightingCalibration | None) -> str:
    if calibration is None:
        return "adaptive_fallback"
    metadata = calibration.metadata or {}
    if metadata.get("paper_exact") is True:
        return "paper_exact_ols"
    supervision = metadata.get("supervision_kind")
    if supervision:
        return f"public_hdri_{supervision}_proxy_ols"
    return "ols_unclassified_provenance"


def descriptor_domain_audit(
    calibration: RelightingCalibration | None,
    descriptor: torch.Tensor | None,
) -> dict[str, Any] | None:
    if calibration is None or descriptor is None:
        return None
    domain = (calibration.metadata or {}).get("descriptor_domain_fit")
    if not isinstance(domain, dict):
        return {"available": False}
    value = descriptor.detach().to(torch.float32).cpu()
    minimum = torch.tensor(domain["minimum"], dtype=torch.float32)
    maximum = torch.tensor(domain["maximum"], dtype=torch.float32)
    mean = torch.tensor(domain["mean"], dtype=torch.float32)
    std = torch.tensor(domain["std"], dtype=torch.float32).clamp_min(1e-6)
    inside = (value >= minimum) & (value <= maximum)
    z_score = (value - mean).abs() / std
    return {
        "available": True,
        "dimensions": value.numel(),
        "inside_training_range_dimensions": int(inside.sum()),
        "inside_training_range_fraction": float(inside.float().mean()),
        "maximum_absolute_z_score": float(z_score.max()),
        "descriptor": value.tolist(),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real DecoupleGS effect rendering requires CUDA")
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0, 1]")
    if args.crop_size <= 0 or args.zoom_size <= 0 or args.difference_gain <= 0:
        raise ValueError("crop, zoom, and difference parameters must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_config = load_scene_config(args.scene_dir)
    sh_degree = int(scene_config.get("model", {}).get("sh_degree", 3))
    affine = bool(scene_config.get("affine", True))
    with (args.scene_dir / "meta_data.json").open() as stream:
        frames = json.load(stream)["frames"]
    if not -len(frames) <= args.frame_index < len(frames):
        raise IndexError(f"frame index {args.frame_index} is outside a {len(frames)}-frame scene")
    frame = frames[args.frame_index]

    background = GaussianModel(sh_degree, affine=affine)
    background_state, scene_iteration = torch.load(
        args.scene_dir / "scene.pth",
        map_location="cuda",
        weights_only=False,
    )
    background.restore(background_state, None)

    registration = RegistrationConfig(
        column_radius=0.35,
        vertical_axis=1,
        vertical_sign=-1,
        horizontal_axes=(0, 2),
        forward_axis=0,
        up_axis=1,
        up_sign=-1,
    )
    relighting_config = RelightingConfig(
        adaptive_strength=args.adaptive_strength,
        shadow_strength=args.shadow_strength,
    )
    runtime = DecoupleRuntime(
        AssetLibrary(),
        registration_config=registration,
        relighting_config=relighting_config,
        adaptive_relighting=True,
        frustum_culling=True,
    )
    bridge = HUGSIMRuntimeBridge(runtime)
    calibration_path = resolve_calibration(args)
    calibration = (
        None
        if calibration_path is None
        else RelightingCalibration.load(calibration_path, map_location="cpu")
    )
    asset_id = "decouple-effects-car"
    dynamic_models: dict[str, ObjModel] = {}
    asset_iteration: int | None = None
    if args.asset.suffix == ".dgs":
        compact = CompactGaussianAsset.load(args.asset)
        runtime.library.add(asset_id, compact, calibration)
        asset_primitives = len(compact)
    else:
        asset_model = ObjModel(sh_degree, feat_mutable=False)
        asset_state = torch.load(args.asset, map_location="cuda", weights_only=False)
        if isinstance(asset_state, (tuple, list)) and len(asset_state) == 2:
            asset_state, asset_iteration = asset_state
        asset_model.restore(list(asset_state), None)
        bridge.register_model(asset_id, asset_model, calibration_path)
        dynamic_models[asset_id] = asset_model
        asset_primitives = int(asset_model.get_xyz.shape[0])

    intrinsics = torch.as_tensor(np.array(frame["intrinsics"]), dtype=torch.float32, device="cuda")
    intrinsics[0, :3] *= args.scale
    intrinsics[1, :3] *= args.scale
    width = max(1, round(int(frame["width"]) * args.scale))
    height = max(1, round(int(frame["height"]) * args.scale))
    pose = canonical_pose(frame, args, registration)
    viewpoint = SimpleNamespace(
        c2w=torch.as_tensor(frame["camtoworld"], dtype=torch.float32, device="cuda"),
        K=intrinsics,
        width=width,
        height=height,
        timestamp=float(frame.get("timestamp", -1.0)),
        dynamics={},
    )
    black = torch.zeros(3, dtype=torch.float32, device="cuda")

    images: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def render_stage(name: str, dynamics: dict[str, torch.Tensor]) -> None:
        viewpoint.dynamics = dynamics
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = render(
                viewpoint,
                None,
                background,
                dynamic_models,
                {},
                black,
                decouple_bridge=bridge,
            )
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        image = output["render"].permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        if not np.isfinite(image).all():
            raise RuntimeError(f"{name} produced non-finite RGB")
        images[name] = image
        Image.fromarray(np.round(image * 255.0).astype(np.uint8)).save(
            args.output_dir / f"{name}.png"
        )
        del output

    render_stage("background", {})
    for name, flags in VARIANTS:
        for field, value in flags.items():
            setattr(runtime, field, value)
        render_stage(name, {asset_id: pose})

    local_descriptor = None
    if bridge._prepared_compact is not None:
        local_descriptor = bridge._prepared_compact.descriptors.get(asset_id)

    stage_names = [name for name, _ in VARIANTS]
    background_image = images["background"]
    object_mask = np.zeros((height, width), dtype=bool)
    for name in stage_names:
        object_mask |= np.abs(images[name] - background_image).max(axis=-1) > 2.0 / 255.0
    crop_box = square_crop_box(object_mask, width, height, args.crop_size)

    pil_images = {
        name: Image.fromarray(np.round(image * 255.0).astype(np.uint8))
        for name, image in images.items()
    }
    full_panels = [("background", pil_images["background"])] + [
        (name, pil_images[name]) for name in stage_names
    ]
    montage(full_panels, args.output_dir / "effects_full.png", columns=3)

    zoom_panels = [
        (label, crop_and_resize(image, crop_box, args.zoom_size))
        for label, image in full_panels
    ]
    montage(zoom_panels, args.output_dir / "effects_zoom.png", columns=3)

    comparisons = list(pairwise(stage_names))
    difference_panels: list[tuple[str, Image.Image]] = []
    step_metrics: dict[str, dict[str, float]] = {}
    threshold = 1.0 / 255.0
    for before_name, after_name in comparisons:
        delta = np.abs(images[after_name] - images[before_name])
        key = f"{before_name}_to_{after_name}"
        step_metrics[key] = {
            "mean_absolute_error": float(delta.mean()),
            "maximum_absolute_error": float(delta.max()),
            "fraction_pixels_changed_gt_1_255": float((delta.max(axis=-1) > threshold).mean()),
        }
        heatmap = difference_heatmap(images[before_name], images[after_name], args.difference_gain)
        heatmap = crop_and_resize(heatmap, crop_box, args.zoom_size)
        difference_panels.append((f"{args.difference_gain:g}x {after_name} - {before_name}", heatmap))
    montage(difference_panels, args.output_dir / "effects_difference.png", columns=2)

    metadata = {
        "scene": str(args.scene_dir.resolve()),
        "asset": str(args.asset.resolve()),
        "scene_iteration": int(scene_iteration),
        "asset_iteration": None if asset_iteration is None else int(asset_iteration),
        "background_gaussians": int(background.get_full_xyz.shape[0]),
        "asset_gaussians": asset_primitives,
        "resolution": [width, height],
        "frame_index": args.frame_index,
        "placement": {
            "distance": args.distance,
            "lateral": args.lateral,
            "camera_height": args.camera_height,
        },
        "relighting": {
            "mode": calibration_mode(calibration),
            "calibration": None if calibration_path is None else str(calibration_path.resolve()),
            "calibration_metadata": None if calibration is None else calibration.metadata,
            "descriptor_domain_audit": descriptor_domain_audit(calibration, local_descriptor),
            "adaptive_strength": args.adaptive_strength,
        },
        "shadow_strength": args.shadow_strength,
        "crop_box_xyxy": crop_box,
        "difference_gain": args.difference_gain,
        "elapsed_seconds": timings,
        "step_metrics": step_metrics,
        "outputs": {
            "full": str((args.output_dir / "effects_full.png").resolve()),
            "zoom": str((args.output_dir / "effects_zoom.png").resolve()),
            "difference": str((args.output_dir / "effects_difference.png").resolve()),
        },
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
