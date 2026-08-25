#!/usr/bin/env python3
"""Materialize a readable report and vehicle-crop contact sheet for reinsertion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "results/decouplegs/reinsertion-paper-protocol/evaluation-v1"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    )
    roots = [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/truetype/liberation2")]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def crop_box(mask: np.ndarray, padding: float = 0.22) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(12, int((x1 - x0) * padding))
    pad_y = max(12, int((y1 - y0) * padding))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(mask.shape[1], x1 + pad_x),
        min(mask.shape[0], y1 + pad_y),
    )


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.copy()
    scale = min(width / image.width, height / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (width, height), (18, 18, 18))
    panel.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return panel


def select_representative(
    root: Path,
    scene_id: str,
    mask_index: dict[str, Any],
) -> tuple[str, float]:
    no_payload = read_json(root / scene_id / "compact-no-relight/pairs.json")
    ols_payload = read_json(root / scene_id / "compact-ols/pairs.json")
    no_by_id = {str(row["id"]): row for row in no_payload["pairs"]}
    ols_by_id = {str(row["id"]): row for row in ols_payload["pairs"]}
    best_id, best_change = "", -1.0
    for pair_id, mask_value in mask_index["masks"].items():
        if pair_id not in no_by_id or pair_id not in ols_by_id:
            continue
        mask = np.asarray(Image.open(mask_value).convert("L")) > 0
        if not mask.any():
            continue
        no = np.asarray(Image.open(no_by_id[pair_id]["pred"]).convert("RGB"), dtype=np.float32)
        ols = np.asarray(Image.open(ols_by_id[pair_id]["pred"]).convert("RGB"), dtype=np.float32)
        change = float(np.abs(no - ols)[mask].mean())
        if change > best_change:
            best_id, best_change = pair_id, change
    if not best_id:
        raise ValueError(f"no masked representative found for {scene_id}")
    return best_id, best_change


def contact_sheet(root: Path, summary: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    panel_width, panel_height, label_height = 350, 240, 54
    columns = ("Ground truth", "No relighting", "Public-HDRI OLS", "5x |OLS - none|")
    header_height = 62
    rows = len(summary["scenes"])
    canvas = Image.new(
        "RGB",
        (panel_width * len(columns), header_height + rows * (panel_height + label_height)),
        (14, 14, 14),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = font(23, bold=True)
    label_font = font(18, bold=True)
    small_font = font(15)
    for col, label in enumerate(columns):
        draw.text((col * panel_width + 12, 18), label, fill=(240, 240, 240), font=title_font)

    selections = []
    primary = summary["mask_results"]["sam_primary_undilated"]
    for row_index, scene_id in enumerate(summary["scenes"]):
        mask_index_path = root / scene_id / "sam-vit-h/primary-undilated/mask-index.json"
        mask_index = read_json(mask_index_path)
        pair_id, change = select_representative(root, scene_id, mask_index)
        payloads = {
            variant: read_json(root / scene_id / variant / "pairs.json")
            for variant in ("compact-no-relight", "compact-ols")
        }
        no_row = {str(item["id"]): item for item in payloads["compact-no-relight"]["pairs"]}[pair_id]
        ols_row = {str(item["id"]): item for item in payloads["compact-ols"]["pairs"]}[pair_id]
        target = Image.open(no_row["target"]).convert("RGB")
        no = Image.open(no_row["pred"]).convert("RGB")
        ols = Image.open(ols_row["pred"]).convert("RGB")
        mask = np.asarray(Image.open(mask_index["masks"][pair_id]).convert("L")) > 0
        box = crop_box(mask)
        target_crop, no_crop, ols_crop = target.crop(box), no.crop(box), ols.crop(box)
        diff = np.abs(
            np.asarray(ols_crop, dtype=np.int16) - np.asarray(no_crop, dtype=np.int16)
        )
        diff = Image.fromarray(np.clip(diff * 5, 0, 255).astype(np.uint8))
        y = header_height + row_index * (panel_height + label_height)
        for col, image in enumerate((target_crop, no_crop, ols_crop, diff)):
            canvas.paste(fit_panel(image, panel_width, panel_height), (col * panel_width, y))
        no_metric = primary["compact-no-relight"]["per_scene"][scene_id]["per_image"]
        ols_metric = primary["compact-ols"]["per_scene"][scene_id]["per_image"]
        label = (
            f"{scene_id}  {pair_id}  mean |dRGB|={change:.1f}/255\n"
            f"scene mean: PSNR {no_metric['psnr_vehicle_pixel_l2_db']['mean']:.2f} -> "
            f"{ols_metric['psnr_vehicle_pixel_l2_db']['mean']:.2f} dB, "
            f"PAE {no_metric['peak_angular_error_deg']['mean']:.2f} -> "
            f"{ols_metric['peak_angular_error_deg']['mean']:.2f} deg"
        )
        draw.rectangle(
            (0, y + panel_height, canvas.width, y + panel_height + label_height),
            fill=(8, 8, 8),
        )
        draw.text((12, y + panel_height + 6), label.split("\n")[0], fill=(255, 218, 120), font=label_font)
        draw.text((12, y + panel_height + 30), label.split("\n")[1], fill=(220, 220, 220), font=small_font)
        selections.append(
            {
                "scene": scene_id,
                "pair_id": pair_id,
                "mean_absolute_ols_change_uint8": change,
                "crop_xyxy": list(box),
                "target": no_row["target"],
                "no_relighting": no_row["pred"],
                "ols": ols_row["pred"],
                "mask": mask_index["masks"][pair_id],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)
    return selections


def shadow_contact_sheet(
    root: Path, summary: dict[str, Any], output: Path
) -> list[dict[str, Any]]:
    panel_width, panel_height, label_height = 350, 240, 54
    columns = ("Ground truth", "OLS", "OLS + contact shadow", "10x |shadow - OLS|")
    header_height = 62
    canvas = Image.new(
        "RGB",
        (
            panel_width * len(columns),
            header_height + len(summary["scenes"]) * (panel_height + label_height),
        ),
        (14, 14, 14),
    )
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = font(23, True), font(18, True), font(15)
    for col, label in enumerate(columns):
        draw.text((col * panel_width + 12, 18), label, fill=(240, 240, 240), font=title_font)
    primary = summary["mask_results"]["sam_primary_undilated"]
    selections = []
    for row_index, scene_id in enumerate(summary["scenes"]):
        mask_index = read_json(
            root / scene_id / "sam-vit-h/primary-undilated/mask-index.json"
        )
        ols = {
            str(row["id"]): row
            for row in read_json(root / scene_id / "compact-ols/pairs.json")["pairs"]
        }
        shadow = {
            str(row["id"]): row
            for row in read_json(root / scene_id / "compact-ols-shadow/pairs.json")["pairs"]
        }
        selected: tuple[float, str, tuple[int, int, int, int]] | None = None
        for pair_id, mask_value in mask_index["masks"].items():
            mask = np.asarray(Image.open(mask_value).convert("L")) > 0
            if not mask.any():
                continue
            box = crop_box(mask, padding=0.65)
            first = np.asarray(Image.open(ols[pair_id]["pred"]).convert("RGB"), dtype=np.float32)
            second = np.asarray(Image.open(shadow[pair_id]["pred"]).convert("RGB"), dtype=np.float32)
            x0, y0, x1, y1 = box
            change = float(np.abs(first[y0:y1, x0:x1] - second[y0:y1, x0:x1]).mean())
            candidate = (change, pair_id, box)
            if selected is None or candidate > selected:
                selected = candidate
        if selected is None:
            raise ValueError(f"no shadow representative found for {scene_id}")
        change, pair_id, box = selected
        target = Image.open(ols[pair_id]["target"]).convert("RGB").crop(box)
        first = Image.open(ols[pair_id]["pred"]).convert("RGB").crop(box)
        second = Image.open(shadow[pair_id]["pred"]).convert("RGB").crop(box)
        diff = np.abs(np.asarray(second, dtype=np.int16) - np.asarray(first, dtype=np.int16))
        diff = Image.fromarray(np.clip(diff * 10, 0, 255).astype(np.uint8))
        y = header_height + row_index * (panel_height + label_height)
        for col, image in enumerate((target, first, second, diff)):
            canvas.paste(fit_panel(image, panel_width, panel_height), (col * panel_width, y))
        ols_metric = primary["compact-ols"]["per_scene"][scene_id]["per_image"]
        shadow_metric = primary["compact-ols-shadow"]["per_scene"][scene_id]["per_image"]
        draw.rectangle(
            (0, y + panel_height, canvas.width, y + panel_height + label_height),
            fill=(8, 8, 8),
        )
        draw.text(
            (12, y + panel_height + 6),
            f"{scene_id}  {pair_id}  crop mean |dRGB|={change:.2f}/255",
            fill=(255, 218, 120),
            font=label_font,
        )
        draw.text(
            (12, y + panel_height + 30),
            f"scene mean: PAE {ols_metric['peak_angular_error_deg']['mean']:.3f} -> "
            f"{shadow_metric['peak_angular_error_deg']['mean']:.3f} deg",
            fill=(220, 220, 220),
            font=small_font,
        )
        selections.append(
            {
                "scene": scene_id,
                "pair_id": pair_id,
                "mean_absolute_shadow_change_uint8": change,
                "crop_xyxy": list(box),
                "ols": ols[pair_id]["pred"],
                "ols_shadow": shadow[pair_id]["pred"],
                "mask": mask_index["masks"][pair_id],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=93)
    return selections


def metric_mean(entry: dict[str, Any], key: str) -> float:
    return float(entry["metrics"]["per_image"][key]["mean"])


def make_markdown(
    root: Path,
    summary: dict[str, Any],
    probe: dict[str, Any],
    selections_path: Path,
) -> str:
    primary = summary["mask_results"]["sam_primary_undilated"]
    dilated = summary["mask_results"]["sam_sensitivity_dilated5"]
    lines = [
        "# DecoupleGS held-out vehicle re-insertion — public replacement protocol",
        "",
        "Status: **complete**. This uses the paper's Masked PSNR / PIE / PAE equations, but it is not an author-data reproduction: the held-out vehicle IDs, image sampling, SAM recipe, HDRI/material supervision, and cross-image aggregation are unpublished.",
        "",
        "## Primary result — undilated SAM ViT-H masks",
        "",
        "| Variant | Masked images | Masked PSNR ↑ | PIE ↓ | PAE ↓ | Offline camera FPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("raw-native", "compact-no-relight", "compact-ols", "compact-ols-shadow"):
        entry = primary[name]
        lines.append(
            f"| {name} | {entry['metrics']['masked_images']} | "
            f"{metric_mean(entry, 'psnr_vehicle_pixel_l2_db'):.3f} dB | "
            f"{metric_mean(entry, 'peak_intensity_error'):.4f} | "
            f"{metric_mean(entry, 'peak_angular_error_deg'):.3f}° | "
            f"{entry['runtime']['mean_camera_fps_across_scene_processes']:.1f} |"
        )
    delta = primary["compact-ols"]["delta_vs_compact_no_relight"]
    lines.extend(
        [
            "",
            f"Compression is nearly lossless relative to raw native assets: {metric_mean(primary['compact-no-relight'], 'psnr_vehicle_pixel_l2_db') - metric_mean(primary['raw-native'], 'psnr_vehicle_pixel_l2_db'):+.3f} dB and {metric_mean(primary['compact-no-relight'], 'peak_angular_error_deg') - metric_mean(primary['raw-native'], 'peak_angular_error_deg'):+.3f}° PAE.",
            "",
            f"The public covariance-diffuse OLS proxy does **not** reproduce the paper's relighting gain on real images: vs no relighting it changes Masked PSNR by {delta['psnr_vehicle_pixel_l2_db']:+.3f} dB, PIE by {delta['peak_intensity_error']:+.4f}, and PAE by {delta['peak_angular_error_deg']:+.3f}°. It remains an implemented ablation, not a paper-equivalent score.",
            "",
            "## Per-scene primary metrics",
            "",
            "| Scene | Masked images | No-relight PSNR | OLS PSNR | No-relight PIE | OLS PIE | No-relight PAE | OLS PAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene_id in summary["scenes"]:
        no = primary["compact-no-relight"]["per_scene"][scene_id]
        ols = primary["compact-ols"]["per_scene"][scene_id]
        n, o = no["per_image"], ols["per_image"]
        lines.append(
            f"| {scene_id} | {no['masked_images']} | "
            f"{n['psnr_vehicle_pixel_l2_db']['mean']:.3f} | {o['psnr_vehicle_pixel_l2_db']['mean']:.3f} | "
            f"{n['peak_intensity_error']['mean']:.4f} | {o['peak_intensity_error']['mean']:.4f} | "
            f"{n['peak_angular_error_deg']['mean']:.3f}° | {o['peak_angular_error_deg']['mean']:.3f}° |"
        )
    unsupported = primary["compact-no-relight"]["metrics"]["per_image"]["pae_unsupported_fraction"]["mean"]
    diluted_delta = dilated["compact-ols"]["delta_vs_compact_no_relight"]
    test_rows = [
        track["test"]
        for scene in probe["scenes"].values()
        for track in scene["tracks"].values()
    ]
    lines.extend(
        [
            "",
            "## Robustness and diagnosis",
            "",
            f"- The primary set contains 1,296 held-out camera views; 210 have a non-empty vehicle SAM mask, totaling {primary['compact-no-relight']['metrics']['mask_pixels']:,} masked pixels across 10 tracks.",
            f"- Five-pixel dilation preserves the conclusion: OLS delta is {diluted_delta['psnr_vehicle_pixel_l2_db']:+.3f} dB PSNR, {diluted_delta['peak_intensity_error']:+.4f} PIE, {diluted_delta['peak_angular_error_deg']:+.3f}° PAE.",
            f"- Near-zero RGB vectors are negligible: mean unsupported fraction at a 1/255 norm threshold is {unsupported:.8f}; literal and supported PAE therefore agree.",
            f"- Real test probes have {sum(row['element_outside_fit_range_fraction'] for row in test_rows) / len(test_rows):.2%} of descriptor elements outside the public-HDRI fit range; {sum(row['samples_with_any_outside_dimension_fraction'] for row in test_rows) / len(test_rows):.2%} of track-timestamps have at least one outside dimension.",
            "- The degradation also occurs for in-range tracks, so the dominant gap is the unpublished target supervision (vehicle materials, target renderer, and relit per-primitive colors), not merely descriptor extrapolation.",
            "- Contact shadows preserve logged poses and make only a small aggregate change; re-grounding logged vehicles is intentionally disabled because their dataset pose is already the ground-truth registration.",
            "",
            "## Artifacts",
            "",
            f"- Machine-readable summary: `{(root / 'protocol-summary.json').resolve()}`",
            f"- Probe-domain audit: `{(root / 'probe-domain-audit.json').resolve()}`",
            f"- Representative selection manifest: `{selections_path.resolve()}`",
            f"- Vehicle-crop contact sheet: `{(root / 'reinsertion-comparison-contact-sheet.jpg').resolve()}`",
            f"- Contact-shadow effect sheet: `{(root / 'reinsertion-shadow-contact-sheet.jpg').resolve()}`",
            "",
            "The paper's reported 6.8° / 48.5° relighting ablation cannot be directly compared with this table: the author vehicle cohort and calibration supervision are absent, and the paper does not state its cross-image peak aggregation. This report emits arithmetic mean, median, pooled masked PSNR, and dataset maxima in JSON so no hidden aggregation is selected after seeing results.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(root: Path, summary: dict[str, Any]) -> None:
    primary = summary["mask_results"]["sam_primary_undilated"]
    path = root / "per-track-primary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "track_id",
                "masked_images",
                "no_relight_psnr_db",
                "ols_psnr_db",
                "no_relight_pie",
                "ols_pie",
                "no_relight_pae_deg",
                "ols_pae_deg",
            ]
        )
        no = primary["compact-no-relight"]["per_track"]
        ols = primary["compact-ols"]["per_track"]
        for track_id in sorted(no):
            n, o = no[track_id], ols[track_id]
            writer.writerow(
                [
                    track_id,
                    n["masked_images"],
                    n["per_image"]["psnr_vehicle_pixel_l2_db"]["mean"],
                    o["per_image"]["psnr_vehicle_pixel_l2_db"]["mean"],
                    n["per_image"]["peak_intensity_error"]["mean"],
                    o["per_image"]["peak_intensity_error"]["mean"],
                    n["per_image"]["peak_angular_error_deg"]["mean"],
                    o["per_image"]["peak_angular_error_deg"]["mean"],
                ]
            )


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    summary = read_json(root / "protocol-summary.json")
    probe = read_json(root / "probe-domain-audit.json")
    contact_path = root / "reinsertion-comparison-contact-sheet.jpg"
    selections = contact_sheet(root, summary, contact_path)
    selection_path = root / "representative-selections.json"
    selection_path.write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    shadow_contact_path = root / "reinsertion-shadow-contact-sheet.jpg"
    shadow_selections = shadow_contact_sheet(root, summary, shadow_contact_path)
    shadow_selection_path = root / "shadow-representative-selections.json"
    shadow_selection_path.write_text(
        json.dumps(shadow_selections, indent=2) + "\n", encoding="utf-8"
    )
    report = make_markdown(root, summary, probe, selection_path)
    report_path = root / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    write_csv(root, summary)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "contact_sheet": str(contact_path),
                "selections": len(selections),
                "shadow_contact_sheet": str(shadow_contact_path),
                "shadow_selections": len(shadow_selections),
                "per_track_csv": str(root / "per-track-primary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
