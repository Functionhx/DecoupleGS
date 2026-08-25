#!/usr/bin/env python3
"""Refine projected vehicle prompts with the official Segment Anything model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_h")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--dilation", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_prompts(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    count, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    prompts = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if not len(xs):
            continue
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        # A projected box center can fall outside a clipped/oblique hull. Use
        # the component pixel nearest its centroid as a guaranteed foreground
        # point prompt.
        centroid = np.array([xs.mean(), ys.mean()])
        distances = (xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2
        nearest = int(np.argmin(distances))
        point = np.array([[xs[nearest], ys[nearest]]], dtype=np.float32)
        prompts.append((box, point))
    return prompts


def prompt_entries(pair: dict, fallback: np.ndarray) -> list[dict]:
    """Prefer instance-separated prompts while accepting schema-v1 pairs."""

    entries = pair.get("instance_prompts")
    if entries:
        return [
            {
                "track_id": str(entry["track_id"]),
                "category": entry.get("category"),
                "mask": np.asarray(Image.open(entry["mask"]).convert("L")),
            }
            for entry in entries
        ]
    return [{"track_id": "union", "category": None, "mask": fallback}]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    first = first > 0
    second = second > 0
    union = np.logical_or(first, second).sum()
    if not union:
        return None
    return float(np.logical_and(first, second).sum() / union)


def main() -> None:
    args = parse_args()
    if args.dilation < 0:
        raise ValueError("--dilation must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    model = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    model.to(device=args.device).eval()
    predictor = SamPredictor(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel = None
    if args.dilation:
        width = 2 * args.dilation + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))

    masks: dict[str, str] = {}
    instance_masks: dict[str, dict[str, str]] = {}
    rows: list[dict] = []
    nonempty = 0
    prediction_scores = []
    for pair in pairs:
        pair_id = str(pair["id"])
        prompt = np.asarray(Image.open(pair["mask"]).convert("L"))
        output = np.zeros_like(prompt, dtype=np.uint8)
        entries = prompt_entries(pair, prompt)
        has_prompt = any(component_prompts(entry["mask"]) for entry in entries)
        pair_instances: dict[str, str] = {}
        pair_rows: list[dict] = []
        if has_prompt:
            image = np.asarray(Image.open(pair["target"]).convert("RGB"))
            predictor.set_image(image)
            for entry in entries:
                track_id = entry["track_id"]
                geometry = entry["mask"]
                refined = np.zeros_like(prompt, dtype=np.uint8)
                component_rows = []
                for box, point in component_prompts(geometry):
                    candidates, scores, _ = predictor.predict(
                        point_coords=point,
                        point_labels=np.ones((1,), dtype=np.int32),
                        box=box,
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    candidate = candidates[best]
                    refined[candidate] = 255
                    score = float(scores[best])
                    prediction_scores.append(score)
                    component_rows.append(
                        {
                            "box_xyxy": [float(value) for value in box],
                            "point_xy": [float(value) for value in point[0]],
                            "predicted_iou": score,
                            "geometry_iou": mask_iou(candidate, geometry),
                        }
                    )
                if kernel is not None and refined.any():
                    refined = cv2.dilate(refined, kernel)
                if refined.any():
                    track_dir = args.output_dir / "instances" / track_id
                    track_dir.mkdir(parents=True, exist_ok=True)
                    destination = track_dir / Path(pair["mask"]).name
                    Image.fromarray(refined).save(destination)
                    pair_instances[track_id] = str(destination.resolve())
                    output = np.maximum(output, refined)
                pair_rows.append(
                    {
                        "track_id": track_id,
                        "category": entry.get("category"),
                        "geometry_pixels": int(np.count_nonzero(geometry)),
                        "sam_pixels": int(np.count_nonzero(refined)),
                        "components": component_rows,
                    }
                )
            nonempty += int(output.any())
            predictor.reset_image()
        filename = Path(pair["mask"]).name
        destination = args.output_dir / filename
        Image.fromarray(output).save(destination)
        masks[pair_id] = str(destination.resolve())
        instance_masks[pair_id] = pair_instances
        rows.append(
            {
                "id": pair_id,
                "target": str(Path(pair["target"]).resolve()),
                "union_pixels": int(np.count_nonzero(output)),
                "instances": pair_rows,
            }
        )

    result = {
        "schema_version": 2,
        "hypothesis": "H-PHOTO-01",
        "pairs_manifest": str(args.pairs.resolve()),
        "pairs_sha256": sha256(args.pairs),
        "protocol": {
            "model": "official Segment Anything",
            "model_type": args.model_type,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(args.checkpoint),
            "prompt": "per-vehicle projected 3D convex hull; XYXY box plus nearest-centroid positive point",
            "candidate_selection": "highest SAM predicted IoU",
            "morphological_dilation_pixels": args.dilation,
            "paper_gap": (
                "metric section discloses precise SAM masks but not model variant, "
                "prompting, dilation, or post-processing; five-pixel dilation is "
                "disclosed separately for static-background cleanup"
            ),
        },
        "pair_count": len(pairs),
        "nonempty_masks": nonempty,
        "mean_predicted_iou": (
            float(np.mean(prediction_scores)) if prediction_scores else None
        ),
        "masks": masks,
        "instance_masks": instance_masks,
        "pairs": rows,
    }
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pair_count", "nonempty_masks", "mean_predicted_iou", "protocol")}, indent=2))


if __name__ == "__main__":
    main()
