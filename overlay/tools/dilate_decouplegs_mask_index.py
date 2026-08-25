#!/usr/bin/env python3
"""Create a deterministic dilation sensitivity index from frozen SAM masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def dilate_file(source: Path, destination: Path, kernel: np.ndarray) -> int:
    array = np.asarray(Image.open(source).convert("L"))
    result = cv2.dilate((array > 0).astype(np.uint8) * 255, kernel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(destination)
    return int(np.count_nonzero(result))


def main() -> None:
    args = parse_args()
    if args.radius <= 0:
        raise ValueError("--radius must be positive")
    payload = json.loads(args.input_index.read_text(encoding="utf-8"))
    base = args.input_index.parent
    width = 2 * args.radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    output_masks: dict[str, str] = {}
    output_instances: dict[str, dict[str, str]] = {}
    pixels: dict[str, int] = {}
    for pair_id, value in payload["masks"].items():
        source = resolve(value, base)
        destination = args.output_dir / "union" / source.name
        pixels[pair_id] = dilate_file(source, destination, kernel)
        output_masks[pair_id] = str(destination.resolve())
        output_instances[pair_id] = {}
        for track_id, instance_value in payload.get("instance_masks", {}).get(
            pair_id, {}
        ).items():
            instance_source = resolve(instance_value, base)
            instance_destination = (
                args.output_dir / "instances" / track_id / instance_source.name
            )
            dilate_file(instance_source, instance_destination, kernel)
            output_instances[pair_id][track_id] = str(instance_destination.resolve())

    protocol = dict(payload.get("protocol", {}))
    protocol.update(
        {
            "morphological_dilation_pixels": args.radius,
            "dilation_shape": "OpenCV MORPH_ELLIPSE",
            "sensitivity_only": True,
            "source_mask_kind": "undilated official SAM prediction",
        }
    )
    result = {
        "schema_version": 2,
        "hypothesis": "H-PHOTO-01",
        "status": "five_pixel_mask_sensitivity_not_primary_metric",
        "source_index": str(args.input_index.resolve()),
        "source_index_sha256": sha256(args.input_index),
        "pairs_manifest": payload.get("pairs_manifest"),
        "pairs_sha256": payload.get("pairs_sha256"),
        "protocol": protocol,
        "pair_count": len(output_masks),
        "nonempty_masks": sum(value > 0 for value in pixels.values()),
        "masks": output_masks,
        "instance_masks": output_instances,
        "mask_pixels": pixels,
    }
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    args.output_index.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "pair_count": result["pair_count"],
                "nonempty_masks": result["nonempty_masks"],
                "radius": args.radius,
                "output_index": str(args.output_index.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
