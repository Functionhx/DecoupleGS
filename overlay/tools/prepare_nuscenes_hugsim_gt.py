#!/usr/bin/env python3
"""Build and materialize the exact nuScenes 12 Hz images used by HUGSIM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.nuscenes_gt import (
    load_hugsim_12hz_manifest,
    validate_against_hugsim_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--hugsim-metadata", type=Path)
    parser.add_argument(
        "--source-list",
        type=Path,
        help="optional tar --files-from list containing each unique raw image path",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        help="raw nuScenes root; when supplied, materialize processed images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="output scene directory for materialized HUGSIM-sized JPEGs",
    )
    return parser.parse_args()


def materialize(mappings, raw_root: Path, output_root: Path) -> None:
    missing: list[Path] = []
    for mapping in mappings:
        source = raw_root / mapping.source_path
        if not source.is_file():
            missing.append(source)
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {source}")
        if mapping.camera == "CAM_BACK":
            image = image[:-80]
        image = cv2.resize(
            image,
            (mapping.output_width, mapping.output_height),
            interpolation=cv2.INTER_LINEAR,
        )
        destination = output_root / mapping.output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"failed to write {destination}")
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} raw images are missing; first paths:\n{preview}"
        )


def main() -> None:
    args = parse_args()
    mappings = load_hugsim_12hz_manifest(
        args.metadata_root, args.scene, frame_count=args.frames
    )
    if args.hugsim_metadata:
        validate_against_hugsim_metadata(mappings, args.hugsim_metadata)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scene": args.scene,
        "frame_count": args.frames,
        "camera_count": 6,
        "mapping_count": len(mappings),
        "mappings": [mapping.to_dict() for mapping in mappings],
    }
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    if args.source_list:
        args.source_list.parent.mkdir(parents=True, exist_ok=True)
        unique_sources = sorted({mapping.source_path for mapping in mappings})
        args.source_list.write_text("\n".join(unique_sources) + "\n", encoding="utf-8")

    if (args.raw_root is None) != (args.output_root is None):
        raise ValueError("--raw-root and --output-root must be supplied together")
    if args.raw_root is not None:
        materialize(mappings, args.raw_root, args.output_root)
        if args.hugsim_metadata is not None:
            # Keep the released camera/dynamic transforms next to the staged
            # images so HUGSIM's native held-out loader is directly usable.
            shutil.copy2(args.hugsim_metadata, args.output_root / "meta_data.json")

    print(
        json.dumps(
            {
                "scene": args.scene,
                "frames": args.frames,
                "cameras": 6,
                "mappings": len(mappings),
                "unique_sources": len({mapping.source_path for mapping in mappings}),
                "manifest": str(args.manifest),
                "materialized": args.raw_root is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
