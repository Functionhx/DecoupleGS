#!/usr/bin/env python3
"""Materialize exact nuScenes keyframe state for open-loop E2E evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decouplegs.nuscenes_protocol import build_open_loop_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--uniad-info",
        type=Path,
        help="Official UniAD temporal train/val pkl for exact planner inputs",
    )
    parser.add_argument(
        "--vad-info",
        type=Path,
        help="Official VAD temporal train/val pkl for exact VAD history",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path):
    try:
        import orjson
    except ImportError:
        return json.loads(path.read_text(encoding="utf-8"))
    return orjson.loads(path.read_bytes())


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    scenes = load_json(args.metadata_root / "scene.json")
    samples = load_json(args.metadata_root / "sample.json")
    sensors = load_json(args.metadata_root / "sensor.json")
    calibrations = load_json(args.metadata_root / "calibrated_sensor.json")

    wanted_samples = {str(mapping["sample_token"]) for mapping in manifest["mappings"]}
    scene = next(value for value in scenes if value.get("name") == args.scene)
    sample_by_token = {str(value["token"]): value for value in samples}
    token = str(scene["first_sample_token"])
    while token:
        wanted_samples.add(token)
        token = str(sample_by_token[token].get("next", ""))
    wanted_camera_data = {
        str(mapping["sample_data_token"]) for mapping in manifest["mappings"]
    }
    channel_by_calibration = {
        str(calibration["token"]): next(
            sensor["channel"]
            for sensor in sensors
            if sensor["token"] == calibration["sensor_token"]
        )
        for calibration in calibrations
    }
    all_sample_data = load_json(args.metadata_root / "sample_data.json")
    sample_data = [
        value
        for value in all_sample_data
        if str(value["token"]) in wanted_camera_data
        or (
            str(value["sample_token"]) in wanted_samples
            and channel_by_calibration.get(str(value["calibrated_sensor_token"]))
            == "LIDAR_TOP"
            and bool(value.get("is_key_frame", False))
        )
    ]
    del all_sample_data
    gc.collect()

    wanted_poses = {str(value["ego_pose_token"]) for value in sample_data}
    all_ego_poses = load_json(args.metadata_root / "ego_pose.json")
    ego_poses = [value for value in all_ego_poses if str(value["token"]) in wanted_poses]
    del all_ego_poses
    gc.collect()

    official_infos = None
    if args.uniad_info is not None:
        with args.uniad_info.open("rb") as stream:
            info_payload = pickle.load(stream)
        official_infos = [
            value
            for value in info_payload["infos"]
            if str(value["token"]) in wanted_samples
        ]
        del info_payload
        gc.collect()

    official_vad_infos = None
    if args.vad_info is not None:
        with args.vad_info.open("rb") as stream:
            vad_info_payload = pickle.load(stream)
        official_vad_infos = [
            value
            for value in vad_info_payload["infos"]
            if str(value["token"]) in wanted_samples
        ]
        del vad_info_payload
        gc.collect()

    protocol = build_open_loop_protocol(
        scene_name=args.scene,
        manifest=manifest,
        raw_root=args.raw_root,
        scenes=scenes,
        samples=samples,
        sample_data=sample_data,
        calibrated_sensors=calibrations,
        sensors=sensors,
        ego_poses=ego_poses,
        official_uniad_infos=official_infos,
        official_vad_infos=official_vad_infos,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "scene": args.scene,
                "frames": len(protocol["frames"]),
                "output": str(args.output.resolve()),
                "state_source": protocol["state_source"],
                "can_bus_source": protocol["can_bus_source"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
