from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class NuScenesFrameMapping:
    """Mapping from one HUGSIM 12 Hz camera frame to the raw nuScenes image."""

    frame_index: int
    camera: str
    source_path: str
    output_path: str
    sample_token: str
    sample_data_token: str
    sample_timestamp_us: float
    sensor_timestamp_us: int
    is_key_frame: bool
    output_width: int
    output_height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _index(records: list[dict[str, Any]], field: str = "token") -> dict[str, dict[str, Any]]:
    return {str(record[field]): record for record in records}


def _follow_asap_camera_sweep(
    first: dict[str, Any],
    frame_offset: int,
    sample_data_by_token: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Mirror ASAP's key-frame plus five camera-sweep selection exactly."""

    selected = first
    for _ in range(frame_offset):
        next_token = str(selected.get("next", ""))
        if not next_token:
            break
        candidate = sample_data_by_token[next_token]
        if bool(candidate["is_key_frame"]):
            break
        selected = candidate
    return selected


def build_hugsim_12hz_manifest(
    *,
    scene_name: str,
    scenes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    sample_data: list[dict[str, Any]],
    calibrated_sensors: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
    frame_count: int = 180,
    interpolation_factor: int = 6,
    downsample: int = 2,
) -> list[NuScenesFrameMapping]:
    """Reconstruct the raw-image selection used by HUGSIM's nuScenes loader.

    ASAP inserts ``interpolation_factor - 1`` samples between adjacent 2 Hz
    nuScenes key frames. HUGSIM then takes the first 180 resulting samples and
    writes the six cameras in ``NUSCENES_CAMERAS`` order.
    """

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if interpolation_factor <= 0:
        raise ValueError("interpolation_factor must be positive")
    if downsample <= 0:
        raise ValueError("downsample must be positive")

    scene_matches = [scene for scene in scenes if scene.get("name") == scene_name]
    if len(scene_matches) != 1:
        raise ValueError(f"expected one scene named {scene_name!r}, found {len(scene_matches)}")

    sample_by_token = _index(samples)
    sample_data_by_token = _index(sample_data)
    sensor_by_token = _index(sensors)
    calibrated_sensor_by_token = _index(calibrated_sensors)
    channel_by_calibration = {
        token: str(sensor_by_token[str(calibration["sensor_token"])]["channel"])
        for token, calibration in calibrated_sensor_by_token.items()
    }
    key_data_by_sample_camera = {
        (str(record["sample_token"]), channel_by_calibration[str(record["calibrated_sensor_token"])]): record
        for record in sample_data
        if bool(record["is_key_frame"])
    }
    current = sample_by_token[str(scene_matches[0]["first_sample_token"])]
    mappings: list[NuScenesFrameMapping] = []
    frame_index = 0

    while frame_index < frame_count:
        next_sample_token = str(current.get("next", ""))
        next_sample = sample_by_token[next_sample_token] if next_sample_token else None

        for frame_offset in range(interpolation_factor):
            if frame_index >= frame_count:
                break
            if frame_offset > 0 and next_sample is None:
                break

            if next_sample is None:
                sample_timestamp_us = float(current["timestamp"])
            else:
                fraction = frame_offset / interpolation_factor
                sample_timestamp_us = float(current["timestamp"]) + fraction * (
                    float(next_sample["timestamp"]) - float(current["timestamp"])
                )

            for camera in NUSCENES_CAMERAS:
                sample_camera_key = (str(current["token"]), camera)
                if sample_camera_key not in key_data_by_sample_camera:
                    raise KeyError(f"sample {current['token']} has no {camera} data")
                key_data = key_data_by_sample_camera[sample_camera_key]
                selected = _follow_asap_camera_sweep(
                    key_data, frame_offset, sample_data_by_token
                )
                output_height = 410 if camera == "CAM_BACK" else 450
                mappings.append(
                    NuScenesFrameMapping(
                        frame_index=frame_index,
                        camera=camera,
                        source_path=str(selected["filename"]),
                        output_path=f"images/{camera}/{frame_index:05d}.jpg",
                        sample_token=str(current["token"]),
                        sample_data_token=str(selected["token"]),
                        sample_timestamp_us=sample_timestamp_us,
                        sensor_timestamp_us=int(selected["timestamp"]),
                        is_key_frame=bool(selected["is_key_frame"]),
                        output_width=1600 // downsample,
                        output_height=output_height * 2 // downsample,
                    )
                )
            frame_index += 1

        if next_sample is None:
            break
        current = next_sample

    if frame_index != frame_count:
        raise ValueError(
            f"scene {scene_name!r} only yielded {frame_index} frames; requested {frame_count}"
        )
    return mappings


def load_hugsim_12hz_manifest(
    metadata_root: str | Path,
    scene_name: str,
    *,
    frame_count: int = 180,
    interpolation_factor: int = 6,
    downsample: int = 2,
) -> list[NuScenesFrameMapping]:
    metadata_root = Path(metadata_root)

    def load_table(name: str) -> list[dict[str, Any]]:
        with (metadata_root / f"{name}.json").open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise TypeError(f"{name}.json must contain a list")
        return value

    # sample_data.json is large; keeping only these three tables bounds peak
    # memory and avoids loading the 1.2 GB annotation/ego-pose tables.
    return build_hugsim_12hz_manifest(
        scene_name=scene_name,
        scenes=load_table("scene"),
        samples=load_table("sample"),
        sample_data=load_table("sample_data"),
        calibrated_sensors=load_table("calibrated_sensor"),
        sensors=load_table("sensor"),
        frame_count=frame_count,
        interpolation_factor=interpolation_factor,
        downsample=downsample,
    )


def validate_against_hugsim_metadata(
    mappings: list[NuScenesFrameMapping], metadata_path: str | Path
) -> None:
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    frames = metadata["frames"]
    if len(frames) != len(mappings):
        raise ValueError(f"frame count mismatch: HUGSIM={len(frames)}, manifest={len(mappings)}")

    start_timestamp = mappings[0].sample_timestamp_us
    for mapping, frame in zip(mappings, frames, strict=True):
        expected_path = "./" + mapping.output_path
        if frame["rgb_path"] != expected_path:
            raise ValueError(
                f"path mismatch at frame {mapping.frame_index}/{mapping.camera}: "
                f"{frame['rgb_path']!r} != {expected_path!r}"
            )
        if int(frame["width"]) != mapping.output_width or int(frame["height"]) != mapping.output_height:
            raise ValueError(
                f"shape mismatch at {expected_path}: "
                f"HUGSIM={frame['width']}x{frame['height']}, "
                f"manifest={mapping.output_width}x{mapping.output_height}"
            )
        expected_seconds = (mapping.sample_timestamp_us - start_timestamp) / 1_000_000.0
        if abs(float(frame["timestamp"]) - expected_seconds) > 1e-5:
            raise ValueError(
                f"timestamp mismatch at {expected_path}: "
                f"{frame['timestamp']} != {expected_seconds}"
            )
