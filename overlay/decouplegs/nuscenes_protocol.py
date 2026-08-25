"""Exact nuScenes keyframe state for planner-facing open-loop evaluation.

The HUGSIM reconstruction metadata is expressed in a canonical OpenCV world
frame and includes interpolated 12 Hz camera poses.  UniAD and VAD, however,
are trained and evaluated on 2 Hz nuScenes keyframes in the top-LiDAR frame.
This module builds a small, auditable protocol file directly from the public
nuScenes metadata tables so rendering variants can share the exact same
planner state and future ego trajectory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from decouplegs.nuscenes_gt import NUSCENES_CAMERAS


def index_records(
    records: Iterable[dict[str, Any]], field: str = "token"
) -> dict[str, dict[str, Any]]:
    return {str(record[field]): record for record in records}


def quaternion_matrix(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    """Convert a nuScenes ``[w, x, y, z]`` quaternion to a rotation matrix."""

    quaternion = np.asarray(tuple(quaternion_wxyz), dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(translation: Iterable[float], rotation_wxyz: Iterable[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_matrix(rotation_wxyz)
    pose[:3, 3] = np.asarray(tuple(translation), dtype=np.float64)
    return pose


def lidar_to_global(
    sample_data: dict[str, Any],
    calibrated_sensor: dict[str, Any],
    ego_pose: dict[str, Any],
) -> np.ndarray:
    del sample_data  # Included in the signature to make table provenance explicit.
    return pose_matrix(ego_pose["translation"], ego_pose["rotation"]) @ pose_matrix(
        calibrated_sensor["translation"], calibrated_sensor["rotation"]
    )


def future_lidar_trajectory(
    lidar_poses: list[np.ndarray], index: int, steps: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Return future LiDAR origins in current ``[right, forward]`` axes."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    trajectory = np.zeros((steps, 2), dtype=np.float64)
    mask = np.zeros(steps, dtype=bool)
    current_inverse = np.linalg.inv(lidar_poses[index])
    count = min(steps, len(lidar_poses) - index - 1)
    for offset in range(1, count + 1):
        relative = current_inverse @ lidar_poses[index + offset]
        # nuScenes top-LiDAR uses x=right, y=forward in this release stack.
        trajectory[offset - 1] = relative[:2, 3]
        mask[offset - 1] = True
    return trajectory, mask


def past_lidar_offsets(
    lidar_poses: list[np.ndarray], index: int, steps: int = 2
) -> np.ndarray:
    """Reproduce VAD's previous-LiDAR per-step displacement features."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    positions = np.zeros((steps + 1, 3), dtype=np.float64)
    differences = np.zeros_like(positions)
    source_index = index
    for output_index in range(steps, -1, -1):
        if source_index >= 0:
            positions[output_index] = lidar_poses[source_index][:3, 3]
            if source_index + 1 < len(lidar_poses):
                differences[output_index] = (
                    lidar_poses[source_index + 1][:3, 3]
                    - lidar_poses[source_index][:3, 3]
                )
            source_index -= 1
        else:
            positions[output_index] = (
                positions[output_index + 1] - differences[output_index + 1]
            )
            differences[output_index] = differences[output_index + 1]
    current_inverse = np.linalg.inv(lidar_poses[index])
    homogeneous = np.concatenate(
        (positions, np.ones((positions.shape[0], 1), dtype=np.float64)), axis=-1
    )
    local = (current_inverse @ homogeneous.T).T[:, :3]
    return np.diff(local, axis=0)[:, :2]


def navigation_command(trajectory: np.ndarray, mask: np.ndarray) -> int:
    valid = trajectory[np.asarray(mask, dtype=bool)]
    if not len(valid):
        return 2
    lateral = float(valid[-1, 0])
    if lateral >= 2.0:
        return 0
    if lateral <= -2.0:
        return 1
    return 2


def camera_parameters(
    camera_data: dict[str, Any],
    camera_calibration: dict[str, Any],
    camera_ego_pose: dict[str, Any],
    lidar_calibration: dict[str, Any],
    lidar_ego_pose: dict[str, Any],
    *,
    output_width: int,
    output_height: int,
    downsample: int = 2,
) -> dict[str, Any]:
    """Build the HUGSIM planner camera contract from native calibration."""

    del camera_data
    camera_to_global = pose_matrix(
        camera_ego_pose["translation"], camera_ego_pose["rotation"]
    ) @ pose_matrix(
        camera_calibration["translation"], camera_calibration["rotation"]
    )
    lidar_to_global_pose = pose_matrix(
        lidar_ego_pose["translation"], lidar_ego_pose["rotation"]
    ) @ pose_matrix(
        lidar_calibration["translation"], lidar_calibration["rotation"]
    )
    lidar_to_camera = np.linalg.inv(camera_to_global) @ lidar_to_global_pose
    intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera intrinsic must have shape (3, 3), got {intrinsic.shape}")
    scaled = intrinsic.copy()
    scaled[:2] /= float(downsample)
    fovx = 2.0 * math.atan(output_width / (2.0 * scaled[0, 0]))
    fovy = 2.0 * math.atan(output_height / (2.0 * scaled[1, 1]))
    return {
        "intrinsic": {
            "H": int(output_height),
            "W": int(output_width),
            "cx": float(scaled[0, 2]),
            "cy": float(scaled[1, 2]),
            # ``load_camera_cfg`` converts YAML degrees to radians before the
            # planner sees this contract; protocol JSON therefore stores the
            # same post-load radian values directly.
            "fovx": float(fovx),
            "fovy": float(fovy),
        },
        "l2c": lidar_to_camera.tolist(),
    }


def official_uniad_camera_state(info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reproduce UniAD's native ``get_data_info`` camera matrices exactly."""

    state: dict[str, dict[str, Any]] = {}
    cameras = info.get("cams", {})
    for camera in NUSCENES_CAMERAS:
        if camera not in cameras:
            raise ValueError(f"official UniAD info lacks camera {camera}")
        camera_info = cameras[camera]
        sensor_to_lidar_rotation = np.asarray(
            camera_info["sensor2lidar_rotation"], dtype=np.float64
        )
        sensor_to_lidar_translation = np.asarray(
            camera_info["sensor2lidar_translation"], dtype=np.float64
        )
        lidar_to_camera_rotation = np.linalg.inv(sensor_to_lidar_rotation)
        lidar_to_camera_translation = (
            sensor_to_lidar_translation @ lidar_to_camera_rotation.T
        )
        lidar_to_camera = np.eye(4, dtype=np.float64)
        # This row/column layout intentionally mirrors
        # NuScenesE2EDataset.get_data_info rather than simplifying the
        # expression and risking a convention change.
        lidar_to_camera[:3, :3] = lidar_to_camera_rotation.T
        lidar_to_camera[3, :3] = -lidar_to_camera_translation
        intrinsic = np.asarray(camera_info["cam_intrinsic"], dtype=np.float64)
        viewpad = np.eye(4, dtype=np.float64)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        state[camera] = {
            "lidar2img": (viewpad @ lidar_to_camera.T).tolist(),
            "intrinsic": viewpad.tolist(),
            "sample_data_token": str(camera_info["sample_data_token"]),
            "timestamp_us": int(camera_info["timestamp"]),
        }
    return state


def official_uniad_planner_state(info: dict[str, Any]) -> dict[str, Any]:
    """Materialize pose and CAN-bus fields exactly as UniAD does at test time."""

    lidar_to_ego_rotation = quaternion_matrix(info["lidar2ego_rotation"])
    ego_to_global_rotation = quaternion_matrix(info["ego2global_rotation"])
    lidar_to_ego_translation = np.asarray(
        info["lidar2ego_translation"], dtype=np.float64
    )
    ego_to_global_translation = np.asarray(
        info["ego2global_translation"], dtype=np.float64
    )
    can_bus = np.asarray(info["can_bus"], dtype=np.float64).copy()
    if can_bus.shape != (18,):
        raise ValueError(f"official UniAD can_bus must have shape (18,), got {can_bus.shape}")
    can_bus[:3] = ego_to_global_translation
    can_bus[3:7] = np.asarray(info["ego2global_rotation"], dtype=np.float64)
    heading = ego_to_global_rotation @ np.asarray((1.0, 0.0, 0.0))
    patch_angle = math.degrees(math.atan2(float(heading[1]), float(heading[0])))
    if patch_angle < 0.0:
        patch_angle += 360.0
    can_bus[-2] = math.radians(patch_angle)
    can_bus[-1] = patch_angle
    return {
        "scene_token": str(info["scene_token"]),
        "l2g_t": (
            lidar_to_ego_translation @ ego_to_global_rotation.T
            + ego_to_global_translation
        ).tolist(),
        "l2g_r_mat": (
            lidar_to_ego_rotation.T @ ego_to_global_rotation.T
        ).tolist(),
        "can_bus": can_bus.tolist(),
        "camera_state": official_uniad_camera_state(info),
    }


def _scene_sample_chain(
    scene_name: str,
    scenes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = [scene for scene in scenes if str(scene.get("name")) == scene_name]
    if len(matches) != 1:
        raise ValueError(f"expected one scene named {scene_name!r}, found {len(matches)}")
    sample_by_token = index_records(samples)
    chain: list[dict[str, Any]] = []
    token = str(matches[0]["first_sample_token"])
    while token:
        sample = sample_by_token[token]
        chain.append(sample)
        token = str(sample.get("next", ""))
    return matches[0], chain


def build_open_loop_protocol(
    *,
    scene_name: str,
    manifest: dict[str, Any],
    raw_root: str | Path,
    scenes: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    sample_data: list[dict[str, Any]],
    calibrated_sensors: list[dict[str, Any]],
    sensors: list[dict[str, Any]],
    ego_poses: list[dict[str, Any]],
    official_uniad_infos: Iterable[dict[str, Any]] | None = None,
    official_vad_infos: Iterable[dict[str, Any]] | None = None,
    planning_steps: int = 6,
) -> dict[str, Any]:
    """Build a compact official-state protocol for one reconstructed clip."""

    raw_root = Path(raw_root)
    scene, sample_chain = _scene_sample_chain(scene_name, scenes, samples)
    sample_index = {str(sample["token"]): index for index, sample in enumerate(sample_chain)}
    sensor_by_token = index_records(sensors)
    calibration_by_token = index_records(calibrated_sensors)
    channel_by_calibration = {
        token: str(sensor_by_token[str(calibration["sensor_token"])]["channel"])
        for token, calibration in calibration_by_token.items()
    }
    data_by_token = index_records(sample_data)
    lidar_by_sample = {
        str(record["sample_token"]): record
        for record in sample_data
        if channel_by_calibration.get(str(record["calibrated_sensor_token"]))
        == "LIDAR_TOP"
        and bool(record.get("is_key_frame", False))
        and str(record["sample_token"]) in sample_index
    }
    ego_pose_by_token = index_records(ego_poses)
    official_info_by_token = (
        None
        if official_uniad_infos is None
        else index_records(official_uniad_infos, field="token")
    )
    official_vad_by_token = (
        None
        if official_vad_infos is None
        else index_records(official_vad_infos, field="token")
    )

    lidar_poses: list[np.ndarray] = []
    for sample in sample_chain:
        token = str(sample["token"])
        data = lidar_by_sample[token]
        calibration = calibration_by_token[str(data["calibrated_sensor_token"])]
        ego_pose = ego_pose_by_token[str(data["ego_pose_token"])]
        lidar_poses.append(lidar_to_global(data, calibration, ego_pose))

    mappings_by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    for mapping in manifest.get("mappings", []):
        if not bool(mapping.get("is_key_frame", False)):
            continue
        mappings_by_frame.setdefault(int(mapping["frame_index"]), {})[
            str(mapping["camera"])
        ] = mapping

    first_timestamp = float(sample_chain[0]["timestamp"])
    positions = np.asarray([pose[:3, 3] for pose in lidar_poses])
    timestamps = np.asarray([float(sample["timestamp"]) / 1e6 for sample in sample_chain])
    speeds = np.linalg.norm(np.gradient(positions, timestamps, axis=0, edge_order=2), axis=-1)
    frames: list[dict[str, Any]] = []
    for frame_index in sorted(mappings_by_frame):
        camera_mappings = mappings_by_frame[frame_index]
        if set(camera_mappings) != set(NUSCENES_CAMERAS):
            missing = set(NUSCENES_CAMERAS) - set(camera_mappings)
            raise ValueError(f"frame {frame_index} lacks cameras {sorted(missing)}")
        sample_tokens = {str(mapping["sample_token"]) for mapping in camera_mappings.values()}
        if len(sample_tokens) != 1:
            raise ValueError(f"frame {frame_index} maps to multiple sample tokens")
        sample_token = sample_tokens.pop()
        chain_index = sample_index[sample_token]
        sample = sample_chain[chain_index]
        lidar_data = lidar_by_sample[sample_token]
        lidar_calibration = calibration_by_token[
            str(lidar_data["calibrated_sensor_token"])
        ]
        ego_pose = ego_pose_by_token[str(lidar_data["ego_pose_token"])]
        trajectory, trajectory_mask = future_lidar_trajectory(
            lidar_poses, chain_index, planning_steps
        )
        vad_history = past_lidar_offsets(lidar_poses, chain_index, steps=2)
        command = navigation_command(trajectory, trajectory_mask)
        can_bus = np.zeros(18, dtype=np.float64)
        can_bus[:3] = np.asarray(ego_pose["translation"], dtype=np.float64)
        can_bus[3:7] = np.asarray(ego_pose["rotation"], dtype=np.float64)
        can_bus[13] = float(speeds[chain_index])
        ego_rotation = quaternion_matrix(ego_pose["rotation"])
        yaw = math.atan2(ego_rotation[1, 0], ego_rotation[0, 0])
        if yaw < 0.0:
            yaw += 2.0 * math.pi
        can_bus[-2] = yaw
        can_bus[-1] = math.degrees(yaw)

        cam_params: dict[str, Any] = {}
        image_paths: dict[str, dict[str, str]] = {}
        for camera in NUSCENES_CAMERAS:
            mapping = camera_mappings[camera]
            camera_data = data_by_token[str(mapping["sample_data_token"])]
            camera_calibration = calibration_by_token[
                str(camera_data["calibrated_sensor_token"])
            ]
            cam_params[camera] = camera_parameters(
                camera_data,
                camera_calibration,
                ego_pose_by_token[str(camera_data["ego_pose_token"])],
                lidar_calibration,
                ego_pose,
                output_width=int(mapping["output_width"]),
                output_height=int(mapping["output_height"]),
            )
            image_paths[camera] = {
                "native": str((raw_root / str(mapping["source_path"])).resolve()),
                "hugsim": str(mapping["output_path"]),
            }

        lidar_pose = lidar_poses[chain_index]
        if official_info_by_token is None:
            planner_state = {
                "scene_token": str(scene["token"]),
                "l2g_t": lidar_pose[:3, 3].tolist(),
                # UniAD stores row-vector local-to-global rotation:
                # l2e_R.T @ e2g_R.T == (e2g_R @ l2e_R).T.
                "l2g_r_mat": lidar_pose[:3, :3].T.tolist(),
                "can_bus": can_bus.tolist(),
                "vad_ego_his_trajs": vad_history.tolist(),
            }
        else:
            if sample_token not in official_info_by_token:
                raise ValueError(
                    f"official UniAD info lacks protocol sample {sample_token}"
                )
            official_info = official_info_by_token[sample_token]
            if int(official_info["timestamp"]) != int(sample["timestamp"]):
                raise ValueError(f"timestamp mismatch for sample {sample_token}")
            planner_state = official_uniad_planner_state(official_info)
            planner_state["vad_ego_his_trajs"] = vad_history.tolist()
        if official_vad_by_token is not None:
            if sample_token not in official_vad_by_token:
                raise ValueError(f"official VAD info lacks protocol sample {sample_token}")
            vad_info = official_vad_by_token[sample_token]
            official_history = np.asarray(
                vad_info["gt_ego_his_trajs"], dtype=np.float64
            )
            if not np.allclose(vad_history, official_history, rtol=0.0, atol=1e-5):
                maximum = float(np.max(np.abs(vad_history - official_history)))
                raise ValueError(
                    f"derived VAD history mismatch for {sample_token}: {maximum} m"
                )
            vad_command = int(np.argmax(np.asarray(vad_info["gt_ego_fut_cmd"])))
            if vad_command != command:
                raise ValueError(
                    f"VAD command mismatch for {sample_token}: {vad_command} != {command}"
                )
            planner_state["vad_ego_his_trajs"] = official_history.tolist()
            planner_state["vad_ego_fut_cmd"] = np.asarray(
                vad_info["gt_ego_fut_cmd"], dtype=np.float64
            ).tolist()
            planner_state["vad_ego_lcf_feat"] = np.asarray(
                vad_info["gt_ego_lcf_feat"], dtype=np.float64
            ).tolist()
        frames.append(
            {
                "frame_index": frame_index,
                "sample_token": sample_token,
                "sample_timestamp_us": int(sample["timestamp"]),
                "timestamp_seconds": (float(sample["timestamp"]) - first_timestamp) / 1e6,
                "image_paths": image_paths,
                "planner_info": {
                    "timestamp": float(sample["timestamp"]) / 1e6,
                    "command": command,
                    "ego_velo": float(speeds[chain_index]),
                    "ego_steer": 0.0,
                    "accelerate": 0.0,
                    "steer_rate": 0.0,
                    "cam_params": cam_params,
                    "planner_state": planner_state,
                },
                "gt_traj": trajectory.tolist(),
                "gt_mask": trajectory_mask.tolist(),
            }
        )

    return {
        "schema_version": 1,
        "benchmark": "decouplegs_nuscenes_open_loop_protocol",
        "scene": scene_name,
        "scene_token": str(scene["token"]),
        "raw_root": str(raw_root.resolve()),
        "frame_protocol": "official_nuscenes_2hz_keyframes",
        "state_source": (
            "nuScenes top-LiDAR calibrated_sensor + ego_pose tables"
            if official_info_by_token is None
            else "official UniAD temporal info (pose, calibration, lidar2img)"
        ),
        "can_bus_source": (
            "H-DATA-06 pose finite differences; official CAN bus pending"
            if official_info_by_token is None
            else "official UniAD temporal info + dataset pose/yaw overwrite"
        ),
        "vad_history_source": (
            "derived from raw nuScenes top-LiDAR sample chain"
            if official_vad_by_token is None
            else "official VAD temporal info (derived trajectory validated within 1e-5 m)"
        ),
        "planning_steps": planning_steps,
        "planning_timestep_seconds": 0.5,
        "frames": frames,
    }
