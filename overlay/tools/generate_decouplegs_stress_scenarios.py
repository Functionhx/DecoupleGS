#!/usr/bin/env python3
"""Generate deterministic paper-scale DecoupleGS closed-loop stress scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


PROFILES = {
    "easy": {
        "agents": (1, 3),
        "progress": (14.0, 46.0),
        "speed": (4.0, 7.0),
        "allow_lane_changes": False,
        "idm": {
            "desired_speed": 8.0,
            "minimum_gap": 5.0,
            "time_headway": 2.5,
            "max_acceleration": 1.2,
            "comfortable_deceleration": 3.0,
        },
        "mobil": {"acceleration_threshold": 999.0, "politeness": 0.5},
        "lighting": {"kind": "clear_day", "exposure": 1.0, "gamma": 1.0},
    },
    "medium": {
        "agents": (4, 8),
        "progress": (10.0, 52.0),
        "speed": (3.0, 8.0),
        "allow_lane_changes": True,
        "idm": {
            "desired_speed": 10.0,
            "minimum_gap": 4.0,
            "time_headway": 1.8,
            "max_acceleration": 1.8,
            "comfortable_deceleration": 3.5,
        },
        "mobil": {"acceleration_threshold": 0.35, "politeness": 0.4},
        "lighting": {"kind": "mild_overcast", "exposure": 0.9, "gamma": 1.05},
    },
    "hard": {
        "agents": (9, 15),
        "progress": (6.0, 56.0),
        "speed": (2.0, 9.0),
        "allow_lane_changes": True,
        "idm": {
            "desired_speed": 11.0,
            "minimum_gap": 2.5,
            "time_headway": 1.15,
            "max_acceleration": 2.4,
            "comfortable_deceleration": 4.5,
        },
        "mobil": {"acceleration_threshold": 0.05, "politeness": 0.2},
        "lighting": {"kind": "dusk", "exposure": 0.62, "gamma": 1.15},
    },
    "extreme": {
        "agents": (16, 22),
        "progress": (4.0, 54.0),
        "speed": (1.0, 11.0),
        "allow_lane_changes": True,
        "idm": {
            "desired_speed": 13.0,
            "minimum_gap": 1.25,
            "time_headway": 0.7,
            "max_acceleration": 3.0,
            "comfortable_deceleration": 6.0,
        },
        "mobil": {
            "acceleration_threshold": -0.15,
            "politeness": 0.0,
            "safe_deceleration": 6.0,
        },
        "lighting": {
            "kind": "adversarial_low_light_glare",
            "exposure": 0.32,
            "gamma": 1.35,
            "glare_strength": 0.35,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Easy/Medium/Hard/Extreme 2 Hz, 20 s DecoupleGS scenarios"
    )
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-name", default="scene-0383")
    parser.add_argument(
        "--scene-names",
        nargs="+",
        help=(
            "Balanced multi-clip pool. When supplied, each difficulty's episodes are "
            "assigned round-robin across these scenes; --scene-name is ignored."
        ),
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260801761)
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        help="Optional YAML containing an assets list; otherwise all usable asset directories are used.",
    )
    return parser.parse_args()


def usable_assets(root: Path, manifest: Path | None) -> list[str]:
    if manifest is not None:
        value = yaml.safe_load(manifest.read_text()) or {}
        assets = list(value.get("assets", value if isinstance(value, list) else []))
    else:
        assets = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "gs.pth").is_file() and (path / "wlh.json").is_file()
        )
    if not assets:
        raise ValueError("no usable 3DRealCar assets found")
    missing = [asset for asset in assets if not (root / asset / "gs.pth").is_file()]
    if missing:
        raise FileNotFoundError(f"asset checkpoints are missing: {missing}")
    return assets


def agent_count(profile: dict, episode: int, episodes: int) -> int:
    minimum, maximum = profile["agents"]
    if episodes == 1:
        return maximum
    return int(round(minimum + (maximum - minimum) * episode / (episodes - 1)))


def generate_scenario(
    difficulty: str,
    episode: int,
    episodes: int,
    assets: list[str],
    scene_name: str,
    base_seed: int,
) -> dict:
    profile = PROFILES[difficulty]
    difficulty_index = list(PROFILES).index(difficulty)
    seed = base_seed + difficulty_index * 100_000 + episode
    rng = np.random.default_rng(seed)
    count = agent_count(profile, episode, episodes)
    lane_offsets = [-3.5, 0.0, 3.5]
    low, high = profile["progress"]

    # Stratification gives high density without initial overlap: each lane has
    # an independent ordered progression and a deterministic sub-metre jitter.
    lane_ids = np.arange(count) % len(lane_offsets)
    rng.shuffle(lane_ids)
    lane_progress = {}
    for lane_id in range(len(lane_offsets)):
        indices = np.flatnonzero(lane_ids == lane_id)
        # Easy episodes can contain fewer actors than lanes.  An empty lane
        # must contribute no progression samples; using max(N, 1) here made
        # strict zip correctly expose a phantom sample for those lanes.
        values = (
            np.linspace(low, high, len(indices) + 2)[1:-1]
            if len(indices)
            else np.empty(0, dtype=np.float64)
        )
        for index, progress in zip(indices, values, strict=True):
            lane_progress[int(index)] = float(progress + rng.uniform(-0.35, 0.35))

    plan_list = []
    for index in range(count):
        controller = {
            "lane_id": int(lane_ids[index]),
            "progress": lane_progress[index],
            "lane_offsets": lane_offsets,
            "allow_lane_changes": profile["allow_lane_changes"],
            "idm": profile["idm"],
            "mobil": profile["mobil"],
            "lane_change_duration": 2.4 if difficulty == "extreme" else 3.0,
            "lane_change_cooldown": 1.0 if difficulty == "extreme" else 2.0,
            "hard_brake_deceleration": 7.0 if difficulty == "extreme" else 6.0,
        }
        if difficulty == "extreme" and index == int(np.argmin(
            [abs(lane_progress[i] - 18.0) + (0 if lane_ids[i] == 1 else 10) for i in range(count)]
        )):
            controller.update(
                hard_brake_start_seconds=float(rng.uniform(5.0, 9.0)),
                hard_brake_duration_seconds=float(rng.uniform(1.0, 2.0)),
            )
        speed = float(rng.uniform(*profile["speed"]))
        asset = assets[(episode * 7 + index * 11 + difficulty_index * 3) % len(assets)]
        plan_list.append(
            [0.0, 0.0, -0.3, 0.0, speed, asset, "IDMMOBILPlanner", controller]
        )

    return {
        "mode": f"{difficulty}_{episode:02d}",
        "plan_list": plan_list,
        "load_HD_map": False,
        "start_euler": [0.0, 0.0, 0.0],
        "start_ab": [0.0, 0.0],
        "start_velo": 1.0,
        "start_steer": 0.0,
        "scene_name": scene_name,
        "iteration": 30000,
        "decouplegs_stress": {
            "schema_version": 1,
            "hypothesis": "H-SCENARIO-01 deterministic paper-scale scenario synthesis",
            "difficulty": difficulty,
            "seed": seed,
            "dynamic_agents": count,
            "traffic_radius_m": 50.0,
            "lighting": profile["lighting"],
            "behavior": "IDM+MOBIL with H-BEHAVIOR-01/02 execution details",
            "asset_unique_pool": len(assets),
        },
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    assets = usable_assets(args.asset_root, args.asset_manifest)
    scene_pool = list(dict.fromkeys(args.scene_names or [args.scene_name]))
    if not scene_pool or any(not name.startswith("scene-") for name in scene_pool):
        raise ValueError("scene names must use the scene-NNNN form")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "data_contract": {
            "hypothesis": "H-DATA-05 balanced deterministic clip assignment",
            "scene_pool": scene_pool,
            "scene_selection": "round_robin_with_difficulty_offset",
            "hugsim_revision": "38433e301bef99dca09f7bbd10f3145449fb66a2",
            "paper_scene_ids_disclosed": False,
        },
        "paper_protocol": {
            "episodes_per_difficulty": args.episodes,
            "episode_seconds": 20,
            "control_frequency_hz": 2,
            "difficulty_agent_ranges": {
                key: list(value["agents"]) for key, value in PROFILES.items()
            },
        },
        "hypotheses": [
            "H-SCENARIO-01 deterministic seeds and density stratification",
            "H-BEHAVIOR-01 smooth polyline IDM+MOBIL execution",
            "H-BEHAVIOR-02 synchronous non-overlap projection",
        ],
        "asset_pool": assets,
        "scenarios": [],
    }
    for difficulty_index, difficulty in enumerate(PROFILES):
        for episode in range(args.episodes):
            scene_name = scene_pool[
                (difficulty_index * args.episodes + episode) % len(scene_pool)
            ]
            scenario = generate_scenario(
                difficulty,
                episode,
                args.episodes,
                assets,
                scene_name,
                args.seed,
            )
            path = args.output / f"{scene_name}-{scenario['mode']}.yaml"
            path.write_text(yaml.safe_dump(scenario, sort_keys=False))
            manifest["scenarios"].append(
                {
                    "path": path.name,
                    "scene": scene_name,
                    "difficulty": difficulty,
                    "seed": scenario["decouplegs_stress"]["seed"],
                    "agents": scenario["decouplegs_stress"]["dynamic_agents"],
                }
            )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenarios": len(manifest["scenarios"]),
                "unique_assets": len(assets),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
