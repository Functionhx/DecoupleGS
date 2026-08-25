import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
import pickle
import json
import time
from sim.utils.launch_ad import fifo_receive, fifo_send, launch, check_alive, terminate
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
from decouplegs.trajectory_stabilizer import (
    RouteTrajectoryStabilizer,
    TrajectoryStabilizerConfig,
)


def to_video(observations, output_path):
    from moviepy import ImageSequenceClip

    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def create_gym_env(
    cfg,
    output,
    *,
    save_video=False,
    run_hugsim_score=True,
    max_steps_override=None,
    planner_process=None,
    ipc_timeout_seconds=120.0,
):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    reset_started = time.perf_counter()
    obs, info = env.reset()
    reset_wall_ms = (time.perf_counter() - reset_started) * 1000.0
    done = False
    cnt = 0
    max_steps = None
    decouple_cfg = cfg.get("decouplegs", None)
    stabilizer_cfg = TrajectoryStabilizerConfig.from_mapping(
        None if decouple_cfg is None else decouple_cfg.get("trajectory_stabilizer", None)
    )
    trajectory_stabilizer = RouteTrajectoryStabilizer(
        np.asarray(env.unwrapped._route_xy, dtype=np.float64),
        float(cfg.kinematic.dt),
        stabilizer_cfg,
    )
    if decouple_cfg is not None and bool(decouple_cfg.get("enabled", False)):
        max_episode_seconds = float(decouple_cfg.get("max_episode_seconds", 20.0))
        if max_episode_seconds <= 0:
            raise ValueError("decouplegs.max_episode_seconds must be positive")
        max_steps = max(1, int(np.ceil(max_episode_seconds / float(cfg.kinematic.dt))))
    if max_steps_override is not None:
        if max_steps_override <= 0:
            raise ValueError("max_steps_override must be positive")
        max_steps = int(max_steps_override)
    save_data = {
        'type': 'closeloop',
        'frames': [],
        'protocol': {
            'control_dt_seconds': float(cfg.kinematic.dt),
            'maximum_steps': max_steps,
            'maximum_episode_seconds': (
                None if max_steps is None else max_steps * float(cfg.kinematic.dt)
            ),
            'route_completion': {
                'paper_definition': 'distance_traveled / distance_planned',
                'legacy_hugsim_field': 'rc',
                'metric_field': 'rc_distance',
                'hypothesis': 'H-RC-01 polyline arc-length projection',
            },
            'reset_wall_ms': reset_wall_ms,
            'implementation_contracts': {
                'H-E2E-03': '3DGS RGB is converted to checkpoint-required Caffe BGR before normalization',
                'H-PHYSICS-01': 'declared acceleration and steering-rate action bounds are enforced',
                'H-PHYSICS-02': 'ego static-collision volume follows -worldY height convention',
                'H-PHYSICS-04': 'forward ego speed is clamped non-negative after acceleration integration',
                'H-BEHAVIOR-03': 'agents retire at finite route endpoints instead of teleporting through overlap correction',
                'H-RC-01': 'route completion and terminal threshold use metric polyline distance (RC >= 0.99)',
                'H-PLAN-01': trajectory_stabilizer.contract(),
                'H-NAV-01': {
                    'description': (
                        'strict baseline samples the command at the current route arc; '
                        'positive lookahead is an explicit R&D ablation'
                    ),
                    'lookahead_m': float(cfg.decouplegs.get('navigation_command_lookahead_m', 0.0)),
                },
                'H-CONTROL-01': 'iLQR reference yaw is atan2(right, forward) after axis swap',
            },
        },
    }

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    infos_save.append(info)
    termination_reason = None
    while not done:
        if save_video:
            observations_save.append(obs['rgb'])

        print('ego pose', info['ego_pos'])

        planner_started = time.perf_counter()
        try:
            fifo_send(
                obs_pipe,
                (obs, info),
                planner_process,
                ipc_timeout_seconds,
            )
            plan_traj = fifo_receive(
                plan_pipe,
                planner_process,
                ipc_timeout_seconds,
            )
        except RuntimeError as error:
            print(f"Planner IPC failure: {error}")
            termination_reason = 'planner_failure'
            done = True
            break
        planner_wall_ms = (time.perf_counter() - planner_started) * 1000.0

        if plan_traj is None:  # AD process crashed or returned no trajectory.
            termination_reason = 'planner_failure'
            done = True
            break

        plan_traj = np.asarray(plan_traj)
        if plan_traj.ndim != 2 or plan_traj.shape[0] == 0 or plan_traj.shape[1] < 2:
            raise ValueError(f"planner trajectory must be [N, >=2], got {plan_traj.shape}")

        # The plan belongs to the state sent to the AD process.  Preserve that
        # state for scoring before advancing the simulator.
        current_info = info
        raw_plan_traj = plan_traj.copy()
        plan_traj, stabilizer_diagnostics = trajectory_stabilizer.stabilize(
            raw_plan_traj,
            current_info,
        )
        imu_plan_traj = plan_traj[:, [1, 0]].copy()
        imu_plan_traj[:, 1] *= -1
        global_traj = traj_transform_to_global(imu_plan_traj, current_info['ego_box'])

        acc, steer_rate = traj2control(plan_traj, current_info)
        action = {'acc': acc, 'steer_rate': steer_rate}
        step_started = time.perf_counter()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        step_wall_ms = (time.perf_counter() - step_started) * 1000.0
        cnt += 1
        reached_limit = cnt > 400 if max_steps is None else cnt >= max_steps
        if reached_limit and not terminated and not truncated:
            termination_reason = 'time_limit'
        else:
            termination_reason = next_info.get('termination_reason')
            if truncated and termination_reason is None:
                termination_reason = 'environment_truncated'
        done = terminated or truncated or reached_limit

        save_data['frames'].append({
            'time_stamp': current_info['timestamp'],
            'is_key_frame': True,
            'ego_box': current_info['ego_box'],
            'ego_velocity_xy': current_info.get('ego_velocity_xy'),
            'obj_boxes': current_info['obj_boxes'],
            'obj_ids': current_info.get('obj_ids'),
            'obj_velocities': current_info.get('obj_velocities'),
            'obj_behavior_states': current_info.get('obj_behavior_states'),
            'obj_names': ['car' for _ in current_info['obj_boxes']],
            'command': current_info.get('command'),
            'navigation_command': current_info.get('navigation_command'),
            'planner_raw_traj_local': raw_plan_traj,
            'controller_traj_local': plan_traj,
            'trajectory_stabilizer': stabilizer_diagnostics,
            'planned_traj': {
                'traj': global_traj,
                'timestep': 0.5
            },
            'collision': bool(current_info.get('collision', False)),
            'rc': float(current_info.get('rc', 0.0)),
            'rc_distance': float(current_info.get('rc_distance', 0.0)),
            'route_distance_traveled': float(
                current_info.get('route_distance_traveled', 0.0)
            ),
            'route_distance_planned': float(
                current_info.get('route_distance_planned', 0.0)
            ),
            'timing_ms': {
                'planner_and_ipc_wall': planner_wall_ms,
                'environment_step_wall': step_wall_ms,
                **{
                    f"simulator_{key}": value
                    for key, value in next_info.get('runtime_timing_ms', {}).items()
                },
            },
            'transition': {
                'reward': float(reward),
                'collision': bool(next_info.get('collision', False)),
                'rc': float(next_info.get('rc', 0.0)),
                'rc_distance': float(next_info.get('rc_distance', 0.0)),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'termination_reason': termination_reason,
                'control_contract': next_info.get('control_contract'),
                'collision_geometry': next_info.get('collision_geometry'),
                'traffic_lifecycle': next_info.get('traffic_lifecycle'),
            },
        })
        obs, info = next_obs, next_info
        infos_save.append(info)

    save_data['episode'] = {
        'steps': cnt,
        'termination_reason': termination_reason,
        'final_info': info,
    }

    # A FIFO opened for writing blocks until a reader connects.  If the
    # planner has already failed/exited, sending the sentinel would deadlock
    # the simulator and hide the episode artifacts.
    if termination_reason != 'planner_failure':
        try:
            fifo_send(obs_pipe, 'Done', planner_process, ipc_timeout_seconds)
        except RuntimeError as error:
            print(f"Planner exited before Done sentinel: {error}")

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    if save_video and observations_save:
        to_video(observations_save, os.path.join(output, 'video.mp4'))
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    if run_hugsim_score and save_data['frames']:
        ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
        scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
        results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
        with open(os.path.join(output, 'eval.json'), 'w') as f:
            json.dump(results, f, default=_json_default, indent=2)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    parser.add_argument(
        '--decouple_config',
        default=None,
        help='Optional DecoupleGS runtime YAML (for example configs/decouplegs.yaml)',
    )
    parser.add_argument(
        '--decouple-option',
        action='append',
        default=[],
        help='Repeatable OmegaConf dot-list override applied after --decouple_config.',
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='Encode a six-camera MP4. Disabled by default for runtime benchmarks.',
    )
    parser.add_argument(
        '--skip_hugsim_score',
        action='store_true',
        help='Skip the legacy HUGSIM PDMS score (paper metrics are evaluated separately).',
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='Optional smoke-test episode limit; paper runs leave this unset.',
    )
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    if args.decouple_config is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.decouple_config))
    if args.decouple_option:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.decouple_option))
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    # Released checkpoints retain the author's absolute training path. The
    # selected base directory is authoritative for portable evaluation.
    cfg.model_path = model_path
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    else:
        raise NotImplementedError
    
    process = launch(ad_path, args.ad_cuda, output)
    try:
        create_gym_env(
            cfg,
            output,
            save_video=args.save_video,
            run_hugsim_score=not args.skip_hugsim_score,
            max_steps_override=args.max_steps,
            planner_process=process,
        )
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        terminate(process)
    finally:
        terminate(process)
    
    # # For debug
    # create_gym_env(cfg, output)
