import numpy as np
import torch
from scipy.spatial.transform import Rotation as SCR
import roma
from collections import namedtuple
from sim.utils.agent_controller import constant_headaway
from sim.utils import agent_controller
from collections import defaultdict
from trajdata import AgentType, UnifiedDataset
from trajdata.maps import MapAPI
from trajdata.simulation import SimulationScene
from sim.utils.sim_utils import rt2pose, pose2rt
from sim.utils.agent_controller import IDM, AttackPlanner, ConstantPlanner, UnicyclePlanner
import os
import json
from decouplegs.behavior import (
    IDMParameters,
    InteractiveTrafficState,
    MOBILParameters,
    PolylineIDMMOBILEngine,
    PolylineLaneNetwork,
)

Model = namedtuple('Models', ['model_path', 'controller', 'controller_args'])


class planner:
    def __init__(self, plan_list, scene_path=None, dt=0.2, unified_map=None, ground=None):
        self.unified_map = unified_map
        self.ground = ground
        self.PREDICT_STEPS = 20
        self.NUM_NEIGHBORS = 3
        
        self.rectify_angle = 0
        if self.unified_map is not None:
            self.rectify_angle = self.unified_map.rectify_angle
        
        # plan_list: a, b, height, yaw, v, model_path, controller, controller_args: dict
        self.stats, self.route, self.controller, self.ckpts, self.wlhs = {}, {}, {}, {}, {}
        self.dt = dt
        self.ATTACK_FREQ = 3
        self.interactive_engine = None
        self.interactive_states = {}
        self.interactive_args = {}
        self.interactive_allow_lane_changes = True
        self.retired_interactive_ids = set()
        self.last_retired_interactive_ids = []
        for iid, args in enumerate(plan_list):
            if args[6] == "UnicyclePlanner":
                # self.ckpts[f"agent_{iid}"] = os.path.join(scene_path, "ckpts", f"dynamic_{args[7]}_chkpnt30000.pth")
                # self.wlhs[f'agent_{iid}'] = [2.0, 4.0, 1.5]
                self.ckpts[f'agent_{iid}'] = os.path.join(args[5], 'gs.pth')
                with open(os.path.join(args[5], 'wlh.json')) as f:
                    self.wlhs[f'agent_{iid}'] = json.load(f)
                uc_configs = args[7]
                self.controller[f"agent_{iid}"] = UnicyclePlanner(os.path.join(scene_path, f"unicycle_{uc_configs['uc_id']}.pth"), speed=uc_configs['speed'])
                a, b, v, pitchroll, yaw, h = self.controller[f"agent_{iid}"].uc_model.forward(0.0)
                self.stats[f'agent_{iid}'] = torch.tensor([a, b, args[2], yaw, v])
                self.route[f'agent_{iid}'] = None
            else:
                model = Model(*args[5:])
                self.stats[f'agent_{iid}'] = torch.tensor(args[:5])  # a, b, height, yaw, v
                self.stats[f'agent_{iid}'][3] += self.rectify_angle
                self.route[f'agent_{iid}'] = None
                self.ckpts[f'agent_{iid}'] = os.path.join(model.model_path, 'gs.pth')
                with open(os.path.join(model.model_path, 'wlh.json')) as f:
                    self.wlhs[f'agent_{iid}'] = json.load(f)
                if model.controller == "IDMMOBILPlanner":
                    self.controller[f'agent_{iid}'] = model.controller
                    self.interactive_args[f'agent_{iid}'] = dict(model.controller_args)
                else:
                    self.controller[f'agent_{iid}'] = getattr(agent_controller, model.controller)(**model.controller_args)
                if model.controller == "AttackPlanner":
                    self.ATTACK_FREQ = model.controller_args["ATTACK_FREQ"]

        if self.interactive_args:
            self._initialize_interactive_engine()

    def _initialize_interactive_engine(self):
        if self.ground is None:
            raise ValueError("IDMMOBILPlanner requires a reconstructed route")
        first_args = next(iter(self.interactive_args.values()))
        self.interactive_allow_lane_changes = bool(
            first_args.get("allow_lane_changes", True)
        )
        lane_offsets = first_args.get("lane_offsets", [-3.5, 0.0, 3.5])
        cam_poses = np.asarray(self.ground[0])
        centerline = np.stack([cam_poses[:, 0, 3], cam_poses[:, 2, 3]], axis=-1)
        network = PolylineLaneNetwork.from_centerline(centerline, lane_offsets)
        idm_fields = set(IDMParameters.__dataclass_fields__)
        mobil_fields = set(MOBILParameters.__dataclass_fields__)
        idm = IDMParameters(
            **{
                key: value
                for key, value in dict(first_args.get("idm", {})).items()
                if key in idm_fields
            }
        )
        mobil = MOBILParameters(
            **{
                key: value
                for key, value in dict(first_args.get("mobil", {})).items()
                if key in mobil_fields
            }
        )
        self.interactive_engine = PolylineIDMMOBILEngine(
            network,
            idm=idm,
            mobil=mobil,
            lane_change_duration=float(first_args.get("lane_change_duration", 3.0)),
            lane_change_cooldown=float(first_args.get("lane_change_cooldown", 2.0)),
            hard_brake_deceleration=float(first_args.get("hard_brake_deceleration", 6.0)),
        )
        lane_ids = list(network.points)
        default_lane = lane_ids[len(lane_ids) // 2]
        for agent_id, controller_args in self.interactive_args.items():
            lane_id = controller_args.get("lane_id", default_lane)
            if lane_id not in network.points:
                raise ValueError(f"unknown lane_id {lane_id!r} for {agent_id}")
            stat = self.stats[agent_id]
            if "progress" in controller_args:
                progress = float(controller_args["progress"])
            else:
                progress, _ = network.project(lane_id, stat[:2].numpy())
            self.interactive_states[agent_id] = InteractiveTrafficState(
                vehicle_id=agent_id,
                lane_id=lane_id,
                progress=progress,
                speed=float(stat[4]),
                length=float(self.wlhs[agent_id][1]),
                lane_change_duration=float(controller_args.get("lane_change_duration", 3.0)),
            )
            self._sync_interactive_stat(agent_id, self.interactive_states[agent_id])

    def _sync_interactive_stat(self, agent_id, state):
        pose = self.interactive_engine.pose(state)
        stat = self.stats[agent_id].clone()
        stat[0] = pose.position[0]
        stat[1] = pose.position[1]
        stat[3] = pose.yaw
        stat[4] = pose.speed
        self.stats[agent_id] = stat

    def _step_interactive_engine(self, t, ego_stats):
        if self.interactive_engine is None:
            return
        ego_lane, ego_progress, _ = self.interactive_engine.network.nearest_lane(
            ego_stats[:2].detach().cpu().numpy()
        )
        ego = InteractiveTrafficState(
            vehicle_id="ego",
            lane_id=ego_lane,
            progress=ego_progress,
            speed=float(ego_stats[3]),
            length=3.0,
        )
        hard_brake_ids = []
        for agent_id, controller_args in self.interactive_args.items():
            start = controller_args.get("hard_brake_start_seconds")
            if start is None:
                continue
            duration = float(controller_args.get("hard_brake_duration_seconds", 1.5))
            if float(start) <= float(t) < float(start) + duration:
                hard_brake_ids.append(agent_id)
        agent_ids = list(self.interactive_states)
        updated = self.interactive_engine.step(
            [self.interactive_states[agent_id] for agent_id in agent_ids],
            self.dt,
            ego=ego,
            hard_brake_ids=hard_brake_ids,
            allow_lane_changes=self.interactive_allow_lane_changes,
        )
        terminal_ids = set(self.interactive_engine.last_terminal_vehicle_ids)
        self.last_retired_interactive_ids = sorted(str(value) for value in terminal_ids)
        for agent_id, state in zip(agent_ids, updated):
            if agent_id in terminal_ids:
                self.retired_interactive_ids.add(agent_id)
                self.interactive_states.pop(agent_id, None)
                self.stats.pop(agent_id, None)
                continue
            self.interactive_states[agent_id] = state
            self._sync_interactive_stat(agent_id, state)

    def update_ground(self, ground):
        self.ground = ground

    def update_agent_route(self):
        assert self.unified_map is not None, "Map shouldn't be None to forecast agent path"
        for iid, stat in self.stats.items():
            path = self.unified_map.get_route(stat)
            if path is None:
                print("path not found at ", self.stats)
            if path is not None:
                self.route[iid] = torch.from_numpy(np.hstack([path[:, :2], path[:, -1:]]))

    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground
        cam_poses = torch.from_numpy(cam_poses)
        cam_dist = np.sqrt(
            (cam_poses[:-1, 0, 3] - u) ** 2 + (cam_poses[:-1, 2, 3] - v) ** 2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]
        
        nearest_w2c = np.linalg.inv(nearest_c2w)
        uv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uv_local[1] = 0
        uv_world = nearest_c2w[:3, :3] @ uv_local + nearest_c2w[:3, 3]
        
        return uv_world[1] + cam_height

    def plan_traj(self, t, ego_stats):
        self._step_interactive_engine(t, ego_stats)
        all_stats = [ego_stats]
        for iid, stat in self.stats.items():
            all_stats.append(stat[[0, 1, 3, 4]])  # a, b, yaw, v
        all_stats = torch.stack(all_stats, dim=0)
        future_states = constant_headaway(all_stats, num_steps=self.PREDICT_STEPS, dt=self.dt)

        b2ws = {}
        for iid, stat in self.stats.items():
            # find closet neighbors
            curr_xy_agents = all_stats[:, :2]
            distance_agents = torch.norm(curr_xy_agents - stat[:2], dim=-1)
            neighbor_idx = torch.argsort(distance_agents)[1:self.NUM_NEIGHBORS + 1]
            neighbors = future_states[neighbor_idx]

            controller = self.controller[iid]
            if type(controller) is IDM:
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], path=self.route[iid], dt=self.dt,
                                              neighbors=neighbors)
            elif type(controller) is AttackPlanner:
                safe_neighbors = neighbors[1:, ...]
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], unified_map=self.unified_map, dt=0.1,
                                              neighbors=safe_neighbors, attacked_states=future_states[0],
                                              new_plan=((t // self.dt) % self.ATTACK_FREQ == 0))
            elif type(controller) is ConstantPlanner:
                next_xyrv = controller.update(state=stat[[0, 1, 3, 4]], dt=self.dt)
            elif type(controller) is UnicyclePlanner:
                next_xyrv = controller.update(dt=self.dt)
            elif controller == "IDMMOBILPlanner":
                next_xyrv = stat[[0, 1, 3, 4]]
            else:
                raise NotImplementedError
            next_stat = torch.zeros_like(stat)
            next_stat[[0, 1, 3, 4]] = next_xyrv.float()
            next_stat[2] = stat[2]
            self.stats[iid] = next_stat

            b2w = np.eye(4)
            h = self.ground_height(next_xyrv[0].numpy(), next_xyrv[1].numpy())
            if type(controller) is UnicyclePlanner:
                # b2w[:3, :3] = SCR.from_euler('xzy', [pitch_roll[0], pitch_roll[1], stat[3]]).as_matrix()
                b2w[:3, :3] = SCR.from_euler('y', [-stat[3]]).as_matrix()
                b2w[:3, 3] = np.array([next_stat[0], h + stat[2], next_stat[1]])
            else:
                b2w[:3, :3] = SCR.from_euler('y', [-stat[3] - np.pi / 2 - self.rectify_angle]).as_matrix()
                b2w[:3, 3] = np.array([next_stat[0], h + stat[2], next_stat[1]])
            b2ws[iid] = torch.tensor(b2w).float().cuda()

        return [b2ws, {}]


class UnifiedMap:
    def __init__(self, datapath, version, scene_name):
        self.datapath = datapath
        self.version = version

        self.dataset = UnifiedDataset(
            desired_data=[self.version],
            data_dirs={
                self.version: self.datapath,
            },
            cache_location="~/.unified_data_cache",
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 50.0),
            desired_dt=0.1,
            num_workers=4,
            verbose=True,
        )

        self.map_api = MapAPI(self.dataset.cache_path)

        self.scene = None
        for scene in list(self.dataset.scenes()):
            if scene.name == scene_name:
                self.scene = scene
        assert self.scene is not None, f"Can't find scene {scene_name}"
        self.vector_map = self.map_api.get_map(
            f"{self.version}:{self.scene.location}"
        )
        self.ego_start_pos, self.ego_start_yaw = self.get_start_pose()
        self.rectify_angle = 0
        if self.ego_start_yaw < 0:
            self.ego_start_yaw += np.pi
            self.rectify_angle = np.pi
        self.PATH_LENGTH = 100

    def get_start_pose(self):
        sim_scene: SimulationScene = SimulationScene(
            env_name=self.version,
            scene_name=f"sim_scene",
            scene=self.scene,
            dataset=self.dataset,
            init_timestep=0,
            freeze_agents=True,
        )
        obs = sim_scene.reset()
        assert obs.agent_name[0] == 'ego', 'The first agent is not ego'
        # We consider position of the first ego frame as origin
        # This suppose is ok when the first frame front camera pose is set as origin
        ego_start_pos = obs.curr_agent_state.position[0]
        ego_start_yaw = obs.curr_agent_state.heading[0]
        return ego_start_pos.numpy(), ego_start_yaw.item()

    def xyzr_local2world(self, stat):
        alpha = np.arctan(stat[0] / stat[1])
        beta = self.ego_start_yaw - alpha
        dist = np.linalg.norm(stat[:2])
        delta_x = dist * np.cos(beta)
        delta_y = dist * np.sin(beta)

        world_stat = np.zeros(4)
        world_stat[0] = delta_x + self.ego_start_pos[0]
        world_stat[1] = delta_y + self.ego_start_pos[1]
        world_stat[3] = stat[3] + self.ego_start_yaw

        return world_stat

    def batch_xyzr_world2local(self, stat):
        beta = np.arctan((stat[:, 1] - self.ego_start_pos[1]) / (stat[:, 0] - self.ego_start_pos[0]))
        alpha = self.ego_start_yaw - beta
        dist = np.linalg.norm(stat[:, :2] - self.ego_start_pos, axis=1)
        delta_x = dist * np.sin(alpha)
        delta_y = dist * np.cos(alpha)

        local_stat = np.zeros_like(stat)
        local_stat[:, 0] = delta_x
        local_stat[:, 1] = delta_y
        local_stat[:, 3] = stat[:, 3] - self.ego_start_yaw

        return local_stat

    def get_route(self, stat):
        # stat: a, b, height, yaw, v
        curr_xyzr = self.xyzr_local2world(stat[:4].numpy())
        
        # lanes = self.vector_map.get_current_lane(curr_xyzr, max_dist=5, max_heading_error=np.pi/3)
        lanes = self.vector_map.get_current_lane(curr_xyzr)

        if len(lanes) > 0:
            curr_lane = lanes[0]
            path = self.batch_xyzr_world2local(curr_lane.center.xyzh)
            total_path_length = np.linalg.norm(curr_lane.center.xy[1:] - curr_lane.center.xy[:-1], axis=1).sum()
            # random select next lanes until reach PATH_LENGTH
            while total_path_length < self.PATH_LENGTH:
                next_lanes = list(curr_lane.next_lanes)
                if len(next_lanes) == 0:
                    break
                next_lane = self.vector_map.get_road_lane(next_lanes[np.random.randint(len(next_lanes))])
                path = np.vstack([path, self.batch_xyzr_world2local(next_lane.center.xyzh)])
                total_path_length += np.linalg.norm(next_lane.center.xy[1:] - next_lane.center.xy[:-1], axis=1).sum()
                curr_lane = next_lane
        else:
            path = None
        return path
