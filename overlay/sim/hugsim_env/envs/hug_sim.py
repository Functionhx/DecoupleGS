import torch
import numpy as np
from copy import deepcopy
import gymnasium
from gymnasium import spaces
from copy import deepcopy
from sim.utils.sim_utils import create_cam, rt2pose, pose2rt, load_camera_cfg, dense_cam_poses
from scipy.spatial.transform import Rotation as SCR
from sim.utils.score_calculator import (
    create_rectangle,
    bg_collision_det,
    bg_collision_point_count,
)
import os
import pickle
from sim.utils.plan import planner, UnifiedMap
from omegaconf import OmegaConf
import math
import time
from gaussian_renderer import GaussianModel
from scene.obj_model import ObjModel
from gaussian_renderer import apply_appearance_affine, render
import open3d as o3d
from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.registration import RegistrationConfig
from decouplegs.relighting import RelightingConfig
from decouplegs.runtime import AssetLibrary, DecoupleRuntime
from decouplegs.sensor_effects import SensorLightingConfig, apply_sensor_lighting


def fg_collision_det(ego_box, objs):
    ego_x, ego_y, _, ego_w, ego_l, ego_h, ego_yaw = ego_box
    ego_poly = create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)
    for obs in objs:
        obs_x, obs_y, _, obs_w, obs_l, _, obs_yaw = obs
        obs_poly = create_rectangle(
            obs_x, obs_y, obs_w, obs_l, obs_yaw)
        if ego_poly.intersects(obs_poly):
            return True
    return False

class HUGSimEnv(gymnasium.Env):
    def __init__(self, cfg, output):
        super().__init__()
        
        decouple_cfg = cfg.get("decouplegs", None)
        decouple_enabled = decouple_cfg is not None and bool(decouple_cfg.get("enabled", False))
        compact_filename = (
            str(decouple_cfg.get("compact_filename", "decouplegs.dgs"))
            if decouple_enabled
            else "decouplegs.dgs"
        )
        plan_list = cfg.scenario.plan_list
        for control_param in plan_list:
            control_param[5] = os.path.join(cfg.base.realcar_path, control_param[5])

        # read ground infos
        with open(os.path.join(cfg.model_path, 'ground_param.pkl'), 'rb') as f:
            #numpy.ndarray, float, list
            cam_poses, cam_heights, commands = pickle.load(f)
            cam_poses, commands = dense_cam_poses(cam_poses, commands)
            self.ground_model = (cam_poses, cam_heights, commands)

        if cfg.scenario.load_HD_map:
            unified_map = UnifiedMap(cfg.base.HD_map.path, cfg.base.HD_map.version, cfg.scenario.scene_name)
        else:
            unified_map = None
        
        self.kinematic = OmegaConf.to_container(cfg.kinematic)
        self.kinematic['min_steer'] = -math.radians(cfg.kinematic.min_steer)
        self.kinematic['max_steer'] = math.radians(cfg.kinematic.max_steer)
        self.kinematic['start_vr']= np.array(cfg.scenario.start_euler) / 180 * np.pi
        self.kinematic['start_vab'] = np.array(cfg.scenario.start_ab)
        self.kinematic['start_velo'] = cfg.scenario.start_velo
        self.kinematic['start_steer'] = cfg.scenario.start_steer

        self.gaussians = GaussianModel(cfg.model.sh_degree, affine=cfg.affine)

        """
        plan_list: a, b, height, yaw, v, model_path, controller, params
        Yaw is based on ego car's orientation. 0 means same direction as ego. 
        Right is positive and left is negative.
        """
        self.planner = planner(plan_list, scene_path=cfg.model_path, unified_map=unified_map, ground=self.ground_model, dt=cfg.kinematic.dt)
        
        (model_params, iteration) = torch.load(os.path.join(cfg.model_path, "scene.pth"), weights_only=False)
        self.gaussians.restore(model_params, None)
        
        dynamic_gaussians = {}
        for plan_id in self.planner.ckpts.keys():
            checkpoint = self.planner.ckpts[plan_id]
            compact_path = os.path.join(os.path.dirname(checkpoint), compact_filename)
            if decouple_enabled and os.path.isfile(compact_path):
                continue
            dynamic_gaussians[plan_id] = ObjModel(cfg.model.sh_degree, feat_mutable=False)
            (model_params, iteration) = torch.load(checkpoint, weights_only=False)
            model_params = list(model_params)
            dynamic_gaussians[plan_id].restore(model_params, None)
            
        semantic_idx = torch.argmax(self.gaussians.get_full_3D_features, dim=-1, keepdim=True)
        ground_xyz = self.gaussians.get_full_xyz[(semantic_idx == 0)[:, 0]].detach().cpu().numpy()
        scene_xyz = self.gaussians.get_full_xyz[((semantic_idx > 1) & (semantic_idx != 10))[:, 0]].detach().cpu().numpy()
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'ground.ply'), ground_pcd)
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_xyz.astype(float))
        o3d.io.write_point_cloud(os.path.join(output, 'scene.ply'), scene_pcd)

        if cfg.scenario.load_HD_map:
            self.planner.update_agent_route()
        
        self.cam_params, cam_align, self.cam_rect = load_camera_cfg(cfg.camera)
        
        self.ego_verts = np.array([[0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1.0,  0.5], [0.5, 1.0, -0.5],
                    [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1.0, -0.5], [-0.5, 1.0, 0.5]])
        self.whl = np.array([1.6, 1.5, 3.0])
        self.ego_verts *= self.whl
        # H-PHYSICS-02: this reconstruction uses camera-style world Y (down),
        # while ego_box converts height to -worldY.  The released environment
        # extended the collision prism along +worldY and therefore counted
        # buried terrain Gaussians as obstacles.  Match the already-published
        # ego_box convention and extend the physical vehicle above ground.
        self.ego_verts[:, 1] *= -1.0
        self.data_type = cfg.data_type
        stress_cfg = cfg.scenario.get("decouplegs_stress", None)
        lighting_cfg = None if stress_cfg is None else stress_cfg.get("lighting", None)
        self.sensor_lighting = (
            None
            if lighting_cfg is None
            else SensorLightingConfig(
                kind=str(lighting_cfg.get("kind", "identity")),
                exposure=float(lighting_cfg.get("exposure", 1.0)),
                gamma=float(lighting_cfg.get("gamma", 1.0)),
                glare_strength=float(lighting_cfg.get("glare_strength", 0.0)),
                vignette_strength=float(lighting_cfg.get("vignette_strength", 0.0)),
            )
        )

        self.action_space = spaces.Dict(
            {
                "steer_rate": spaces.Box(self.kinematic['min_steer'], self.kinematic['max_steer'], dtype=float),
                "acc": spaces.Box(self.kinematic['min_acc'], self.kinematic['max_acc'], dtype=float)
            }
        )
        self.observation_space = spaces.Dict(
            {
                'rgb': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=255, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W'], 3), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'semantic': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=50, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.uint8
                    ) for cam_name, params in self.cam_params.items()
                }),
                'depth': spaces.Dict({
                    cam_name: spaces.Box(
                        low=0, high=1000, 
                        shape=(params['intrinsic']['H'], params['intrinsic']['W']), dtype=np.float32
                    ) for cam_name, params in self.cam_params.items()
                }),
            }
        )
        self.fric = self.kinematic['fric']

        self.start_vr = self.kinematic['start_vr']
        self.start_vab = self.kinematic['start_vab']
        self.start_velo = self.kinematic['start_velo']
        self.start_steer = self.kinematic['start_steer']
        self.vr = deepcopy(self.kinematic['start_vr'])
        self.vab = deepcopy(self.kinematic['start_vab'])
        self.velo = deepcopy(self.kinematic['start_velo'])
        self.steer = deepcopy(self.kinematic['start_steer'])
        self.dt = self.kinematic['dt']

        bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
        self.render_fn = render
        self.render_kwargs = {
            "pc": self.gaussians,
            "bg_color": torch.tensor(bg_color, dtype=torch.float32, device="cuda"),
            "dynamic_gaussians": dynamic_gaussians,
            "unicycles": {} # dummy input, unicycle planner is used for unicycle models
        }
        self.decouple_batch_cameras = False
        if decouple_enabled:
            max_vertical_distance = decouple_cfg.get("max_vertical_distance", None)
            probe_radius = decouple_cfg.get("probe_radius", 9.0)
            shadow_mask_epsilon = decouple_cfg.get("shadow_mask_epsilon", None)
            registration_cfg = RegistrationConfig(
                heading_weight=float(decouple_cfg.get("heading_weight", 2.5)),
                map_resolution=float(decouple_cfg.get("map_resolution", 0.1)),
                column_radius=float(decouple_cfg.get("column_radius", 0.35)),
                opacity_threshold=float(decouple_cfg.get("ground_opacity_threshold", 0.0)),
                max_vertical_distance=(
                    None if max_vertical_distance is None else float(max_vertical_distance)
                ),
                # HUGSIM uses X/Z as the road plane and physical up is -Y;
                # 3DRealCar assets likewise extend upward along local -Y.
                vertical_axis=1,
                vertical_sign=int(decouple_cfg.get("world_up_sign", -1)),
                horizontal_axes=(0, 2),
                # HUGSIM/3DRealCar canonical vehicles are length-aligned to X.
                forward_axis=0,
                up_axis=1,
                up_sign=int(decouple_cfg.get("asset_up_sign", -1)),
            )
            relighting_cfg = RelightingConfig(
                probe_sigma=float(decouple_cfg.get("probe_sigma", 3.0)),
                probe_radius=None if probe_radius is None else float(probe_radius),
                shadow_strength=float(decouple_cfg.get("shadow_strength", 0.55)),
                shadow_exponent=float(decouple_cfg.get("shadow_exponent", 4.0)),
                shadow_decay=float(decouple_cfg.get("shadow_decay", 2.0)),
                shadow_ground_band=float(decouple_cfg.get("shadow_ground_band", 0.25)),
                shadow_mask_epsilon=(
                    None
                    if shadow_mask_epsilon is None
                    else float(shadow_mask_epsilon)
                ),
                adaptive_strength=float(decouple_cfg.get("adaptive_strength", 0.65)),
                adaptive_reference_intensity=float(
                    decouple_cfg.get("adaptive_reference_intensity", 0.5)
                ),
                adaptive_min_gain=float(decouple_cfg.get("adaptive_min_gain", 0.25)),
                adaptive_max_gain=float(decouple_cfg.get("adaptive_max_gain", 2.0)),
            )
            runtime = DecoupleRuntime(
                AssetLibrary(),
                registration_config=registration_cfg,
                relighting_config=relighting_cfg,
                rotate_sh=bool(decouple_cfg.get("rotate_sh", True)),
                relighting=bool(decouple_cfg.get("relighting", True)),
                adaptive_relighting=bool(decouple_cfg.get("adaptive_relighting", True)),
                contact_shadows=bool(decouple_cfg.get("contact_shadows", True)),
                opacity_grounding=bool(decouple_cfg.get("opacity_grounding", True)),
                semantic_grounding=bool(decouple_cfg.get("semantic_grounding", True)),
                frustum_culling=bool(decouple_cfg.get("frustum_culling", True)),
            )
            background_visibility = None
            visibility_path = decouple_cfg.get("background_visibility_path", None)
            if visibility_path is not None:
                visibility_path = os.path.expanduser(str(visibility_path))
                if not os.path.isabs(visibility_path):
                    visibility_path = os.path.join(cfg.model_path, visibility_path)
                visibility_state = torch.load(visibility_path, map_location="cpu", weights_only=True)
                if isinstance(visibility_state, dict):
                    visibility_state = visibility_state.get("visibility")
                if not isinstance(visibility_state, torch.Tensor):
                    raise TypeError(
                        "background_visibility_path must contain a tensor or {'visibility': tensor}"
                    )
                background_visibility = visibility_state
            bridge = HUGSIMRuntimeBridge(
                runtime,
                background_visibility=background_visibility,
                compact_asset_batch_size=int(
                    decouple_cfg.get("compact_asset_batch_size", 8)
                ),
                compact_radius_clip=float(decouple_cfg.get("compact_radius_clip", 0.0)),
                compact_lod_radius_clip=(
                    None
                    if decouple_cfg.get("compact_lod_radius_clip", None) is None
                    else float(decouple_cfg.get("compact_lod_radius_clip"))
                ),
                compact_lod_start_distance=float(
                    decouple_cfg.get("compact_lod_start_distance", 20.0)
                ),
                static_background_cache=bool(
                    decouple_cfg.get("static_background_cache", True)
                ),
                static_background_cache_entries=int(
                    decouple_cfg.get("static_background_cache_entries", 1)
                ),
                static_background_cache_min_observations=int(
                    decouple_cfg.get("static_background_cache_min_observations", 2)
                ),
                incremental_merge_max_dynamic_ratio=float(
                    decouple_cfg.get("incremental_merge_max_dynamic_ratio", 1.25)
                ),
            )
            calibration_filename = str(decouple_cfg.get("calibration_filename", "relighting.pt"))
            for plan_id, checkpoint in self.planner.ckpts.items():
                asset_dir = os.path.dirname(checkpoint)
                compact_path = os.path.join(asset_dir, compact_filename)
                calibration_path = os.path.join(asset_dir, calibration_filename)
                if os.path.isfile(compact_path):
                    bridge.register_compact_asset(
                        plan_id,
                        compact_path,
                        calibration_path if os.path.isfile(calibration_path) else None,
                    )
                else:
                    bridge.register_model(
                        plan_id,
                        dynamic_gaussians[plan_id],
                        calibration_path if os.path.isfile(calibration_path) else None,
                    )
            self.render_kwargs["decouple_bridge"] = bridge
            self.render_kwargs["rgb_only"] = bool(
                decouple_cfg.get("rgb_only_sensor", True)
            )
            self.decouple_batch_cameras = bool(
                decouple_cfg.get("batch_camera_sensor", True)
            )
        gaussians = self.gaussians
        semantic_idx = torch.argmax(gaussians.get_3D_features, dim=-1, keepdim=True)
        opacities = gaussians.get_opacity[:, 0]
        mask = ((semantic_idx > 1) & (semantic_idx != 10))[:, 0] & (opacities > 0.8)
        self.points = gaussians.get_xyz[mask]

        self.last_accel = 0
        self.last_steer_rate = 0
        self._last_requested_action = {"acc": 0.0, "steer_rate": 0.0}
        self._last_applied_action = {"acc": 0.0, "steer_rate": 0.0}

        self.timestamp = 0
        self._last_runtime_timing_ms = {}
        self._last_bg_collision_point_count = 0
        self._previous_obj_centers = {}

        # H-RC-01: the released HUGSIM code estimates route completion from
        # the nearest camera *index*.  That is sensitive to non-uniform camera
        # spacing, whereas the paper defines RC from travelled/planned metric
        # distance.  Keep the legacy value for compatibility and expose an
        # arc-length projection alongside it for the paper-facing evaluator.
        cam_poses = self.ground_model[0]
        self._route_xy = np.stack(
            [cam_poses[:, 2, 3], -cam_poses[:, 0, 3]], axis=-1
        ).astype(np.float64)
        route_delta = np.diff(self._route_xy, axis=0)
        self._route_segment_lengths = np.linalg.norm(route_delta, axis=-1)
        self._route_cumulative = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self._route_segment_lengths)]
        )
        self._route_commands = np.asarray(self.ground_model[2])
        self.navigation_command_lookahead_m = float(
            0.0
            if decouple_cfg is None
            else decouple_cfg.get("navigation_command_lookahead_m", 0.0)
        )
        if self.navigation_command_lookahead_m < 0.0:
            raise ValueError("navigation_command_lookahead_m must be non-negative")
        self._route_start_arc = 0.0
        self._route_max_arc = 0.0
    
    def ground_height(self, u, v):
        cam_poses, cam_height, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - u)**2 + (cam_poses[:, 2, 3] - v)**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        nearest_c2w = cam_poses[nearest_cam_idx]

        nearest_w2c = np.linalg.inv(nearest_c2w)
        uhv_local = nearest_w2c[:3, :3] @ np.array([u, 0, v]) + nearest_w2c[:3, 3]
        uhv_local[1] = 0
        uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
        
        return uhv_world[1]
    
    @property
    def route_completion(self):
        cam_poses, _, _ = self.ground_model
        cam_dist = np.sqrt(
            (cam_poses[:, 0, 3] - self.vab[0])**2 + (cam_poses[:, 2, 3] - self.vab[1])**2
        )
        nearest_cam_idx = np.argmin(cam_dist, axis=0)
        return (nearest_cam_idx + 1) / (cam_poses.shape[0] * 0.9), cam_dist[nearest_cam_idx]

    def _project_route_arc(self, xy):
        """Project an IMU-frame XY point to the reconstructed route polyline."""

        if len(self._route_xy) < 2:
            return 0.0, float("inf")
        point = np.asarray(xy, dtype=np.float64)
        starts = self._route_xy[:-1]
        delta = self._route_xy[1:] - starts
        denom = np.maximum(self._route_segment_lengths**2, 1e-12)
        ratio = np.clip(np.sum((point - starts) * delta, axis=-1) / denom, 0.0, 1.0)
        projection = starts + ratio[:, None] * delta
        distance = np.linalg.norm(projection - point, axis=-1)
        index = int(np.argmin(distance))
        arc = self._route_cumulative[index] + ratio[index] * self._route_segment_lengths[index]
        return float(arc), float(distance[index])

    @property
    def metric_route_completion(self):
        """Distance-based RC matching the equation in the supplement."""

        current_arc, route_distance = self._project_route_arc(
            np.asarray(self.ego_box[:2], dtype=np.float64)
        )
        self._route_max_arc = max(self._route_max_arc, current_arc)
        travelled = max(0.0, self._route_max_arc - self._route_start_arc)
        planned = max(
            float(self._route_cumulative[-1] - self._route_start_arc),
            1e-9,
        )
        return min(travelled / planned, 1.0), route_distance, travelled, planned
        

    @property
    def vt(self):
        vt = np.zeros(3)
        vt[[0, 2]] = self.vab
        vt[1] = self.ground_height(self.vab[0], self.vab[1])
        return vt
    
    @property
    def ego(self):
        return rt2pose(self.vr, self.vt)
    
    @property
    def ego_state(self):
        return torch.tensor([self.vab[0], self.vab[1], self.vr[1], self.velo])
    
    @property
    def ego_box(self):
        return [self.vt[2], -self.vt[0], -self.vt[1], self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]

    @property
    def objs_list(self):
        obj_boxes = []
        objs = self.render_kwargs['planning'][0]
        for obj_id, obj_b2w in objs.items():
            yaw = SCR.from_matrix(obj_b2w[:3, :3].detach().cpu().numpy()).as_euler('YXZ')[0]
            # X, Y, Z in IMU, w, l, h
            wlh = self.planner.wlhs[obj_id]
            obj_boxes.append([obj_b2w[2, 3].item(), -obj_b2w[0, 3].item(), -obj_b2w[1, 3].item(), wlh[0], wlh[1], wlh[2], -yaw-0.5*np.pi])
        return obj_boxes

    def _get_obs(self):
        rgbs, semantics, depths = {}, {}, {}
        v2front = self.cam_params['CAM_FRONT']["v2c"]
        viewpoints = []
        for cam_name, params in self.cam_params.items():
            intrinsic, v2c = params['intrinsic'], params['v2c']
            c2front = v2front @ np.linalg.inv(v2c) @ self.cam_rect
            c2w = self.ego @ c2front
            viewpoint = create_cam(intrinsic, c2w)
            viewpoint.timestamp = self.timestamp
            viewpoints.append((cam_name, viewpoint))

        bridge = self.render_kwargs.get("decouple_bridge")
        if (
            self.decouple_batch_cameras
            and bridge is not None
            and self.render_kwargs.get("rgb_only", False)
            and not self.render_kwargs.get("dynamic_gaussians")
            and not self.render_kwargs.get("unicycles")
        ):
            planning = self.render_kwargs.get("planning", ({}, {}))
            transforms = dict(planning[0]) if len(planning) else {}
            with torch.no_grad():
                batch = bridge.render_compact_batch(
                    [viewpoint for _, viewpoint in viewpoints],
                    self.gaussians,
                    transforms,
                    self.render_kwargs["bg_color"],
                    auxiliary=False,
                )
            if batch is not None:
                for index, (cam_name, viewpoint) in enumerate(viewpoints):
                    rendered = apply_appearance_affine(
                        viewpoint, self.gaussians, batch["render"][index]
                    )
                    rgb = (
                        rendered.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    if self.sensor_lighting is not None:
                        rgb = apply_sensor_lighting(rgb, cam_name, self.sensor_lighting)
                    smt = np.zeros(rgb.shape[:2], dtype=np.uint8)
                    depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                    if (self.data_type == 'waymo' or self.data_type == 'kitti360') and 'BACK' in cam_name:
                        rgb = np.zeros_like(rgb)
                    rgbs[cam_name] = rgb
                    semantics[cam_name] = smt
                    depths[cam_name] = depth
                return {'rgb': rgbs, 'semantic': semantics, 'depth': depths}

        for cam_name, viewpoint in viewpoints:
            with torch.no_grad():
                render_pkg = self.render_fn(viewpoint=viewpoint, prev_viewpoint=None, **self.render_kwargs)
            rgb = (torch.permute(render_pkg['render'].clamp(0, 1), (1,2,0)).detach().cpu().numpy() * 255).astype(np.uint8)
            if self.sensor_lighting is not None:
                rgb = apply_sensor_lighting(rgb, cam_name, self.sensor_lighting)
            if render_pkg['feats'] is None:
                smt = np.zeros(rgb.shape[:2], dtype=np.uint8)
            else:
                smt = torch.argmax(render_pkg['feats'], dim=0).detach().cpu().numpy().astype(np.uint8)
            if render_pkg['depth'] is None:
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)
            else:
                depth = render_pkg['depth'][0].detach().cpu().numpy()
            if (self.data_type == 'waymo' or self.data_type == 'kitti360') and 'BACK' in cam_name:
                rgbs[cam_name] = np.zeros_like(rgb)
                semantics[cam_name] = np.zeros_like(smt)
                depths[cam_name] = np.zeros_like(depth)
            else:
                rgbs[cam_name] = rgb
                semantics[cam_name] = smt
                depths[cam_name] = depth

        return {
                'rgb': rgbs, 
                'semantic': semantics,
                'depth': depths,
                }
    
    def _get_info(self):
        wego_r, wego_t = pose2rt(self.ego)
        current_route_arc, _ = self._project_route_arc(
            np.asarray(self.ego_box[:2], dtype=np.float64)
        )
        command_route_arc = min(
            current_route_arc + self.navigation_command_lookahead_m,
            float(self._route_cumulative[-1]),
        )
        command_index = min(
            int(np.searchsorted(self._route_cumulative, command_route_arc, side='left')),
            len(self._route_commands) - 1,
        )
        command = self._route_commands[command_index]
        obj_ids = list(self.planner.stats)
        obj_boxes = self.objs_list
        obj_velocities = []
        current_obj_centers = {}
        for obj_id, obj_box in zip(obj_ids, obj_boxes):
            stat = self.planner.stats[obj_id]
            center = np.asarray(obj_box[:2], dtype=np.float64)
            current_obj_centers[obj_id] = center
            previous = self._previous_obj_centers.get(obj_id)
            if previous is None:
                yaw, speed = float(stat[3]), float(stat[4])
                velocity = [speed * math.cos(yaw), speed * math.sin(yaw)]
            else:
                velocity = ((center - previous) / self.dt).tolist()
            obj_velocities.append(velocity)
        self._previous_obj_centers = current_obj_centers

        obj_behavior_states = []
        interactive_engine = self.planner.interactive_engine
        if interactive_engine is None:
            obj_behavior_states = [None for _ in obj_ids]
        else:
            ego_lane, _, _ = interactive_engine.network.nearest_lane(self.vab)
            for obj_id in obj_ids:
                state = self.planner.interactive_states.get(obj_id)
                if state is None:
                    obj_behavior_states.append(None)
                    continue
                effective_lane = (
                    state.target_lane if state.target_lane is not None else state.lane_id
                )
                fraction = (
                    0.0
                    if state.target_lane is None
                    else float(
                        np.clip(
                            state.lane_change_elapsed
                            / max(state.lane_change_duration, 1e-9),
                            0.0,
                            1.0,
                        )
                    )
                )
                obj_behavior_states.append(
                    {
                        'vehicle_id': str(obj_id),
                        'lane_id': state.lane_id,
                        'source_lane': state.source_lane,
                        'target_lane': state.target_lane,
                        'effective_lane': effective_lane,
                        'ego_lane': ego_lane,
                        'targets_ego_lane': effective_lane == ego_lane,
                        'lane_change_fraction_linear': fraction,
                        'progress_m': state.progress,
                        'speed_mps': state.speed,
                        'acceleration_mps2': state.acceleration,
                    }
                )
        ego_velocity = [
            float(self.velo * math.cos(self.vr[1])),
            float(-self.velo * math.sin(self.vr[1])),
        ]
        rc_distance, route_lateral_error, distance_traveled, distance_planned = (
            self.metric_route_completion
        )
        return {
            'ego_pos'  : wego_t.tolist(),
            'ego_rot'  : wego_r.tolist(),
            'ego_velo' : self.velo,
            'ego_steer': self.steer,
            'accelerate': self.last_accel,
            'steer_rate': self.last_steer_rate,
            'timestamp': self.timestamp,
            'command': command,
            'navigation_command': {
                'hypothesis': 'H-NAV-01 current route-arc command; optional explicit lookahead ablation',
                'lookahead_m': self.navigation_command_lookahead_m,
                'current_route_arc_m': current_route_arc,
                'command_route_arc_m': command_route_arc,
                'command_index': command_index,
            },
            'ego_box': self.ego_box,
            'obj_boxes': obj_boxes,
            'obj_ids': obj_ids,
            'obj_velocities': obj_velocities,
            'obj_behavior_states': obj_behavior_states,
            'ego_velocity_xy': ego_velocity,
            'rc_distance': rc_distance,
            'route_lateral_error': route_lateral_error,
            'route_distance_traveled': distance_traveled,
            'route_distance_planned': distance_planned,
            'runtime_timing_ms': dict(self._last_runtime_timing_ms),
            'sensor_lighting': (
                None
                if self.sensor_lighting is None
                else {
                    key: getattr(self.sensor_lighting, key)
                    for key in self.sensor_lighting.__dataclass_fields__
                }
            ),
            'control_contract': {
                'hypothesis': 'H-PHYSICS-01 enforce declared Gym action bounds',
                'requested': dict(self._last_requested_action),
                'applied': dict(self._last_applied_action),
            },
            'collision_geometry': {
                'hypothesis': 'H-PHYSICS-02 collision height follows -worldY ego_box convention',
                'background_point_count': self._last_bg_collision_point_count,
                'background_point_threshold': 100,
            },
            'traffic_lifecycle': {
                'hypothesis': 'H-BEHAVIOR-03 retire agents at finite route endpoints',
                'retired_this_step': list(self.planner.last_retired_interactive_ids),
                'retired_total': sorted(str(value) for value in self.planner.retired_interactive_ids),
            },
            'cam_params': self.cam_params,
            # 'ego_verts': verts,
        }
    
    def reset(self, seed=None, options=None):
        self.vr = deepcopy(self.start_vr)
        self.vab = deepcopy(self.start_vab)
        self.velo = deepcopy(self.start_velo)
        self.steer = deepcopy(self.start_steer)
        self.last_accel = 0
        self.last_steer_rate = 0
        self._last_requested_action = {"acc": 0.0, "steer_rate": 0.0}
        self._last_applied_action = {"acc": 0.0, "steer_rate": 0.0}
        self._last_bg_collision_point_count = 0
        self._previous_obj_centers = {}
        self.timestamp = 0
        start_xy = np.asarray(self.ego_box[:2], dtype=np.float64)
        self._route_start_arc, _ = self._project_route_arc(start_xy)
        self._route_max_arc = self._route_start_arc

        if self.planner is not None:
            plan_started = time.perf_counter()
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)
            planner_ms = (time.perf_counter() - plan_started) * 1000.0

        render_started = time.perf_counter()
        observation = self._get_obs()
        render_ms = (time.perf_counter() - render_started) * 1000.0
        self._last_runtime_timing_ms = {
            'background_agents': planner_ms if self.planner is not None else 0.0,
            'sensor_render': render_ms,
        }
        info = self._get_info()
        legacy_rc, _ = self.route_completion
        info['rc'] = legacy_rc
        info['collision'] = False
        info['termination_reason'] = None

        return observation, info
    
    def step(self, action):
        self.timestamp += self.dt
        if self.planner is not None:
            plan_started = time.perf_counter()
            self.render_kwargs['planning'] = self.planner.plan_traj(self.timestamp, self.ego_state)
            planner_ms = (time.perf_counter() - plan_started) * 1000.0
        else:
            planner_ms = 0.0
        requested_steer_rate = float(action['steer_rate'])
        requested_acc = float(action['acc'])
        # H-PHYSICS-01: enforce the declared Gym action-space contract. The
        # upstream environment integrated raw iLQR outputs with wider bounds.
        steer_rate = float(np.clip(
            requested_steer_rate,
            self.kinematic['min_steer'],
            self.kinematic['max_steer'],
        ))
        acc = float(np.clip(
            requested_acc,
            self.kinematic['min_acc'],
            self.kinematic['max_acc'],
        ))
        self._last_requested_action = {
            "acc": requested_acc,
            "steer_rate": requested_steer_rate,
        }
        self._last_applied_action = {"acc": acc, "steer_rate": steer_rate}
        self.last_steer_rate, self.last_accel = steer_rate, acc
        L = self.kinematic['Lr'] + self.kinematic['Lf']
        # H-PHYSICS-04: the bicycle state represents forward road speed.
        # Braking at a stopped/terminal reference must not integrate through
        # zero and make the ego vehicle reverse without a reverse command.
        self.velo = max(0.0, self.velo + acc * self.dt)
        self.steer += steer_rate * self.dt
        theta = self.vr[1]
        # print(theta / np.pi * 180, self.steer / np.pi * 180)
        self.vab[0] = self.vab[0] + self.velo * np.sin(theta) * self.dt
        self.vab[1] = self.vab[1] + self.velo * np.cos(theta) * self.dt
        self.vr[1] = theta + self.velo * np.tan(self.steer) / L * self.dt

        terminated = False
        termination_reasons = []
        reward = 0
        verts = (self.ego[:3, :3] @ self.ego_verts.T).T + self.ego[:3, 3]
        verts = torch.from_numpy(verts.astype(np.float32)).cuda()
        
        self._last_bg_collision_point_count = bg_collision_point_count(self.points, verts)
        bg_collision = self._last_bg_collision_point_count > 100
        if bg_collision:
            terminated = True
            termination_reasons.append('background_collision')
            print('Collision with background')
            reward = -100

        fg_collision = fg_collision_det(self.ego_box, self.objs_list)
        if fg_collision:
            terminated = True
            termination_reasons.append('agent_collision')
            print('Collision with foreground')
            reward = -100

        legacy_rc, _ = self.route_completion
        rc_distance, route_lateral_error, _, _ = self.metric_route_completion
        if route_lateral_error > 10:
            terminated=True
            termination_reasons.append('off_route')
            print('Far from preset trajectory')
            reward = -50
            
        # H-RC-01: the paper defines completion from travelled/planned metric
        # distance.  The released HUGSIM camera-index score reaches one at
        # roughly 90% of the route and otherwise terminates valid episodes
        # prematurely.
        if rc_distance >= 0.99:
            terminated = True
            termination_reasons.append('route_complete')
            print('Complete')
            reward = 1000

        render_started = time.perf_counter()
        observation = self._get_obs()
        render_ms = (time.perf_counter() - render_started) * 1000.0
        self._last_runtime_timing_ms = {
            'background_agents': planner_ms,
            'sensor_render': render_ms,
        }
        info = self._get_info()
        info['rc'] = legacy_rc
        info['collision'] = bg_collision or fg_collision
        info['termination_reason'] = '+'.join(termination_reasons) if termination_reasons else None
        
        return observation, reward, terminated, False, info
