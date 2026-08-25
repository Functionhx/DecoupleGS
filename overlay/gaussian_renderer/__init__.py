import torch
from scene.gaussian_model import GaussianModel
from scene.ground_model import GroundModel
from gsplat.rendering import rasterization
import roma
from scene.cameras import Camera
from torch import Tensor
from decouplegs.hugsim import HUGSIMRuntimeBridge
from decouplegs.registration import ground_asset_from_pose
from decouplegs.transforms import transform_gaussians

def euler2matrix(yaw):
    return torch.tensor([
        [torch.cos(-yaw), 0, torch.sin(-yaw)],
        [0, 1, 0],
        [-torch.sin(-yaw), 0, torch.cos(-yaw)]
    ]).cuda()

def cat_bgfg(bg, fg, only_xyz=False):
    if only_xyz:
        if bg.ground_model is None:
            bg_feats = [bg.get_xyz]
        else:
            bg_feats = [bg.get_full_xyz]
    else:
        if bg.ground_model is None:
            bg_feats = [bg.get_xyz, bg.get_opacity, bg.get_scaling, bg.get_rotation, bg.get_features, bg.get_3D_features]
        else:
            bg_feats = [bg.get_full_xyz, bg.get_full_opacity, bg.get_full_scaling, bg.get_full_rotation, bg.get_full_features, bg.get_full_3D_features]

    
    if len(fg) == 0:
        return bg_feats
    
    output = []
    for fg_feat, bg_feat in zip(fg, bg_feats):
        if fg_feat is None:
            output.append(bg_feat)
        else:
            if bg_feat.shape[1] != fg_feat.shape[1]:
                fg_feat = fg_feat[:, :bg_feat.shape[1], :]
            output.append(torch.cat((bg_feat, fg_feat), dim=0))
    
    return output

def concatenate_all(all_fg):
    output = []
    for feat in list(zip(*all_fg)):
        output.append(torch.cat(feat, dim=0))
    return output

def proj_uv(xyz, cam):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intr = torch.as_tensor(cam.K[:3, :3]).float().to(device)  # (3, 3)
    w2c = torch.linalg.inv(cam.c2w)[:3, :]  # (3, 4)

    c_xyz = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
    i_xyz = (intr @ c_xyz.mT).mT  # (N, 3)
    uv = i_xyz[:, [1,0]] / i_xyz[:, -1:].clip(1e-3) # (N, 2)
    return uv


def unicycle_b2w(timestamp, model):
    pred = model(timestamp)
    if pred is None:
        return None
    pred_a, pred_b, pred_v, pitchroll, pred_yaw, pred_h = pred
    rt = torch.eye(4).float().cuda()
    rt[:3,:3] = roma.euler_to_rotmat('xzy', [-pitchroll[0]+torch.pi/2, -pitchroll[1]+torch.pi/2, -pred_yaw+torch.pi/2])
    rt[1, 3], rt[0, 3], rt[2, 3] = pred_h, pred_a, pred_b
    return rt


def apply_appearance_affine(viewpoint: Camera, pc: GaussianModel, image: Tensor) -> Tensor:
    if not pc.affine:
        return image
    cam_xyz, cam_dir = viewpoint.c2w[:3, 3].cuda(), viewpoint.c2w[:3, 2].cuda()
    o_enc = pc.pos_enc(cam_xyz[None, :] / 60)
    d_enc = pc.dir_enc(cam_dir[None, :])
    appearance = pc.appearance_model(torch.cat([o_enc, d_enc], dim=1)) * 1e-1
    affine_weight = appearance[:, :9].view(3, 3) + torch.eye(3, device=appearance.device)
    affine_bias = appearance[:, -3:]
    colors = image.view(3, -1).permute(1, 0)
    return (colors @ affine_weight + affine_bias).clip(0, 1).permute(1, 0).view_as(image)


def render(viewpoint:Camera, prev_viewpoint:Camera, pc:GaussianModel, dynamic_gaussians:dict, 
            unicycles:dict, bg_color:Tensor, render_optical=False, planning=[],
            decouple_bridge:HUGSIMRuntimeBridge=None, rasterizer_packed=False,
            rgb_only=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    timestamp = viewpoint.timestamp

    all_fg = [None, None, None, None, None, None]
    prev_all_fg = [None]

    if unicycles is None or len(unicycles) == 0:
        track_dict = viewpoint.dynamics
        if prev_viewpoint is not None:
            prev_track_dict = prev_viewpoint.dynamics
    else:
        track_dict, prev_track_dict = {}, {}
        for track_id, B2W in viewpoint.dynamics.items():
            if track_id in unicycles:
                B2W = unicycle_b2w(timestamp, unicycles[track_id]['model'])
            track_dict[track_id] = B2W
            if prev_viewpoint is not None:
                prev_B2W = unicycle_b2w(prev_viewpoint.timestamp, unicycles[track_id]['model'])
                prev_track_dict[track_id] = prev_B2W
    if len(planning) > 0:
        for plan_id, B2W in planning[0].items():
            track_dict[plan_id] = B2W
        if prev_viewpoint is not None:
            for plan_id, B2W in planning[1].items():
                prev_track_dict[plan_id] = B2W

    if decouple_bridge is not None and not render_optical:
        compact = decouple_bridge.render_compact(
            viewpoint,
            pc,
            track_dict,
            bg_color,
            auxiliary=not rgb_only,
        )
        if compact is not None:
            rendered_image = apply_appearance_affine(viewpoint, pc, compact["render"])
            return {
                "render": rendered_image,
                "feats": compact["feats"],
                "depth": compact["depth"],
                "opticalflow": None,
                "alphas": compact["alphas"],
                "viewspace_points": compact["viewspace_points"],
                "info": compact["info"],
            }
    
    if decouple_bridge is not None:
        composed = decouple_bridge.compose(viewpoint, pc, dynamic_gaussians, track_dict)
        merged = composed.merged
        xyz = merged.means
        opacities = merged.opacities[:, None]
        scales = merged.scales
        rotations = merged.quats
        shs = merged.sh
        if merged.semantics is None:
            raise ValueError("HUGSIM unified rendering requires semantic features")
        feats3D = merged.semantics
        if render_optical and prev_viewpoint is not None:
            previous_parts = [composed.background.means]
            for track_id, current_asset in composed.assets.items():
                if track_id not in prev_track_dict:
                    previous_parts.append(current_asset.means)
                    continue
                local_asset = decouple_bridge.runtime.library.get(track_id, xyz)
                previous_transform = prev_track_dict[track_id].to(xyz)
                if decouple_bridge.runtime.opacity_grounding:
                    try:
                        previous_transform = ground_asset_from_pose(
                            local_asset,
                            previous_transform,
                            composed.background.means,
                            composed.background.opacities,
                            decouple_bridge.runtime.registration_config,
                        ).transform
                    except ValueError:
                        pass
                previous_parts.append(
                    transform_gaussians(local_asset, previous_transform, rotate_sh=False).means
                )
            prev_xyz = torch.cat(previous_parts, dim=0)
    else:
        all_fg, prev_all_fg = [], []
        for track_id, B2W in track_dict.items():
            w_dxyz = (B2W[:3, :3] @ dynamic_gaussians[track_id].get_xyz.T).T + B2W[:3, 3]

            drot = roma.quat_wxyz_to_xyzw(dynamic_gaussians[track_id].get_rotation)
            drot = roma.unitquat_to_rotmat(drot)
            w_drot = roma.quat_xyzw_to_wxyz(roma.rotmat_to_unitquat(B2W[:3, :3] @ drot))
            fg = [w_dxyz,
                dynamic_gaussians[track_id].get_opacity,
                dynamic_gaussians[track_id].get_scaling,
                w_drot,
                # dynamic_gaussians[track_id].get_rotation,
                dynamic_gaussians[track_id].get_features,
                dynamic_gaussians[track_id].get_3D_features]

            all_fg.append(fg)

            if render_optical and prev_viewpoint is not None:
                if track_id in prev_track_dict:
                    prev_B2W = prev_track_dict[track_id]
                    prev_w_dxyz = torch.mm(prev_B2W[:3, :3], dynamic_gaussians[track_id].get_xyz.T).T + prev_B2W[:3, 3]
                    prev_all_fg.append([prev_w_dxyz])
                else:
                    prev_all_fg.append([w_dxyz])

        all_fg = concatenate_all(all_fg)
        xyz, opacities, scales, rotations, shs, feats3D = cat_bgfg(pc, all_fg)

        if render_optical and prev_viewpoint is not None:
            prev_all_fg = concatenate_all(prev_all_fg)
            prev_xyz = cat_bgfg(pc, prev_all_fg, only_xyz=True)[0]

    if render_optical and prev_viewpoint is not None:
        uv = proj_uv(xyz, viewpoint)
        prev_uv = proj_uv(prev_xyz, prev_viewpoint)
        delta_uv = prev_uv - uv
        delta_uv = torch.cat([delta_uv, torch.ones_like(delta_uv[:, :1], device=delta_uv.device)], dim=-1)
    else:
        delta_uv = torch.zeros_like(xyz)

    if rgb_only:
        render_mode = 'RGB'
    elif render_optical:
        render_mode = 'RGB+ED+S+F'
    else:
        render_mode = 'RGB+ED+S'
        
    renders, render_alphas, info = rasterization(
        means=xyz,
        quats=rotations,
        scales=scales,
        opacities=opacities[:, 0],
        colors=shs,
        viewmats=torch.linalg.inv(viewpoint.c2w)[None, ...],  # [C, 4, 4]
        Ks=viewpoint.K[None, :3, :3],  # [C, 3, 3]
        width=viewpoint.width,
        height=viewpoint.height,
        smts=None if rgb_only else feats3D[None, ...],
        flows=None if rgb_only else delta_uv[None, ...],
        render_mode=render_mode,
        sh_degree=pc.active_sh_degree,
        near_plane=0.01,
        far_plane=500,
        packed=rasterizer_packed,
        backgrounds=bg_color[None, :],
    )

    renders = renders[0]
    rendered_image = renders[..., :3].permute(2,0,1)
    depth = None if rgb_only else renders[..., 3][None, ...]
    smt = None if rgb_only else renders[..., 4:(4+feats3D.shape[-1])].permute(2,0,1)
    
    refined_image = apply_appearance_affine(viewpoint, pc, rendered_image)

    return {"render": refined_image,
            "feats": smt,
            "depth": depth,
            "opticalflow": renders[..., -2:].permute(2,0,1) if render_optical else None,
            "alphas": render_alphas,
            "viewspace_points": info["means2d"],
            "info": info,
            }


def render_ground(viewpoint:Camera, pc:GroundModel, bg_color:Tensor):
    xyz, opacities, scales = pc.get_xyz, pc.get_opacity, pc.get_scaling
    rotations, shs, feats3D = pc.get_rotation, pc.get_features, pc.get_3D_features

    K = viewpoint.K[None, :3, :3]
    renders, render_alphas, info = rasterization(
        means=xyz,
        quats=rotations,
        scales=scales,
        opacities=opacities[:, 0],
        colors=shs,
        viewmats=torch.linalg.inv(viewpoint.c2w)[None, ...],  # [C, 4, 4]
        Ks=K,  # [C, 3, 3]
        width=viewpoint.width,
        height=viewpoint.height,
        smts=feats3D[None, ...],
        render_mode='RGB+ED+S',
        sh_degree=pc.active_sh_degree,
        near_plane=0.01,
        far_plane=500,
        packed=False,
        backgrounds=bg_color[None, :],
    )

    renders = renders[0]
    rendered_image = renders[..., :3].permute(2,0,1)
    depth = renders[..., 3][None, ...]
    smt = renders[..., 4:(4+feats3D.shape[-1])].permute(2,0,1)

    return {"render": rendered_image,
            "feats": smt,
            "depth": depth,
            "opticalflow": None,
            "alphas": render_alphas,
            "viewspace_points": info["means2d"],
            "info": info,
            }
