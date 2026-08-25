import os
import torch
from utils.loss_utils import l1_loss, ssim, ssim_loss
from gaussian_renderer import render
from scene import Scene, GaussianModel
import uuid
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from argparse import ArgumentParser
from torch.nn import CrossEntropyLoss
import json
import pickle
import torchvision
from utils.dataset import HUGSIM_dataset, hugsim_collate, tocuda
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm as std_tqdm
from utils.dynamic_utils import create_unicycle_model
tqdm = partial(std_tqdm, dynamic_ncols=True)

results = {'train': {}, 'test': {}}

# metrics
m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

def training(cfg):

    if cfg.semantic:
        semantic_ce = CrossEntropyLoss()
    
    with open(os.path.join(cfg.source_path, 'ground_param.pkl'), 'rb') as f:
        cam_poses, _, _ = pickle.load(f)
        cam_positions = torch.tensor(cam_poses[:, :3, 3]).float().cuda()

    first_iter = 0
    prepare_output(cfg)
    (ground_model_params, _) = torch.load(os.path.join(cfg.model_path, "ckpts", f"ground_chkpnt30000.pth"))
    gaussians = GaussianModel(cfg.model.sh_degree, feat_mutable=True, affine=cfg.affine, ground_args=ground_model_params)
    scene = Scene(
        cfg,
        gaussians,
        data_type=cfg.data_type,
        ignore_dynamic=bool(cfg.ignore_dynamic),
    )
    
    scene.gaussians.training_setup(cfg.opt)
    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
        dynamic_gaussian.training_setup(cfg.opt)
    
    if cfg.unicycle:
        unicycles = create_unicycle_model(scene.getTrainCameras(), cfg.model_path, cfg.uc_fit_iter, cfg.uc_opt_pos, cfg.data_type)
    else:
        unicycles = {}

    bg_color = [1, 1, 1] if cfg.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    progress_bar = tqdm(range(first_iter, cfg.train.iterations), desc="Training progress")
    first_iter += 1

    os.makedirs(os.path.join(scene.model_path, "save_train"), exist_ok=True)

    train_cams = scene.getTrainCameras().copy()
    train_dataset = HUGSIM_dataset(train_cams, cfg.data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=hugsim_collate)

    for iteration in range(first_iter, cfg.train.iterations + 1):        

        iter_start.record()

        scene.gaussians.update_learning_rate(iteration)
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_gaussian.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                dynamic_gaussian.oneupSHdegree()

        view_iid, prev_iid, gt_image, gt_semantic, gt_flow, gt_depth, mask = next(iter(train_dataloader))
        gt_image, gt_flow, gt_depth, mask = gt_image.cuda(), tocuda(gt_flow), tocuda(gt_depth), tocuda(mask)
        viewpoint_cam = train_cams[view_iid]

        # Render
        render_pkg = render(viewpoint_cam, None, gaussians, scene.dynamic_gaussians, unicycles, background)
        
        # gsplat
        image, viewspace_point_tensor, info = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["info"]
        radii = info["radii"][0]
        visibility_filter = radii > 0
        viewspace_point_tensor.retain_grad()

        if iteration % 500 == 0:
            torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_train", f"{iteration}_{viewpoint_cam.image_name}.png"))

        # Loss
        loss = 0

        if cfg.semantic and gt_semantic is not None:
            gt_semantic = gt_semantic.cuda()
            semantic_map = render_pkg["feats"]
            if mask is not None and cfg.ignore_dynamic:
                gt_semantic[:, ~mask] = torch.argmax(semantic_map[:, ~mask], dim=0).detach()
            semantic_loss = semantic_ce(semantic_map.permute(1,2,0).view(-1, 20), gt_semantic.view(-1)) * 0.01
            loss += semantic_loss

        if mask is not None and cfg.ignore_dynamic:
            gt_image[:, ~mask] = image[:, ~mask].detach()
            
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - cfg.opt.lambda_dssim) * Ll1 + cfg.opt.lambda_dssim * ssim_loss(image, gt_image)
        loss += rgb_loss

        distort_3d_loss = 0 
        N_sample=10
        grid_length = 0.2
        if iteration > 2000:
            ground_mask = torch.argmax(gaussians.get_3D_features, dim=1) <= 1
            w2c = torch.linalg.inv(viewpoint_cam.c2w)
            points = gaussians.get_xyz[ground_mask]
            c_points = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            points_gd = gaussians.ground_model.get_xyz
            c_points_gd = (w2c[:3, :3] @ points_gd.T).T + w2c[:3, 3]
            biases = -2 + 4 * torch.rand(N_sample, device='cuda')
            for bias in biases:
                mask = (bias < c_points[:, 2]) & (c_points[:, 2] < (bias + grid_length)) 
                if torch.sum(mask) == 0:
                    continue
                ys = c_points[mask, 1]
                mask_gd = (bias < c_points_gd[:, 2]) & (c_points_gd[:, 2] < (bias + grid_length)) 
                ys_gd = c_points_gd[mask_gd, 1]
                distort_3d_loss += torch.mean((ys - torch.mean(ys_gd))**2)
            distort_3d_loss /= N_sample
            loss += distort_3d_loss

        reg_loss = 0
        if cfg.uc_opt_pos and (len(unicycles) > 0) and (1000 < iteration) and (iteration < 15000):
            for track_id, unicycle_pkg in unicycles.items():
                model = unicycle_pkg['model']
                reg_loss += 1e-3 * model.reg_loss() + 1e-4 * model.pos_loss()
            reg_loss = reg_loss / len(unicycles)
            loss += reg_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                postfix = {"RGB": f"{rgb_loss:.{4}f}"}
                if cfg.semantic:
                    postfix["Semantic"] = f"{semantic_loss:.{4}f}"
                if reg_loss != 0:
                    postfix["UniReg"] = f"{reg_loss:.{4}f}"
                if distort_3d_loss != 0:
                    postfix['dist3d'] = f"{distort_3d_loss:.{4}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == cfg.train.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize()
            
            if (iteration in cfg.train.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                os.makedirs(scene.model_path + '/ckpts', exist_ok=True)
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"ckpts/chkpnt{str(iteration)}.pth"))

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    torch.save((dynamic_gaussian.capture(), iteration), os.path.join(scene.model_path, f"ckpts/dynamic_{iid}_chkpnt{iteration}.pth"))
                for track_id, unicycle_pkg in unicycles.items():
                    model = unicycle_pkg['model']
                    torch.save(model.capture(), os.path.join(scene.model_path, f"ckpts/unicycle_{track_id}.pth"))
                    model.visualize(os.path.join(scene.model_path, "unicycle", f"{track_id}_{iteration}.png"))

                validation(iteration, scene, render, [scene.dynamic_gaussians, unicycles, background])

                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Optimizer step
            if iteration < cfg.train.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if gaussians.ground_optimizer is not None:
                    gaussians.ground_optimizer.step()
                    gaussians.ground_optimizer.zero_grad(set_to_none = True)

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    dynamic_gaussian.optimizer.step()
                    dynamic_gaussian.optimizer.zero_grad(set_to_none = True)

                if cfg.unicycle and cfg.uc_opt_pos and iteration > 1000:
                    for track_id, unicycle_pkg in unicycles.items():
                        unicycle_optimizer = unicycle_pkg['optimizer']
                        unicycle_optimizer.step()
                        unicycle_optimizer.zero_grad(set_to_none = True)
                        # if iteration % 5000 == 0:
                        #     for g in unicycle_optimizer.param_groups:
                        #         g['lr'] /= 2

            # Densification
            if iteration < cfg.opt.densify_until_iter:

                # gsplat
                grad = viewspace_point_tensor.grad[0].clone()
                grad[..., 0] *= info['width'] / 2.0
                grad[..., 1] *= info['height'] / 2.0
                # Keep track of max radii in image-space for pruning
                current_index = gaussians.get_xyz.shape[0]
                gaussians.max_radii2D[visibility_filter[:current_index]] = torch.max(gaussians.max_radii2D[visibility_filter[:current_index]], radii[:current_index][visibility_filter[:current_index]])
                gaussians.add_densification_stats_grad(grad[:current_index], visibility_filter[:current_index])
                last_index = current_index

                for iid in viewpoint_cam.dynamics.keys():
                    dynamic_gaussian = scene.dynamic_gaussians[iid]
                    current_index = last_index + dynamic_gaussian.get_xyz.shape[0]
                    visible_mask = visibility_filter[last_index:current_index]
                    dynamic_gaussian.max_radii2D[visible_mask] = torch.max(dynamic_gaussian.max_radii2D[visible_mask], radii[last_index:current_index][visible_mask])
                    dynamic_gaussian.add_densification_stats_grad(grad[last_index:current_index], visible_mask)
                    last_index = current_index

                if iteration > cfg.opt.densify_from_iter and iteration % cfg.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > cfg.opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(cfg.opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, cam_pos=cam_positions)
                    # for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    #     dynamic_gaussian.densify_and_prune(cfg.opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                
                if iteration % cfg.opt.opacity_reset_interval == 0 or (cfg.model.white_background and iteration == cfg.opt.densify_from_iter):
                    gaussians.reset_opacity()
                    # for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    #     dynamic_gaussian.reset_opacity()

def prepare_output(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "unicycle"), exist_ok=True)
    OmegaConf.save(args, os.path.join(args.model_path, "cfg.yaml"))

def validation(iteration, scene, renderFunc, renderArgs):
    os.makedirs(os.path.join(scene.model_path, "save_test"), exist_ok=True)

    # Report test and samples of training set
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                            {'name': 'train', 'cameras': scene.getTrainCameras()})

    for config in validation_configs:
        l1_test = 0
        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        for viewpoint in config['cameras']:
            gt_image = viewpoint.original_image.cuda()
            image = torch.clamp(renderFunc(viewpoint, None, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
            l1_test += l1_loss(image, gt_image).mean()
            image = image[None, ...]
            gt_image = gt_image[None, ...]
            psnr_test += m_psnr(image, gt_image)
            ssim_test += m_ssim(image, gt_image)
            lpips_test += m_lpips(image, gt_image)

            if config['name'] == 'test':
                torchvision.utils.save_image(image, os.path.join(scene.model_path, "save_test", f"{viewpoint.image_name}.png"))
        
        psnr_test /= len(config['cameras'])
        ssim_test /= len(config['cameras'])
        lpips_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])          
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {format(l1_test, '.4f')} "
                f"PSNR {format(psnr_test, '.4f')} SSIM {format(ssim_test, '.4f')} Lpips {format(lpips_test, '.4f')}")
        
        results[config['name']][iteration] = {
            'psnr': psnr_test.item(),
            'ssim': ssim_test.item(),
            'lpips': lpips_test.item(),
            'l1': l1_test.item()
        }

    torch.cuda.empty_cache()

    with open(os.path.join(scene.model_path, 'results.json'), 'w') as wf:
        json.dump(results, wf, indent=4)

def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--base_cfg", type=str, default="./configs/gs_base.yaml")
    parser.add_argument("--data_cfg", type=str, default="./configs/nusc.yaml")
    parser.add_argument("--source_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--ignore_dynamic",
        action="store_true",
        help="Train a clean static background using the generated dynamic masks",
    )
    parser.add_argument(
        "--disable_affine",
        action="store_true",
        help="Disable HUGSIM's appearance MLP and use vanilla 3DGS colors",
    )
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.base_cfg), OmegaConf.load(args.data_cfg))
    if len(args.source_path) > 0:
        cfg.source_path = args.source_path
    if len(args.model_path) > 0:
        cfg.model_path = args.model_path
    if args.ignore_dynamic:
        cfg.ignore_dynamic = True
    if args.disable_affine:
        cfg.affine = False
        
    print("Optimizing " + args.model_path)
    training(cfg)
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
