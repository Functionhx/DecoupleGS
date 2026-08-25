import numpy as np
import json
import os
from imageio.v2 import imread, imwrite
import argparse
import cv2

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument(
        '--dilation', type=int, default=5,
        help='Dynamic-mask dilation radius in pixels (DecoupleGS uses 5).',
    )
    parser.add_argument(
        '--sam_checkpoint', type=str, default=None,
        help='Optional Segment Anything checkpoint. Projected 3D boxes are used as prompts.',
    )
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    return parser.parse_args()

def checkcorner(corner, h, w):
    if np.all(corner < 0) or (corner[0] >= h and corner[1] >= w):
        return False
    else:
        return True

def main():
    args = get_opts()
    basedir = args.data_path
    predictor = None
    if args.sam_checkpoint is not None:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as error:
            raise RuntimeError(
                'SAM masking requested, but segment-anything is not installed'
            ) from error
        sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
        sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        predictor = SamPredictor(sam)
    os.makedirs(os.path.join(basedir, 'masks'), exist_ok=True)
    if args.data_type == 'kitti360':
        cameras = ['cam_0', 'cam_1', 'cam_2', 'cam_3']
    elif args.data_type == 'pandaset':
        AVAILABLE_CAMERAS = ("front", "front_left", "front_right", "back", "left", "right")
        cameras = [cam + "_camera" for cam in AVAILABLE_CAMERAS]
    elif args.data_type == 'nuscenes':
        cameras = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", 
                             "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    elif args.data_type == 'waymo':
        cameras = ['cam_1', 'cam_2', 'cam_3']
    else:
        raise NotImplementedError
    for cam in cameras:
        os.makedirs(os.path.join(basedir, 'masks', cam), exist_ok=True)
    # Opening JSON file
    with open(os.path.join(basedir, "meta_data.json")) as f:
        meta_data = json.load(f)

    verts = meta_data['verts']
    for f in meta_data['frames']:
        rgb_path = f['rgb_path']
        c2w = np.array(f['camtoworld'])
        intr = np.array(f['intrinsics'])
        w2c = np.linalg.inv(c2w)

        smt = np.load(os.path.join(basedir, rgb_path.replace('images', 'semantics').replace('.jpg', '.npy')).replace('.png', '.npy'))
        car_mask = (smt == 11) | (smt == 12) | (smt == 13) | (smt == 14) | (smt == 15) | (smt == 18)
        mask = np.zeros_like(car_mask).astype(np.bool_)
        if predictor is not None:
            image = cv2.imread(os.path.join(basedir, rgb_path))
            if image is None:
                raise FileNotFoundError(os.path.join(basedir, rgb_path))
            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        for iid, rt in f['dynamics'].items():
            H, W = mask.shape[0], mask.shape[1]
            rt = np.array(rt)
            points = np.array(verts[iid])
            points = (rt[:3, :3] @ points.T).T + rt[:3, 3]
            xyz_cam = (w2c[:3, :3] @ points.T).T + w2c[:3, 3]
            valid_depth = xyz_cam[:, 2] > 0
            xyz_screen = (intr[:3, :3] @ xyz_cam.T).T + intr[:3, 3]
            xy_screen  = xyz_screen[:, :2] / xyz_screen[:, 2][:, None]
            valid_x = (xy_screen[:, 0] >= 0) & (xy_screen[:, 0] < W)
            valid_y = (xy_screen[:, 1] >= 0) & (xy_screen[:, 1] < H)
            valid_pixel = valid_x & valid_y & valid_depth

            if valid_pixel.any():
                xy_screen = np.round(xy_screen).astype(int)
                if predictor is not None:
                    visible_xy = xy_screen[valid_pixel]
                    box = np.array([
                        np.clip(visible_xy[:, 0].min(), 0, W - 1),
                        np.clip(visible_xy[:, 1].min(), 0, H - 1),
                        np.clip(visible_xy[:, 0].max(), 0, W - 1),
                        np.clip(visible_xy[:, 1].max(), 0, H - 1),
                    ], dtype=np.float32)
                    sam_masks, scores, _ = predictor.predict(
                        box=box,
                        multimask_output=True,
                    )
                    mask |= sam_masks[np.argmax(scores)]
                else:
                    bbox_mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(bbox_mask, [xy_screen[[0, 1, 4, 5, 0]]], 1)
                    cv2.fillPoly(bbox_mask, [xy_screen[[2, 3, 6, 7, 2]]], 1)
                    cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 7, 5, 0]]], 1)
                    cv2.fillPoly(bbox_mask, [xy_screen[[1, 3, 6, 4, 1]]], 1)
                    cv2.fillPoly(bbox_mask, [xy_screen[[0, 2, 3, 1, 0]]], 1)
                    cv2.fillPoly(bbox_mask, [xy_screen[[5, 4, 6, 7, 5]]], 1)
                    bbox_mask = bbox_mask & car_mask
                    mask |= bbox_mask != 0

        if args.dilation > 0:
            diameter = 2 * args.dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        save_path = os.path.join(basedir, rgb_path.replace('images', 'masks'))
        np.save(save_path.replace('.jpg', '.npy').replace('.png', '.npy'), ~mask)
        imwrite(save_path+'.png', (~mask).astype(np.uint8) * 255)

if __name__ == "__main__":
    main()
