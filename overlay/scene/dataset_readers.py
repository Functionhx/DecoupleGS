import os
from typing import NamedTuple
import numpy as np
import json
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch.nn.functional as F
from imageio.v2 import imread
import torch


class CameraInfo(NamedTuple):
    K: np.array
    c2w: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic2d: np.array
    optical_image: np.array
    depth: torch.tensor
    mask: np.array
    timestamp: int
    dynamics: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    verts: dict

def getNerfppNorm(cam_info, data_type):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        cam_centers.append(cam.c2w[:3, 3:4]) # cam_centers in world coordinate

    radius = 10

    return {'radius': radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' in vertices:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        print('Create random colors')
        shs = np.ones((positions.shape[0], 3)) * 0.5
        colors = SH2RGB(shs)
    normals = np.zeros((positions.shape[0], 3))
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readHUGSIMCameras(path, data_type, ignore_dynamic):
    train_cam_infos, test_cam_infos = [], []
    with open(os.path.join(path, 'meta_data.json')) as json_file:
        meta_data = json.load(json_file)

        verts = {}
        if 'verts' in meta_data and not ignore_dynamic:
            verts_list = meta_data['verts']
            for k, v in verts_list.items():
                verts[k] = np.array(v)

        frames = meta_data['frames']
        for idx, frame in enumerate(frames):
            c2w = np.array(frame['camtoworld'])

            rgb_path = os.path.join(path, frame['rgb_path'].replace('./', ''))

            rgb_split = rgb_path.split('/')
            image_name = '_'.join([rgb_split[-2], rgb_split[-1][:-4]])
            image = imread(rgb_path)

            semantic_2d = None
            semantic_pth = rgb_path.replace("images", "semantics").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(semantic_pth):
                semantic_2d = np.load(semantic_pth)
                semantic_2d[(semantic_2d == 14) | (semantic_2d == 15)] = 13

            optical_path = rgb_path.replace("images", "flow").replace('.png', '_flow.npy').replace('.jpg', '_flow.npy')
            if os.path.exists(optical_path):
                optical_image = np.load(optical_path)
            else:
                optical_image = None

            depth_path = rgb_path.replace("images", "depth").replace('.png', '.pt').replace('.jpg', '.pt')
            if os.path.exists(depth_path):
                depth = torch.load(depth_path, weights_only=True)
            else:
                depth = None

            mask = None
            mask_path = rgb_path.replace("images", "masks").replace('.png', '.npy').replace('.jpg', '.npy')
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

            timestamp = frame.get('timestamp', -1)

            intrinsic = np.array(frame['intrinsics'])
            
            dynamics = {}
            if 'dynamics' in frame and not ignore_dynamic:
                dynamics_list = frame['dynamics']
                for iid in dynamics_list.keys():
                    dynamics[iid] = torch.tensor(dynamics_list[iid]).cuda()
                
            cam_info = CameraInfo(K=intrinsic, c2w=c2w, image=np.array(image),
                                image_path=rgb_path, image_name=image_name, height=image.shape[0],
                                width=image.shape[1], semantic2d=semantic_2d, 
                                optical_image=optical_image, depth=depth, mask=mask, timestamp=timestamp, dynamics=dynamics)
            
            if data_type == 'kitti360':
                if idx < 20:
                    train_cam_infos.append(cam_info)
                elif idx % 20 < 16:
                    train_cam_infos.append(cam_info)
                elif idx % 20 >= 16:
                    test_cam_infos.append(cam_info)
                else:
                    continue

            elif data_type == 'kitti':
                if idx < 10 or idx >= len(frames) - 4:
                    train_cam_infos.append(cam_info)
                elif idx % 4 < 2:
                    train_cam_infos.append(cam_info)
                elif idx % 4 == 2:
                    test_cam_infos.append(cam_info)
                else:
                    continue

            elif data_type == "nuscenes":
                if idx % 30 >= 24:
                    test_cam_infos.append(cam_info)
                else:
                    train_cam_infos.append(cam_info)

            elif data_type == "waymo":
                if idx % 15 >= 12:
                    test_cam_infos.append(cam_info)
                else:
                    train_cam_infos.append(cam_info)

            elif data_type == "pandaset":
                if idx > 30 and idx % 30 >= 24:
                    test_cam_infos.append(cam_info)
                else:
                    train_cam_infos.append(cam_info)
            
            else:
                raise NotImplementedError

    return train_cam_infos, test_cam_infos, verts


def readHUGSIMInfo(path, data_type, ignore_dynamic):
    train_cam_infos, test_cam_infos, verts = readHUGSIMCameras(path, data_type, ignore_dynamic)

    print(f'Loaded {len(train_cam_infos)} train cameras and {len(test_cam_infos)} test cameras')
    nerf_normalization = getNerfppNorm(train_cam_infos, data_type)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path)
        except Exception as e:
            raise RuntimeError(f'failed to load point cloud {ply_path}') from e
    else:
        # Released checkpoints can be evaluated without the training-time
        # initialization cloud. Scene raises below if a fresh training run is
        # attempted from such a source directory.
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           verts=verts)
    return scene_info


sceneLoadTypeCallbacks = {
    "HUGSIM": readHUGSIMInfo,
}
