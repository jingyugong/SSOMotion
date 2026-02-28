import os
import glob
import json
import numpy as np
import torch
import trimesh
from pathlib import Path
from tools.project_config_tools import random_scene_test_dir, host_device
from tools.navigation_tools import get_navmesh
from tools.motion_tools import get_global_orient_from_forward_z_up, place_stand_pose_on_floor
from deps.dimos.test_navmesh import path_find
from data_loaders.scene2motion.realtime_scene import RealtimeScene

random_scene_test_room_list = [
    '5.33_5.94_2_1677228749.1353848',
    '5.54_5.34_0_1677228775.5952885',
    '5.76_8.87_1_1677228764.1464393',
    '5.97_6.83_1_1677228757.1440427',
    '6.27_4.97_0_1677228807.6371164',
]
def get_test_scene_point_from_json(json_name):
    with open(json_name, "r") as f:
        data = json.load(f)
    data = np.array([data[k] for k in ['x', 'y', 'z']])
    rotmat = np.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0],
    ])
    data = np.matmul(rotmat, data[...,None]).reshape(-1)
    return data

class RandomSceneTestWalk(torch.utils.data.Dataset):
    def __init__(self, room_name):
        self.room_name = room_name
        self.start_points = sorted(glob.glob(os.path.join(random_scene_test_dir, room_name, 'pair*_start.json')))
        self.start_points = [get_test_scene_point_from_json(x) for x in self.start_points]
        self.target_points = sorted(glob.glob(os.path.join(random_scene_test_dir, room_name, 'pair*_target.json')))
        self.target_points = [get_test_scene_point_from_json(x) for x in self.target_points]
        self.navmesh_type = "loose"
        return

    def __getitem__(self, index):
        start_point = self.start_points[index]
        target_point = self.target_points[index]
        start_target = np.stack([start_point, target_point])
        navmesh_path = Path(os.path.join(random_scene_test_dir, self.room_name, 'navmesh_{}.ply'.format(self.navmesh_type)))
        scene_path = Path(os.path.join(random_scene_test_dir, self.room_name, 'mesh.ply'))
        floor_height = 0.
        if self.navmesh_type == 'loose':
            navmesh = get_navmesh(navmesh_path, scene_path, agent_radius=0.2, floor_height=floor_height)
        else:
            navmesh = get_navmesh(navmesh_path, scene_path, agent_radius=0.05, floor_height=floor_height)
        scene_mesh = trimesh.load(scene_path, force='mesh')
        wpath = path_find(navmesh, start_target[0], start_target[1], visualize=False, scene_mesh=scene_mesh)
        gender = np.random.choice(['male', 'female'])
        init_betas = np.random.normal(0., 1., 10).astype(np.float32)
        init_global_orient = get_global_orient_from_forward_z_up(wpath[1] - wpath[0])
        pose, transl, pelvis_joint = place_stand_pose_on_floor(gender, init_betas, init_global_orient, wpath[0], floor_height)
        wpath[:,2] = pelvis_joint[2]
        init_global_orient, pose, transl = init_global_orient[None,...], pose[None,...], transl[None,...]
        wpath_poset = [np.concatenate([init_global_orient, pose, transl], axis=-1)]
        wpath_poset += [None] * (len(wpath) - 1)
        wpath_action = ["walk"] * len(wpath)
        scene_hints = {
            'floor_height': floor_height,
            'scene_global_occ': RealtimeScene(os.path.join(random_scene_test_dir + '_occ', self.room_name, "compact_occ.npy"), os.path.join(random_scene_test_dir + '_occ', self.room_name, "compact_occ_info.json"), semantic_anno_type="matterport3d")
        }
        ret = {
            'gender': gender,
            'init_betas': init_betas,
            'init_transl': transl,
            'init_pose': np.concatenate([init_global_orient, pose], axis=-1),
            'wpath': torch.tensor(wpath, dtype=torch.float32, device=host_device),
            'wpath_poset':wpath_poset,
            'wpath_action': wpath_action,
            'scene_hints': scene_hints,
            'navmesh_path': str(navmesh_path).replace('loose', 'tight'),
        }
        return ret
    def __len__(self):
        return len(self.start_points)
