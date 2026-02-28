import os
import glob
import json
import numpy as np
import torch
import pickle
import trimesh
from pathlib import Path
from tools.io_tools import export_batch_human_mesh
from tools.project_config_tools import shapenet_real_dir, host_device
from tools.navigation_tools import get_navmesh
from tools.motion_tools import get_global_orient_from_forward_z_up, place_stand_pose_on_floor, place_stand_pose_near_lie, place_stand_pose_before_sit, load_body_mesh_model_batch, extract_smplx_from_feature
from data_loaders.scene2motion.realtime_scene import RealtimeScene
from deps.dimos.test_navmesh import path_find

shapenet_scene_test_object_list = [
    'Armchairs/9faefdf6814aaa975510d59f3ab1ed64',
    'Armchairs/cacb9133bc0ef01f7628281ecb18112',
    'Armchairs/ea918174759a2079e83221ad0d21775',
    'L-Sofas/5cea034b028af000c2843529921f9ad7',
    'Sofas/1dd6e32097b09cd6da5dde4c9576b854',
    'Sofas/71fd7103997614db490ad276cd2af3a4',
    'Sofas/277231dcb7261ae4a9fe1734a6086750',
    'StraightChairs/2ed17abd0ff67d4f71a782a4379556c7',
    'StraightChairs/68dc37f167347d73ea46bea76c64cc3d',
    'StraightChairs/d93760fda8d73aaece101336817a135f',
]

class ShapenetSceneTestLie(torch.utils.data.Dataset):
    def __init__(self, object_name):
        self.sit_before_lie = True
        self.object_category, self.object_id = object_name.split('/')
        self.interaction_candidates = sorted(glob.glob(os.path.join(shapenet_real_dir, object_name, 'lie', 'selected', '*.pkl')))
        with open(os.path.join(shapenet_real_dir, object_name, 'sdf_gradient.pkl'), 'rb') as f:
            self.sdf = pickle.load(f)
        return
    def __getitem__(self, idx):
        floor_height = 0.
        navmesh_path = os.path.join(shapenet_real_dir, self.object_category, self.object_id, 'navmesh_loose.ply')
        interaction_pkl_file = self.interaction_candidates[idx]
        with open(interaction_pkl_file, 'rb') as f:
            interaction = pickle.load(f)
        interaction_smplx_param = interaction['smplx_param']
        if self.sit_before_lie:
            sit_pkl_file = interaction_pkl_file.replace('lie', 'sit_before_lie').replace(".pkl", "_sit.pkl")
            with open(sit_pkl_file, 'rb') as f:
                sit = pickle.load(f)
            sit_smplx_param = sit["smplx_param"]
        init_betas = interaction_smplx_param['betas']
        if not 'gender' in interaction_smplx_param.keys():
            gender = 'neutral'
        else:
            gender = interaction_smplx_param['gender']
        interaction_poset = np.concatenate([interaction_smplx_param['global_orient'], interaction_smplx_param['body_pose'], interaction_smplx_param['transl']], axis=-1)
        if self.sit_before_lie:
            sit_poset = np.concatenate([sit_smplx_param['global_orient'], sit_smplx_param['body_pose'], sit_smplx_param['transl']], axis=-1)
        if self.sit_before_lie:
            stand_poset = place_stand_pose_before_sit(sit_poset, init_betas, floor_height, action_order="walk_sit", gender=gender)
            wpath_poset = np.concatenate([stand_poset, sit_poset, interaction_poset, sit_poset, stand_poset], axis=0)
            wpath_action = ["walk", "sit", "lie", "sit", "walk"]
            num_posets = 5
        else:
            stand_poset = place_stand_pose_near_lie(navmesh_path, interaction_poset, init_betas, floor_height, action_order="walk_lie", gender=gender)
            wpath_poset = np.concatenate([stand_poset, interaction_poset, stand_poset], axis=0)
            wpath_action = ["walk", "lie", "walk"]
            num_posets = 3
        body_mesh_model_batch = load_body_mesh_model_batch(num_posets, gender=gender)
        wpath = extract_smplx_from_feature(torch.tensor(wpath_poset[None,...]), body_mesh_model_batch, return_type='joints', betas=torch.tensor(init_betas).repeat(num_posets, 1))[0, :, 0, :].detach().numpy()
        wpath_poset = [wpath_poset[i:i+1,:] for i in range(num_posets)]
        sdf_gradient_pkl = os.path.join(shapenet_real_dir, self.object_category, self.object_id, 'sdf_gradient.pkl')
        with open(sdf_gradient_pkl, 'rb') as f:
            sdf_gradient = pickle.load(f)
        scene_hints = {
            'floor_height': floor_height,
            'scene_global_occ': RealtimeScene(os.path.join(shapenet_real_dir + "_occ", self.object_category, self.object_id, "compact_occ.npy"), os.path.join(shapenet_real_dir + "_occ", self.object_category, self.object_id, "compact_occ_info.json"), semantic_anno_type="matterport3d"),
            'scene_sdf': sdf_gradient['grid'][None,...],
            'scene_sdf_centroid': sdf_gradient['centroid'],
            'scene_sdf_scale': sdf_gradient['scale'],
        }
        ret = {
            'gender': gender,
            'init_betas': init_betas[0],
            'init_transl': stand_poset[:,66:69],
            'init_pose': stand_poset[:,0:66],
            'wpath': torch.tensor(wpath, dtype=torch.float32, device=host_device),
            'wpath_poset':wpath_poset,
            'wpath_action':wpath_action,
            'scene_hints': scene_hints,
            'navmesh_path': navmesh_path,
        }
        return ret

    def __len__(self):
        return len(self.interaction_candidates)

class ShapenetSceneTestSit(torch.utils.data.Dataset):
    def __init__(self, object_name):
        self.object_category, self.object_id = object_name.split('/')
        self.interaction_candidates = sorted(glob.glob(os.path.join(shapenet_real_dir, object_name, 'sit', 'selected', '*.pkl')))
        with open(os.path.join(shapenet_real_dir, object_name, 'sdf_gradient.pkl'), 'rb') as f:
            self.sdf = pickle.load(f)
        return

    def __getitem__(self, idx):
        floor_height = 0.
        interaction_pkl_file = self.interaction_candidates[idx]
        with open(interaction_pkl_file, 'rb') as f:
            interaction = pickle.load(f)
        interaction_smplx_param = interaction['smplx_param']
        init_betas = interaction_smplx_param['betas']
        if not 'gender' in interaction_smplx_param.keys():
            gender = 'neutral'
        else:
            gender = interaction_smplx_param['gender']
        interaction_poset = np.concatenate([interaction_smplx_param['global_orient'], interaction_smplx_param['body_pose'], interaction_smplx_param['transl']], axis=-1)
        stand_poset = place_stand_pose_before_sit(interaction_poset, init_betas, floor_height, action_order="walk_sit", gender=gender)
        wpath_poset = np.concatenate([stand_poset, interaction_poset, stand_poset], axis=0)
        body_mesh_model_batch = load_body_mesh_model_batch(3, gender=gender)
        wpath = extract_smplx_from_feature(torch.tensor(wpath_poset[None,...]), body_mesh_model_batch, return_type='joints', betas=torch.tensor(init_betas).repeat(3, 1))[0, :, 0, :].detach().numpy()
        wpath_poset = [wpath_poset[0:1,:], wpath_poset[1:2,:], wpath_poset[2:3,:]]
        wpath_action = ["walk", "sit", "walk"]
        sdf_gradient_pkl = os.path.join(shapenet_real_dir, self.object_category, self.object_id, 'sdf_gradient.pkl')
        with open(sdf_gradient_pkl, 'rb') as f:
            sdf_gradient = pickle.load(f)
        scene_hints = {
            'floor_height': floor_height,
            'scene_global_occ': RealtimeScene(os.path.join(shapenet_real_dir + "_occ", self.object_category, self.object_id, "compact_occ.npy"), os.path.join(shapenet_real_dir + "_occ", self.object_category, self.object_id, "compact_occ_info.json"), semantic_anno_type="matterport3d"),
            'scene_sdf': sdf_gradient['grid'][None,...],
            'scene_sdf_centroid': sdf_gradient['centroid'],
            'scene_sdf_scale': sdf_gradient['scale'],
        }
        ret = {
            'gender': gender,
            'init_betas': init_betas[0],
            'init_transl': stand_poset[:,66:69],
            'init_pose': stand_poset[:,0:66],
            'wpath': torch.tensor(wpath, dtype=torch.float32, device=host_device),
            'wpath_poset':wpath_poset,
            'wpath_action':wpath_action,
            'scene_hints': scene_hints,
            'navmesh_path': None,
        }
        return ret

    def __len__(self):
        return len(self.interaction_candidates)
