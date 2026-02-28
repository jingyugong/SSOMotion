import os
import sys
from utils.misc import to_torch
from .dataset import Dataset
import glob
import numpy as np
import torch
from tools.io_tools import load_joint_hint_from_file, random_mask_train
from tools.project_config_tools import action_enumerator, local_grid_info

class MixedMotion(Dataset):
    dataname = "mixedmotion"
    def __init__(self, datapath="dataset/processed_datasets", split="train", controlnet=False, **kargs):
        self.datapath = datapath
        self.controlnet = controlnet
        kargs["pose_rep"] = "rotvec"
        kargs["max_len"] = 30
        kargs["num_frames"] = -1
        super().__init__(**kargs)
        data_subset_collections = [
            ["babel/walk", "babel/turn", "humanml3d/jog_run", "humanml3d/walk_turn"] + ["humanise/walk"] * 20, 
            ["babel/sit", "humanml3d/sit"] + ["humanise/sit"] * 20, 
            ["humanise/stand_up"] * 20, 
            ["babel/lie", "babel/lie"] + ["humanml3d/lie"] * 20
        ]
        total_num_actions = len(data_subset_collections)
        self.num_actions = total_num_actions
        self._data_file_paths = []
        self._actions = []
        for action_id, data_subset_collection in zip(range(self.num_actions), data_subset_collections): 
            for data_subset in data_subset_collection:
                new_file_paths = sorted(glob.glob(os.path.join(datapath, data_subset, "*.npz")))
                self._data_file_paths.extend(new_file_paths)
                self._actions.extend([action_id] * len(new_file_paths))
        reorder_idx = np.random.permutation(len(self._data_file_paths))
        self._data_file_paths = [self._data_file_paths[i] for i in reorder_idx]
        self._actions = [self._actions[i] for i in reorder_idx]
        self._train = list(range(len(self._data_file_paths)))
        self._num_frames_in_video = [dict(np.load(self._data_file_paths[0], allow_pickle=True))["transl"].shape[0]] * len(self._data_file_paths)

        keep_actions = np.arange(0, total_num_actions)
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}
        self._action_classes = action_enumerator

        return

    def _load(self, ind, frame_idx):
        file_path = self._data_file_paths[ind]
        data_dict = dict(np.load(file_path, allow_pickle=True))
        transl = data_dict["transl"][frame_idx]
        pose = data_dict["pose"][frame_idx,:66]
        nframes = transl.shape[0]
        pose_rep = self.pose_rep
        transl = to_torch(transl)
        pose = to_torch(pose)
        if pose_rep == "rotvec":
            pass
        elif pose_rep == "rot6d":
            pose = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose.view(nframes, -1, 3))).view(nframes, -1)
        else:
            raise NotImplementedError
        ret = torch.concat([pose, transl], dim=1).unsqueeze(-1)
        ret = ret.permute(1, 2, 0).contiguous()
        return ret.float()

    def get_joint_hint(self, ind):
        file_path = self._data_file_paths[ind]
        skeleton = load_joint_hint_from_file(file_path)
        n_frames, n_joints, n_dims = skeleton.shape
        hint = random_mask_train(skeleton, n_joints=n_joints)
        case_idx = np.random.randint(3)
        if case_idx == 0:
            pass
        elif case_idx == 1:
            hint[:1,:,:] = skeleton[:1,:,:]
        else:
            hint[:5,:,:] = skeleton[:5,:,:]
        hint = hint.reshape(n_frames, -1)
        return hint

    def get_direction_hint(self, ind, action):
        file_path = self._data_file_paths[ind]
        joints = load_joint_hint_from_file(file_path)
        if action == "walk":
            direction = joints[-1, 0:1, :] - joints[0, 0:1, :]
            direction = direction.repeat(repeats=joints.shape[1], axis=0)
        elif action in ["sit", "stand_up", "lie"]:
            direction = joints[-1, :, :] - joints[0, :, :]
        else:
            raise ValueError("Unexcepted action type.")
        direction_hint = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(direction, axis=1, keepdims=True).clip(0, 1) + 1e-8)
        direction_hint = direction_hint.reshape(-1) 
        return direction_hint

    def get_void_scene_hint(self):
        x_grid_steps, y_grid_steps, z_grid_steps = local_grid_info["grid_steps"]
        ret = {
            "scene_valid": torch.zeros(1, dtype=torch.float32),
            "x_map1_depth": torch.zeros((1, z_grid_steps, y_grid_steps), dtype=torch.float32),
            "x_map2_depth": torch.zeros((1, z_grid_steps, y_grid_steps), dtype=torch.float32),
            "y_map1_depth": torch.zeros((1, z_grid_steps, x_grid_steps), dtype=torch.float32),
            "y_map2_depth": torch.zeros((1, z_grid_steps, x_grid_steps), dtype=torch.float32),
            "z_map1_depth": torch.zeros((1, y_grid_steps, x_grid_steps), dtype=torch.float32),
            "x_map1_rgb": torch.zeros((3, z_grid_steps, y_grid_steps), dtype=torch.float32),
            "x_map2_rgb": torch.zeros((3, z_grid_steps, y_grid_steps), dtype=torch.float32),
            "y_map1_rgb": torch.zeros((3, z_grid_steps, x_grid_steps), dtype=torch.float32),
            "y_map2_rgb": torch.zeros((3, z_grid_steps, x_grid_steps), dtype=torch.float32),
            "z_map1_rgb": torch.zeros((3, y_grid_steps, x_grid_steps), dtype=torch.float32),
            "x_map1_sem": torch.zeros((1, z_grid_steps, y_grid_steps), dtype=torch.int32),
            "x_map2_sem": torch.zeros((1, z_grid_steps, y_grid_steps), dtype=torch.int32),
            "y_map1_sem": torch.zeros((1, z_grid_steps, x_grid_steps), dtype=torch.int32),
            "y_map2_sem": torch.zeros((1, z_grid_steps, x_grid_steps), dtype=torch.int32),
            "z_map1_sem": torch.zeros((1, y_grid_steps, x_grid_steps), dtype=torch.int32),
        }
        return ret
        

"""
mixed_coarse_action_enumerator = {
    0: "walk",
    1: "sit",
    2: "stand_up",
    3: "lie",
}
"""
if __name__ == "__main__":
    dataset = MixedMotion(datapath="/home/gongjingyu/gcode/RGBD/code/guided-motion-diffusion/dataset/processed_datasets", split="train", pose_rep="rotvec", num_frames=-1, max_len=30)
    print(dataset[0])
