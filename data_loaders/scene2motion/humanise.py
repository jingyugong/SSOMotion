import os
import glob
import pickle
import tqdm
import numpy as np
import torch
import clip
from trimesh import transform_points
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion as Q
from tools.project_config_tools import project_dir, host_device, action_enumerator, default_voxel_size, local_grid_info
from tools.occ_tools import OccInfo, Occupancy
from tools.motion_tools import load_body_mesh_model_batch, extract_smplx_from_feature
from tools.coordinate_tools import calc_calibrate_offset, get_new_coordinate
from tools.io_tools import random_mask_train
from tools.visualization_tools import CategoryLabel, CategoryLabelConverter
from utils.smplx_util import SMPLX_Util

class HumaniseMotion(torch.utils.data.Dataset):
    def __init__(
        self,
        split="train",
        num_frames=30,
        controlnet=True,
        motion_stride=5,
        **kargs
    ):
        self.load_action = kargs.get('action_name', '')
        if self.load_action not in list(action_enumerator.values()):
            self.load_action = '*'
        super().__init__()
        self.data_device = 'cpu'#host_device
        self.anno_dir = os.path.join(project_dir, "dataset/HUMANISE/align_data_release")
        self.occ_dir = os.path.join(project_dir, "dataset/HUMANISE/occ")
        self.motion_dir = os.path.join(project_dir, "dataset/HUMANISE/pure_motion")

        self.id2action = action_enumerator
        self.action2id = {v:k for k,v in self.id2action.items()}
        self.num_actions = len(action_enumerator)

        self.label_semantics = CategoryLabel(semantic_anno_type="scannet")

        self.split = split
        self.num_frames = num_frames
        self.motion_stride = motion_stride

        self.bodymodel_one = load_body_mesh_model_batch(1, body_type='smplx', gender='neutral', device=self.data_device)
        self.bodymodel_batch = load_body_mesh_model_batch(self.num_frames, body_type='smplx', gender='neutral', device=self.data_device)

        self.all_anno_files = sorted(glob.glob(os.path.join(self.anno_dir, self.load_action + "/*/anno.pkl")))
        self._get_anno_according_split()
        #prepare motion data
        self._prepare_motion_data_list()
        return

    def _get_anno_according_split(self):
        self.anno_list = []
        self.scene_dict = {}
        self.motion_dict = {}
        for i, anno_file in enumerate(self.all_anno_files):
            with open(anno_file, "rb") as f:
                anno = pickle.load(f)
            for anno_item in anno:
                action = anno_item["action"].replace(" ", "_")
                scene_id = anno_item["scene"]
                motion_id = anno_item["motion"]
                if self.split == "train" and int(scene_id[5:9]) >= 600:
                    continue
                if self.split == "test" and int(scene_id[5:9]) < 600:
                    continue
                self.anno_list.append(anno_item)
                if scene_id not in self.scene_dict:
                    scene_occ_path = os.path.join(self.occ_dir, scene_id, scene_id + "_vh_clean.compact.npy")
                    scene_info = OccInfo(scene_occ_path[:-4] + ".json")
                    scene_global_occ = Occupancy()
                    scene_global_occ.initialize_from_compact_npy(scene_occ_path, scene_info.grid_size)
                    self.scene_dict[scene_id] = (scene_info, scene_global_occ)
                if motion_id not in self.motion_dict:
                    with open(os.path.join(self.motion_dir, action, motion_id, "motion.pkl"), "rb") as f:
                        motion = pickle.load(f)
                    self.motion_dict[motion_id] = motion
        return

    def _prepare_motion_data_list(self):
        """
        sample motion according to humanise annotation, and translate the motion to coordinate where occ center is at origin
        """
        self.index2loc = []
        self.anno_motion_list = []
        for anno_index, anno_item in tqdm.tqdm(enumerate(self.anno_list)):
            action = anno_item["action"].replace(" ", "_")
            scene_id = anno_item["scene"]
            scene_trans = anno_item["scene_translation"]

            motion_id = anno_item["motion"]
            motion_trans = anno_item["translation"]
            motion_rot = anno_item["rotation"]
            anchor_frame_index = self._get_anchor_frame_index(action)
            (
                gender,         # str, nerutal
                trans,          # np.ndarray, <L, 3>
                orient,         # np.ndarray, <L, 3>
                betas,          # np.ndarray, <16>
                body_pose,      # np.ndarray, <L, 63>
                hand_pose,      # np.ndarray, <L, 90>
                jaw_pose,       # np.ndarray, <L, 3>
                eye_pose,       # np.ndarray, <L, 6>
                joints,         # np.ndarray, <L, 127, 3>
            ) = self.motion_dict[motion_id]
            sample_trans, sample_orient, sample_pelvis = self._transform_smplx_from_origin_to_sampled_position(motion_trans, motion_rot, trans, orient, joints[:, 0, :], anchor_frame_index)
            betas = betas[:10]
            scene_info, _ = self.scene_dict[scene_id]
            occ_center = scene_info.min_bound + scene_trans + (np.array(scene_info.grid_size) - 1)/2 * scene_info.voxel_size
            sample_trans -= occ_center

            motion_item = {
                "betas": betas.astype(np.float32),
                "translation": sample_trans.astype(np.float32),
                "orientation": sample_orient.astype(np.float32),
                "body_pose": body_pose.astype(np.float32),
                "hand_pose": hand_pose.astype(np.float32),
            }
            self.anno_motion_list.append(motion_item)

            total_num_frames = sample_trans.shape[0]
            new_tuples = [(anno_index, offset) for offset in range(0, total_num_frames-self.num_frames+1, self.motion_stride)]
            self.index2loc.extend(new_tuples)
        return

    def _transform_smplx_from_origin_to_sampled_position(
        self,
        sampled_trans: np.ndarray,
        sampled_rotat: np.ndarray,
        origin_trans: np.ndarray,
        origin_orient: np.ndarray,
        origin_pelvis: np.ndarray,
        anchor_frame: int = 0,
    ):
        """Convert original smplx parameters to transformed smplx parameters

        Args:
            sampled_trans: sampled valid position
            sampled_rotat: sampled valid rotation
            origin_trans: original trans param array
            origin_orient: original orient param array
            origin_pelvis: original pelvis trajectory
            anchor_frame: the anchor frame index for transform motion, this value is very important!!!

        Return:
            Transformed trans, Transformed orient, Transformed pelvis
        """
        position = sampled_trans
        rotat = sampled_rotat

        T1 = np.eye(4, dtype=np.float32)
        T1[0:2, -1] = -origin_pelvis[anchor_frame, 0:2]
        T2 = Q(axis=[0, 0, 1], angle=rotat).transformation_matrix.astype(np.float32)
        T3 = np.eye(4, dtype=np.float32)
        T3[0:3, -1] = position
        T = T3 @ T2 @ T1

        trans_t = []
        orient_t = []
        for i in range(len(origin_trans)):
            t_, o_ = SMPLX_Util.convert_smplx_verts_transfomation_matrix_to_body(
                T, origin_trans[i], origin_orient[i], origin_pelvis[i]
            )
            trans_t.append(t_)
            orient_t.append(o_)

        trans_t = np.array(trans_t)
        orient_t = np.array(orient_t)
        pelvis_t = transform_points(origin_pelvis, T)
        return trans_t, orient_t, pelvis_t

    def _get_anchor_frame_index(self, action: str):
        if action == "sit":
            return -1
        elif action == "stand_up":
            return 0
        elif action == "walk":
            return -1
        elif action == "lie":
            return -1
        else:
            raise Exception("Unexcepted action type.")

    def _get_direction_hint_from_joints(self, joints, action):
        if action == "walk":
            direction = joints[-1, 0:1, :] - joints[0, 0:1, :]
            direction = direction.repeat(repeats=joints.shape[1], axis=0)
        elif action in ["sit", "stand_up", "lie"]:
            direction = joints[-1, :, :] - joints[0, :, :]
        else:
            raise ValueError("Unexcepted action type.")
        direction_hint = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(direction, axis=1, keepdims=True).clip(0, 1) + 1e-8)
        return direction_hint
        
    def __getitem__(self, index):
        anno_index, offset = self.index2loc[index]

        motion_item = self.anno_motion_list[anno_index]
        trans = motion_item["translation"][offset:offset+self.num_frames].copy()
        orient = motion_item["orientation"][offset:offset+self.num_frames].copy()
        betas = motion_item["betas"].copy()
        body_pose = motion_item["body_pose"][offset:offset+self.num_frames].copy()
        hand_pose = motion_item["hand_pose"][offset:offset+self.num_frames].copy()

        anno_item = self.anno_list[anno_index]
        scene_id = anno_item["scene"]
        scene_info, scene_global_occ = self.scene_dict[scene_id]

        trans_local, orient_local, scene_local_occ, transf_rotmat, transf_transl = self._localize_motion_scene(trans, orient, body_pose, betas, scene_info, scene_global_occ)
        scene_local_occ_tensor = torch.tensor(scene_local_occ, dtype=torch.float)
        x_map1, x_map2 = self._decompx(scene_local_occ_tensor)
        y_map1, y_map2 = self._decompy(scene_local_occ_tensor)
        z_map1, _ = self._decompz(scene_local_occ_tensor, offset=10)
        scene_hint = {
            "scene_valid": torch.ones(1, dtype=torch.float32),
            "x_map1_depth": x_map1[0:1] * scene_info.voxel_size,
            "x_map2_depth": x_map2[0:1] * scene_info.voxel_size,
            "y_map1_depth": y_map1[0:1] * scene_info.voxel_size,
            "y_map2_depth": y_map2[0:1] * scene_info.voxel_size,
            "z_map1_depth": z_map1[0:1] * scene_info.voxel_size,
            "x_map1_rgb": x_map1[1:4],
            "x_map2_rgb": x_map2[1:4],
            "y_map1_rgb": y_map1[1:4],
            "y_map2_rgb": y_map2[1:4],
            "z_map1_rgb": z_map1[1:4],
            "x_map1_sem": x_map1[4:5].to(torch.int32),
            "x_map2_sem": x_map2[4:5].to(torch.int32),
            "y_map1_sem": y_map1[4:5].to(torch.int32),
            "y_map2_sem": y_map2[4:5].to(torch.int32),
            "z_map1_sem": z_map1[4:5].to(torch.int32),
        }
        if self.split in ["test"]:
            scene_hint["transf_rotmat"] = transf_rotmat
            scene_hint["transf_transl"] = transf_transl

        action = anno_item["action"].replace(" ", "_")
        motion_feature_69dim = torch.tensor(np.concatenate([orient_local, body_pose, trans_local], axis=1), dtype=torch.float, device=self.data_device).unsqueeze(0)
        joints = extract_smplx_from_feature(motion_feature_69dim, self.bodymodel_batch, return_type="joints").squeeze(0).detach().cpu().numpy()[:,:22,:]
        if self.split == "train":
            joint_hint = random_mask_train(joints, n_joints=22)
            case_idx = np.random.randint(3)
            if case_idx == 0:
                pass
            elif case_idx == 1:
                joint_hint[:1,:,:] = joints[:1,:,:]
            else:
                joint_hint[:5,:,:] = joints[:5,:,:]
            joint_hint = joint_hint.reshape(self.num_frames, -1)
        elif self.split == "test":
            joint_hint = joints.copy().reshape(self.num_frames, -1)
            joint_hint[1:,:] = 0
        else:
            raise NotImplementedError
        direction_hint = self._get_direction_hint_from_joints(joints, action).reshape(-1)

        inp = torch.tensor(np.concatenate([orient_local, body_pose, trans_local], axis=1), dtype=torch.float).unsqueeze(-1).permute(1, 2, 0).contiguous()
        output = {
            "inp": inp,
            "action": self.action2id[action],
            "action_text": action,
            "joint_hint": joint_hint,
            "direction_hint": direction_hint,
            "scene_hint": scene_hint,
        }

        return output

    def _decompx(
        self,
        local_scene: torch.Tensor,  # shape [C, X, Y, Z] = [5, 101, 101, 81]
        offset: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor]":  # shape [Cout, Y, Z] = [5, 101, 81]
        _, x_dim, _, _ = local_scene.shape
        plane = int((x_dim + 1)/2) + offset
        local_scene = torch.permute(local_scene, (0, 3, 2, 1))

        d = torch.linspace(1, plane, plane, dtype=torch.int32)
        scene_height = local_scene[3:4, :, :, :plane].clamp(0, 1) * d
        scene = torch.cat((plane - scene_height, local_scene[..., :plane]), dim=0)
        idx = scene_height.max(dim=-1)[1].expand(5, -1, -1).unsqueeze(-1)
        btm2plane = scene.gather(-1, idx).squeeze(-1).permute(0, 2, 1)
        btm2plane = torch.rot90(btm2plane, k=1, dims=[1, 2])

        d = torch.linspace(x_dim - plane + 1, 1, x_dim - plane + 1, dtype=torch.int32)
        scene_height = local_scene[3:4, :, :, plane - 1 :].clamp(0, 1) * d
        scene = torch.cat((x_dim - plane + 1 - scene_height, local_scene[..., plane - 1 :]), dim=0)
        idx = scene_height.max(dim=-1)[1].expand(5, -1, -1).unsqueeze(-1)
        top2plane = scene.gather(-1, idx).squeeze(-1).permute(0, 2, 1)
        top2plane = torch.rot90(top2plane, k=1, dims=[1, 2])

        return (btm2plane, top2plane)

    def _decompy(
        self,
        local_scene: torch.Tensor,  # shape [C, X, Y, Z] = [4, 101, 101, 81]
        offset: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor]":  # shape [Cout, X, Z] = [5, 101, 81]
        return self._decompx(local_scene.permute(0, 2, 1, 3), offset)

    def _decompz(
        self,
        local_scene: torch.Tensor,
        offset: int = 0,
    ) -> "tuple[torch.Tensor, torch.Tensor]":  # shape [Cout, X, Y] = [5, 101, 101]
        return self._decompx(local_scene.permute(0, 3, 1, 2), offset)

    def _localize_motion_scene(self, trans, orient, body_pose, betas, scene_info, scene_global_occ):
        pose = np.concatenate([orient, body_pose], axis=1)
        transf_rotmat, transf_transl = get_new_coordinate(self.bodymodel_one, betas, trans[:1, :], pose[:1, :], local_device=self.data_device)
        ### calibrate offset
        delta_T = calc_calibrate_offset(self.bodymodel_batch, betas, pose, local_device=self.data_device)
        ### get new global_orient
        global_ori = R.from_rotvec(pose[:,:3]).as_matrix() # to [t,3,3] rotation mat
        global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
        orient_local = R.from_matrix(global_ori_new).as_rotvec()
        ### get new trans
        trans_local = np.einsum('ij,tj->ti', transf_rotmat.T, trans+delta_T-transf_transl)-delta_T

        ### grid sample
        transf_rotmat = torch.tensor(transf_rotmat, dtype=torch.float)
        transf_transl = torch.tensor(transf_transl, dtype=torch.float)
        bounds_negative = torch.tensor(local_grid_info['bounds_negative'])
        bounds_positive = torch.tensor(local_grid_info['bounds_positive'])
        grid_steps = torch.tensor(local_grid_info['grid_steps'])
        scene = torch.from_numpy(scene_global_occ.scene).unsqueeze(0).permute(0, 4, 3, 2, 1).float()

        xyz = map(
            lambda x, y, step: torch.linspace(x, y, step), -bounds_negative, bounds_positive, grid_steps
        )
        grids = torch.stack(torch.meshgrid(*xyz, indexing="ij"), dim=-1)
        grids = grids * default_voxel_size
        grids.unsqueeze_(0)
        grids = torch.einsum("chwdt,kt->chwdk", grids, transf_rotmat)
        grids = grids + transf_transl
        #grids = grids - (torch.tensor(scene_info.min_bound) + torch.tensor(scene_info.max_bound)) / 2 # refer _prepare_motion_data_list function where occ_center is calculated from human motion
        grids_scale = 2 * (torch.tensor(scene_info.grid_size)) / (torch.tensor(scene_info.grid_size) - 1) / (torch.tensor(scene_info.max_bound) - torch.tensor(scene_info.min_bound))
        grids = grids * grids_scale
        grids = grids.to(torch.float32)

        scene_local_occ = torch.nn.functional.grid_sample(scene, grids, mode="nearest", padding_mode="zeros", align_corners=True)[0].cpu().numpy()
        """
        ### debug occ
        from tools.occ_tools import export_local_occ
        from tools.io_tools import export_batch_human_mesh
        poses = np.concatenate([trans_local[0:1,:], orient_local[0:1,:], betas.reshape(1, -1), body_pose[0:1,:], np.zeros((1, 24))], axis=1)
        export_local_occ(scene_local_occ, 0.04, "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/visualization_results/figs/intro")
        export_batch_human_mesh(poses, ["/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/visualization_results/figs/intro/000000_test.obj"])
        exit()
        """

        return trans_local, orient_local, scene_local_occ, transf_rotmat, transf_transl

    def __len__(self):
        return len(self.index2loc)

if __name__ == "__main__":
    humanise = HumaniseMotion()
    data_item = humanise[0]
    semantic_map = data_item["scene_hint"]["z_map1_sem"].squeeze(0).numpy()
    label_global2local = CategoryLabelConverter(src_semantic_anno_type="global", tgt_semantic_anno_type="scannet")
    semantic_map = label_global2local.convert_semantic_map_src2tgt(semantic_map)
    semantic_map = humanise.label_semantics.label2color2d(semantic_map)
    import matplotlib.pyplot as plt
    plt.imshow(semantic_map)
    plt.show()
