import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from tools.project_config_tools import default_voxel_size, local_grid_info 
from tools.coordinate_tools import calc_calibrate_offset, get_new_coordinate

class BasicScene(object):
    def __init__(
            self,
            ):
        return

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
        grids = grids - (torch.tensor(scene_info.min_bound) + torch.tensor(scene_info.max_bound)) / 2
        grids_scale = 2 * (torch.tensor(scene_info.grid_size)) / (torch.tensor(scene_info.grid_size) - 1) / (torch.tensor(scene_info.max_bound) - torch.tensor(scene_info.min_bound))
        grids = grids * grids_scale
        grids = grids.to(torch.float32)

        scene_local_occ = torch.nn.functional.grid_sample(scene, grids, mode="nearest", padding_mode="zeros", align_corners=True)[0].cpu().numpy()

        return trans_local, orient_local, scene_local_occ

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
