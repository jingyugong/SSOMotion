import torch
import torch.nn.functional as F
from .gaussian_diffusion import _extract_into_tensor
from .respace import SpacedDiffusion
from tools.io_tools import load_markers
from tools.motion_tools import load_body_mesh_model_batch, extract_smplx_from_feature

class SceneDiffusion(SpacedDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        super().__init__(use_timesteps, **kwargs)
        self.guidance_loss_config = {
            'walk': {
                'hint_weight': 1.,
                'xy_hint_weight': 1.,
                'skating_weight': 1e-2,
                'penetration_weight': 0.,
                'foot_contact_weight': 1e-1,
            },
            'sit': {
                'hint_weight': 1.,
                'xy_hint_weight': 1.,
                'smoothness_weight': 1e-3,
                'skating_weight': 3e-4,
                'penetration_weight': 1e3,
            },
            'stand_up': {
                'hint_weight': 1.,
                'xy_hint_weight': 1.,
                'smoothness_weight': 1e-3,
                'skating_weight': 3e-4,
                'penetration_weight': 1e3,
            },
            'lie': {
                'hint_weight': 1e2,
                'xy_hint_weight': 1.,
                'smoothness_weight': 1e-3,
                'penetration_weight': 1e2,
            },
        }
        self.smoothness_acc_tol = 50.
        self.skating_vel_tol = 0.5
        self.scene_pene_tol = 0.
        self.foot_contact_tol = 0.01
        self.body_contact_tol = 0.03

    def indices_from_markers(self, markers, marker_type='SSM2'):
        if not hasattr(self, 'marker_indices'):
            self.marker_indices = load_markers(marker_type=marker_type)['markersets'][0]['indices']
        indices = []
        for marker in markers:
            indices.append(self.marker_indices[marker])
        return indices 

    def obtain_all_marker_indices(self, marker_type='SSM2'):
        if not hasattr(self, 'marker_indices'):
            self.marker_indices = load_markers(marker_type=marker_type)['markersets'][0]['indices']
        return list(self.marker_indices.values())

    def set_foot_indices(self, device):
        if not hasattr(self, 'left_foot_indices'):
            left_foot_markers = ['LHEE', 'LTOE', 'LRSTBEEF']
            self.left_foot_indices = torch.tensor(self.indices_from_markers(left_foot_markers), device=device)
        if not hasattr(self, 'right_foot_indices'):
            right_foot_markers = ['RHEE', 'RTOE', 'RRSTBEEF']
            self.right_foot_indices = torch.tensor(self.indices_from_markers(right_foot_markers), device=device)
        if not hasattr(self, 'left_foot_joint_indices'):
            self.left_foot_joint_indices = torch.tensor([7, 10], device=device)
        if not hasattr(self, 'right_foot_joint_indices'):
            self.right_foot_joint_indices = torch.tensor([8, 11], device=device)
        return

    def calc_skating_loss(self, joints, vertices, guidance_loss_config, fps=30):
        use_vertices = False
        use_joints = True
        self.set_foot_indices(vertices.device)
        ret = 0.
        if use_vertices:
            foot_indices = torch.cat([self.left_foot_indices, self.right_foot_indices], dim=0)
            foot_vertices = vertices[:, :, foot_indices, :]
            foot_velocity = torch.norm(foot_vertices[:,1:,:,:]-foot_vertices[:,:-1,:,:], dim=-1) * fps
            min_foot_velocity = foot_velocity.min(dim=-1)[0]
            ret += torch.relu(min_foot_velocity-self.skating_vel_tol).mean()
        if use_joints:
            foot_joint_indices = torch.cat([self.left_foot_joint_indices, self.right_foot_joint_indices], dim=0)
            foot_joints = joints[:, :, foot_joint_indices, :]
            foot_velocity = torch.norm(foot_joints[:,1:,:,:]-foot_joints[:,:-1,:,:], dim=-1) * fps
            min_foot_velocity = foot_velocity.min(dim=-1)[0]
            ret += torch.relu(min_foot_velocity-self.skating_vel_tol).mean()
        return ret * guidance_loss_config['skating_weight']

    def calc_smoothness_loss(self, joints, guidance_loss_config, fps=30):
        velocity = (joints[:,1:,:,:]-joints[:,:-1,:,:]) * fps
        acceleration = (velocity[:,1:,:,:]-velocity[:,:-1,:,:]) * fps
        ret = torch.relu(acceleration - self.smoothness_acc_tol).mean()
        return ret * guidance_loss_config['smoothness_weight']

    def calc_body_penetration_contact_loss(self, vertices, scene_hint, guidance_loss_config):
        """
        vertices shape: (batch_size, num_frames, num_vertices, 3)
        scene_hint['scene_sdf'] shape: (batch_size, channel, d1, d2, d3)
        scene_hint['scene_sdf_centroid'] shape: (batch_size, 3)
        scene_hint['scene_sdf_scale'] shape: (batch_size) or (batch_size, 3)
        scene_hint['transf_rotmat'] shape: (batch_size, 3, 3)
        scene_hint['transf_transl'] shape: (batch_size, 1, 3)
        """
        assert vertices.dim() == 4
        assert scene_hint['scene_sdf'].dim() == 5
        assert scene_hint['scene_sdf_centroid'].dim() == 2
        assert scene_hint['scene_sdf_scale'].dim() in [1, 2]
        assert scene_hint['transf_rotmat'].dim() == 3
        assert scene_hint['transf_transl'].dim() == 3
        batch_size, n_frames, _, _ = vertices.shape
        assert scene_hint['scene_sdf'].shape[0] == batch_size
        assert scene_hint['scene_sdf_centroid'].shape[0] == batch_size
        assert scene_hint['scene_sdf_scale'].shape[0] == batch_size
        assert scene_hint['transf_rotmat'].shape[0] == batch_size
        assert scene_hint['transf_transl'].shape[0] == batch_size
        if scene_hint['scene_sdf_scale'].dim() == 2:
            assert scene_hint['scene_sdf_scale'].shape[1] == 3
            scale_last_dim_len = 3
        else:
            scale_last_dim_len = 1

        for k in scene_hint.keys():
            if scene_hint[k].device != vertices.device:
                scene_hint[k] = scene_hint[k].to(vertices.device)
        if not hasattr(self, 'all_marker_indices'):
            self.all_marker_indices = self.obtain_all_marker_indices(marker_type='SSM2')
        marker_vertices = vertices[:, :, self.all_marker_indices, :]
        marker_vertices = torch.matmul(marker_vertices, scene_hint['transf_rotmat'].view(batch_size,1,3,3).transpose(2,3)) + scene_hint['transf_transl'].view(batch_size,1,1,3)
        marker_vertices = (marker_vertices - scene_hint['scene_sdf_centroid'].view(batch_size, 1, 1, 3))/scene_hint['scene_sdf_scale'].view(batch_size, 1, 1, scale_last_dim_len)
        _, _, n_markers, _ = marker_vertices.shape
        sdf_values = F.grid_sample(
                scene_hint['scene_sdf'],
                marker_vertices[:,:,:,[2,1,0]].unsqueeze(3),
                padding_mode='border',
                align_corners=True,
        ).view(batch_size, n_frames, n_markers)
        ret = 0.
        if 'penetration_weight' in guidance_loss_config:
            ret += torch.relu(-sdf_values-self.scene_pene_tol).mean() * guidance_loss_config['penetration_weight']
        if 'body_contact_weight' in guidance_loss_config:
            min_sdf_values = sdf_values.min(dim=-1)[0]
            ret += torch.relu(torch.abs(min_sdf_values)-self.body_contact_tol).mean() * guidance_loss_config['body_contact_weight']
        return ret

    def calc_foot_contact_loss(self, vertices, scene_hint, guidance_loss_config):
        batch_size, _, _, _ = vertices.shape
        for k in scene_hint.keys():
            if scene_hint[k].device != vertices.device:
                scene_hint[k] = scene_hint[k].to(vertices.device)
        self.set_foot_indices(vertices.device)
        foot_indices = torch.cat([self.left_foot_indices, self.right_foot_indices], dim=0)
        foot_vertices = vertices[:, :, foot_indices, :]
        foot_vertices = torch.matmul(foot_vertices, scene_hint['transf_rotmat'].view(batch_size, 1, 3, 3).transpose(2,3)) + scene_hint['transf_transl'].view(batch_size,1,1,3)
        foot_vertex_heights = foot_vertices[:, :, :, 2]
        lowest_vertices = foot_vertex_heights.min(dim=-1)[0]
        ret = torch.relu((lowest_vertices-scene_hint['floor_height']).abs()-self.foot_contact_tol).mean()
        return ret * guidance_loss_config['foot_contact_weight']

    def gradients(self, x, t, model, model_kwargs, all_hint, via_x_t=False):
        guidance_loss_config = self.guidance_loss_config[model_kwargs['y']['action_text'][0]]
        with torch.enable_grad():
            x.requires_grad_(True)
            if not via_x_t and t[0] > 0:
                model_output = model(x, self._scale_timesteps(t-1), **model_kwargs)
                x_ = model_output
            else:
                x_ = x

            x_ = x_.permute(0, 3, 2, 1).contiguous()
            x_ = x_.squeeze(2)
            if x_.shape[2] == 69:
                n_joints = 22
            else:
                raise NotImplementedError
            batch_size, num_frames, _ = x_.shape
            smplx_output = extract_smplx_from_feature(x_, self.body_mesh_model_batch)
            joints = smplx_output.joints.view(batch_size, num_frames, -1, 3)[:,:,:n_joints,:]
            vertices = smplx_output.vertices.view(batch_size, num_frames, -1, 3) 

            loss = 0.
            if 'hint_weight' in guidance_loss_config and guidance_loss_config['hint_weight'] > 0. and 'joint_hint' in all_hint:
                loss += torch.norm((joints - all_hint['joint_hint']) * all_hint['mask_hint'], dim=-1).mean() * guidance_loss_config['hint_weight']
            if 'xy_hint_weight' in guidance_loss_config and guidance_loss_config['xy_hint_weight'] > 0. and 'xy_hint' in all_hint:
                loss += torch.norm((joints[...,:2] - all_hint['xy_hint'][...,:2]) * all_hint['mask_xy_hint'], dim=-1).mean() * guidance_loss_config['xy_hint_weight']
            if 'smoothness_weight' in guidance_loss_config and guidance_loss_config['smoothness_weight'] > 0.:
                loss += self.calc_smoothness_loss(joints, guidance_loss_config)
            if 'skating_weight' in guidance_loss_config and guidance_loss_config['skating_weight'] > 0.:
                loss += self.calc_skating_loss(joints, vertices, guidance_loss_config)
            if ('penetration_weight' in guidance_loss_config and guidance_loss_config['penetration_weight'] > 0.) or ('body_contact_weight' in guidance_loss_config and guidance_loss_config['body_contact_weight'] > 0.):
                loss += self.calc_body_penetration_contact_loss(vertices, model_kwargs['y']['scene_hint'], guidance_loss_config)
            if 'foot_contact_weight' in guidance_loss_config and guidance_loss_config['foot_contact_weight'] > 0.:
                loss += self.calc_foot_contact_loss(vertices, model_kwargs['y']['scene_hint'], guidance_loss_config)
            if isinstance(loss, torch.Tensor):
                grad = torch.autograd.grad([loss], [x])[0]
            elif isinstance(loss, float):
                grad = torch.zeros_like(x)
            else:
                raise NotImplementedError
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            #grad[..., 0] = 0
            x.detach()
        return loss, grad

    def guide(self, x, t, model=None, model_kwargs=None, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        if x.shape[1] == 69:
            n_joint = 22
        else:
            raise NotImplementedError
        if not hasattr(self, "body_mesh_model_batch"):
            self.body_mesh_model_batch = load_body_mesh_model_batch(x.shape[0]*x.shape[3]).eval().to(x.device)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 0
            else:
                n_guide_steps = 1

        # process hint
        all_hint = {}
        if 'guidance_hint' in model_kwargs['y']:
            if model_kwargs['y']['guidance_hint'].device != x.device:
                model_kwargs['y']['guidance_hint'] = model_kwargs['y']['guidance_hint'].to(x.device)
            hint = model_kwargs['y']['guidance_hint'].clone().detach()
            mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).abs().sum(dim=-1, keepdim=True) != 0
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint
            all_hint['joint_hint'] = hint
            all_hint['mask_hint'] = mask_hint
        if 'xy_guidance_hint' in model_kwargs['y']:
            if model_kwargs['y']['xy_guidance_hint'].device != x.device:
                model_kwargs['y']['xy_guidance_hint'] = model_kwargs['y']['xy_guidance_hint'].to(x.device)
            xy_hint = model_kwargs['y']['xy_guidance_hint'].clone().detach()
            mask_xy_hint = xy_hint.view(xy_hint.shape[0], xy_hint.shape[1], n_joint, 3).abs().sum(dim=-1, keepdim=True) != 0
            xy_hint = xy_hint.view(xy_hint.shape[0], xy_hint.shape[1], n_joint, 3) * mask_xy_hint
            all_hint['xy_hint'] = xy_hint
            all_hint['mask_xy_hint'] = mask_xy_hint
        
        if not train:
            if 'mask_hint' in all_hint and 'mask_xy_hint' in all_hint:
                scale = min(self.calc_grad_scale(all_hint['mask_hint']), self.calc_grad_scale(all_hint['mask_xy_hint']))
            elif 'mask_hint' in all_hint:
                scale = self.calc_grad_scale(all_hint['mask_hint'])
            elif 'mask_xy_hint' in all_hint:
                scale = self.calc_grad_scale(all_hint['mask_xy_hint'])
            else:
                raise KeyError("mask_hint and mask_xy_hint not in all_hint, please set hint or xy_hint")

        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, t, model, model_kwargs, all_hint, via_x_t=self.guide_via_x_t)
            grad = model_variance * grad
            # print(loss.sum())
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        return x.detach()
