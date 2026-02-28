import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
sys.path.append("..")
import smplx
from utils import dist_util
from tools.motion_tools import load_body_mesh_model_batch, time_variant_motion_updating, extract_smplx_from_feature

#multi-round motion process tools
from tools.coordinate_tools import get_new_coordinate, point_coordinate_transform, motion_coordinate_transform, coordinate_inv_transform
from tools.guidance_tools import wpath2hints
from tools.io_tools import extract_smplx_joints
from tools.model_tools import pack_model_kwargs
from tools.project_config_tools import host_device, action_name_to_action

class MultiRoundMotionAgent:
    def __init__(
            self,
            demo_data,
            model,
            diffusion,
            single_round_n_frames=30,
            max_frames=30,
            controlnet=True,
            inpainting=False,
            mdm_guidance_param=1,
            mdm_batch_size=1,
            n_his_frames=5,
            last_pose_max_lasting_frames=1,
            compile_model=False,
        ):
        self.demo_data = demo_data
        self.scene_global_occ = self.demo_data['scene_hints']['scene_global_occ']
        del self.demo_data['scene_hints']['scene_global_occ']
        self.gender = self.demo_data['gender']
        self.betas = self.demo_data['init_betas']
        self.model = model
        if compile_model:
            self.model = torch.compile(self.model, mode='reduce-overhead')
        self.diffusion = diffusion
        self.single_round_n_frames = single_round_n_frames
        self.max_frames = max_frames
        self.controlnet = controlnet
        self.inpainting = inpainting
        self.mdm_guidance_param = mdm_guidance_param
        self.mdm_batch_size = mdm_batch_size
        self.n_his_frames = n_his_frames
        self.last_pose_max_lasting_frames = last_pose_max_lasting_frames
        self.task_finished = False
        self.task_schedule = self.obtain_task_schedule(self.demo_data)
        self.task_idx = 0
        self.curr_motion = np.concatenate([self.demo_data['init_pose'], self.demo_data['init_transl']], axis=1)
        self.sample_text_seq = []
        self.map_fine_state_to_lasting = {
            'walk':1,
            'sit_up':1,
            'sit_down':10,
            'lie_up':1,
            'lie_down':10,
            'lie_down_from_stand':10,
        }
        self.map_fine_state_to_coarse_state = {
            'walk': 'walk',
            'sit_up': 'stand_up',
            'sit_down': 'sit',
            'lie_up': 'lie',
            'lie_down': 'lie',
            'lie_down_from_stand': 'lie',
        }

        self.bodymodel_one = load_body_mesh_model_batch(1, gender=self.gender, device=host_device)
        if compile_model:
            self.bodymodel_one = torch.compile(self.bodymodel_one, mode='reduce-overhead')
        return

    def obtain_task_schedule(self, demo_data):
        task_schedule = []
        task_idx_schedule = []
        wpath = demo_data['wpath']
        wpath_poset = demo_data['wpath_poset']
        wpath_action = demo_data['wpath_action']
        curr_list = [0]
        state = wpath_action[0]
        for idx in range(1, len(wpath_action)):
            if wpath_action[idx] == state:
                curr_list.append(idx)
            else:
                if len(curr_list) > 1:
                    task_idx_schedule.append((state, curr_list))
                curr_list = [idx-1, idx]
                task_idx_schedule.append((self.judge_action(state, wpath_action[idx]), curr_list))

                curr_list = [idx]
                state = wpath_action[idx]
        if len(curr_list) > 1:
            task_idx_schedule.append((state, curr_list))
        for state, curr_list in task_idx_schedule:
            task_schedule_item = {}
            task_schedule_item['state'] = state
            task_schedule_item['wpath'] = wpath[curr_list[0]:curr_list[-1]+1]
            task_schedule_item['wpath_poset'] = wpath_poset[curr_list[0]:curr_list[-1]+1]
            task_schedule_item['wpath_action'] = wpath_action[curr_list[0]:curr_list[-1]+1]
            task_schedule.append(task_schedule_item)
        return task_schedule

    def judge_action(self, state1, state2):
        assert state1 != state2
        if state1 == 'walk':
            if state2 == 'sit':
                ret = 'sit_down'
            elif state2 == 'lie':
                ret = "lie_down_from_stand"
            else:
                raise ValueError
        elif state1 == 'sit':
            if state2 == 'walk':
                ret = 'sit_up'
            elif state2 == 'lie':
                ret = 'lie_down'
            else:
                raise ValueError
        elif state1 == 'lie':
            ret = 'lie_up'
        else:
            raise ValueError 
        return ret

    def fetch_local_occ_hint(self, poset):
        assert poset.shape[1] == 69
        global_orient = poset[:1, :3].copy()
        body_pose = poset[:1, 3:66].copy()
        transl = poset[:1, 66:69].copy()
        human_pose = np.concatenate([transl, global_orient, body_pose], axis=-1)
        local_occ = self.scene_global_occ.fetch_local_scene_occ_hint(human_pose, self.betas, self.gender)
        for k, v in local_occ.items():
            local_occ[k] = v.to(host_device)
        return local_occ

    def fetch_direction_hints(self, state, hints, poset_hints, poset_list):
        if state in ["walk"]:
            idx = np.argmax((np.sum(np.abs(hints).reshape(self.single_round_n_frames, -1), axis=-1) > 1e-8) * np.arange(1, self.single_round_n_frames+1, 1))
            direction = hints[idx:idx+1,0,:] - hints[0:1,0,:]
            direction = direction.repeat(repeats=hints.shape[1], axis=0)
        elif state in ["sit", "stand_up", "lie"]:
            src_poset = poset_hints[0].copy().reshape(1, -1)
            tgt_poset = poset_list[-1].copy().reshape(1, -1)
            src_joints = extract_smplx_joints(np.concatenate([src_poset, self.betas.reshape(1, -1)], axis=-1), self.gender)[:, :22, :]
            tgt_joints = extract_smplx_joints(np.concatenate([tgt_poset, self.betas.reshape(1, -1)], axis=-1), self.gender)[:, :22, :]
            direction = tgt_joints[0] - src_joints[0]
        direction_hints = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8) * (np.linalg.norm(direction, axis=1, keepdims=True).clip(0, 1) + 1e-8)
        return direction_hints

    def next_iter_parameters(self):
        all_parameters = {}
        curr_task = self.task_schedule[self.task_idx]
        curr_task_fine_state = curr_task['state']
        curr_task_state = self.map_fine_state_to_coarse_state[curr_task_fine_state]

        self.curr_n_his_frames = min(self.n_his_frames, self.curr_motion.shape[0])
        self.ref_coordinate_poset = self.curr_motion[-self.curr_n_his_frames, :].reshape(1, -1)
        self.curr_transf_rotmat, self.curr_transf_transl = get_new_coordinate(self.bodymodel_one, self.demo_data['init_betas'], self.ref_coordinate_poset[:,66:], self.ref_coordinate_poset[:,:66])
        self.curr_transf_inv_rotmat, self.curr_transf_inv_transl = coordinate_inv_transform(self.curr_transf_rotmat, self.curr_transf_transl)

        his_motion_w = self.curr_motion[-self.curr_n_his_frames:,:]
        his_transl_l, his_pose_l = motion_coordinate_transform(self.gender, self.betas, his_motion_w[:,66:], his_motion_w[:,:66], self.curr_transf_rotmat, self.curr_transf_transl)
        his_joints_l = extract_smplx_joints(np.concatenate([his_pose_l, his_transl_l, self.betas.reshape(1,-1).repeat(self.curr_n_his_frames,axis=0)], axis=1), gender=self.gender)[:,:22,:]
        his_poset_l = np.concatenate([his_pose_l, his_transl_l], axis=1)

        wpath_w = curr_task['wpath'].detach().cpu().numpy()
        wpath_l = point_coordinate_transform(wpath_w, self.curr_transf_rotmat, self.curr_transf_transl)
        wpath_poset_w = curr_task['wpath_poset']
        wpath_poset_valid = [x is not None for x in wpath_poset_w]
        wpath_poset_filled_w = np.concatenate([x if x is not None else np.zeros((1, 69), dtype=np.float32) for x in wpath_poset_w], axis=0)
        wpath_transl_filled_l, wpath_pose_filled_l = motion_coordinate_transform(self.gender, self.betas, wpath_poset_filled_w[:,66:], wpath_poset_filled_w[:,:66], self.curr_transf_rotmat, self.curr_transf_transl)
        wpath_joints_filled_l = extract_smplx_joints(np.concatenate([wpath_pose_filled_l, wpath_transl_filled_l, self.betas.reshape(1,-1).repeat(len(wpath_poset_w),axis=0)], axis=1), gender=self.gender)[:,:22,:]
        wpath_poset_filled_l = np.concatenate([wpath_pose_filled_l, wpath_transl_filled_l], axis=1)
        wpath_joints_l = [x if valid else None for x, valid in zip(wpath_joints_filled_l, wpath_poset_valid)]
        wpath_poset_l = [x if valid else None for x, valid in zip(wpath_poset_filled_l, wpath_poset_valid)]

        xy_guidance_hints = np.zeros((self.single_round_n_frames, 22, 3))
        guidance_hints = np.zeros((self.single_round_n_frames, 22, 3))
        hints = np.zeros((self.single_round_n_frames, 22, 3)) 
        poset_hints = np.zeros((self.single_round_n_frames, 69))
        if curr_task_state in ['walk', 'sit', 'stand_up', 'lie']:
            res_local_wpath = np.concatenate([his_joints_l[-1:,0,:], wpath_l[:,:]], axis=0)
            res_local_wpath_joints = [his_joints_l[-1]] + wpath_joints_l[:]
            res_local_wpath_poset = [his_poset_l[-1]] + wpath_poset_l[:]
            res_xy_guidance_hints, res_guidance_hints, res_hints, res_poset_hints, res_wpath_frame_idx, res_wpath  = wpath2hints(res_local_wpath, res_local_wpath_joints, res_local_wpath_poset, curr_task_state, self.single_round_n_frames-self.curr_n_his_frames+1, last_pose_lasting_frames=min(self.map_fine_state_to_lasting[curr_task_fine_state], self.last_pose_max_lasting_frames))
            wpath_frame_idx = [x+self.curr_n_his_frames-1 for x in res_wpath_frame_idx]

            xy_guidance_hints[self.curr_n_his_frames-1:,:,:] = res_xy_guidance_hints
            guidance_hints[self.curr_n_his_frames-1:,:,:] = res_guidance_hints
            hints[self.curr_n_his_frames-1:,:,:] = res_hints
            poset_hints[self.curr_n_his_frames-1:,:] = res_poset_hints
            guidance_hints[0:self.curr_n_his_frames,:,:] = his_joints_l
            poset_hints[0:self.curr_n_his_frames,:] = his_poset_l
            hints[0,0,:] = his_joints_l[0,0,:]
            direction_hints = self.fetch_direction_hints(curr_task_state, hints, poset_hints, res_local_wpath_poset)
        else:
            raise NotImplementedError

        self.curr_n_keep_frames = wpath_frame_idx[-1]+1
        if res_wpath.shape[0] == 0:
            self.task_idx += 1
            if self.task_idx >= len(self.task_schedule):
                self.task_finished = True
        else:
            n_finished_wpath_nodes = curr_task = self.task_schedule[self.task_idx]['wpath'].shape[0] - res_wpath.shape[0]
            self.task_schedule[self.task_idx]['wpath'] = self.task_schedule[self.task_idx]['wpath'][n_finished_wpath_nodes:]
            self.task_schedule[self.task_idx]['wpath_poset'] = self.task_schedule[self.task_idx]['wpath_poset'][n_finished_wpath_nodes:]

        action_text = [curr_task_state]
        action = [action_name_to_action[tmp] for tmp in action_text]
        scene_hints = self.demo_data['scene_hints']
        scene_hints['transf_rotmat'] = self.curr_transf_rotmat
        scene_hints['transf_transl'] = self.curr_transf_transl
        scene_occ_hint = self.fetch_local_occ_hint(self.ref_coordinate_poset)
        scene_hints.update(scene_occ_hint)
        if not self.inpainting:
            poset_hints = None
        model_kwargs = pack_model_kwargs(hints, xy_guidance_hints, guidance_hints, poset_hints, direction_hints, scene_hints, action_text, action, self.single_round_n_frames)

        all_parameters['model_kwargs'] = model_kwargs
        return all_parameters

    def update_single_round_motion(
            self,
            motion_single_round,
        ):
        assert motion_single_round['sample'].shape[0] == 1 #batch inference is not supported
        sample = motion_single_round['sample'][0]
        self.sample_text_seq.extend(motion_single_round['sample_text'])
        sample = sample[:self.curr_n_keep_frames,:]
        self.curr_motion = time_variant_motion_updating(self.curr_motion, sample, self.curr_n_his_frames)
        return

    def infer_single_round(
            self,
            model_kwargs,
        ):
        all_text = []
        sample_fn = self.diffusion.p_sample_loop

        sample_l = sample_fn(
            self.model,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (self.mdm_batch_size, self.model.njoints, self.model.nfeats, self.max_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # Recover XYZ *positions* from HumanML3D vector representation
        if self.model.data_rep == 'mixed_vec':
            sample_l = sample_l.permute(0, 3, 1, 2).squeeze(-1)
        else:
            raise NotImplementedError

        sample_l = sample_l.cpu().numpy()
        sample_transl_l, sample_pose_l = sample_l[:, :, 66:].reshape(-1, 3), sample_l[:, :, :66].reshape(-1, 66)
        sample_transl_w, sample_pose_w = motion_coordinate_transform(self.gender, self.betas, sample_transl_l, sample_pose_l, self.curr_transf_inv_rotmat, self.curr_transf_inv_transl)
        sample_w = np.concatenate([sample_pose_w, sample_transl_w], axis=1).reshape(*sample_l.shape)
        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        sample_text = model_kwargs['y'][text_key]

        ret = {
            'sample': sample_w,
            'sample_text': sample_text,
            #'lengths': model_kwargs['y']['lengths'].cpu().numpy(),
        }
        return ret

    def kwargs_to_host_device(self, model_kwargs):
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor) and v.device != host_device:
                model_kwargs[k] = v.to(host_device)
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor) and vv.device != host_device:
                        v[kk] = vv.to(host_device)
        return model_kwargs

    def run_one_loop(self):
        while not self.task_finished:
            print(self.task_idx)
            all_parameters = self.next_iter_parameters()
            model_kwargs = all_parameters['model_kwargs']
            if self.mdm_guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(self.mdm_batch_size, device=dist_util.dev()) * self.mdm_guidance_param
            model_kwargs = self.kwargs_to_host_device(model_kwargs)
            motion_single_round = self.infer_single_round(model_kwargs)
            self.update_single_round_motion(motion_single_round)
            """
            self.curr_motion = motion_single_round['sample']
            self.sample_text_seq.extend(motion_single_round['sample_text'])
            self.task_finished = True
            """
        return motion_single_round

    def get_results(self):
        ret = {
            'sample': self.curr_motion[None,...].astype(np.float32),
            'sample_text': '_'.join(self.sample_text_seq),
            'sample_lengths': [self.curr_motion.shape[-2]],
        }
        return ret

    def reset_agent(self):
        self.task_finished = False
        self.task_schedule = self.obtain_task_schedule(self.demo_data)
        self.task_idx = 0
        self.curr_motion = np.concatenate([self.demo_data['init_pose'], self.demo_data['init_transl']], axis=1)
        self.sample_text_seq = []
        return

class MultiRoundMotionAgentforEval(MultiRoundMotionAgent):
    def get_results(self):
        self.curr_motion = self.curr_motion.astype(np.float32)
        device = host_device
        action_name = 'walk' 
        for task in self.task_schedule:
            for task_action in task['wpath_action']:
                if task_action != 'walk':
                    action_name = task_action
                    break
        if action_name == 'walk':
            goal_point = self.task_schedule[-1]['wpath'][-1].detach().cpu().numpy()
            batch_size, _ = self.curr_motion.shape
            body_mesh_model_batch = load_body_mesh_model_batch(batch_size, body_type='smplx', gender=self.gender, device=device)
            data = torch.tensor(self.curr_motion, device=device).unsqueeze(0)
            betas = torch.tensor(self.betas, device=device).view(1, -1).repeat(batch_size, 1)
            joints = extract_smplx_from_feature(data, body_mesh_model_batch, return_type='joints', betas=betas).detach().cpu().numpy()[0,:,:55,:]
            dist_xy = np.linalg.norm((joints-goal_point)[:,:,:2], axis=2).min(axis=1)
            idx = np.argmin(dist_xy)
            self.curr_motion = self.curr_motion[:idx+1]
        ret = {
            'sample': self.curr_motion[None,...],
            'sample_text': '_'.join(self.sample_text_seq),
            'sample_lengths': [self.curr_motion.shape[-2]],
        }
        return ret
