import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
sys.path.append("..")
import smplx
import trimesh
from tools.project_config_tools import project_dir
from utils import dist_util

#body process tools
def load_template_pose(action):
    ret = np.load(project_dir + "/assets/template_poses/" + action + ".npy").astype(np.float32)
    return ret

def load_body_mesh_model_batch(batch_size, body_type='smplx', gender='neutral', model_dir=os.path.join(project_dir,'body_models'), device='cpu'):
    body_mesh_model_batch = smplx.create(model_dir, 
                                   model_type=body_type,
                                   gender=gender, ext='npz',
                                   num_pca_comps=12,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=batch_size
                                   )
    if device == 'cuda':
        return body_mesh_model_batch.cuda()
    else:
        return body_mesh_model_batch

def extract_smplx_from_feature(data, body_mesh_model_batch, return_type='output', betas=None):
    batch_size, num_frames, feature_dim = data.shape
    data = data.view(batch_size*num_frames, feature_dim)
    body_param_rec = {}
    if feature_dim == 69:
        body_param_rec['transl'] = data[:, 66:]
        body_param_rec['global_orient'] = data[:, 0:3]
        body_param_rec['body_pose'] = data[:, 3:66]
        if betas is None:
            body_param_rec['betas'] = torch.zeros((batch_size*num_frames, 10), dtype=torch.float32, device=data.device)
        else:
            pass
    else:
        raise NotImplementedError
    smplx_out = body_mesh_model_batch(return_verts=True, **body_param_rec)
    if return_type == 'output':
        ret = smplx_out
    elif return_type == 'joints':
        ret = smplx_out.joints
        _, num_joints, _ = ret.shape
        ret = ret.view(batch_size, num_frames, num_joints, 3)
    elif return_type == 'vertices':
        ret = smplx_out.vertices
        _, num_vertices, _ = ret.shape
        ret = ret.view(batch_size, num_frames, num_vertices, 3)
    else:
        raise NotImplementedError
    return ret

def get_global_orient_from_forward_z_up(forward):
    forward_dir = forward.copy()
    forward_dir[2] = 0
    forward_dir = forward_dir / np.linalg.norm(forward_dir)
    up = np.array([0, 0, 1])
    rotmat = np.stack([np.cross(np.array([0, 0, 1]), forward_dir), np.array([0, 0, 1]), forward_dir]).T
    global_orient = R.from_matrix(rotmat[None,...]).as_rotvec().astype(np.float32)
    return global_orient.reshape(-1)

def place_stand_pose_on_floor(gender, betas, global_orient, target_point, floor_height):
    body_model_one_batch = load_body_mesh_model_batch(1, gender=gender)
    stand_pose = load_template_pose('stand').reshape(-1, 63)
    betas = betas.reshape(-1, 10)
    global_orient = global_orient.reshape(-1, 3)
    transl = np.zeros((1, 3), dtype=np.float32)
    smplx_params = {'global_orient': torch.tensor(global_orient), 'body_pose': torch.tensor(stand_pose), 'betas': torch.tensor(betas), 'transl': torch.tensor(transl)}
    smplx_out = body_model_one_batch(**smplx_params)
    pelvis_joint = smplx_out.joints.detach().numpy()[0,0,:]
    transl[0,:2] = target_point[:2] -pelvis_joint[:2]
    transl[0,2] = floor_height - np.min(smplx_out.vertices.detach().numpy()[0,:,2])
    stand_pose = stand_pose.reshape(-1)
    transl = transl.reshape(-1)
    pelvis_joint = pelvis_joint + transl
    return stand_pose, transl, pelvis_joint

def place_stand_pose_near_lie(navmesh_path, poset, betas, floor_height, action_order, gender="neutral"):
    body_model_one_batch = load_body_mesh_model_batch(1, gender=gender)
    smplx_params = {'global_orient': torch.tensor(poset[:,:3]), 'body_pose': torch.tensor(poset[:,3:66]), 'betas': torch.tensor(betas), 'transl': torch.tensor(poset[:,66:69])}
    smplx_out = body_model_one_batch(**smplx_params)
    pelvis = smplx_out.joints.detach().numpy()[0,0,:]
    navmesh = trimesh.load(navmesh_path, force="mesh")
    target_point, _, _ = trimesh.proximity.closest_point(navmesh, np.array(pelvis)[None,...])
    target_point = target_point.reshape(-1)
    forward = target_point - pelvis
    global_orient = get_global_orient_from_forward_z_up(forward)
    stand_pose, transl, _ = place_stand_pose_on_floor(gender, betas, global_orient, target_point, floor_height)
    ret = np.concatenate([global_orient, stand_pose, transl], axis=-1)[None,...]
    return ret 

def place_stand_pose_before_sit(poset, betas, floor_height, action_order, gender="neutral"):
    body_model_one_batch = load_body_mesh_model_batch(1, gender=gender)
    smplx_params = {'global_orient': torch.tensor(poset[:,:3]), 'body_pose': torch.tensor(poset[:,3:66]), 'betas': torch.tensor(betas), 'transl': torch.tensor(poset[:,66:69])}
    smplx_out = body_model_one_batch(**smplx_params)
    pelvis = smplx_out.joints.detach().numpy()[0,0,:]
    if action_order == "sit_walk":
        r = np.random.uniform(0,0.2) + 0.6
    else:
        r = np.random.uniform(0,0.2) + 0.8
    forward_dir = R.from_rotvec(poset[:,:3]).as_matrix()[0,:,2]
    forward_dir[2] = 0
    forward_dir = forward_dir / np.linalg.norm(forward_dir)
    target_point = pelvis + r * forward_dir

    stand_pose = load_template_pose('stand').reshape(-1, 63)
    rotmat = np.stack([np.cross(np.array([0, 0, 1]), forward_dir), np.array([0, 0, 1]), forward_dir]).T
    global_orient = R.from_matrix(rotmat[None,...]).as_rotvec().astype(np.float32)
    transl = np.zeros((1, 3), dtype=np.float32)
    smplx_params = {'global_orient': torch.tensor(global_orient), 'body_pose': torch.tensor(stand_pose), 'betas': torch.tensor(betas), 'transl': torch.tensor(transl)}
    smplx_out = body_model_one_batch(**smplx_params)
    transl[0,:2] = target_point[:2] -smplx_out.joints.detach().numpy()[0,0,:2]
    transl[0,2] = floor_height - np.min(smplx_out.vertices.detach().numpy()[0,:,2])
    
    ret = np.concatenate([global_orient, stand_pose, transl], axis=-1)
    return ret

#motion process tools
def motion_blending(motion1, motion2, blending_coeff):
    ret = np.zeros_like(motion1)
    n_frames = motion1.shape[0] 
    feature_dim = motion1.shape[-1]
    if feature_dim == 69:
        motion1_pose, motion1_transl = motion1[:, :66], motion1[:, 66:]
        motion2_pose, motion2_transl = motion2[:, :66], motion2[:, 66:]
        ret_transl = motion1_transl * (1 - blending_coeff[:, None]) + motion2_transl * blending_coeff[:, None]
        motion1_pose_rotmat = R.from_rotvec(motion1_pose.reshape(-1,3)).as_matrix()
        motion2_pose_rotmat = R.from_rotvec(motion2_pose.reshape(-1,3)).as_matrix()
        transform_rotmat = np.matmul(motion2_pose_rotmat, motion1_pose_rotmat.transpose(0,2,1))
        transform_rotvec = R.from_matrix(transform_rotmat).as_rotvec().reshape(-1, 66)
        transform_rotvec_blended = (transform_rotvec * blending_coeff[:, None]).reshape(-1, 3)
        transform_rotmat_blended = R.from_rotvec(transform_rotvec_blended).as_matrix()
        ret_pose_rotmat = np.matmul(transform_rotmat_blended, motion1_pose_rotmat)
        ret_pose = R.from_matrix(ret_pose_rotmat).as_rotvec().reshape(-1, 66)
        ret[:, :66] = ret_pose
        ret[:, 66:] = ret_transl
    else:
        raise NotImplementedError
    return ret

def time_variant_motion_blending(motion1, motion2):
    assert motion1.shape == motion2.shape, f"{motion1.shape}!={motion2.shape}"
    blending_coeff = np.linspace(0, 1, motion1.shape[0]+2)
    blending_coeff = blending_coeff[1:-1]
    ret = motion_blending(motion1, motion2, blending_coeff)
    return ret
    
def time_variant_motion_updating(prev_motion, curr_motion, n_his_frames):
    ret = prev_motion.copy()
    ret[-n_his_frames:] = time_variant_motion_blending(prev_motion[-n_his_frames:], curr_motion[:n_his_frames])
    ret = np.concatenate([ret, curr_motion[n_his_frames:]], axis=0)
    return ret

def results_from_full_motion_parameters(motion):
    device = 'cuda'
    _, batch_size, _ = motion['smplx_params'].shape
    gender = motion['gender']
    smplx_params = motion['smplx_params']
    body_mesh_model_batch = load_body_mesh_model_batch(batch_size, gender=gender, device=device)
    smplx_params = torch.tensor(motion['smplx_params']).to(device)
    betas = torch.tensor(motion['betas']).to(device).view(1, -1).repeat(batch_size, 1)
    smplx_output = extract_smplx_from_feature(smplx_params, body_mesh_model_batch, betas=betas)
    vertices, joints = smplx_output.vertices, smplx_output.joints[:, :55, :]
    vertices = vertices.detach().cpu().numpy()
    joints = joints.detach().cpu().numpy()
    return vertices, joints, body_mesh_model_batch.faces
