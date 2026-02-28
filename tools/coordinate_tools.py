import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R
from tools.motion_tools import load_body_mesh_model_batch
from tools.io_tools import extract_smplx_joints
from tools.project_config_tools import host_device

def calc_calibrate_offset(body_mesh_model, betas, pose, local_device=host_device):
    '''
    The factors to influence this offset is not clear. Maybe it is shape and pose dependent.
    Therefore, we calculate such delta_T for each individual body mesh.
    It takes a batch of body parameters
    input:
        body_params: dict, basically the input to the smplx model
        smplx_model: the model to generate smplx mesh, given body_params
    Output:
        the offset for params transform
    '''
    n_batches = pose.shape[0]
    bodyconfig = {}
    bodyconfig['body_pose'] = torch.tensor(pose[:,3:], device=local_device).float()
    bodyconfig['betas'] = torch.tensor(betas, device=local_device).float().unsqueeze(0).repeat(n_batches,1)
    bodyconfig['transl'] = torch.zeros([n_batches,3], dtype=torch.float32, device=local_device)
    bodyconfig['global_orient'] = torch.zeros([n_batches,3], dtype=torch.float32, device=local_device)
    smplx_out = body_mesh_model(return_verts=True, **bodyconfig)
    delta_T = smplx_out.joints[:,0,:] # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy() #[t, 3]

    return delta_T

def get_new_coordinate(body_mesh_model, betas, transl, pose, local_device=host_device):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the pelvis
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    bodyconfig = {}
    bodyconfig['transl'] = torch.tensor(transl, device=local_device).float()
    bodyconfig['global_orient'] = torch.tensor(pose[:,:3], device=local_device).float()
    bodyconfig['body_pose'] = torch.tensor(pose[:,3:], device=local_device).float()
    bodyconfig['betas'] = torch.tensor(betas, device=local_device).unsqueeze(0).float()
    smplxout = body_mesh_model(**bodyconfig)
    joints = smplxout.joints.squeeze().detach().cpu().numpy()
    x_axis = joints[2,:] - joints[1,:]
    x_axis[-1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.array([0,0,1])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis/np.linalg.norm(y_axis)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    transl_new = joints[:1,:] # put the local origin to pelvis

    return global_ori_new, transl_new

def point_coordinate_transform(points, transf_rotmat, transf_transl):
    points = points.copy() - transf_transl.reshape(-1, 3)
    points = np.matmul(points, transf_rotmat)
    return points

def motion_coordinate_transform(gender, betas, transl, pose, transf_rotmat, transf_transl, local_device=host_device):
    transl = transl.copy()
    pose = pose.copy()
    bodymodel_batch = load_body_mesh_model_batch(transl.shape[0], gender=gender, device=local_device)
    ### calibrate offset
    delta_T = calc_calibrate_offset(bodymodel_batch, betas[:10], pose[:,:66], local_device=local_device)
    ### get new global_orient
    global_ori = R.from_rotvec(pose[:,:3]).as_matrix() # to [t,3,3] rotation mat
    global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
    pose[:,:3] = R.from_matrix(global_ori_new).as_rotvec()
    ### get new transl
    transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl+delta_T-transf_transl)-delta_T
    return transl, pose

def coordinate_inv_transform(transf_rotmat, transf_transl):
    return np.linalg.inv(transf_rotmat), -np.matmul(np.linalg.inv(transf_rotmat), transf_transl.reshape(3,1)).reshape(3)

def swap_left_right(data):
    data_m = copy.deepcopy(data)
    right_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_chain = [0, 3, 6, 9, 12, 15, 17, 19]
    
    n_frames, _ = data_m['pose'].shape
    if n_frames == 0:
        return None
    
    data_m['pose'] = data_m['pose'].reshape(n_frames, -1, 3)
    
    tmp = data_m['pose'][:, right_chain, :]
    data_m['pose'][:, right_chain, :] = data_m['pose'][:, left_chain, :]
    data_m['pose'][:, left_chain, :] = tmp

    data_m['pose'][:, :, 1:3] *= -1
    data_m['pose'] = data_m['pose'].reshape(n_frames, -1)

    rot = R.from_rotvec(data_m['global_orient']).as_matrix()
    mat = np.eye(3)
    mat[0, 0] = -1
    rot = np.matmul(mat, rot)
    rot = np.matmul(rot, mat)
    data_m['global_orient'] = R.from_matrix(rot).as_rotvec()
    betas = data['betas'].copy()
    assert len(betas.shape) in [1, 2]
    if len(betas.shape) == 1:
        betas = betas.reshape(1, -1)
    if betas.shape[0] != data['transl'].shape[0]:
        betas = np.repeat(betas, data['transl'].shape[0], axis=0)
    hand_pose = np.zeros((data['transl'].shape[0], 24))

    motion_103dim = np.concatenate([data['transl'], data['global_orient'], betas, data['pose'], hand_pose], axis=-1)
    motion_m_103dim = np.concatenate([data_m['transl'], data_m['global_orient'], betas, data_m['pose'], hand_pose], axis=-1)
    motion_joints = extract_smplx_joints(motion_103dim, data['gender'])
    motion_m_joints = extract_smplx_joints(motion_m_103dim, data['gender'])

    target = motion_joints[:, 0, :]
    target[:,0] *= -1
    source = motion_m_joints[:, 0, :]
    data_m['transl'][:, :] +=  target - source

    return data_m

def get_mirrored_motion(transl, pose, betas, gender):
    transl = transl.copy()
    pose = pose.copy()
    residual_pose = pose[:,66:]
    data = {}
    data['transl'] = transl
    data['global_orient'] = pose[:,:3]
    data['pose'] = pose[:,3:66]
    data['betas'] = betas
    data['gender'] = gender
    data_m = swap_left_right(data)
    transl_M = data_m['transl']
    pose_M = np.concatenate([data_m['global_orient'], data_m['pose'], residual_pose], axis=-1)
    return transl_M, pose_M

