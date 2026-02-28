import sys
import os
sys.path.append("..")
import numpy as np
import torch
import trimesh
import json
import pickle
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from tools.motion_tools import load_body_mesh_model_batch
from tools.utils import BodyParamParser
from tools.project_config_tools import project_dir

def convert_joints_smplx2smpl(smplx_joints):
    smpl_joints = np.zeros((smplx_joints.shape[0], 24, 3))
    smpl_joints[:, :22, :] = smplx_joints [:, :22, :]
    smpl_joints[:, 22, :] = np.mean(smplx_joints[:, 25:39, :], axis = 1)   # left hand
    smpl_joints[:, 23, :] = np.mean(smplx_joints[:, 40:54, :], axis = 1)   # right hand
    return smpl_joints

def extract_smplx_joints(feature, gender='neutral'):
    assert len(feature.shape) == 2
    batch_size, feature_dim = feature.shape
    assert feature_dim in [69, 72, 79, 103]
    if feature_dim in [69, 79]:
        pose_data = np.zeros((feature.shape[0], 103))  # 3: translation + 3-6: global orient + 6-16: body shape + 16-79: body pose + 79-103: left land + right hand
        pose_data[:, :3] = feature[:, 66:69]
        pose_data[:, 3:6] = feature[:, 0:3]
        pose_data[:, 16:79] = feature[:, 3:66]
        if feature_dim == 79:
            pose_data[:, 6:16] = feature[:, 69:79]
        feature = pose_data
        feature_dim = 103

    device = 'cuda'
    if feature_dim == 72:
        vposer, _ = load_vposer(os.path.join(project_dir,'body_models/vposer_v1_0'), model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        vposer=vposer.to(device)
        vposer.eval()
    body_mesh_model = load_body_mesh_model_batch(batch_size, gender=gender)
    body_mesh_model.eval()
    # smplx_faces = body_mesh_model.faces
    body_mesh_model=body_mesh_model.to(device)
    if type(feature) == np.ndarray:
        feature = torch.tensor(feature, dtype=torch.float32)
    feature = feature.view(-1, feature_dim).to(device)
    if feature_dim == 72:
        body_param_rec = BodyParamParser.body_params_encapsulate_batch_hand(feature, body_pose_dim=32)
        body_param_rec['body_pose'] = vposer.decode(body_param_rec['body_pose'], output_type='aa').view(1, -1)
    else:
        body_param_rec = BodyParamParser.body_params_encapsulate_batch_hand(feature, body_pose_dim=63)
    smplx_output = body_mesh_model(return_verts=True, **body_param_rec) # joint
    return smplx_output.joints.cpu().detach().numpy()

def export_batch_human_mesh(feature, save_paths, gender='neutral', cam_extrinsic=None, save_type='obj', ):
    assert len(feature.shape) == 2
    batch_size, feature_dim = feature.shape
    assert feature_dim in [72, 103]

    device = 'cuda'
    if feature_dim == 72:
        vposer, _ = load_vposer(os.path.join(project_dir,'body_models/vposer_v1_0'), model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        vposer=vposer.to(device)
        vposer.eval()
    body_mesh_model = load_body_mesh_model_batch(batch_size, gender=gender)
    body_mesh_model.eval()
    smplx_faces = body_mesh_model.faces
    body_mesh_model=body_mesh_model.to(device)
    if type(feature) == np.ndarray:
        feature = torch.tensor(feature, dtype=torch.float32)
    feature = feature.view(-1, feature_dim).to(device)
    if feature_dim == 72:
        body_param_rec = BodyParamParser.body_params_encapsulate_batch_hand(feature, body_pose_dim=32)
        body_param_rec['body_pose'] = vposer.decode(body_param_rec['body_pose'], output_type='aa').view(1, -1)
    else:
        body_param_rec = BodyParamParser.body_params_encapsulate_batch_hand(feature, body_pose_dim=63)
    smplx_output = body_mesh_model(return_verts=True, **body_param_rec)
    body_verts_batch = smplx_output.vertices

    if type(save_paths) is str:
        assert save_type in ["obj", "ply"], "save type must be obj or ply."
        save_paths = [os.path.join(save_paths, "{:06d}.".format(i) + save_type) for i in range(batch_size)]

    parent_dir = os.path.dirname(save_paths[0])
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    for i in range(batch_size):
        out_mesh =trimesh.Trimesh(body_verts_batch[i].detach().cpu().numpy(),smplx_faces,process=False)
        if cam_extrinsic is not None:
            cam_ext = cam_extrinsic.numpy()
            cam_ext = torch.tensor(cam_ext,dtype=torch.float32).view(4, 4)
            out_mesh.apply_transform(cam_ext)
        out_mesh.export(save_paths[i])
    return

def load_feature69dim_from_file(file_path):
    data_dict = dict(np.load(file_path, allow_pickle=True))
    pose = data_dict["pose"][:,:66]
    transl = data_dict["transl"]
    ret = np.concatenate([pose, transl], axis=1)
    return ret

def load_joint_hint_from_file(file_path):
    data_dict = dict(np.load(file_path, allow_pickle=True))
    skeleton = data_dict["joints"]
    return skeleton

def load_random_joint_hint_from_file(file_path):
    skeleton = load_joint_hint_from_file(file_path)
    n_frames, n_joints, n_dims = skeleton.shape
    hint = random_mask_train(skeleton, n_joints=n_joints)
    hint = hint.reshape(n_frames, -1)
    return hint

def load_markers(marker_type='SSM2'):
    marker_subpath = "assets/markers/" + marker_type + ".json"
    with open(os.path.join(project_dir, marker_subpath), 'r') as f:
        markers = json.load(f)
    return markers

def random_mask_train(joints, n_joints=22):
    if n_joints == 22:
        #controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        controllable_joints = np.arange(0, n_joints)
    else:
        raise NotImplementedError
    num_joints = len(controllable_joints)
    # joints: length, 22, 3
    num_joints_control = np.random.choice(num_joints, 1)
    # only use one joint during training
    num_joints_control = 1
    choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
    choose_joint = controllable_joints[choose_joint]

    length = joints.shape[0]
    choose_seq_num = np.random.choice(length - 1, 1) + 1
    choose_seq = np.random.choice(length, choose_seq_num, replace=False)
    choose_seq.sort()
    mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

    for cj in choose_joint:
        mask_seq[choose_seq, cj] = True

    # normalize
    joints = joints * mask_seq
    return joints

def load_pkl(file_name, read_type="rb"):
    with open(file_name, read_type) as f:
        ret = pickle.load(f)
    return ret
