import os
import numpy as np
import tqdm
import pickle
import torch
from tools.motion_tools import load_body_mesh_model_batch, extract_smplx_from_feature

rotmat = np.array([[[-1, 0, 0], [0, 0, 1], [0, 1, 0]]]).astype(np.float32)

def transform_motion(result_paths, save_dir):
    for idx, result_path in tqdm.tqdm(enumerate(result_paths)):
        with open(result_path, 'rb') as f:
           result = pickle.load(f)
        motion = result['motion']
        betas = torch.tensor(motion['betas'].reshape(1, *motion['betas'].shape)).repeat(10, 1).to(torch.float32)

        motion_feature = torch.tensor(motion['smplx_params'].reshape(1,30,69)).to(torch.float32)
        smplx_output = extract_smplx_from_feature(motion_feature, body_mesh_model_batch, betas=betas)
        joints = smplx_output.joints.detach().cpu().numpy()[:, :22, :]

        joints = np.matmul(joints, rotmat)
        save_path = f'{save_dir}/sample{idx}.npy'
        np.save(save_path, joints)
    return

if __name__ == "__main__":
    n_frames = 30
    body_mesh_model_batch = load_body_mesh_model_batch(n_frames, gender='neutral')
    for pred_gt in ["pred", "gt"]:
        result_paths = [f'./save/results_for_eval/cmdm_action2motion_qkv/humanise/sample{i}/rep0/{pred_gt}/results.pkl' for i in range(0, 21919)] #total 21919
        save_dir = f'./save/results_for_eval/cmdm_action2motion_qkv/humanise_in_humanml3d/{pred_gt}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        transform_motion(result_paths, save_dir)
    

