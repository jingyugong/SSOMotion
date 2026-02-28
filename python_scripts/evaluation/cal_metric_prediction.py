import numpy as np
import glob
import pickle
import torch
import tqdm
from tools.motion_tools import load_body_mesh_model_batch, extract_smplx_from_feature

def calc_metric_for_prediction(pred_result_paths):
    metrics = {'mpjpe': [], 'mpvpe': [], 'dist': [], 'inner_mpjpe': [], 'inner_mpvpe': [], 'inner_dist': []}
    instructions = []
    for pred_result_path in tqdm.tqdm(pred_result_paths):
        gt_result_path = pred_result_path.replace('pred', 'gt')
        with open(pred_result_path, 'rb') as f:
            pred_result = pickle.load(f)
        with open(gt_result_path, 'rb') as f:
            gt_result = pickle.load(f)
        pred_motion = pred_result['motion']
        gt_motion = gt_result['motion']

        betas = torch.tensor(pred_motion['betas'].reshape(1, *pred_motion['betas'].shape)).repeat(10, 1).to(torch.float32)

        pred_motion_feature = torch.tensor(pred_motion['smplx_params'].reshape(1,30,69)).to(torch.float32)
        pred_smplx = extract_smplx_from_feature(pred_motion_feature, body_mesh_model_batch, betas=betas)
        pred_joints = pred_smplx.joints.detach().cpu().numpy()[:, :22, :]
        pred_vertices = pred_smplx.vertices.detach().cpu().numpy()

        gt_motion_feature = torch.tensor(gt_motion['smplx_params'].reshape(1,30,69)).to(torch.float32)
        gt_smplx = extract_smplx_from_feature(gt_motion_feature, body_mesh_model_batch, betas=betas)
        gt_joints = gt_smplx.joints.detach().cpu().numpy()[:, :22, :]
        gt_vertices = gt_smplx.vertices.detach().cpu().numpy()

        mpjpe_i = np.linalg.norm(gt_joints - pred_joints, axis=-1).mean()
        mpvpe_i = np.linalg.norm(gt_vertices - pred_vertices, axis=-1).mean()
        dist_i = np.linalg.norm(gt_joints[-1, 0, :] - pred_joints[-1, 0, :])

        inner_mpjpe_i = np.linalg.norm(gt_joints - gt_joints[0:1, :, :], axis=-1).mean()
        inner_mpvpe_i = np.linalg.norm(gt_vertices - gt_vertices[0:1, :, :], axis=-1).mean()
        inner_dist_i = np.linalg.norm(gt_joints[-1, 0, :] - gt_joints[0, 0, :])

        metrics['mpjpe'].append(mpjpe_i)
        metrics['mpvpe'].append(mpvpe_i)
        metrics['dist'].append(dist_i)
        metrics['inner_mpjpe'].append(inner_mpjpe_i)
        metrics['inner_mpvpe'].append(inner_mpvpe_i)
        metrics['inner_dist'].append(inner_dist_i)
        instructions.append(pred_result['action_text'])

    return metrics, instructions

def select_sample_according_to_metrics(metrics, action_ids, num_sample_per_class=30):
    tol = 1e-8
    selected_sample_indices = []
    for action_id in range(4):
        sub_indices = np.where(action_ids == action_id)[0]
        penalties = []
        for i in sub_indices: 
            penalty = (metrics['mpjpe'][i] + tol) * (metrics['mpvpe'][i] + tol) / (metrics['inner_mpjpe'][i] + tol) / (metrics['inner_mpvpe'][i] + tol)
            penalties.append(penalty)

        sub_sample_indices = np.argsort(penalties)[:num_sample_per_class]
        selected_sample_indices.extend(sub_indices[sub_sample_indices].tolist())
    return selected_sample_indices

def instruction_to_action_id(instructions):
    action_ids = []
    for instruction in instructions:
        if "lie" in instruction:
            action_ids.append(3)
        elif "stand up" in instruction:
            action_ids.append(2)
        elif "sit" in instruction:
            action_ids.append(1)
        else:
            action_ids.append(0)
    action_ids = np.array(action_ids)
    return action_ids

if __name__ == "__main__":
    #glob_pattern = f'./save/results_for_eval/cmdm_action2motion_qkv/humanise/sample*/rep*/pred/results.pkl'
    #pred_result_paths = sorted(glob.glob(glob_pattern))
    pred_result_paths = [f'./save/results_for_eval/cmdm_action2motion_qkv/humanise/sample{i}/rep0/pred/results.pkl' for i in range(0, 21919)] #total 21919
    num_sample_per_class = 10
    n_frames = 30
    body_mesh_model_batch = load_body_mesh_model_batch(n_frames, gender='neutral')
    metrics, instructions = calc_metric_for_prediction(pred_result_paths)
    action_ids = instruction_to_action_id(instructions)
    selected_sample_indices = select_sample_according_to_metrics(metrics, action_ids, num_sample_per_class=num_sample_per_class)
    selected_samples = [pred_result_paths[i].split('/')[-4] for i in selected_sample_indices]
    selected_sample_instructions = [instructions[i] for i in selected_sample_indices]

    for k, v in metrics.items():
        metrics[k] = np.mean(v)
    print(metrics)

    for i in range(len(selected_samples)):
        print(selected_samples[i], selected_sample_instructions[i])
