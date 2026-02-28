import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import trimesh
import pyrender
import json
import pickle
import time
from tqdm import tqdm
from pathlib import  Path
import torch
import smplx
import glob
from smplx import joint_names
from tools.motion_tools import results_from_full_motion_parameters

def check_inside(navmesh, points_scene):
    floor_height = navmesh.vertices[0, 2]
    points_scene[:, :, 2] = floor_height
    B, J, _ = points_scene.shape

    # https://stackoverflow.com/a/2049593/14532053
    points_2d = torch.cuda.FloatTensor(points_scene[:, :, :2]).reshape(B * J, 1, 2)  # [P=b*j, 1, 2]
    triangles = torch.cuda.FloatTensor(np.stack([navmesh.vertices[navmesh.faces[:, 0], :2],
                                        navmesh.vertices[navmesh.faces[:, 1], :2],
                                        navmesh.vertices[navmesh.faces[:, 2], :2]], axis=-1)).permute(0, 2, 1)[None, ...]  # [1, F, 3, 2]
    def sign(p1, p2, p3):
        return (p1[:, :, 0] - p3[:, :, 0]) * (p2[:, :, 1] - p3[:, :, 1]) - (p2[:, :, 0] - p3[:, :, 0]) * (p1[:, :, 1] - p3[:, :, 1])

    d1 = sign(points_2d, triangles[:, :, 0, :], triangles[:, :, 1, :])
    d2 = sign(points_2d, triangles[:, :, 1, :], triangles[:, :, 2, :])
    d3 = sign(points_2d, triangles[:, :, 2, :], triangles[:, :, 0, :])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    inside_triangle = ~(has_neg & has_pos) #[P, F]
    inside_mesh = inside_triangle.any(-1).reshape((B, J)).detach().cpu().numpy()

    return inside_mesh

def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}


def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}


def rollout_primitives(motion_primitives):
    gender = motion_primitives[0]['gender']
    model_path = "./data/models_smplx_v1_1/models"
    body_model = smplx.create(model_path=model_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=10,
                              ).to(device='cuda')
    joints_list = []
    vertices_list = []
    for idx, motion_primitive in enumerate(motion_primitives):
        # pelvis_original = body_model(betas=torch.cuda.FloatTensor(motion_primitive['betas']).repeat(10, 1)).joints[:, 0, :].detach().cpu().numpy()  # [10, 3]
        smplx_param = motion_primitive['smplx_params'][0]  #[10, 96]
        smplx_param_dict = {
            'betas': torch.cuda.FloatTensor(motion_primitive['betas']).repeat(10, 1),
            'transl': smplx_param[:, :3],
            'global_orient': smplx_param[:, 3:6],
            'body_pose': smplx_param[:, 6:69]
        }
        smplx_param_dict = params2torch(smplx_param_dict)
        output = body_model(**smplx_param_dict)
        vertices, joints = output.vertices, output.joints[:, :55, :]

        rotation = torch.cuda.FloatTensor(motion_primitive['transf_rotmat']).reshape((1, 3, 3)) # [1, 3, 3]
        transl = torch.cuda.FloatTensor(motion_primitive['transf_transl']).reshape((1, 1, 3)) # [1, 1, 3]
        vertices = torch.einsum('bij,bpj->bpi', rotation, vertices) + transl
        joints = torch.einsum('bij,bpj->bpi', rotation, joints) + transl

        if idx == 0:
            start_frame = 0
        elif motion_primitive['mp_type'] == '1-frame':
            start_frame = 1
        elif motion_primitive['mp_type'] == '2-frame':
            start_frame = 2
        else:
            print(motion_primitive['mp_type'])
            start_frame = 0
        vertices_list.append(vertices[start_frame:, :, :])
        joints_list.append(joints[start_frame:, :, :])

    vertices_rollout = torch.cat(vertices_list, dim=0).detach().cpu().numpy()
    joints_rollout = torch.cat(joints_list, dim=0).detach().cpu().numpy()
    return vertices_rollout, joints_rollout, body_model.faces

def visualize(vertices, joints, faces, navmesh):
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(navmesh))
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    body_node = None
    for frame_idx in tqdm(range(len(vertices))):
        body_mesh = trimesh.Trimesh(
            faces=faces,
            vertices=vertices[frame_idx],
            vertex_colors=np.array([200, 200, 200, 255])
        )
        joints_mesh = []
        for joint in joints[frame_idx]:
            trans_mat = np.eye(4)
            trans_mat[:3, 3] = joint
            sm = trimesh.creation.uv_sphere(radius=0.02)
            sm.visual.vertex_colors = [1.0, 0.0, 0.0]
            sm.apply_transform(trans_mat)
            joints_mesh.append(sm)

        viewer.render_lock.acquire()
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(body_mesh + joints_mesh), name='body')
        scene.add_node(body_node)
        viewer.render_lock.release()

def calc_metric_primitives(result_list):
    foot_joints_names = ["left_ankle", "left_foot", "right_ankle", "right_foot", ]
    foot_joints_idx = [joint_names.JOINT_NAMES.index(joint) for joint in foot_joints_names]
    ankle_joints_names = ["left_ankle", "right_ankle", ]
    ankle_joints_idx = [joint_names.JOINT_NAMES.index(joint) for joint in ankle_joints_names]

    metrics = {
        'num_frames': [],
        'seconds': [],
        'contact': [],
        'pene_foot': [],
        'pene_ankle': [],
        'pene_vertices': [],
        'pene_vertices_mean': [],
        'min_dist': [],
        'last_dist': [],
        'success': [],
    }
    for result_idx, result_path in enumerate(tqdm(result_list)):
        with open(result_path, 'rb') as f:
            motion_data = pickle.load(f)
        navmesh_path = motion_data['navmesh_path']
        wpath = motion_data['wpath']
        final_goal = wpath[-1]
        navmesh = trimesh.load(navmesh_path, force='mesh')

        vertices, joints, faces = results_from_full_motion_parameters(motion_data['motion'])
        #vertices, joints, faces = rollout_primitives(motion_data['motion'])  # [T, V, 3], [T, J, 3]
        if result_idx == 0 and debug:
            visualize(vertices, joints, faces, navmesh)
        h = 1 / frame_rate
        foot_joints = joints[:, foot_joints_idx, :]
        ankle_joints = joints[:, ankle_joints_idx, :]
        dists_xy = np.linalg.norm((joints - final_goal)[:, :, :2], axis=-1).min(axis=-1)  # [T, J] -> [J]
        foot_inside = check_inside(navmesh, foot_joints).all(axis=-1)  # [T, 4] -> [T,]
        foot_inside = foot_inside.mean()
        ankle_inside = check_inside(navmesh, ankle_joints).all(axis=-1)  # [T, 4] -> [T,]
        ankle_inside = ankle_inside.mean()
        vertices_inside = check_inside(navmesh, vertices)  # [T, V]
        vertices_inside_mean = vertices_inside.mean(axis=-1).mean()
        vertices_inside = vertices_inside.all(axis=-1).mean()
        foot_joints_speed = np.linalg.norm(foot_joints[2:, :, :] - foot_joints[:-2, :, :], axis=-1) / (
                    2 * h)  # [T-2, 4]
        foot_joints_speed = np.clip(foot_joints_speed.min(axis=-1) - 0.075, a_min=0, a_max=32767)  # [T-2, ]
        foot_joints_abs_height = np.clip(np.abs(foot_joints[:, :, 2].min(axis=-1)) - 0.05, a_min=0,
                                         a_max=32767)  # [T, 4] -> [T, ]
        contact_score = np.exp(-foot_joints_speed) * np.exp(-foot_joints_abs_height[1:-1])
        contact_score = contact_score.mean()

        metrics['num_frames'].append(vertices.shape[0])
        metrics['seconds'].append(vertices.shape[0] / frame_rate)
        metrics['pene_foot'].append(foot_inside)
        metrics['pene_ankle'].append(ankle_inside)
        metrics['pene_vertices'].append(vertices_inside)
        metrics['pene_vertices_mean'].append(vertices_inside_mean)
        metrics['last_dist'].append(dists_xy[-1])
        metrics['min_dist'].append(dists_xy.min())
        metrics['success'].append(1 if dists_xy.min() < 0.2 else 0)
        metrics['contact'].append(contact_score)

    group_size = 10 #generate 10 samples per group
    num_samples = len(metrics['contact'])

    best_indices = [
        i + np.argmax(
            np.array(metrics['contact'][i:i+group_size]) + np.array(metrics['pene_vertices_mean'][i:i+group_size])
        )
        for i in range(0, num_samples, group_size)
    ]

    metrics = {
        k: float(np.mean([metrics[k][i] for i in best_indices])) if len(metrics[k]) else 0
        for k in metrics
    }
    metrics['indices'] = [int(i % group_size) for i in best_indices]

    return metrics

debug = 0
if __name__ == '__main__':
    frame_rate = 30.0
    scene_name_list = [
        '5.33_5.94_2_1677228749.1353848',
        '5.54_5.34_0_1677228775.5952885',
        '5.76_8.87_1_1677228764.1464393',
        '5.97_6.83_1_1677228757.1440427',
        '6.27_4.97_0_1677228807.6371164',
    ]
    framelabel_search_results = []
    result_base_dir = Path('results/locomotion')
    num_pair = 10
    
    glob_pattern = f'./save/results_for_eval/cmdm_action2motion_qkv/random_scene_test/*/path*/rep*/results.pkl'
    framelabel_search_results = glob.glob(glob_pattern)
    framelabel_search_results = sorted(framelabel_search_results)
    metrics = {
        'framelabel_search': calc_metric_primitives(framelabel_search_results),
    }

    print(metrics)
    with open(os.path.join('/'.join(glob_pattern.split('/')[:5]),'metric_locomotion.json'), 'w') as f:
        json.dump(metrics, f)
