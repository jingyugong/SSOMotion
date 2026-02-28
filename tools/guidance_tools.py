import os
from pathlib import Path
import numpy as np
import torch
import json
from data_loaders.scene2motion.realtime_scene import RealtimeScene



def load_prox_scenehints(scene_name, device='cuda'):
    from tools.project_config_tools import prox_data_dir
    from tools.navigation_tools import get_prox_floor_height
    floor_height = get_prox_floor_height(os.path.join(prox_data_dir, "scene_segmentation", scene_name+".pkl"), os.path.join(prox_data_dir, "scenes", scene_name+".ply"))
    with open(os.path.join(prox_data_dir, "sdf", scene_name+".json"), 'r') as f:
        sdf_config = json.load(f)
    dim = sdf_config['dim']
    centroid = (np.array(sdf_config["max"]) + np.array(sdf_config["min"]))/2
    scale = (np.array(sdf_config["max"]) - np.array(sdf_config["min"]))/2
    sdf = np.load(os.path.join(prox_data_dir, "sdf", scene_name+"_sdf.npy")).reshape(1, dim, dim, dim)
    scene_global_occ = RealtimeScene(os.path.join(prox_data_dir, "occ", scene_name, "compact_occ.npy"), os.path.join(prox_data_dir, "occ", scene_name, "compact_occ_info.json"), semantic_anno_type="matterport3d")
    ret = {
        'floor_height': floor_height,
        'scene_sdf': torch.tensor(sdf, dtype=torch.float32, device=device),
        'scene_sdf_dim': dim,
        'scene_sdf_centroid': torch.tensor(centroid, dtype=torch.float32, device=device),
        'scene_sdf_scale': torch.tensor(scale, dtype=torch.float32, device=device),
        'scene_global_occ': scene_global_occ,
        'transf_rotmat': None,
        'transf_transl': None,
    }
    return ret


def load_replica_scenehints(scene_name, device='cuda'):
    from tools.project_config_tools import replica_dir
    from tools.navigation_tools import get_replica_floor_height
    replica_dir = Path(replica_dir)
    floor_height = get_replica_floor_height(replica_dir / scene_name)
    with open(replica_dir / scene_name / f"{scene_name}_sdf.json", 'r') as f:
        sdf_config = json.load(f)
    dim = sdf_config['dim']
    centroid = sdf_config['centroid']
    scale = sdf_config['scale']
    sdf = np.load(replica_dir / scene_name / f"{scene_name}_sdf.npy").reshape(1, dim, dim, dim)
    scene_global_occ = RealtimeScene(os.path.join(replica_dir, "occ", scene_name, "compact_occ.npy"), os.path.join(replica_dir, "occ", scene_name, "compact_occ_info.json"), semantic_anno_type="replica")
    ret = {
        'floor_height': floor_height,
        'scene_sdf': torch.tensor(sdf, dtype=torch.float32, device=device),
        'scene_sdf_dim': dim,
        'scene_sdf_centroid': torch.tensor(centroid, dtype=torch.float32, device=device),
        'scene_sdf_scale': torch.tensor(scale, dtype=torch.float32, device=device),
        'scene_global_occ': scene_global_occ,
        'transf_rotmat': None,
        'transf_transl': None,
    }
    return ret


def dimosdata2scenehints(data):
    ret = {}
    if 'floor_height' in data:
        ret['floor_height'] = data['floor_height']
    if 'obj_sdf' in data:
        _, c, d1, d2, d3 = data['obj_sdf']['grid'].shape
        ret['scene_sdf'] = data['obj_sdf']['grid'].view(c, d1, d2, d3)
        _, c, d1, d2, d3 = data['obj_sdf']['gradient_grid'].shape
        ret['scene_sdf_gradient'] = data['obj_sdf']['gradient_grid'].view(c, d1, d2, d3)
        ret['scene_sdf_dim'] = data['obj_sdf']['dim']
        ret['scene_sdf_centroid'] = data['obj_sdf']['centroid'].view(3)
        ret['scene_sdf_scale'] = data['obj_sdf']['scale']
    return ret

def data2scenehints(data, transf_rotmat, transf_transl, src_fmt='dimos'):
    if src_fmt == "dimos":
        ret = dimosdata2scenehints(data)
    else:
        raise NotImplementedError
    ret['transf_rotmat'] = transf_rotmat
    ret['transf_transl'] = transf_transl
    return ret

def wpath2hints(wpath, wpath_joints, wpath_poset, scene_action_name, n_frames=30, fps=30, last_pose_lasting_frames=1):
    xy_guidance_hints = np.zeros((n_frames, 22, 3))
    guidance_hints = np.zeros((n_frames, 22, 3))
    hints = np.zeros((n_frames, 22, 3))
    poset_hints = np.zeros((n_frames, 69))
    wpath_frame_idxs = []

    if scene_action_name in ["walk", "sit", "stand_up"]:
        dist_use_all_joints = False 
    elif scene_action_name in ["lie"]:
        dist_use_all_joints = isinstance(wpath_joints, list) 
        if isinstance(wpath_joints, list):
            for wpath_joint in wpath_joints:
                dist_use_all_joints = dist_use_all_joints and (wpath_joint is not None)
    if dist_use_all_joints:
        wpath_dists = np.mean(np.linalg.norm(np.array(wpath_joints)[1:,:] - np.array(wpath_joints)[:-1,:], axis=-1), axis=-1)
    else:
        wpath_dists = np.linalg.norm(wpath[1:,:] - wpath[:-1,:], axis=-1)
    wpath_dists = np.concatenate([np.array([0]), wpath_dists], axis=0)
    wpath_cum_dists = np.cumsum(wpath_dists)

    if scene_action_name == "walk":
        #speed = np.random.uniform(1.1, 1.4)
        speed = np.random.uniform(0.7, 1.1)
    elif scene_action_name == "sit":
        speed = np.random.uniform(0.45, 0.6)
    elif scene_action_name == "stand_up":
        speed = np.random.uniform(0.75, 0.85)
    elif scene_action_name == "lie":
        if dist_use_all_joints:
            speed = np.random.uniform(0.75, 0.85)
        else:
            speed = np.random.uniform(0.45, 0.65)
    else:
        raise NotImplementedError
    farthest_dist = speed * (n_frames-1) / fps

    for i, cum_dist in enumerate(wpath_cum_dists):
        if cum_dist <= farthest_dist:
            frame_idx = int(cum_dist / speed * fps)
            wpath_frame_idxs.append(frame_idx)
            if isinstance(wpath_joints, list) and wpath_joints[i] is not None:
                guidance_hints[frame_idx,:,:] = wpath_joints[i]
            if isinstance(wpath_poset, list) and wpath_poset[i] is not None:
                poset_hints[frame_idx,:] = wpath_poset[i]
            hints[frame_idx,0,:] = wpath[i,:]
            if scene_action_name in ["walk"]:
                xy_guidance_hints[frame_idx,0,:] = wpath[i,:]
            elif scene_action_name in ["sit", "stand_up", "lie"]:
                guidance_hints[frame_idx,0,:] = wpath[i]
        else:
            ratio = (farthest_dist - wpath_cum_dists[i-1]) / wpath_dists[i]
            inner_point = ratio * wpath[i-1,:] + (1 - ratio) * wpath[i,:]
            hints[-1,0,:] = inner_point
            if scene_action_name in ["walk"]:
                xy_guidance_hints[-1,0,:] = inner_point
            elif scene_action_name in ["sit", "stand_up", "lie"]:
                guidance_hints[-1,0,:] = inner_point
            wpath_frame_idxs.append(n_frames-1)
            break
    last_frame_at_goal = farthest_dist >= wpath_cum_dists[-1]
    if last_frame_at_goal:
        if scene_action_name in ["walk"]:
            xy_guidance_hints[-1,0,:] = wpath[-1, :]
        elif scene_action_name in ["sit", "stand_up", "lie"]:
            guidance_hints[-1,0,:] = wpath[-1, :]
        last_pose_idx = wpath_frame_idxs[-1]
        last_pose_lasting_frames = min(last_pose_lasting_frames, n_frames-last_pose_idx)
        if last_pose_lasting_frames > 1:
            xy_guidance_hints[last_pose_idx:last_pose_idx+last_pose_lasting_frames,:,:] = xy_guidance_hints[last_pose_idx:last_pose_idx+1,:,:]
            guidance_hints[last_pose_idx:last_pose_idx+last_pose_lasting_frames,:,:] = guidance_hints[last_pose_idx:last_pose_idx+1,:,:]
            hints[last_pose_idx:last_pose_idx+last_pose_lasting_frames,:,:] = hints[last_pose_idx:last_pose_idx+1,:,:]
            poset_hints[last_pose_idx:last_pose_idx+last_pose_lasting_frames,:] = poset_hints[last_pose_idx:last_pose_idx+1,:]
            wpath_frame_idxs[-1] += (last_pose_lasting_frames-1)
        i += 1
    residual_wpath = wpath[i:,:]
    return xy_guidance_hints, guidance_hints, hints, poset_hints, wpath_frame_idxs, residual_wpath

if __name__ == "__main__":
    """
    wpath = np.array([
        [0, 0, 0],
        [2, 2, 0],
        [5.,5., 0],
        ])
    hints, residual_wpath = wpath2hints(wpath)
    print(hints)
    print(residual_wpath)
    """
    print(load_replica_scenehints("apartment_0"))

