import os
import numpy as np
import torch
from pathlib import Path
import trimesh
from tools.project_config_tools import *
from tools.io_tools import extract_smplx_joints, load_pkl
from tools.motion_tools import load_body_mesh_model_batch, place_stand_pose_before_sit
from tools.coordinate_tools import get_new_coordinate, point_coordinate_transform, motion_coordinate_transform
from tools.guidance_tools import wpath2hints, data2scenehints, load_prox_scenehints, load_replica_scenehints
from tools.navigation_tools import adjust4walkable
from deps.dimos.exp_GAMMAPrimitive.utils.environments import BatchGeneratorSceneTest, BatchGeneratorInteractionTest

def single_round_demos(demo_id, n_frames):
    scene_action = demo_id.split('_')[-1]
    init_data = fetch_single_round_init_data(demo_id)

    init_betas = init_data['betas'].detach().cpu().numpy()
    init_transl = init_data['transl'].detach().cpu().numpy()
    init_pose = np.concatenate((init_data['global_orient'].detach().cpu().numpy(),init_data['body_pose'].detach().cpu().numpy()), axis=1)
    gender = init_data['gender']
    bodymodel_one = load_body_mesh_model_batch(1, gender=gender)

    transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, init_betas, init_transl, init_pose)
    if scene_action == "walk":
        world_transl = init_transl
        world_pose = init_pose
    elif scene_action == "sit":
        world_transl = np.concatenate([init_transl, init_data['target_body']['transl'].detach().cpu().numpy()], axis=0)
        world_pose = np.concatenate([init_pose, np.concatenate((init_data['target_body']['global_orient'].detach().cpu().numpy(),init_data['target_body']['body_pose'].detach().cpu().numpy()), axis=1)], axis=0)
    local_transl, local_pose = motion_coordinate_transform(gender, init_betas, world_transl, world_pose, transf_rotmat, transf_transl)
    local_joints = extract_smplx_joints(np.concatenate([local_pose, local_transl, init_betas.reshape(-1,10).repeat(local_pose.shape[0], axis=0)], axis=1), gender=gender)[:,:22,:]
    local_wpath = point_coordinate_transform(init_data['wpath'].detach().cpu().numpy(), transf_rotmat, transf_transl)
    if scene_action == 'walk':
        local_wpath_joints = [local_joints[0]] + [None] * (len(local_wpath) - 1)
    elif scene_action == 'sit':
        local_wpath_joints = [local_joints[0], local_joints[1]]
    xy_guidance_hints, guidance_hints, hints, _, _, _  = wpath2hints(local_wpath, local_wpath_joints, None, scene_action, n_frames)
    scene_hints = data2scenehints(init_data, transf_rotmat, transf_transl, src_fmt='dimos')

    ret = {
        'hints': hints,
        'xy_guidance_hints': xy_guidance_hints,
        'guidance_hints': guidance_hints,
        'scene_hints': scene_hints,
        'transf_rotmat': transf_rotmat,
        'transf_transl': transf_transl,
        'gender': gender,
        'init_betas': init_betas,
    }
    return ret

def fetch_single_round_init_data(demo_id):
    if demo_id == 'test_room_walk':
        batch_gen = BatchGeneratorSceneTest(dataset_path="", body_model_path="dataset/dimos_data/models_smplx_v1_1/models/")
        ret = batch_gen.next_body(visualize=0,use_zero_pose=1,use_zero_shape=1,scene_path=Path("dataset/dimos_data/test_room/room.ply"), floor_height=0.0, navmesh_path=Path("dataset/dimos_data/test_room/navmesh_tight.ply"), wpath_path=Path("dataset/dimos_data/test_room/test_room_path_0.pkl"), path_name="test_room_path_0", last_motion_path=None, clip_far=1, random_orient=0, res=16, extent=0.8)
    elif demo_id == 'test_room_sit':
        batch_gen = BatchGeneratorInteractionTest(dataset_path="", body_model_path="dataset/dimos_data/models_smplx_v1_1/models/")
        ret = batch_gen.next_body(visualize=0,use_zero_pose=1,use_zero_shape=1,scene_path=Path("dataset/dimos_data/test_room/room.ply"), floor_height=0.0, sdf_path=Path("dataset/dimos_data/test_room/sofa_sdf_gradient.pkl"), mesh_path="dataset/dimos_data/test_room/sofa.ply", target_body_path="../DIMOS/results/tmp/test_room/inter_sofa_sit/target_body.pkl", last_motion_path=None, target_point_path="", start_point_path="../DIMOS/results/tmp/test_room/inter_sofa_sit/target_point.pkl")
    else:
        raise NotImplementedError
    return ret


def prox_multi_round_demos(demo_id):
    demo_id_2_pose_list = {
        "BasementSittingBooth+sit-sofa_walk+0" : ["sit-sofa-1/selected/2.pkl", "stand-floor-0/selected/1.pkl"],
        "MPH1Library+walk+0" : ["stand-floor-0/selected/3.pkl", "stand-floor-0/selected/1.pkl"],
        "MPH8+walk_sit-bed+0" : ["stand-floor-0/selected/1.pkl", "stand-floor-0/selected/2.pkl", "sit-bed-0/selected/0.pkl"],
        "MPH8+walk_sit-bed_lie-bed+0" : ["stand-floor-0/selected/1.pkl", "stand-floor-0/selected/2.pkl", "sit-bed-0/selected/0.pkl", "lie-bed-0/selected/1.pkl"],
        "MPH11+walk_sit-sofa+0" : ["stand-floor-0/selected/0.pkl", "sit-sofa-0/selected/1.pkl"],
        "MPH11+walk_sit-sofa_lie-sofa+0" : ["stand-floor-0/selected/0.pkl", "sit-sofa-0/selected/2.pkl", "lie-sofa-0/selected/0.pkl"],
        "MPH16+walk+0" : ["stand-floor-0/selected/5.pkl", "stand-floor-0/selected/2.pkl"],
        "MPH16+walk_sit-chair+0" : ["stand-floor-0/selected/5.pkl", "sit-chair-0/selected/0.pkl"],
        "MPH16+walk_sit-bed+0" : ["stand-floor-0/selected/5.pkl", "sit-bed-0/selected/0.pkl"],
        "MPH16+sit-bed_walk_sit-chair+0" : ["sit-bed-0/selected/0.pkl", "stand-floor-0/selected/1.pkl", "sit-chair-0/selected/0.pkl"],
        "MPH112+walk_sit-bed+0" : ["stand-floor-0/selected/1.pkl", "sit-bed-0/selected/1.pkl"],
        "N0Sofa+walk_sit-sofa_walk+0" : ["stand-floor-0/selected/2.pkl", "sit-sofa-1/selected/2.pkl", "stand-floor-0/selected/1.pkl"],
        "N3Library+walk_sit-chair+0": ["stand-floor-0/selected/0.pkl", "sit-chair-1/selected/0.pkl"],
        "N3Office+sit-chair_walk+0" : ["sit-chair-0/selected/0.pkl", "stand-floor-0/selected/1.pkl"],
        "N3OpenArea+sit-sofa_walk_sit-chair+0" : ["sit-sofa-0/selected/2.pkl", "stand-floor-0/selected/1.pkl", "sit-chair-1/selected/0.pkl"],
        "N3OpenArea+lie-sofa_sit-sofa_walk_sit-chair+0" : ["lie-sofa-0/selected/0.pkl", "sit-sofa-0/selected/2.pkl", "stand-floor-0/selected/1.pkl", "sit-chair-1/selected/0.pkl"],
        "Werkraum+walk_sit-chair_walk+0" : ["stand-floor-0/selected/1.pkl", "sit-chair-0/selected/0.pkl", "stand-floor-0/selected/0.pkl"],
    }

    scene_name = demo_id.split("+")[0]
    scene_hints = load_prox_scenehints(scene_name, device=host_device)
    floor_height = scene_hints['floor_height']

    action_pose_list = demo_id_2_pose_list[demo_id]
    action_map = {"sit": "sit", "stand": "walk", "lie": "lie"}
    action_list = [action_map[action_pose_file.split("-")[0]] for action_pose_file in action_pose_list]
    smplx_pose_list = [load_pkl(os.path.join(prox_data_dir, "coins_poses", scene_name, action_pose_file))['smplx_param'] for action_pose_file in action_pose_list]
    ret = general_multi_round_demo_prepare(scene_name, scene_hints, floor_height, action_list, smplx_pose_list)
    return ret


def replica_multi_round_demos(demo_id):
    demo_id_2_pose_list = {
        "hotel_0+sit-bed_sit-chair+0" : ["sit-bed-0/selected/2.pkl", "sit-chair-1/selected/2.pkl"],
        "hotel_0+sit-bed_sit-chair+1" : ["sit-bed-0/selected/2.pkl", "sit-chair-1/selected/1.pkl"],
        "office_0+sit-chair_walk_sit-sofa+0" : ["sit-chair-0/selected/0.pkl", "stand-floor-0/selected/2.pkl", "sit-sofa-1/selected/4.pkl"],
        "office_0+sit-sofa_walk_sit-chair+0" : ["sit-sofa-1/selected/0.pkl", "stand-floor-0/selected/0.pkl", "sit-chair-0/selected/0.pkl"],
        "office_0+sit-sofa_walk_sit-sofa+0" : ["sit-sofa-1/selected/4.pkl", "stand-floor-0/selected/2.pkl", "sit-sofa-1/selected/4.pkl"],
        "office_0+sit-chair_sit-sofa+0" : ["sit-chair-0/selected/0.pkl", "sit-sofa-1/selected/4.pkl"],
        "office_1+sit-chair_walk+0" : ["sit-chair-0/selected/2.pkl", "stand-floor-1/selected/0.pkl"],
        "office_2+sit-chair_walk+0" : ["sit-chair-3/selected/1.pkl", "stand-floor-0/selected/1.pkl"],
        "office_2+walk_sit-chair+0" : ["stand-floor-0/selected/1.pkl", "sit-chair-3/selected/1.pkl"],
        "office_2+walk_sit-chair+1" : ["stand-floor-0/selected/1.pkl", "sit-chair-3/selected/2.pkl"],
        "office_3+walk_sit-sofa+0" : ["stand-floor-0/selected/1.pkl", "sit-sofa-0/selected/1.pkl"],
        "office_4+sit-chair_walk+0" : ["sit-chair-3/selected/0.pkl", "stand-floor-0/selected/1.pkl"],
        "room_0+sit-chair_sit-sofa+0" : ["sit-chair-1/selected/0.pkl", "sit-sofa-0/selected/0.pkl"],
        "room_0+sit-sofa_sit-stool+0" : ["sit-sofa-1/selected/3.pkl", "sit-stool-0/selected/1.pkl"],
        "room_0+sit-stool_sit-sofa+0" : ["sit-stool-0/selected/1.pkl", "sit-sofa-0/selected/2.pkl"],
        "room_0+sit-stool_sit-sofa_lie-sofa+0" : ["sit-stool-0/selected/1.pkl", "sit-sofa-1/selected/3.pkl", "lie-sofa-1/selected/0.pkl"],
        "room_0+sit-stool_sit-sofa_lie-sofa+1" : ["sit-stool-0/selected/1.pkl", "sit-sofa-0/selected/2.pkl", "lie-sofa-0/selected/2.pkl"],
        "room_1+walk_sit-bed+0" : ["stand-floor-0/selected/0.pkl", "sit-bed-0/selected/1.pkl"]
    }

    action_pose_list = demo_id_2_pose_list[demo_id]
    action_map = {"sit": "sit", "stand": "walk", "lie": "lie"}
    action_list = [action_map[action_pose_file.split("-")[0]] for action_pose_file in action_pose_list]

    scene_name = demo_id.split("+")[0]
    scene_path = os.path.join(replica_dir, "coins_poses", scene_name)
    scene_hints = load_replica_scenehints(scene_name, device=host_device)
    floor_height = scene_hints['floor_height']
    
    smplx_pose_list = [load_pkl(os.path.join(scene_path, action_pose_file))['smplx_param'] for action_pose_file in action_pose_list]
    ret = general_multi_round_demo_prepare(scene_name, scene_hints, floor_height, action_list, smplx_pose_list)
    return ret 


def general_multi_round_demo_prepare(scene_name, scene_hints, floor_height, action_list, smplx_pose_list):
    gender = "neutral"
    init_betas = smplx_pose_list[0]['betas']
    init_transl = smplx_pose_list[0]['transl']
    init_pose = np.concatenate([smplx_pose_list[0]['global_orient'], smplx_pose_list[0]['body_pose']], axis=-1)

    poset_list = [np.concatenate([data['global_orient'], data['body_pose'], data['transl']], axis=-1) for data in smplx_pose_list]
    wpath = [extract_smplx_joints(np.concatenate([poset_list[0], init_betas], axis=-1), gender)[:,0].reshape(-1)]
    wpath_poset = [poset_list[0]]
    wpath_action = [action_list[0]]
    for i in range(len(action_list)-1):
        curr_state = action_list[i]
        if action_list[i+1] != curr_state:
            if (curr_state == "sit" and action_list[i+1] == "lie") or (curr_state == "lie" and action_list[i+1] == "sit"):
                pass
            elif (curr_state == "walk" and action_list[i+1] == "sit") or (curr_state == "sit" and action_list[i+1] == "walk"):
                if curr_state == "walk":
                    inter_poset = place_stand_pose_before_sit(poset_list[i+1], init_betas, floor_height, "walk_sit")
                else:
                    inter_poset = place_stand_pose_before_sit(poset_list[i], init_betas, floor_height, "sit_walk")
                wpath.append(extract_smplx_joints(np.concatenate([inter_poset, init_betas], axis=-1), gender)[:,0].reshape(-1))
                if control_stand_poset_before_sit: # type: ignore
                    wpath_poset.append(inter_poset)
                else:
                    wpath_poset.append(None)
                wpath_action.append("walk")
            else:
                raise NotImplementedError
        else:
            if (curr_state == "sit" and np.linalg.norm(poset_list[i][-3:] - poset_list[i+1][-3:]) >= 0.5):
                inter_poset1 = place_stand_pose_before_sit(poset_list[i], init_betas, floor_height, "sit_walk")
                inter_poset2 = place_stand_pose_before_sit(poset_list[i+1], init_betas, floor_height, "walk_sit")
                wpath.append(extract_smplx_joints(np.concatenate([inter_poset1, init_betas], axis=-1), gender)[:,0].reshape(-1))
                wpath.append(extract_smplx_joints(np.concatenate([inter_poset2, init_betas], axis=-1), gender)[:,0].reshape(-1))
                if control_stand_poset_before_sit: # type: ignore
                    wpath_poset += [inter_poset1, inter_poset2]
                else:
                    wpath_poset += [None, None]
                wpath_action += ["walk", "walk"]
            elif (curr_state == "lie" and np.linalg.norm(poset_list[i][-3:] - poset_list[i+1][-3:]) >= 0.5):
                raise NotImplementedError
            else:
                pass
        wpath.append(extract_smplx_joints(np.concatenate([poset_list[i+1], init_betas], axis=-1), gender)[:,0].reshape(-1))
        wpath_poset.append(poset_list[i+1])
        wpath_action.append(action_list[i+1])
    wpath, wpath_poset, wpath_action = adjust4walkable(wpath, wpath_poset, wpath_action, scene_name, floor_height)
    ret = {
        'gender': gender,
        'init_betas': init_betas.reshape(10),
        'init_transl': init_transl,
        'init_pose': init_pose,
        'wpath': torch.tensor(np.array(wpath), dtype=torch.float32, device=host_device),
        'wpath_poset': wpath_poset,
        'wpath_action': wpath_action,
        'scene_hints': scene_hints,
    }
    return ret

def test_room_multi_round_demos(demo_id):
    if demo_id == 'test_room+sit_down_and_up+0':
        init_data = fetch_single_round_init_data('test_room_sit')

        gender = init_data['gender']
        init_betas = init_data['betas'].detach().cpu().numpy()
        init_transl = init_data['transl'].detach().cpu().numpy()
        init_pose = np.concatenate((init_data['global_orient'].detach().cpu().numpy(),init_data['body_pose'].detach().cpu().numpy()), axis=1)

        wpath = init_data['wpath']
        wpath = torch.cat([wpath, wpath[:1]], dim=0)
        wpath_poset = [np.concatenate([init_pose, init_transl], axis=1), np.concatenate([init_data['target_body']['global_orient'].detach().cpu().numpy(),init_data['target_body']['body_pose'].detach().cpu().numpy(), init_data['target_body']['transl'].detach().cpu().numpy()], axis=1), None]
        wpath_action = ['walk', 'sit', 'walk']
        scene_hints = data2scenehints(init_data, None, None, src_fmt='dimos')
    elif demo_id == 'test_room+walk_sit+0':
        init_data = fetch_single_round_init_data("test_room_walk")

        gender = init_data['gender']
        init_betas = init_data['betas'].detach().cpu().numpy()
        init_transl = init_data['transl'].detach().cpu().numpy()
        init_pose = np.concatenate((init_data['global_orient'].detach().cpu().numpy(),init_data['body_pose'].detach().cpu().numpy()), axis=1)

        wpath = init_data['wpath']
        wpath_poset = [np.concatenate([init_pose, init_transl], axis=1)] + [None] * (wpath.shape[0] - 1)
        wpath_action = ['walk'] * wpath.shape[0]

        stage2_demo = multi_round_demos("test_room_sit_down_and_up")
        wpath = torch.cat([wpath, stage2_demo['wpath']], dim=0)
        wpath_poset = wpath_poset + [None] + stage2_demo['wpath_poset'][1:]
        wpath_action = wpath_action + stage2_demo['wpath_action']
        scene_hints = stage2_demo['scene_hints']
    ret = {
        'gender': gender,
        'init_betas': init_betas,
        'init_transl': init_transl,
        'init_pose': init_pose,
        'wpath': wpath,
        'wpath_poset': wpath_poset,
        'wpath_action': wpath_action,
        'scene_hints': scene_hints,
    }
    return ret

def multi_round_demos(demo_id):
    scene_name = demo_id.split("+")[0]
    print(scene_name)
    if scene_name == "test_room":
        ret = test_room_multi_round_demos(demo_id)
    elif scene_name in prox_room_list:
        ret = prox_multi_round_demos(demo_id) 
    elif scene_name in replica_room_list:
        ret = replica_multi_round_demos(demo_id)
    else:
        raise NotImplementedError
    return ret
