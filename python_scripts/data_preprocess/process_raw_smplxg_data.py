import numpy as np
import os
import sys
#sys.path.append("../..")
import glob
import json
import tqdm
import smplx
import torch
import pickle
from copy import deepcopy
from collections import *
import pandas
import codecs
from pandas.core.common import flatten
from scipy.spatial.transform import Rotation as R
from tools.project_config_tools import project_dir, host_device
from tools.coordinate_tools import calc_calibrate_offset, get_new_coordinate, get_mirrored_motion

amass_dataset_rename_dict = {
    'ACCAD': 'ACCAD',
    'BMLmovi': 'BMLmovi',
    'BioMotionLab_NTroje': 'BMLrub',
    'MPI_HDM05': 'HDM05',
    'CMU': 'CMU',
    'Eyes_Japan_Dataset': 'Eyes_Japan_Dataset',
    'HumanEva': 'HumanEva',
    'TCD_handMocap': 'TCDHands',
    'KIT': 'KIT',
    'Transitions_mocap': 'Transitions',
    'MPI_Limits': 'PosePrior',
    'MPI_mosh': 'MoSh',
    'DFaust_67': 'DFaust',
    'SSM_synced': 'SSM',
}

def actions_in_text(all_content, actions, excluded_actions=[]):
    for action in excluded_actions:
        if action in all_content:
            return False
    for action in actions:
        if action in all_content:
            return True
    return False

def get_body_model(body_type, gender, batch_size,device='cpu'):
    '''
    body_type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = os.path.join(project_dir, "body_models")
    body_model = smplx.create(body_model_path, model_type=body_type,
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
        return body_model.cuda()
    else:
        return body_model

def get_cats(ann, split):
    """
    DIMOS/exp_GAMMAPrimitive/utils/utils_canonicalize_babel.py
    """
    # Get sequence labels and frame labels if they exist
    seq_l, frame_l = [], []
    if 'extra' not in split:
        if ann['seq_ann'] is not None:
            seq_l = flatten([seg['act_cat'] for seg in ann['seq_ann']['labels']])
        if ann['frame_ann'] is not None:
            frame_l = flatten([seg['act_cat'] for seg in ann['frame_ann']['labels']])
    else:
        # Load all labels from (possibly) multiple annotators
        if ann['seq_anns'] is not None:
            seq_l = flatten([seg['act_cat'] for seq_ann in ann['seq_anns'] for seg in seq_ann['labels']])
        if ann['frame_anns'] is not None:
            frame_l = flatten([seg['act_cat'] for frame_ann in ann['frame_anns'] for seg in frame_ann['labels']])

    return list(seq_l), list(frame_l)

def get_babel_seq_files(babel, action='sit'):
    """
    DIMOS/exp_GAMMAPrimitive/utils/utils_canonicalize_babel.py
    """
    act_anns = defaultdict(list)  # { seq_id_1: [ann_1_1, ann_1_2], seq_id_2: [ann_2_1], ...}
    n_act_spans = 0
    dur = 0
    dur_frame = 0
    file_paths = []
    for spl in babel:
        for sid in babel[spl]:
            seq_l, frame_l = get_cats(babel[spl][sid], spl)
            # print(seq_l + frame_l)
            if 'frame_ann' in babel[spl][sid] and babel[spl][sid]['frame_ann'] is not None:
                for seg in babel[spl][sid]['frame_ann']['labels']:
                    if action in seg["act_cat"]:
                        dur_frame += seg['end_t'] - seg['start_t']
            if action in seq_l + frame_l:
                # Store all relevant mocap sequence annotations
                act_anns[sid].append(babel[spl][sid])
                # # Individual spans of the action in the sequence
                n_act_spans += Counter(seq_l + frame_l)[action]
                dur += babel[spl][sid]['dur']
                file_path = os.path.join(*(babel[spl][sid]['feat_p'].split(os.path.sep)[1:]))
                dataset_name = file_path.split(os.path.sep)[0]
                if dataset_name in amass_dataset_rename_dict:
                    file_path = file_path.replace(dataset_name, amass_dataset_rename_dict[dataset_name])
                file_path = file_path.replace('poses', 'stageii')  # file naming suffix changed in different amass versions
                # replace space
                file_path = file_path.replace(" ", "_")  # set replace count to string length, so all will be replaced
                file_paths.append(file_path)
    print('# Seqs. containing action {0} = {1}'.format(action, len(act_anns)))
    print('# Segments containing action {0} = {1}'.format(action, n_act_spans))
    print('dur:', dur)
    print('dur_frame:', dur_frame)
    print('datasets:', set([file_path.split(os.path.sep)[0] for file_path in file_paths]))

    return file_paths

def process_raw_sequence_data(transl, pose, betas, gender, bodymodel_batch, bodymodel_one, start_frame, sample_saved_idx, save_full_dir):
    end_frame = start_frame + tgt_len_subseq
    sub_transl = deepcopy(transl[start_frame:end_frame])
    sub_pose = deepcopy(pose[start_frame:end_frame])

    data_out = process_raw_subsequence_data(sub_transl, sub_pose, betas, gender, bodymodel_batch, bodymodel_one)
    sample_saved_path = os.path.join(save_full_dir, "{:06d}.npz".format(sample_saved_idx))
    np.savez_compressed(sample_saved_path, **data_out)

    if save_mirrored_data:
        sub_transl_M, sub_pose_M = get_mirrored_motion(sub_transl, sub_pose, betas, gender)
        data_out_M = process_raw_subsequence_data(sub_transl_M, sub_pose_M, betas, gender, bodymodel_batch, bodymodel_one)
        sample_saved_path = os.path.join(save_full_dir, "{:06d}_M.npz".format(sample_saved_idx))
        np.savez_compressed(sample_saved_path, **data_out_M)
    return

def process_raw_subsequence_data(sub_transl, sub_pose, betas, gender, bodymodel_batch, bodymodel_one, need_new_coordinate=True):
    sub_transl = sub_transl.copy()
    sub_pose = sub_pose.copy()
    if need_new_coordinate:
        ## perform transformation from the world coordinate to the amass coordinate
        ### get transformation from amass space to world space
        transf_rotmat, transf_transl = get_new_coordinate(bodymodel_one, betas[:10], sub_transl[:1,:], sub_pose[:1,:66])
        ### calibrate offset
        delta_T = calc_calibrate_offset(bodymodel_batch, betas[:10], sub_pose[:,:66])
        ### get new global_orient
        global_ori = R.from_rotvec(sub_pose[:,:3]).as_matrix() # to [t,3,3] rotation mat
        global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
        sub_pose[:,:3] = R.from_matrix(global_ori_new).as_rotvec()
        ### get new transl
        sub_transl = np.einsum('ij,tj->ti', transf_rotmat.T, sub_transl+delta_T-transf_transl)-delta_T

    data_out = {}
    data_out['transl'] = sub_transl
    data_out['pose'] = sub_pose
    data_out['betas'] = betas
    data_out['gender'] = gender
    data_out['fps'] = tgt_fps
    if tgt_save_skeleton:
        body_param = {}
        body_param['transl'] = torch.FloatTensor(sub_transl).cuda()
        body_param['global_orient'] = torch.FloatTensor(sub_pose[:,:3]).cuda()
        body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(tgt_len_subseq,1).cuda()
        body_param['body_pose'] = torch.FloatTensor(sub_pose[:, 3:66]).cuda()
        smplxout = bodymodel_batch(return_verts=True, **body_param)
        joints = smplxout.joints[:,:22,:].detach().squeeze().cpu().numpy()
        data_out['joints'] = joints
    return data_out

def process_humanise_raw_data():
    for action, save_base_dir in zip(humanise_selected_actions, humanise_save_dirs):
        file_full_paths = sorted(glob.glob(os.path.join(humanise_dir, action, "*/motion.pkl")))
        save_full_dir = os.path.join(processed_datasets_dir, save_base_dir)
        print(save_full_dir)
        if not os.path.exists(save_full_dir):
            os.makedirs(save_full_dir)
        prompts_file = os.path.join(save_full_dir, 'prompts.txt')
        sample_saved_idx = 0
        for file_path in tqdm.tqdm(file_full_paths):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            gender = data[0].item()
            betas = data[3][:10]
            transl = data[1]
            pose = np.concatenate([data[2], data[4]], axis=1)

            n_frames = transl.shape[0]
            if n_frames < tgt_len_subseq:
                continue
            start_frame = 0
            if gender == 'male':
                bodymodel_one = bm_one_male
                bodymodel_batch = bm_batch_male
            elif gender == 'female':
                bodymodel_one = bm_one_female
                bodymodel_batch = bm_batch_female
            else:
                bodymodel_one = bm_one_neutral
                bodymodel_batch = bm_batch_neutral
            while start_frame+tgt_len_subseq <= n_frames:
                process_raw_sequence_data(transl, pose, betas, gender, bodymodel_batch, bodymodel_one, start_frame, sample_saved_idx, save_full_dir)
                with open(prompts_file, 'a') as f:
                    f.write(action + '\n')
                    if save_mirrored_data:
                        f.write(action + '\n')

                start_frame += tgt_sample_stride
                sample_saved_idx += 1
    return

def process_babel_raw_data():
    babel_data = {}
    for split in ["train", "val"]:
        babel_data[split] = json.load(open(os.path.join(babel_dir, split + ".json")))
    for action, save_base_dir in zip(babel_selected_actions, babel_save_dirs):
        file_base_paths = get_babel_seq_files(babel_data, action=action)
        file_full_paths = [os.path.join(amass_dir, seq_path) for seq_path in file_base_paths]
        save_full_dir = os.path.join(processed_datasets_dir, save_base_dir)
        print(save_full_dir)
        if not os.path.exists(save_full_dir):
            os.makedirs(save_full_dir)
        prompts_file = os.path.join(save_full_dir, 'prompts.txt')
        sample_saved_idx = 0
        for file_path in tqdm.tqdm(file_full_paths):
            if os.path.basename(file_path) == "shape.npz":
                continue
            if not os.path.exists(file_path):
                print(file_path, ' not exist!')
                continue
            data = dict(np.load(file_path, allow_pickle=True))
            if not 'mocap_frame_rate' in data:
                continue
            fps = int(data['mocap_frame_rate'])
            assert fps%tgt_fps == 0
            downsample_rate = int(fps / tgt_fps)
            #load motion data
            gender = str(data['gender'].astype(str))
            betas = data['betas'][:10]
            transl = data['trans'][::downsample_rate]
            pose = data['poses'][::downsample_rate]
            n_frames = transl.shape[0]
            if n_frames < tgt_len_subseq:
                continue
            start_frame = 0
            if gender == 'male':
                bodymodel_one = bm_one_male
                bodymodel_batch = bm_batch_male
            elif gender == 'female':
                bodymodel_one = bm_one_female
                bodymodel_batch = bm_batch_female
            else:
                bodymodel_one = bm_one_neutral
                bodymodel_batch = bm_batch_neutral
            while start_frame+tgt_len_subseq <= n_frames:
                process_raw_sequence_data(transl, pose, betas, gender, bodymodel_batch, bodymodel_one, start_frame, sample_saved_idx, save_full_dir)
                with open(prompts_file, 'a') as f:
                    f.write(action + '\n')
                    if save_mirrored_data:
                        f.write(action + '\n')

                start_frame += tgt_sample_stride
                sample_saved_idx += 1
    return

def process_humanml3d_raw_data():
    index_file = pandas.read_csv(humanml3d_index_csv)
    total_amount = index_file.shape[0]
    for actions, excluded_actions, save_base_dir in zip(humanml3d_selected_actions, humanml3d_excluded_actions,humanml3d_save_dirs):
        save_full_dir = os.path.join(processed_datasets_dir, save_base_dir)
        print(save_full_dir)
        if not os.path.exists(save_full_dir):
            os.makedirs(save_full_dir)
        prompts_file = os.path.join(save_full_dir, 'prompts.txt')
        sample_saved_idx = 0
        for i in tqdm.tqdm(range(total_amount)):
            text_annotation_file = os.path.join(humanml3d_dir, "texts", "{:06d}.txt".format(i))
            with codecs.open(text_annotation_file) as f:
                all_content = "$".join(f.readlines())
                all_content.replace("\n", "")
            if not actions_in_text(all_content, actions, excluded_actions):
                continue
            file_path = index_file.loc[i]["source_path"]
            file_path = file_path.replace("./pose_data", amass_dir)
            file_path = file_path.replace(" ", "_")
            file_path = file_path.replace("poses.npy", "stageii.npz")
            sub_dataset_name = file_path.split("/")[len(amass_dir.split("/"))]
            if sub_dataset_name in amass_dataset_rename_dict:
                file_path = file_path.replace(sub_dataset_name, amass_dataset_rename_dict[sub_dataset_name])
            if not os.path.exists(file_path):
                if not "humanact12" in file_path:
                    print(file_path, ' not exist!')
                continue
            data = dict(np.load(file_path, allow_pickle=True))
            if not 'mocap_frame_rate' in data:
                continue
            fps = int(data['mocap_frame_rate'])
            assert fps%tgt_fps == 0
            downsample_rate = int(fps / tgt_fps)
            #load motion data
            gender = str(data['gender'].astype(str))
            betas = data['betas'][:10]
            load_start_frame = 0
            """
            https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
            """
            if 'Eyes_Japan_Dataset' in file_path:
                load_start_frame = 3*fps
            if 'MPI_HDM05' in file_path:
                load_start_frame = 3*fps
            if 'TotalCapture' in file_path:
                load_start_frame = 1*fps
            if 'MPI_Limits' in file_path:
                load_start_frame = 1*fps
            if 'Transitions_mocap' in file_path:
                load_start_frame = int(0.5*fps)
            transl = data['trans'][load_start_frame::downsample_rate]
            pose = data['poses'][load_start_frame::downsample_rate]
            motion_start_frame = index_file.loc[i]["start_frame"]#frame at 20fps
            motion_end_frame = index_file.loc[i]["end_frame"]#frame at 20fps
            start_frame = int(motion_start_frame * tgt_fps / 20)
            n_frames = transl.shape[0]
            if motion_end_frame != -1:
                n_frames = min(n_frames,int(motion_end_frame * tgt_fps / 20))
            if gender == 'male':
                bodymodel_one = bm_one_male
                bodymodel_batch = bm_batch_male
            elif gender == 'female':
                bodymodel_one = bm_one_female
                bodymodel_batch = bm_batch_female
            else:
                bodymodel_one = bm_one_neutral
                bodymodel_batch = bm_batch_neutral
            while start_frame+tgt_len_subseq <= n_frames:
                process_raw_sequence_data(transl, pose, betas, gender, bodymodel_batch, bodymodel_one, start_frame, sample_saved_idx, save_full_dir)
                with open(prompts_file, 'a') as f:
                    f.write(all_content + '\n')
                    if save_mirrored_data:
                        f.write(all_content.replace("left", "place_holder").replace("right", "left").replace("place_holder", "right") + '\n')

                start_frame += tgt_sample_stride
                sample_saved_idx += 1
    return 

def process_raw_data(process_datsets):
    if "babel" in process_datsets:
        process_babel_raw_data()
    if "humanml3d" in process_datsets:
        process_humanml3d_raw_data()
    if "humanise" in process_datasets:
        process_humanise_raw_data()
    return

if __name__ == "__main__":
    save_mirrored_data = True
    processed_datasets_dir = os.path.join(project_dir, "dataset/processed_datasets")
    process_datasets = ["humanise"]
    #AMASS DATASET
    if "babel" in process_datasets or "humanml3d" in process_datasets:
        amass_dir = os.path.join(project_dir, "dataset/amass/smplx_g")
        amass_subset = ["ACCAD","BMLmovi","BMLrub","CMU","DFaust","EKUT","Eyes_Japan_Dataset","HDM05","HumanEva","KIT","MoSh","PosePrior","SFU","SSM","TCDHands","TotalCapture","Transitions"]
    #BABEL DATASET
    if "babel" in process_datasets:
        babel_dir = os.path.join(project_dir, "dataset/amass/babel_v1-0_release/babel_v1.0_release")
        babel_selected_actions = ["sit", "lie", "walk", "turn", "stand up"]
        babel_save_dirs = ["babel/sit", "babel/lie", "babel/walk", "babel/turn", "babel/stand_up"]
    if "humanml3d" in process_datasets:
        humanml3d_dir = os.path.join(project_dir, "dataset/amass/humanml3d")
        humanml3d_index_csv = os.path.join(project_dir, "dataset/amass/humanml3d_index.csv")
        humanml3d_selected_actions = [["walk", "turn"], ["jog", "run"], ["sit"], ["lie", "lying"], ["jump"]]
        humanml3d_excluded_actions = [["sit", "lie", "lying"], ["sit", "lie", "lying"], [], [], []]
        humanml3d_save_dirs = ["humanml3d/walk_turn", "humanml3d/jog_run", "humanml3d/sit", "humanml3d/lie", "humanml3d/jump"]
    if "humanise" in process_datasets:
        humanise_dir = os.path.join(project_dir, "dataset/HUMANISE/pure_motion")
        humanise_selected_actions = ["walk", "sit", "stand_up", "lie"]
        humanise_save_dirs = ["humanise/walk", "humanise/sit", "humanise/stand_up", "humanise/lie"]

    tgt_fps = 30
    tgt_time = 1
    tgt_len_subseq = int(tgt_fps * tgt_time)
    tgt_sample_stride = int(tgt_fps * 0.5)
    tgt_save_skeleton = True

    bm_one_male = get_body_model('smplx', 'male', 1, device=host_device)
    bm_one_female = get_body_model('smplx', 'female', 1, device=host_device)
    bm_one_neutral = get_body_model('smplx', 'neutral', 1, device=host_device)
    bm_batch_male = get_body_model('smplx', 'male', tgt_len_subseq, device=host_device)
    bm_batch_female = get_body_model('smplx', 'female', tgt_len_subseq, device=host_device)
    bm_batch_neutral = get_body_model('smplx', 'neutral', tgt_len_subseq, device=host_device)

    process_raw_data(process_datasets)

