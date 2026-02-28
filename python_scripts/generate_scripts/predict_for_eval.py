
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import pickle
import json
from pathlib import Path
from utils.parser_util import generate_in_scene_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader, VoidDataset
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from tools.project_config_tools import project_dir
from tools.io_tools import load_random_joint_hint_from_file, load_feature69dim_from_file,convert_joints_smplx2smpl, extract_smplx_joints
from tools.motion_tools import load_body_mesh_model_batch
from tools.agent_tools import MultiRoundMotionAgentforEval
from tools.coordinate_tools import get_new_coordinate, point_coordinate_transform, motion_coordinate_transform, coordinate_inv_transform, calc_calibrate_offset 
from tools.guidance_tools import wpath2hints, data2scenehints
from data_loaders.scene2motion.humanise import HumaniseMotion

def convert_to_world_coordinate(motion_l, transf_rotmat, transf_transl, bodymodel_batch):
    betas = np.zeros((10), dtype=np.float32)
    delta_T = calc_calibrate_offset(bodymodel_batch, betas, motion_l[:,:66], local_device='cpu')
    orient_l = R.from_rotvec(motion_l[:,:3]).as_matrix()
    orient_w = np.einsum("ij, tjk->tik", transf_rotmat.reshape(3,3), orient_l)
    orient_w = R.from_matrix(orient_w).as_rotvec()
    trans_l = motion_l[:,66:]
    trans_w = np.einsum("ij, tj->ti", transf_rotmat.reshape(3,3), trans_l + delta_T) + transf_transl.reshape(-1, 3) - delta_T
    motion_w = np.concatenate([orient_w, motion_l[:,3:66], trans_w], axis=-1)
    return motion_w

def main():
    args = generate_in_scene_args()
    args.action_name = args.single_round_demo_id.split("_")[-1]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['kit', 'humanml']:
        max_frames = 196
    elif args.dataset in ['mixedmotion', 'mixedscenemotion', 'humanise']:
        max_frames = 30
    else:
        max_frames = 60
    if args.dataset == 'kit':
        fps = 12.5
    elif args.dataset in ['mixedmotion', 'mixedscenemotion', 'humanise']:
        fps = 30
    else:
        fps = 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    bodymodel_batch = load_body_mesh_model_batch(n_frames, body_type='smplx', gender='neutral', device='cpu')
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = VoidDataset()
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if args.test_scene_type == 'humanise':
        eval_dataset = get_dataset_loader(name=args.test_scene_type, batch_size=1, num_frames=max_frames, controlnet=args.controlnet, split='test', num_workers=0, shuffle=False)
        inner_dataset = eval_dataset.dataset
    else:
        raise NotImplementedError
    for pair_i, data_item in enumerate(eval_dataset):
        anno_index, index_offset = inner_dataset.index2loc[pair_i]
        anno_item = inner_dataset.anno_list[anno_index]
        scene_id = anno_item["scene"]
        motion_id = anno_item["motion"]
        action_label = anno_item["action"]
        action_text = anno_item["utterance"]
        gt_motion, cond = data_item
        cond['y'] = {key: val.to(args.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
        transf_rotmat = cond['y']['scene_hint']['transf_rotmat'].detach().cpu().numpy()
        transf_transl = cond['y']['scene_hint']['transf_transl'].detach().cpu().numpy()
        gt_motion = torch.transpose(gt_motion, 1, 3).view(n_frames,-1).detach().cpu().numpy()
        gt_motion_w = convert_to_world_coordinate(gt_motion, transf_rotmat, transf_transl, bodymodel_batch)
        sample_fn = diffusion.p_sample_loop
        for rep_i in range(args.num_repetitions):
            print(f'### Sample [{pair_i}] [{rep_i}]')
            out_path = os.path.join(args.output_dir, f'sample{pair_i}', f'rep{rep_i}')
            if os.path.exists(out_path):
                continue
            print(f'### Sampling [repetitions #{rep_i}]')

            sample_l = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (1, model.njoints, model.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample_l = torch.transpose(sample_l, 3, 1).view(n_frames, -1).detach().cpu().numpy()
            sample_w = convert_to_world_coordinate(sample_l, transf_rotmat, transf_transl, bodymodel_batch)
            with open(os.path.join(project_dir, "dataset", "HUMANISE", "occ", scene_id, f"{scene_id}_vh_clean.compact.json"), 'rb') as f:
                occ_info = json.load(f)
            occ_center = (np.array(occ_info["min_bound"])+np.array(occ_info["max_bound"]))/2

            for result_src in ["gt", "pred"]:
                if result_src == "gt":
                    sample_save = gt_motion_w
                elif result_src == "pred":
                    sample_save = sample_w
                sample_save[:, 66:] = sample_save[:, 66:] + occ_center.reshape(1,3)
                pickle_data = {
                    'motion':{
                        'gender': 'neutral',
                        'betas': np.zeros((10)),
                        'smplx_params': sample_save.reshape(1, *sample_save.shape),
                    },
                    'scene_id': scene_id,
                    'motion_id': motion_id,
                    'action_label': action_label,
                    'action_text': action_text,
                }
                out_path = os.path.join(args.output_dir, f'sample{pair_i}', f'rep{rep_i}', result_src)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                pickle_path = os.path.join(out_path, 'results.pkl')
                with open(pickle_path, 'wb') as f:
                    pickle.dump(pickle_data, f)

                all_motions = sample_save.reshape(1, *sample_save.shape)
                all_text = [action_text]
                all_lengths = np.array([n_frames])
                npy_path = os.path.join(out_path, 'results.npy')
                print(f"saving results file to [{npy_path}]")
                np.save(npy_path,
                        {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                         'num_samples': 1, 'num_repetitions': 1})
                with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
                    fw.write('\n'.join(all_text))
                with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
                    fw.write('\n'.join([str(l) for l in all_lengths]))
                if model.data_rep == 'mixed_vec':
                    all_motions_smplx = np.zeros((all_motions.shape[0], all_motions.shape[1], 24, 3))
                    for i in range(all_motions.shape[0]):
                        all_motions_smplx[i] = convert_joints_smplx2smpl(extract_smplx_joints(all_motions[i]))
                    xforward2zforward = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
                    all_motions = np.matmul(all_motions_smplx, xforward2zforward)
                    all_motions = all_motions.transpose(0, 2, 3, 1)
                print(f"saving visualizations to [{out_path}]...")
                skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

                sample_files = []

                sample_print_template, row_print_template, all_print_template, \
                sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)

                for sample_i in range(args.num_samples):
                    rep_files = []
                    caption = all_text[sample_i]
                    length = all_lengths[sample_i]
                    print(length)
                    motion = all_motions[sample_i].transpose(2, 0, 1)[:length]
                    save_file = sample_file_template.format(sample_i, 0)
                    print(sample_print_template.format(caption, sample_i, 0, save_file))
                    animation_save_path = os.path.join(out_path, save_file)
                    #plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=fps)
                abs_path = os.path.abspath(out_path)
                print(f'[Done] Results are at [{abs_path}]')
    return

def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template

if __name__ == "__main__":
    main()
