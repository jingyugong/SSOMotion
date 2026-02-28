import argparse
import os
import sys
import shutil
from tqdm import tqdm
import numpy as np
sys.path.append("../..")
from tools.io_tools import export_batch_human_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", type=str, required=True, help='results.npy to be rendered.')
    parser.add_argument("--sample_i", type=int, default=0, help='')
    parser.add_argument("--rep_i", type=int, default=0, help='')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    sample = np.load(params.npy_path, allow_pickle=True)[None][0]
    absl_i = params.rep_i * sample["num_samples"] + params.sample_i
    sample_i = sample["motion"][absl_i]
    pose_data = np.zeros((sample_i.shape[0], 103))
    pose_data[:, :3] = sample_i[:, 66:]
    pose_data[:, 3:6] = sample_i[:, 0:3]
    pose_data[:, 16:79] = sample_i[:, 3:66]

    dirname = os.path.dirname(params.npy_path)
    export_dir =  os.path.join(dirname, "sample{:02d}_rep{:02d}_obj".format(params.sample_i, params.rep_i))
    export_batch_human_mesh(pose_data, export_dir)
