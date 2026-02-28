import argparse
import os
import sys
import shutil
from tqdm import tqdm
import numpy as np
import pickle
sys.path.append("../..")
from tools.io_tools import export_batch_human_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help='pose pickle to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    with open(params.pkl_path, 'rb') as f:
        data = pickle.load(f)
    data["smplx_param"]["transl"] += np.array([[0, 0, -0.12]])
    with open (params.pkl_path, 'wb') as f:
        pickle.dump(data, f)
    smplx_data = data["smplx_param"]
    pose_data = np.zeros((1, 103))
    pose_data[:, :3] = smplx_data["transl"]
    pose_data[:, 3:6] = smplx_data["global_orient"]
    pose_data[:, 16:79] = smplx_data["body_pose"]

    export_path =  [params.pkl_path.replace("pkl", "ply")]
    export_batch_human_mesh(pose_data, export_path)
