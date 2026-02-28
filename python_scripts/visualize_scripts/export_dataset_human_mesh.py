import smplx
import sys
import os
import numpy as np
import trimesh
sys.path.append("../..")
from tools.io_tools import export_batch_human_mesh
from tools.project_config_tools import project_dir

if __name__=="__main__":
    export_dir = os.path.join(project_dir, "save/visualization_results")
    pose_data_dict = dict(np.load(os.path.join(project_dir, "dataset/processed_datasets/babel/sit/003000.npz"), allow_pickle=True))
    n_frames = pose_data_dict["pose"].shape[0]
    gender = str(pose_data_dict["gender"])
    pose_data = np.zeros((n_frames, 103))
    pose_data[:,:3] = pose_data_dict["transl"]
    pose_data[:,3:6] = pose_data_dict["pose"][:,:3]
    pose_data[:,6:16] = pose_data_dict["betas"][:].reshape(1,-1)
    pose_data[:,16:79] = pose_data_dict["pose"][:,3:66]
    export_batch_human_mesh(pose_data, export_dir, gender=gender)
