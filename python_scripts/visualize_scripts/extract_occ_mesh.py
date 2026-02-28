import sys
import os
import trimesh
import numpy as np
import argparse
import glob
from pathlib import Path
from tools.occ_tools import OccInfo, Occupancy
from tools.project_config_tools import prox_data_dir, replica_dir, random_scene_test_dir, shapenet_real_dir

def extract_general_occ_mesh(scene_path: Path, scene_info_path: Path, output_mesh_path: Path, semantic_anno_type: str):
    scene_info = OccInfo(scene_info_path)
    occ = Occupancy(semantic_anno_type)
    occ.initialize_from_compact_npy(scene_path, scene_info.grid_size)
    colors = occ.to_semantic_colors()
    scene = occ.scene[...,-1]

    scene = trimesh.voxel.VoxelGrid(
        encoding=trimesh.voxel.encoding.DenseEncoding(scene.astype(bool)),
        transform=trimesh.transformations.scale_and_translate(
            scale=scene_info.voxel_size,
            translate=scene_info.min_bound
        ),
    )
    mesh = scene.as_boxes(colors)
    face_colors = np.array(mesh.visual.face_colors, dtype=np.float32) / 255.
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.float32)

    for face_idx, face in enumerate(mesh.faces):
        for vert_idx in face:
            if vertex_colors[vert_idx].sum() == 0:
                vertex_colors[vert_idx] = face_colors[face_idx]
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    mesh.export(output_mesh_path, file_type="ply", include_attributes=True)
    return

def process_general_occ_folder(folder_path, semantic_anno_type):
    scene_path = os.path.join(folder_path, "compact_occ.npy")
    scene_info_path = os.path.join(folder_path, "compact_occ_info.json")
    output_mesh_path = os.path.join(folder_path, "occ.ply")
    if os.path.exists(scene_path) and os.path.exists(scene_info_path):
        print(f"Processing {folder_path}")
        extract_general_occ_mesh(scene_path, scene_info_path, output_mesh_path, semantic_anno_type=semantic_anno_type)
    else:
        print(f"Skipping {folder_path}: missing required files.")
    return

def extract_random_scene_test_occ_mesh():
    semantic_anno_type="matterport3d"
    random_scene_test_occ_dir = random_scene_test_dir + "_occ"
    for folder_name in sorted(os.listdir(random_scene_test_occ_dir)):
        folder_path = os.path.join(random_scene_test_occ_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        process_general_occ_folder(folder_path, semantic_anno_type)
    return

def extract_prox_occ_mesh():
    semantic_anno_type="matterport3d"
    prox_occ_dir = os.path.join(prox_data_dir, "occ")
    for folder_name in sorted(os.listdir(prox_occ_dir)):
        folder_path = os.path.join(prox_occ_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        process_general_occ_folder(folder_path, semantic_anno_type)
    return

def extract_replica_occ_mesh():
    semantic_anno_type="replica"
    replica_occ_dir = os.path.join(replica_dir, "occ")
    for folder_name in sorted(os.listdir(replica_occ_dir)):
        folder_path = os.path.join(replica_occ_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        process_general_occ_folder(folder_path, semantic_anno_type)
    return

def extract_shapenet_occ_mesh():
    semantic_anno_type="matterport3d"
    shapenet_real_occ_dir = shapenet_real_dir + "_occ"
    for file_name in sorted(glob.glob(os.path.join(shapenet_real_occ_dir, "*", "*", "compact_occ.npy"))):
        folder_path = os.path.dirname(file_name)
        process_general_occ_folder(folder_path, semantic_anno_type)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=["prox"], help="datasets to be processed.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    datasets = args.datasets
    if "prox" in datasets:
        extract_prox_occ_mesh()
    elif "replica" in datasets:
        extract_replica_occ_mesh()
    elif "random_scene_test" in datasets:
        extract_random_scene_test_occ_mesh()
    elif "shapenet" in datasets:
        extract_shapenet_occ_mesh()
