import os
import json
import glob
import numpy as np
import open3d as o3d
import trimesh
from tools.project_config_tools import prox_data_dir, replica_dir, random_scene_test_dir, shapenet_real_dir, default_voxel_size
from tools.occ_tools import convert_pc_to_occ

def process_prox_scene_occ():
    prox_occ_dir = os.path.join(prox_data_dir, "occ")
    semantic_anno_files = sorted(glob.glob(os.path.join(prox_data_dir, "scenes_semantics", "*withlabels.ply")))
    for semantic_anno_file in semantic_anno_files:
        scene_file = semantic_anno_file.replace("_withlabels", "")
        scene_name = scene_file.split("/")[-1].split(".")[0]
        scene_occ_dir = os.path.join(prox_occ_dir, scene_name)
        if not os.path.exists(scene_occ_dir):
            os.makedirs(scene_occ_dir)
        scene_occ_path = os.path.join(scene_occ_dir, "compact_occ.npy")
        scene_occ_info_path = os.path.join(scene_occ_dir, "compact_occ_info.json")

        semantic_anno = o3d.io.read_triangle_mesh(semantic_anno_file)
        semantic_pc = np.asarray(semantic_anno.vertices)
        semantic_label = np.mean((np.asarray(semantic_anno.vertex_colors) * 255 / 5), axis=1).astype(int)
        semantic_label[semantic_label>=41] = 41

        scene = o3d.io.read_triangle_mesh(scene_file)
        scene_pc = np.asarray(scene.vertices)
        scene_color = np.asarray(scene.vertex_colors) * 255

        scene_occ, scene_occ_info = convert_pc_to_occ(scene_pc, scene_color, semantic_pc, semantic_label, default_voxel_size)
        np.save(scene_occ_path, scene_occ)
        with open(scene_occ_info_path, "w") as f:
            json.dump(scene_occ_info, f, indent=4)
    return

def process_replica_scene_occ():
    replica_occ_dir = os.path.join(replica_dir, "occ")
    scene_instances_dirs = sorted(glob.glob(os.path.join(replica_dir, "*", "instances")))
    scene_dirs = [os.path.dirname(x) for x in scene_instances_dirs]
    
    for scene_dir in scene_dirs:
        instances_dir = os.path.join(scene_dir, "instances")
        scene_name = os.path.basename(scene_dir)
        scene_occ_dir = os.path.join(replica_occ_dir, scene_name)
        if not os.path.exists(scene_occ_dir):
            os.makedirs(scene_occ_dir)
        scene_occ_path = os.path.join(scene_occ_dir, "compact_occ.npy")
        scene_occ_info_path = os.path.join(scene_occ_dir, "compact_occ_info.json")

        all_semantic_points = []
        all_semantic_labels = []
        with open(os.path.join(scene_dir, "habitat", "info_semantic.json"), "r") as f:
            info_semantic = json.load(f)
        all_objects = info_semantic["objects"]
        for obj in all_objects:
            semantic_id = obj["class_id"]
            instance_id = obj["id"]
            instance_file = os.path.join(instances_dir, f"{instance_id}.ply")
            
            instance_mesh = trimesh.load(instance_file)
            if isinstance(instance_mesh, trimesh.Scene):
                instance_points = []
                for geometry in instance_mesh.geometry.values():
                    if hasattr(geometry, 'vertices'):
                        instance_points.append(geometry.vertices)
                if instance_points:
                    instance_points = np.vstack(instance_points)
            else:
                instance_points = instance_mesh.vertices
            
            if len(instance_points) > 0:
                instance_labels = np.full(len(instance_points), semantic_id, dtype=int)
                all_semantic_points.append(instance_points)
                all_semantic_labels.append(instance_labels)
        
        semantic_pc = np.vstack(all_semantic_points)
        semantic_label = np.concatenate(all_semantic_labels)

        scene_file = os.path.join(scene_dir, "mesh.ply")
        scene_mesh = trimesh.load(scene_file)
        scene_pc = scene_mesh.vertices
        scene_color = scene_mesh.visual.vertex_colors[:, :3]

        scene_occ, scene_occ_info = convert_pc_to_occ(scene_pc, scene_color, semantic_pc, semantic_label, default_voxel_size)
        np.save(scene_occ_path, scene_occ)
        with open(scene_occ_info_path, "w") as f:
            json.dump(scene_occ_info, f, indent=4)
    return

def process_random_scene_test_occ():
    random_scene_test_occ_dir = random_scene_test_dir + "_occ"
    random_scene_test_files = sorted(glob.glob(os.path.join(random_scene_test_dir, "*", "mesh.ply")))
    for random_scene_test_file in random_scene_test_files:
        scene_occ_dir = os.path.join(random_scene_test_occ_dir, random_scene_test_file.split('/')[-2])
        if not os.path.exists(scene_occ_dir):
            os.makedirs(scene_occ_dir)
        scene_occ_path = os.path.join(scene_occ_dir, "compact_occ.npy")
        scene_occ_info_path = os.path.join(scene_occ_dir, "compact_occ_info.json")

        scene = o3d.io.read_triangle_mesh(random_scene_test_file)
        scene_pc = np.asarray(scene.vertices)
        extra_pc = scene.sample_points_uniformly(number_of_points=1000000)
        extra_pc = np.asarray(extra_pc.points)
        scene_pc = np.concatenate([scene_pc, extra_pc], axis=0)
        scene_color = np.floor(np.ones_like(scene_pc) * 0.5 * 255)

        semantic_pc = scene_pc.copy()
        semantic_pc_z = semantic_pc[:, 2]
        semantic_label = np.ones_like(semantic_pc_z, dtype=np.int32) * 2
        semantic_label[semantic_pc_z > 1e-2] = 36

        scene_occ, scene_occ_info = convert_pc_to_occ(scene_pc, scene_color, semantic_pc, semantic_label, default_voxel_size)
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(scene_occ[:, 0], scene_occ[:, 1], scene_occ[:, 2])
        plt.show()
        exit()
        """
        np.save(scene_occ_path, scene_occ)
        with open(scene_occ_info_path, "w") as f:
            json.dump(scene_occ_info, f, indent=4)
    return

shapenet2matterport3d_id = {
    "Armchairs": 3,
    "StraightChairs": 3,
    "L-Sofas": 10,
    "Sofas": 10,
}

def process_shapenet_scene_occ():
    shapenet_real_occ_dir = shapenet_real_dir + "_occ"
    shapenet_real_files = sorted(glob.glob(os.path.join(shapenet_real_dir, "*", "*", "scene_mesh.ply")))
    for shapenet_real_file in shapenet_real_files:
        category = shapenet_real_file.split('/')[-3]
        scene_occ_dir = os.path.join(shapenet_real_occ_dir, *shapenet_real_file.split('/')[-3:-1])
        if not os.path.exists(scene_occ_dir):
            os.makedirs(scene_occ_dir)
        scene_occ_path = os.path.join(scene_occ_dir, "compact_occ.npy")
        scene_occ_info_path = os.path.join(scene_occ_dir, "compact_occ_info.json")

        scene = o3d.io.read_triangle_mesh(shapenet_real_file)
        scene_pc = np.asarray(scene.vertices)
        extra_pc = scene.sample_points_uniformly(number_of_points=1000000)
        extra_pc = np.asarray(extra_pc.points)
        scene_pc = np.concatenate([scene_pc, extra_pc], axis=0)
        scene_color = np.floor(np.ones_like(scene_pc) * 0.5 * 255)

        semantic_pc = scene_pc.copy()
        semantic_pc_z = semantic_pc[:, 2]
        semantic_label = np.ones_like(semantic_pc_z, dtype=np.int32) * 2
        semantic_label[semantic_pc_z > 1e-2] = shapenet2matterport3d_id[category]

        scene_occ, scene_occ_info = convert_pc_to_occ(scene_pc, scene_color, semantic_pc, semantic_label, default_voxel_size)
        np.save(scene_occ_path, scene_occ)
        with open(scene_occ_info_path, "w") as f:
            json.dump(scene_occ_info, f, indent=4)
    return

if __name__ == "__main__":
    datasets = ["replica"]
    if "prox" in datasets:
        process_prox_scene_occ()
    elif "replica" in datasets:
        process_replica_scene_occ()
    elif "random_scene_test" in datasets:
        process_random_scene_test_occ()
    elif "shapenet" in datasets:
        process_shapenet_scene_occ()
