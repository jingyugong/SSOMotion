import os
import json
import trimesh
import numpy as np
from pathlib import Path
from deps.dimos.synthesize.get_scene import ReplicaScene

class ReplicaSceneV2(ReplicaScene):
    def __init__(self, scene_name, replica_folder, build=False, zero_floor=False):
        self.name = scene_name
        if not isinstance(replica_folder, Path):
            replica_folder = Path(replica_folder)
        self.replica_folder = replica_folder
        self.room_folder = replica_folder / scene_name
        self.instance_folder = replica_folder / scene_name / 'instances'
        self.ply_path = replica_folder / scene_name / 'habitat' / 'mesh_semantic.ply'
        self.mesh = trimesh.load_mesh(self.ply_path, process=False)  # process=True will change vertices and cause error!

        # per instance category labels
        json_path = os.path.join(replica_folder, scene_name, 'habitat', 'info_semantic.json')
        with open(json_path, 'r') as f:
            self.semantic_mapping = json.load(f)
        self.instance_to_category = self.semantic_mapping['id_to_label']
        self.num_instances = len(self.semantic_mapping['id_to_label'])
        self.category_ids = [self.instance_to_category[instance_id] for instance_id in range(self.num_instances)]
        self.category_names = [self.semantic_mapping['classes'][category_id - 1]['name'] for category_id in
                               self.category_ids]
        self.instances = self.get_instances(build=build)

        # load sdf, https: // github.com / mohamedhassanmus / POSA / blob / de21b40f22316cfb02ec43021dc5f325547c41ca / src / data_utils.py  # L99
        replica_sdf_folder = Path(replica_folder)
        with open(replica_sdf_folder / scene_name / (scene_name + '_sdf.json'), 'r') as f:
            sdf_data = json.load(f)
            grid_dim = sdf_data['dim']
            grid_min = (np.array(sdf_data['centroid']) - sdf_data['scale']).astype(np.float32)
            grid_max = (np.array(sdf_data['centroid']) + sdf_data['scale']).astype(np.float32)
            voxel_size = (grid_max - grid_min) / grid_dim
        sdf = np.load(replica_sdf_folder / scene_name / (scene_name + '_sdf.npy')).astype(np.float32)
        sdf = sdf.reshape((grid_dim, grid_dim, grid_dim, 1))
        self.sdf = sdf
        self.sdf_config = {'grid_min': grid_min, 'grid_max': grid_max, 'grid_dim': grid_dim}

        self.floor_height = self.raw_floor_height = floor_height = self.get_floor_height()
        if zero_floor:
            self.mesh.vertices[:, 2] -= floor_height
            for instance in self.instances:
                instance.vertices[:, 2] -= floor_height
            self.sdf_config['grid_min'] -= np.array([0, 0, floor_height])
            self.sdf_config['grid_max'] -= np.array([0, 0, floor_height])
            self.floor_height = self.get_floor_height()

        self.mesh_with_accessory = {}

    def get_category_indices(self, cagegory_name):
        indices = np.where(np.array(self.category_names) == cagegory_name)[0]
        return indices
