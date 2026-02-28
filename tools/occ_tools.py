import os
import numpy as np
import json
from tools.visualization_tools import CategoryLabel

def export_local_occ(occ, voxel_size, scene_occ_dir):
    num_channels, x_size, y_size, z_size = occ.shape
    assert num_channels == 4
    scene_occ_path = os.path.join(scene_occ_dir, "compact_occ.npy")
    scene_occ_info_path = os.path.join(scene_occ_dir, "compact_occ_info.json")
    x_s, y_s, z_s = np.nonzero(occ[3, :, :, :])
    scene_color = []
    semantic_label = []
    for x, y, z in zip(x_s, y_s, z_s):
        scene_color.append(occ[0:3, x, y, z])
        semantic_label.append(occ[3, x, y, z])
    scene_color = np.array(scene_color)
    semantic_label = np.array(semantic_label).astype(np.int32)
    x_s = (x_s - (x_size - 1) / 2) * voxel_size
    y_s = (y_s - (y_size - 1) / 2) * voxel_size
    z_s = (z_s - (z_size - 1) / 2) * voxel_size
    scene_pc = np.concatenate([x_s.reshape(-1, 1), y_s.reshape(-1, 1), z_s.reshape(-1, 1)], axis=1)
    scene_pc += 0.5 * voxel_size

    scene_occ, scene_occ_info = convert_pc_to_occ(scene_pc, scene_color, scene_pc, semantic_label, voxel_size)
    np.save(scene_occ_path, scene_occ)
    with open(scene_occ_info_path, "w") as f:
        json.dump(scene_occ_info, f, indent=4)
    return

def convert_pc_to_occ(scene_pc, scene_color, semantic_pc, semantic_label, voxel_size):
    min_bound = np.min(scene_pc, axis=0) - 0.5 * voxel_size
    max_bound = np.max(scene_pc, axis=0) + 0.5 * voxel_size
    grid_size = np.ceil((max_bound - min_bound) / voxel_size)
    max_bound = min_bound + grid_size * voxel_size
    voxel_index_to_color = {}
    voxel_index_to_label = {}
    for i in range(scene_pc.shape[0]):
        occ_index = np.floor((scene_pc[i] - min_bound) / voxel_size).astype(np.int32)
        occ_index = tuple(occ_index)
        if occ_index not in voxel_index_to_color:
            voxel_index_to_color[occ_index] = []
        voxel_index_to_color[occ_index].append(scene_color[i])

    for i in range(semantic_pc.shape[0]):
        if ((semantic_pc[i] - min_bound) < 0).any() or ((semantic_pc[i] - max_bound) >= 0).any():
            continue
        occ_index = np.floor((semantic_pc[i] - min_bound) / voxel_size).astype(np.int32)
        occ_index = tuple(occ_index)
        if occ_index not in voxel_index_to_label:
            voxel_index_to_label[occ_index] = []
        voxel_index_to_label[occ_index].append(semantic_label[i])

    occ_data = []
    for occ_index in voxel_index_to_color:
        if occ_index not in voxel_index_to_label:
            continue
        color = np.mean(np.array(voxel_index_to_color[occ_index]), axis=0)
        label = np.bincount(np.array(voxel_index_to_label[occ_index])).argmax()
        occ_data.append(list(occ_index) + list(color) + [255] + [label])
    occ_data = np.array(occ_data, dtype=np.int32)
    occ_info = {"voxel_size": voxel_size, "grid_size": [int(x) for x in grid_size], "min_bound": list(min_bound), "max_bound": list(max_bound)}
    return occ_data, occ_info

class OccInfo:
    def __init__(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
        self.voxel_size = data["voxel_size"]
        self.grid_size = tuple(data["grid_size"])
        self.min_bound = np.array(data["min_bound"])
        self.max_bound = np.array(data["max_bound"])
        return

    def to_json(self, json_path):
        data = {
            "voxel_size": self.voxel_size,
            "grid_size": list(self.grid_size),
            "min_bound": self.min_bound.tolist(),
            "max_bound": self.max_bound.tolist(),
        }
        with open(path, "w") as file:
            json.dump(data, file, indent=4)
        return

class Occupancy:
    def __init__(self, semantic_anno_type="scannet"):
        self.label_tool = CategoryLabel(semantic_anno_type=semantic_anno_type)
        return

    def initialize_from_compact_npy(self, compact_npy_path, grid_size):
        self.scene = np.zeros((*grid_size, 4), dtype=np.int32)
        data = np.load(compact_npy_path)
        indices = tuple(data[:, :3].T)
        self.scene[indices] = np.hstack((data[:, 3:6], data[:, 7:8]))
        return

    def to_semantic_colors(self):
        scene_semantic = self.scene[:, :, :, 3]
        xs, ys, zs = np.nonzero(scene_semantic)
        colors = np.zeros((*scene_semantic.shape, 3), dtype=np.int32)
        for x, y, z in zip(xs, ys, zs):
            colors[x, y, z] = self.label_tool.label2color0d(scene_semantic[x, y, z])
        return colors
