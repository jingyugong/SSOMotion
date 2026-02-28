import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from data_loaders.scene2motion.basic_scene import BasicScene
from tools.project_config_tools import host_device
from tools.io_tools import export_batch_human_mesh
from tools.motion_tools import load_body_mesh_model_batch
from tools.occ_tools import OccInfo, Occupancy, export_local_occ
from tools.visualization_tools import CategoryLabel, CategoryLabelConverter

class RealtimeScene(BasicScene):
    def __init__(
            self,
            compact_occ_npy_path,
            occ_info_path,
            semantic_anno_type="scannet"
            ):
        self.data_device=host_device
        self.semantic_anno_type = semantic_anno_type
        self.label_semantics = CategoryLabel(semantic_anno_type=semantic_anno_type)
        self.label_local2global = CategoryLabelConverter(src_semantic_anno_type=semantic_anno_type, tgt_semantic_anno_type="global")

        self.occ_info = OccInfo(occ_info_path)
        self.scene_global_occ = Occupancy(semantic_anno_type=semantic_anno_type)
        self.scene_global_occ.initialize_from_compact_npy(compact_occ_npy_path, self.occ_info.grid_size)
        self.bodymodel_ones = {
            "male": load_body_mesh_model_batch(1, body_type='smplx', gender='male', device=host_device),
            "female": load_body_mesh_model_batch(1, body_type='smplx', gender='female', device=host_device),
            "neutral": load_body_mesh_model_batch(1, body_type='smplx', gender='neutral', device=host_device)
            }
        return

    def fetch_local_scene_occ_hint(
            self,
            human_pose,
            betas=None,
            gender="neutral",
            ):
        self.bodymodel_batch = self.bodymodel_one = self.bodymodel_ones[gender]

        trans = human_pose[:1, :3]
        orient = human_pose[:1, 3:6]
        body_pose = human_pose[:1, 6:]
        if betas is None:
            betas = np.zeros(10)
        trans_local, orient_local, scene_local_occ = self._localize_motion_scene(trans, orient, body_pose, betas, self.occ_info, self.scene_global_occ)
        scene_local_occ_tensor = torch.tensor(scene_local_occ, dtype=torch.float)
        x_map1, x_map2 = self._decompx(scene_local_occ_tensor)
        y_map1, y_map2 = self._decompy(scene_local_occ_tensor)
        z_map1, _ = self._decompz(scene_local_occ_tensor, offset=10)
        scene_hint = {
            "scene_valid": torch.ones(1, dtype=torch.float32),
            "x_map1_depth": x_map1[0:1] * self.occ_info.voxel_size,
            "x_map2_depth": x_map2[0:1] * self.occ_info.voxel_size,
            "y_map1_depth": y_map1[0:1] * self.occ_info.voxel_size,
            "y_map2_depth": y_map2[0:1] * self.occ_info.voxel_size,
            "z_map1_depth": z_map1[0:1] * self.occ_info.voxel_size,
            "x_map1_rgb": x_map1[1:4],
            "x_map2_rgb": x_map2[1:4],
            "y_map1_rgb": y_map1[1:4],
            "y_map2_rgb": y_map2[1:4],
            "z_map1_rgb": z_map1[1:4],
            "x_map1_sem": self.label_local2global.convert_semantic_map_src2tgt(x_map1[4:5].to(torch.int32)),
            "x_map2_sem": self.label_local2global.convert_semantic_map_src2tgt(x_map2[4:5].to(torch.int32)),
            "y_map1_sem": self.label_local2global.convert_semantic_map_src2tgt(y_map1[4:5].to(torch.int32)),
            "y_map2_sem": self.label_local2global.convert_semantic_map_src2tgt(y_map2[4:5].to(torch.int32)),
            "z_map1_sem": self.label_local2global.convert_semantic_map_src2tgt(z_map1[4:5].to(torch.int32)),
        }
        return scene_hint

def export_scene_hints(scene_hint, label_global2local):
    from tools.project_config_tools import project_dir
    for k in ["x_map1_sem", "x_map2_sem", "y_map1_sem", "y_map2_sem", "z_map1_sem"]:
        semantic_map = scene_hint[k].squeeze(0).numpy()
        semantic_map = label_global2local.convert_semantic_map_src2tgt(semantic_map)
        semantic_map = scene.label_semantics.label2color2d(semantic_map)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(semantic_map)
        output_file = os.path.join(project_dir, "save", "visualization_results", "figs", "intro", k + ".png")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    for k in ["x_map1_depth", "x_map2_depth", "y_map1_depth", "y_map2_depth", "z_map1_depth"]:
        depth_map = scene_hint[k].squeeze(0).numpy()
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(depth_map)
        output_file = os.path.join(project_dir, "save", "visualization_results", "figs", "intro", k + ".png")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    for k in ["x_map1_rgb", "x_map2_rgb", "y_map1_rgb", "y_map2_rgb", "z_map1_rgb"]:
        rgb_map = scene_hint[k].squeeze(0).numpy()
        rgb_map = rgb_map.transpose(1, 2, 0) / 255.
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(rgb_map)
        output_file = os.path.join(project_dir, "save", "visualization_results", "figs", "intro", k + ".png")
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
    return

if __name__ == "__main__":
    semantic_anno_type = "matterport3d"
    scene = RealtimeScene("/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/dataset/proxs/occ/MPH16/compact_occ.npy", "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/dataset/proxs/occ/MPH16/compact_occ_info.json", semantic_anno_type=semantic_anno_type)
    with open("/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/dataset/proxs/coins_poses/MPH16/stand-floor-0/selected/5.pkl", "rb") as f:
        data = pickle.load(f)
        smplx_param = data["smplx_param"]
    human_pose = np.concatenate([smplx_param["transl"], smplx_param["global_orient"], smplx_param["body_pose"]], axis=-1)
    scene_hint = scene.fetch_local_scene_occ_hint(human_pose)
    label_global2local = CategoryLabelConverter(src_semantic_anno_type="global", tgt_semantic_anno_type=semantic_anno_type)
    export_scene_hints(scene_hint, label_global2local)
