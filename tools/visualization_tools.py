import numpy as np
import cv2
import torch
from tools.project_config_tools import global_category2label

# NYU40 color palette
SCANNET_COLOR_PALETTE = [
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183), #ceiling
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144),
]

SCANNET_CATEGORY2LABEL = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "cabinet": 3,
    "bed": 4,
    "chair": 5,
    "sofa": 6,
    "table": 7,
    "door": 8,
    "window": 9,
    "bookshelf": 10,
    "picture": 11,
    "counter": 12,
    "blinds": 13,
    "desk": 14,
    "shelves": 15,
    "curtain": 16,
    "dresser": 17,
    "pillow": 18,
    "mirror": 19,
    "floor_mat": 20,
    "clothes": 21,
    "ceiling": 22,
    "books": 23,
    "refrigerator": 24,
    "television": 25,
    "paper": 26,
    "towel": 27,
    "shower_curtain": 28,
    "box": 29,
    "whiteboard": 30,
    "person": 31,
    "nightstand": 32,
    "toilet": 33,
    "sink": 34,
    "lamp": 35,
    "bathtub": 36,
    "bag": 37,
    "other_structure": 38,
    "other_furniture": 39,
    "other_prop": 40,
}

MATTERPORT3D_COLOR_PALETTE = [
    "#ffffff",
    "#aec7e8",
    "#708090",
    "#98df8a",
    "#c5b0d5",
    "#ff7f0e",
    "#d62728",
    "#1f77b4",
    "#bcbd22",
    "#ff9896",
    "#2ca02c",
    "#e377c2",
    "#de9ed6",
    "#9467bd",
    "#8ca252",
    "#843c39",
    "#9edae5",
    "#9c9ede",
    "#e7969c",
    "#637939",
    "#8c564b",
    "#dbdb8d",
    "#d6616b",
    "#cedb9c",
    "#e7ba52",
    "#393b79",
    "#a55194",
    "#ad494a",
    "#b5cf6b",
    "#5254a3",
    "#bd9e39",
    "#c49c94",
    "#f7b6d2",
    "#6b6ecf",
    "#ffbb78",
    "#c7c7c7",
    "#8c6d31",
    "#e7cb94",
    "#ce6dbd",
    "#17becf",
    "#7f7f7f",
    "#000000",
]
MATTERPORT3D_COLOR_PALETTE = [(eval("0x"+tmp[1:3]), eval("0x"+tmp[3:5]), eval("0x"+tmp[5:7])) for tmp in MATTERPORT3D_COLOR_PALETTE]

MATTERPORT3D_CATEGORY2LABEL = {
    "void": 0,
    "wall": 1,
    "floor": 2,
    "chair": 3,
    "door": 4,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "window": 9,
    "sofa": 10,
    "bed": 11,
    "curtain": 12,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "stairs": 16,
    "ceiling": 17,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "mirror": 21,
    "tv_monitor": 22,
    "shower": 23,
    "column": 24,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "lighting": 28,
    "beam": 29,
    "railing": 30,
    "shelving": 31,
    "blinds": 32,
    "gym_equipment": 33,
    "seating": 34,
    "board_panel": 35,
    "furniture": 36,
    "appliances": 37,
    "clothes": 38,
    "objects": 39,
    "misc": 40,
    "unlabeled": 41,
}

REPLICA_COLOR_PALETTE = [
    (0, 0, 0),
    (196, 51, 182),
    (91, 135, 229),
    (229, 91, 104),
    (247, 182, 210),
    (91, 229, 110),
    (141, 91, 229),
    (255, 187, 120),
    (112, 128, 144),
    (196, 156, 148),
    (197, 176, 213),
    (148, 103, 189),
    (229, 91, 223),
    (219, 219, 141),
    (31, 119, 180),
    (192, 229, 91),
    (88, 218, 137),
    (58, 98, 137),
    (177, 82, 239),
    (255, 127, 14),
    (188, 189, 34),
    (237, 204, 37),
    (41, 206, 32),
    (62, 143, 148),
    (34, 14, 130),
    (143, 45, 115),
    (137, 63, 14),
    (23, 190, 207),
    (16, 212, 139),
    (90, 119, 201),
    (125, 30, 141),
    (78, 71, 183),
    (186, 197, 62),
    (227, 119, 194),
    (38, 100, 128),
    (120, 31, 243),
    (154, 59, 103),
    (214, 39, 40),
    (169, 137, 78),
    (143, 245, 111),
    (152, 223, 138),
    (37, 230, 205),
    (14, 16, 155),
    (208, 49, 84),
    (237, 80, 38),
    (138, 175, 62),
    (158, 218, 229),
    (38, 96, 167),
    (190, 77, 246),
    (208, 193, 72),
    (55, 220, 57),
    (10, 125, 140),
    (76, 38, 202),
    (191, 28, 135),
    (211, 120, 42),
    (118, 174, 76),
    (17, 242, 171),
    (20, 65, 247),
    (208, 61, 222),
    (162, 62, 60),
    (210, 235, 62),
    (45, 152, 72),
    (35, 107, 149),
    (160, 89, 237),
    (227, 56, 125),
    (169, 143, 81),
    (42, 143, 20),
    (25, 160, 151),
    (82, 75, 227),
    (253, 59, 222),
    (240, 130, 89),
    (123, 172, 47),
    (71, 194, 133),
    (24, 94, 205),
    (134, 16, 179),
    (159, 32, 52),
    (213, 208, 88),
    (64, 158, 70),
    (18, 163, 194),
    (65, 29, 153),
    (255, 152, 150),
    (177, 10, 109),
    (152, 83, 7),
    (83, 175, 30),
    (44, 160, 44),
    (18, 199, 153),
    (61, 81, 208),
    (213, 85, 216),
    (170, 53, 42),
    (161, 192, 38),
    (23, 241, 91),
    (12, 103, 170),
    (151, 41, 245),
    (174, 199, 232),
    (133, 51, 80),
    (184, 162, 91),
    (50, 138, 38),
    (31, 237, 236),
    (39, 19, 208),
    (223, 27, 180),
    (254, 141, 85),
    (97, 144, 39)
]

REPLICA_CATEGORY2LABEL = {
    "void": 0,
    "backpack": 1,
    "base-cabinet": 2,
    "basket": 3,
    "bathtub": 4,
    "beam": 5,
    "beanbag": 6,
    "bed": 7,
    "bench": 8,
    "bike": 9,
    "bin": 10,
    "blanket": 11,
    "blinds": 12,
    "book": 13,
    "bottle": 14,
    "box": 15,
    "bowl": 16,
    "camera": 17,
    "cabinet": 18,
    "candle": 19,
    "chair": 20,
    "chopping-board": 21,
    "clock": 22,
    "cloth": 23,
    "clothing": 24,
    "coaster": 25,
    "comforter": 26,
    "computer-keyboard": 27,
    "cup": 28,
    "cushion": 29,
    "curtain": 30,
    "ceiling": 31,
    "cooktop": 32,
    "countertop": 33,
    "desk": 34,
    "desk-organizer": 35,
    "desktop-computer": 36,
    "door": 37,
    "exercise-call": 38,
    "faucet": 39,
    "floor": 40,
    "handbag": 41,
    "hair-dryer": 42,
    "handrail": 43,
    "indoor-plant": 44,
    "knife-block": 45,
    "kitchen-utensil": 46,
    "lamp": 47,
    "laptop": 48,
    "major-appliance": 49,
    "mat": 50,
    "microwave": 51,
    "monitor": 52,
    "mouse": 53,
    "nightstand": 54,
    "pan": 55,
    "panel": 56,
    "paper-towel": 57,
    "phone": 58,
    "picture": 59,
    "pillar": 60,
    "pillow": 61,
    "pipe": 62,
    "plant-stand": 63,
    "plate": 64,
    "pot": 65,
    "rack": 66,
    "refrigerator": 67,
    "remote-control": 68,
    "scarf": 69,
    "sculpture": 70,
    "shelf": 71,
    "shoe": 72,
    "shower-stall": 73,
    "sink": 74,
    "small-appliance": 75,
    "sofa": 76,
    "stair": 77,
    "stool": 78,
    "switch": 79,
    "table": 80,
    "table-runner": 81,
    "tablet": 82,
    "tissue-paper": 83,
    "toilet": 84,
    "toothbrush": 85,
    "towel": 86,
    "tv-screen": 87,
    "tv-stand": 88,
    "umbrella": 89,
    "utensil-holder": 90,
    "vase": 91,
    "vent": 92,
    "wall": 93,
    "wall-cabinet": 94,
    "wall-plug": 95,
    "wardrobe": 96,
    "window": 97,
    "rug": 98,
    "logo": 99,
    "bag": 100,
    "set-of-clothing": 101,
}

class CategoryLabel:
    def __init__(self, semantic_anno_type="scannet"):
        if semantic_anno_type == "scannet":
            self.color_palette = SCANNET_COLOR_PALETTE
            self.category2label = SCANNET_CATEGORY2LABEL
        elif semantic_anno_type == "matterport3d":
            self.color_palette = MATTERPORT3D_COLOR_PALETTE
            self.category2label = MATTERPORT3D_CATEGORY2LABEL
        elif semantic_anno_type == "replica":
            self.color_palette = REPLICA_COLOR_PALETTE
            self.category2label = REPLICA_CATEGORY2LABEL
        else:
            raise ValueError("unknown annotation type: {}".format(semantic_anno_type))
        self.label2category = {str(v): k for k, v in self.category2label.items()}
        return

    def fetch_category(self, label: int) -> str:
        return self.label2category[str(label)]

    def label2color0d(self, label: int) -> np.ndarray:
        color = np.array(self.color_palette[label]) 
        return color

    def label2color2d(self, label: np.ndarray) -> np.ndarray:
        assert len(label.shape) == 2
        colors = np.zeros((*label.shape, 3), dtype=np.int32)
        xs, ys = np.nonzero(label)
        for x, y in zip(xs, ys):
            colors[x, y] = self.label2color0d(label[x, y])
        return colors

class CategoryLabelConverter:
    def __init__(
            self,
            src_semantic_anno_type="scannet",
            tgt_semantic_anno_type="global",
            ):
        if src_semantic_anno_type == "scannet":
            self.src_category2label = SCANNET_CATEGORY2LABEL
        elif src_semantic_anno_type == "matterport3d":
            self.src_category2label = MATTERPORT3D_CATEGORY2LABEL
        elif src_semantic_anno_type == "replica":
            self.src_category2label = REPLICA_CATEGORY2LABEL
        elif src_semantic_anno_type == "global":
            self.src_category2label = global_category2label
        else:
            raise NotImplementedError
        self.src_label2category = {str(v): k for k, v in self.src_category2label.items()}

        if tgt_semantic_anno_type == "scannet":
            self.tgt_category2label = SCANNET_CATEGORY2LABEL
        elif tgt_semantic_anno_type == "matterport3d":
            self.tgt_category2label = MATTERPORT3D_CATEGORY2LABEL
        elif tgt_semantic_anno_type == "replica":
            self.tgt_category2label = REPLICA_CATEGORY2LABEL
        elif tgt_semantic_anno_type == "global":
            self.tgt_category2label = global_category2label
        else:
            raise NotImplementedError
        self.tgt_label2category = {str(v): k for k, v in self.tgt_category2label.items()}

        self.src2tgt_label_map = {}
        for k, v in self.src_label2category.items():
            if v in self.tgt_category2label:
                self.src2tgt_label_map[k] = self.tgt_category2label[v]
            else:
                self.src2tgt_label_map[k] = 0
        return

    def convert_semantic_map_src2tgt(
            self,
            semantic_map
            ):
        is_tensor = isinstance(semantic_map, torch.Tensor)
        if is_tensor:
            semantic_map = semantic_map.detach().cpu().numpy()

        origin_ndim = semantic_map.ndim
        if origin_ndim == 3:
            semantic_map = semantic_map.squeeze(0)

        tgt_semantic_map = semantic_map.copy()
        h, w = semantic_map.shape
        for i in range(h):
            for j in range(w):
                tgt_semantic_map[i, j] = self.src2tgt_label_map[str(semantic_map[i, j])]
        if origin_ndim == 3:
            tgt_semantic_map = tgt_semantic_map.reshape((1, h, w))
        if is_tensor:
            tgt_semantic_map = torch.from_numpy(tgt_semantic_map)
        return tgt_semantic_map

def frames_to_video(frame_list, save_path, fps=30):
    size = (frame_list[0].shape[1], frame_list[0].shape[0])
    if save_path[-3:] == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    elif save_path[-3:] == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    VideoWriter = cv2.VideoWriter(save_path, fourcc, fps, size)
    for i, frame in enumerate(frame_list):
        VideoWriter.write(frame.astype(np.uint8))
    VideoWriter.release()
    return
