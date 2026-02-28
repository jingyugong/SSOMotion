import os
import socket
import torch


class __GMDM_Project_Config():
    def __init__(self) -> None:
        self.get_project_dir_according_to_host()
        self.get_host_device_info()
        self.init_prox()
        self.init_random()
        self.init_replica()
        self.init_shapenet_real()
        self.init_control()
        self.init_settings()
        self.init_ablation()
        self.register_to_global_space()

    def get_project_dir_according_to_host(self):
        self.hostname = socket.gethostname()
        self.map_hostname_to_dir = {
            "lagrange" : "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion",
            "JINGYUdeMacBook-Pro.local" : "/Users/jingyugong/gcode/RGBD/code/OccupancyMotion",
            "fineserver" : "/home/gjy/gcode/RGBD/code/OccupancyMotion",
            "bohr" : "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion",
        }
        self.map_hostname_to_prefix = {
            "lagrange" : "dataset",
            "JINGYUdeMacBook-Pro.local" : "dataset",
            "fineserver" : "dataset",
            "bohr" : "dataset",
        }
        self.project_dir = self.map_hostname_to_dir.get(self.hostname, None)
    

    def init_ablation(self):
        self.ablation_semantic = False

    def init_prox(self):
        "can load .yaml config here in the furture"
        self.prox_data_dir = os.path.join(self.project_dir, "dataset", "proxs")
        self.prox_room_list = ['BasementSittingBooth', 'MPH16', 'N0SittingBooth', 
                               'N3Office', 'MPH112', 'MPH1Library', 'N0Sofa', 
                               'N3OpenArea', 'MPH11', 'MPH8', 'N3Library', 'Werkraum']


    def init_random(self):
        self.random_scene_test_dir = os.path.join(self.project_dir, "dataset", "dimos_scenes", "random_scene_test")
        self.test_room_list = ['test_room']


    def init_shapenet_real(self):
        self.shapenet_real_dir = os.path.join(self.project_dir, self.map_hostname_to_prefix.get(self.hostname, ""), "shapenet", "shapenet_real")
        self.shapenet_obj_list = {
            'Armchairs': ['9faefdf6814aaa975510d59f3ab1ed64',
                'cacb9133bc0ef01f7628281ecb18112',
                'ea918174759a2079e83221ad0d21775',],
            'L-Sofas': ['5cea034b028af000c2843529921f9ad7',],
            'Sofas': ['1dd6e32097b09cd6da5dde4c9576b854',
                '71fd7103997614db490ad276cd2af3a4',
                '277231dcb7261ae4a9fe1734a6086750',],
            'StraightChairs':['2ed17abd0ff67d4f71a782a4379556c7',
                '68dc37f167347d73ea46bea76c64cc3d',
                'd93760fda8d73aaece101336817a135f']
        }


    def init_replica(self):
        self.replica_dir = os.path.join(self.project_dir, self.map_hostname_to_prefix.get(self.hostname, ""), "replica")
        self.replica_room_list = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'hotel_0']


    def init_control(self):
        self.control_stand_poset_before_sit = True

    def get_host_device_info(self):
        if torch.cuda.is_available():
            self.host_device = "cuda"
            self.host_device_count = torch.cuda.device_count()
        else:
            self.host_device = "cpu"
            self.host_device_count = os.cpu_count()


    def register_to_global_space(self):
        for attr, value in vars(self).items():
            globals()[attr] = value
        
    def init_settings(self):
        self.default_voxel_size = 0.04
        self.action_enumerator = {
            0: "walk",
            1: "sit",
            2: "stand_up",
            3: "lie",
        }
        self.action_name_to_action = {k: v for v, k in self.action_enumerator.items()}
        self.local_grid_info = {
            'bounds_negative': [50, 50, 40],
            'bounds_positive': [50, 50, 40],
            'grid_steps': [101, 101, 81],
        }
        self.global_category2label = {
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
            #new classes in matterport3d
            "cushion": 41,
            "chest_of_drawers": 42,
            "plant": 43,
            "stairs": 44,
            "stool": 45,
            "tv_monitor": 46,
            "shower": 47,
            "column": 48,
            "fireplace": 49,
            "lighting": 50,
            "beam": 51,
            "railing": 52,
            "shelving": 53,
            "gym_equipment": 54,
            "seating": 55,
            "board_panel": 56,
            "furniture": 57,
            "appliances": 58,
            "objects": 59,
            "misc": 60,
            "unlabeled": 61,
            #new classes in replica
            "bottle": 62,
            "monitor": 63,
            "hair-dryer": 64,
            "shoe": 65,
            "rack": 66,
            "desktop-computer": 67,
            "basket": 68,
            "cooktop": 69,
            "bowl": 70,
            "beanbag": 71,
            "coaster": 72,
            "rug": 73,
            "comforter": 74,
            "desk-organizer": 75,
            "exercise-call": 76,
            "microwave": 77,
            "toothbrush": 78,
            "handbag": 79,
            "mouse": 80,
            "tv-stand": 81,
            "wardrobe": 82,
            "candle": 83,
            "pillar": 84,
            "bin": 85,
            "faucet": 86,
            "backpack": 87,
            "table-runner": 88,
            "bike": 89,
            "chopping-board": 90,
            "shower-stall": 91,
            "blanket": 92,
            "remote-control": 93,
            "cloth": 94,
            "clock": 95,
            "countertop": 96,
            "panel": 97,
            "cup": 98,
            "bench": 99,
            "mat": 100,
            "wall-cabinet": 101,
            "vase": 102,
            "logo": 103,
            "phone": 104,
            "book": 105,
            "base-cabinet": 106,
            "handrail": 107,
            "wall-plug": 108,
            "indoor-plant": 109,
            "plant-stand": 110,
            "sculpture": 111,
            "pipe": 112,
            "umbrella": 113,
            "switch": 114,
            "tissue-paper": 115,
            "knife-block": 116,
            "tv-screen": 117,
            "pan": 118,
            "scarf": 119,
            "laptop": 120,
            "stair": 121,
            "plate": 122,
            "shelf": 123,
            "utensil-holder": 124,
            "kitchen-utensil": 125,
            "pot": 126,
            "small-appliance": 127,
            "paper-towel": 128,
            "computer-keyboard": 129,
            "set-of-clothing": 130,
            "tablet": 131,
            "camera": 132,
            "vent": 133,
            "major-appliance": 134,
            "clothing": 135,
        }


__config = __GMDM_Project_Config()
del __config
