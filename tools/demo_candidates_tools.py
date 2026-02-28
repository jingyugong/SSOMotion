import os
from tools.io_tools import load_pkl
from tools.project_config_tools import replica_dir, host_device, replica_room_list
from tools.guidance_tools import load_replica_scenehints
from tools.demo_tools import general_multi_round_demo_prepare


def generate_replica_multi_round(demo_id):
    
    def parse_replica_demo_id(demo_id:str):
        scene_name = demo_id.split("+")[0]
        scene_pose_path = os.path.join(replica_dir, "coins_poses", scene_name)
        scene_hints = load_replica_scenehints(scene_name, device=host_device)
        floor_height = scene_hints['floor_height']
        action_obj_list = demo_id.split("+")[1].split("_")
        action_pose_list = []
        for pairs in action_obj_list:
            act, obj, pair_idx, pkl_idx = pairs.split("-")
            if os.path.exists(os.path.join(scene_pose_path, f"{act}-{obj}-{pair_idx}")):
                action_pose_list.append(f"{act}-{obj}-{pair_idx}/selected/{pkl_idx}.pkl")
            else:
                print(f"No such action-object pair as {pairs} in {scene_name}.")
        return scene_name, scene_pose_path, scene_hints, floor_height, action_pose_list
    

    scene_name, scene_path, scene_hints, floor_height, action_pose_list = parse_replica_demo_id(demo_id)

    if action_pose_list == []:
        scene_name, scene_path, scene_hints, floor_height, action_pose_list = parse_replica_demo_id(
            "office_0+sit-sofa-1-0_stand-floor-0-0_sit-chair-0-0+0"
        )

    action_map = {"sit": "sit", "stand": "walk", "lie": "lie"}
    action_list = [action_map[action_pose_file.split("-")[0]] for action_pose_file in action_pose_list]

    scene_name = demo_id.split("+")[0]
    scene_path = os.path.join(replica_dir, "coins_poses", scene_name)
    scene_hints = load_replica_scenehints(scene_name, device=host_device)
    floor_height = scene_hints['floor_height']
    
    smplx_pose_list = [load_pkl(os.path.join(scene_path, action_pose_file))['smplx_param'] for action_pose_file in action_pose_list]
    ret = general_multi_round_demo_prepare(scene_name, scene_hints, floor_height, action_list, smplx_pose_list)
    return ret 


def generate_multi_round_demo_candidates(demo_id):
    scene_name = demo_id.split("+")[0]
    print(scene_name)
    if scene_name in replica_room_list:
        ret = generate_replica_multi_round(demo_id)
    else:
        raise NotImplementedError
    return ret
