import trimesh
import pickle
import numpy as np
from plyfile import PlyData
from tools.project_config_tools import *
from deps.dimos.test_navmesh import *


def project_to_navmesh(navmesh, points):
    closest, _, _ = trimesh.proximity.closest_point(navmesh, points)
    return closest


def get_replica_instances(scene_dir, build=False):
    instance_folder = scene_dir / "instances"
    ply_path = scene_dir / 'habitat' / 'mesh_semantic.ply'
    mesh_path = scene_dir / "mesh.ply"
    mesh = trimesh.load(mesh_path)
    json_path = os.path.join(scene_dir, 'habitat', 'info_semantic.json')
    with open(json_path, 'r') as f:
        semantic_mapping = json.load(f)
    instance_to_category = semantic_mapping['id_to_label']
    num_instances = len(semantic_mapping['id_to_label'])
    if not os.path.exists(instance_folder) or build:
        instance_meshes = []
        os.makedirs(instance_folder, exist_ok=True)
        # per face instance labels
        # https: // github.com / facebookresearch / Replica - Dataset / issues / 17  # issuecomment-538757418
        file_in = PlyData.read(ply_path)
        faces_in = file_in.elements[1]
        objects_vertices = {}
        for f in faces_in:
            object_id = f[1]
            if not object_id in objects_vertices:
                objects_vertices[object_id] = []
            objects_vertices[object_id] += list(f[0])
        for key in objects_vertices:
            objects_vertices[key] = list(set(objects_vertices[key]))

        for instance_id in range(num_instances):
            instance_path = os.path.join(instance_folder, str(instance_id) + '.ply')
            category_id = instance_to_category[instance_id]
            category_name = semantic_mapping['classes'][category_id - 1]['name']
            if category_id < 0:  # empty instance
                instance_mesh = trimesh.Trimesh()
            else:
                vertex_ids = np.array(objects_vertices[instance_id], dtype=np.int32)
                vertex_ids = vertex_ids[vertex_ids <= len(mesh.vertices)-1]
                vertex_valid = np.zeros(len(mesh.vertices))
                vertex_valid[vertex_ids] = 1
                face_ids = np.nonzero(vertex_valid[mesh.faces].sum(axis=-1) == 3)[0]
                instance_mesh = mesh.submesh([face_ids], append=True)
            instance_mesh.export(instance_path)
            instance_meshes.append(instance_mesh)  # directly return submesh cannot be converted to open3d, error vertex normal not writable

    instance_meshes = []
    for instance_id in range(num_instances):
        instance_path = os.path.join(instance_folder, str(instance_id) + '.ply')
        scene_or_mesh = trimesh.load_mesh(instance_path, process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            if len(scene_or_mesh.geometry) == 0:
                instance_mesh = trimesh.Trimesh()  # empty scene
            else:
                # we lose texture information here
                instance_mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                            for g in scene_or_mesh.geometry.values()))
        else:
            instance_mesh = scene_or_mesh
        instance_meshes.append(instance_mesh)
    return mesh, instance_meshes, semantic_mapping


def get_replica_floor_height(scene_dir):
    mesh, instances, semantic_mapping = get_replica_instances(scene_dir=scene_dir)
    instance_to_category = semantic_mapping['id_to_label']
    num_instances = len(semantic_mapping['id_to_label'])
    category_ids = [instance_to_category[instance_id] for instance_id in range(num_instances)]
    category_names = [semantic_mapping['classes'][category_id - 1]['name'] for category_id in
                            category_ids]
    floor_pointclouds = [np.asarray(instances[instance_id].vertices) for instance_id in range(num_instances) if category_names[instance_id] == 'floor']
    if len(floor_pointclouds) == 0:
        floor_height = mesh.bounds[0, 2]
    else:
        max_idx = np.argmax(np.array(
            [pointcloud.shape[0] for pointcloud in floor_pointclouds]
        ))
        max_floor = floor_pointclouds[max_idx]
        floor_height = max_floor[:, 2].mean()  # mean of z coord of points of max sized floor
    return floor_height


def get_prox_floor_height(scene_segmentation_path, scene_path):
    with open(scene_segmentation_path, "rb") as f:
        scene_segmentation = pickle.load(f)
    floor_pointclouds = [np.asarray(obj['mesh']['vertices']) for obj in scene_segmentation['object_nodes'] if obj['category_name'] == 'floor']
    if len(floor_pointclouds) == 0:
        scene_mesh = trimesh.load(scene_path, force='mesh')
        floor_height = np.min(scene_mesh.vertices[:, 2]) + 0.1
    else:
        max_idx = np.argmax(np.array(
            [pointcloud.shape[0] for pointcloud in floor_pointclouds]
        ))
        max_floor = floor_pointclouds[max_idx]
        floor_height = max_floor[:, 2].mean()  # mean of z coord of points of max sized floor
    return floor_height

def get_navmesh(navmesh_path, scene_path, agent_radius, floor_height=0.0, visualize=False):
    if navmesh_path.exists():
        navmesh = trimesh.load(navmesh_path, force='mesh')
    else:
        scene_mesh = trimesh.load(scene_path, force='mesh')
        """assume the scene coords are z-up"""
        scene_mesh.vertices[:, 2] -= floor_height
        scene_mesh.apply_transform(zup_to_shapenet)
        navmesh = create_navmesh(scene_mesh, export_path=navmesh_path, agent_radius=agent_radius, visualize=visualize)
    navmesh.vertices[:, 2] = 0
    return navmesh

def get_locomotion_wpath(start_point, target_point, navmesh_loose_path, navmesh_tight_path=None, scene_path=None, floor_height=0, visualize=False):
    # get loose navmesh for path planning
    navmesh_loose = get_navmesh(navmesh_loose_path, scene_path, agent_radius=0.2, floor_height=floor_height, visualize=visualize)
    # get tight navmesh for local map sensing
    navmesh_tight = get_navmesh(navmesh_tight_path, scene_path, agent_radius=0.05, floor_height=floor_height, visualize=visualize) if navmesh_tight_path is not None else None

    start_target = np.stack([start_point, target_point])
    start_target_xy = start_target.copy()
    start_target_xy[:,2] = 0
    # find collision free path
    scene_mesh = trimesh.load(scene_path, force='mesh') if scene_path is not None else None
    wpath = path_find(navmesh_loose, start_target_xy[0], start_target_xy[1], visualize=visualize, scene_mesh=scene_mesh)
    wpath[:, 2] = np.linspace(start_point[2], target_point[2], wpath.shape[0])
    return wpath

def get_navmesh_anytype(navmesh_type, scene_name, scene_path, floor_height):
    if navmesh_type == 'loose':
        agent_radius = 0.2
    elif navmesh_type == 'middle':
        agent_radius = 0.12
    elif navmesh_type == 'tight':
        agent_radius = 0.06
    else:
        raise ValueError
    if scene_name in prox_room_list:
        navmesh_path = Path(prox_data_dir + '/navmeshes/' + scene_name + '/' + scene_name + '_navmesh_' + navmesh_type + '.ply')
    elif scene_name in replica_room_list:
        navmesh_path = Path(replica_dir + '/navmeshes/' + scene_name + '/' + scene_name + '_navmesh_' + navmesh_type + '.ply')
    else:
        raise NotImplementedError
    navmesh = get_navmesh(navmesh_path, scene_path, agent_radius=agent_radius, floor_height=floor_height)
    return navmesh

def adjust4walkable(wpath, wpath_poset, wpath_action, scene_name, floor_height):
    post_wpath = []
    post_wpath_poset = []
    post_wpath_action = []
    if scene_name in prox_room_list:
        scene_path = Path(prox_data_dir + '/scenes/' + scene_name + '.ply')
    elif scene_name in replica_room_list:
        scene_path = Path(replica_dir, scene_name, 'mesh.ply')
    else:
        raise NotImplementedError
    navmesh_loose = get_navmesh_anytype('loose', scene_name, scene_path, floor_height)
    navmesh_middle = get_navmesh_anytype('middle', scene_name, scene_path, floor_height)
    navmesh_tight = get_navmesh_anytype('tight', scene_name, scene_path, floor_height)
    navmesh_try_list = [navmesh_loose, navmesh_middle, navmesh_tight]
    scene_mesh = trimesh.load(scene_path, force='mesh')
    for i in range(len(wpath) - 1):
        if wpath_action[i] == "walk" and wpath_action[i+1] == "walk":
            start_target = np.stack([wpath[i], wpath[i+1]]).copy()
            start_target_xy = start_target.copy() 
            start_target_xy[:,2] = 0
            tmp_wpath = path_find(navmesh_loose, start_target_xy[0], start_target_xy[1], visualize=False, scene_mesh=scene_mesh)
            if len(tmp_wpath) == 0:
                print(start_target_xy)
                new_start_target_xy = project_to_navmesh(navmesh_loose, start_target_xy)
                changes = new_start_target_xy - start_target_xy
                changes[:, 2] = 0
                start_target += changes
                start_target_xy += changes
                wpath[i] += changes[0]
                if wpath_poset[i] is not None:
                    wpath_poset[i][0, 66:69] += changes[0]
                wpath[i+1] += changes[1]
                if wpath_poset[i+1] is not None:
                    wpath_poset[i+1][0, 66:69] += changes[1]
                for navmesh_try in navmesh_try_list:
                    tmp_wpath = path_find(navmesh_try, start_target_xy[0], start_target_xy[1], visualize=False, scene_mesh=scene_mesh)
                    if len(tmp_wpath) != 0:
                        break
            if len(tmp_wpath) == 0:
                tmp_wpath = start_target
                print("path not found")
            tmp_wpath[:, 2] = np.linspace(start_target[0, 2], start_target[-1, 2], len(tmp_wpath))
            if i == len(wpath) - 2:
                post_wpath.extend(tmp_wpath)
                post_wpath_poset.append(wpath_poset[i])
                post_wpath_poset.extend([None]*(len(tmp_wpath)-2))
                post_wpath_poset.append(wpath_poset[i+1])
                post_wpath_action.extend(["walk"]*(len(tmp_wpath)))
            else:
                post_wpath.extend(tmp_wpath[:-1])
                post_wpath_poset.append(wpath_poset[i])
                post_wpath_poset.extend([None]*(len(tmp_wpath)-2))
                post_wpath_action.extend(["walk"]*(len(tmp_wpath)-1))
        else:
            post_wpath.append(wpath[i])
            post_wpath_poset.append(wpath_poset[i])
            post_wpath_action.append(wpath_action[i])
            if i == len(wpath) - 2:
                post_wpath.append(wpath[i+1])
                post_wpath_poset.append(wpath_poset[i+1])
                post_wpath_action.append(wpath_action[i+1])
    return post_wpath, post_wpath_poset, post_wpath_action


if __name__ == "__main__":
    visualize = False
    scene_name = 'test_room'
    scene_dir = Path('dataset/dimos_data/test_room')
    scene_path = scene_dir / 'room.ply'
    floor_height = 0
    navmesh_tight_path = scene_dir / 'navmesh_tight.ply'
    navmesh_loose_path = scene_dir / 'navmesh_loose.ply'

    """automatic path finding"""
    # specify start and target location
    start_point = np.array([-1.7, 2.35, 0])
    target_point = np.array([-1.4, 0.54, 0])
    wpath = get_locomotion_wpath(start_point, target_point, navmesh_loose_path, navmesh_tight_path, scene_path, floor_height, visualize)
    print(wpath)
