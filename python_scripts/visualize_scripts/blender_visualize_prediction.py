import bpy
import os
import glob
import json
import pickle
from math import pi

def set_light_and_camera():
    #set light and camera
    render_config_file = os.path.join(project_dir, "save", "results_for_eval", method_prefix, "humanise", sample_id, "render_config.json")
    if os.path.exists(render_config_file):
        with open(render_config_file) as f:
            render_config = json.load(f)
        light_position = render_config["light"]
        cam_position = render_config["cam_xyz"]
        cam_rotation = render_config["cam_xyz_rot"]
        bpy.data.objects['Light'].location[0] = light_position[0]
        bpy.data.objects['Light'].location[1] = light_position[1]
        bpy.data.objects['Light'].location[2] = light_position[2]
        bpy.data.objects['Light'].data.energy = 5000
        bpy.data.objects['Camera'].location[0] = cam_position[0]
        bpy.data.objects['Camera'].location[1] = cam_position[1]
        bpy.data.objects['Camera'].location[2] = cam_position[2]
        bpy.data.objects['Camera'].rotation_euler[0] = cam_rotation[0]*pi/180
        bpy.data.objects['Camera'].rotation_euler[1] = cam_rotation[1]*pi/180
        bpy.data.objects['Camera'].rotation_euler[2] = cam_rotation[2]*pi/180
    return

color_list = [
    [0xff, 0x00, 0x00], #dark red
    [0xff, 0x20, 0x00], #red
    [0xff, 0x40, 0x00], #orange
    [0xff, 0xb0, 0x00], #yellow
    [0x40, 0xff, 0x00], #light green
    [0x00, 0xff, 0x00], #green
    [0x00, 0x40, 0xff], #blue
    [0x00, 0x00, 0xff], #dark blue
    [0x20, 0x00, 0xff], #purple
    [0x40, 0x00, 0xff], #dark purple
]

def sample_id_to_color(sample_id):
    color_idx = int(sample_id[6:]) % len(color_list)
    color = color_list[color_idx]
    color = [color[0]/255., color[1]/255., color[2]/255.]
    return color

def fetch_motion_seq_colors(n_frames):
    start_color = (1., 1., 1., 1.)
    color_3d = sample_id_to_color(sample_id)
    target_color = (color_3d[0], color_3d[1], color_3d[2], 1.) 
    colors = []
    for i in range(n_frames):
        ratio = i / (n_frames - 1)
        color = (start_color[0] * (1 - ratio) + target_color[0] * ratio,
                 start_color[1] * (1 - ratio) + target_color[1] * ratio,
                 start_color[2] * (1 - ratio) + target_color[2] * ratio,
                 start_color[3] * (1 - ratio) + target_color[3] * ratio)
        colors.append(color)
    return colors


def load_body_meshes():
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    n_frame_stride = int(n_frames / visualize_split / n_person_per_frame)
    frame_visible = [False] * n_frames
    for i in range(0, n_frames, n_frame_stride):
        frame_visible[i] = True
    frame_visible[-1] = True
    human_body_colors = fetch_motion_seq_colors(n_frames)
    imported_models = []
    bin_idxs = []
    for i, file_name in enumerate(all_files):
        if not frame_visible[i]:
            continue
        if human_fmt == "obj":
            bpy.ops.wm.obj_import(filepath=dir_name+"/"+file_name, forward_axis='Y', up_axis='Z')
        elif human_fmt == "ply":
            bpy.ops.wm.ply_import(filepath=dir_name+"/"+file_name, directory=dir_name, files=[{"name":file_name, "name":file_name}])
        imported_model = bpy.context.selected_objects[0]
        mat = bpy.data.materials.new(f"{i:0>6}")
        mat.diffuse_color = human_body_colors[i]
        imported_model.active_material = mat
        for f in imported_model.data.polygons:
            f.use_smooth=True
        imported_models.append(imported_model)
        bin_idx = int(i / n_frames * visualize_split)
        bin_idxs.append(bin_idx)
    for split in range(visualize_split):
        bpy.context.scene.frame_set(split)
        for i, ob in enumerate(imported_models):
            # If our iteration has reached our designated frame, mark it as visible
            if bin_idxs[i] == split:
                ob.hide_viewport = ob.hide_render = False
                ob.keyframe_insert(data_path="hide_viewport")
                ob.keyframe_insert(data_path="hide_render")
            # Otherwise, set it to invisible
            else:
                ob.hide_viewport = ob.hide_render = True
                ob.keyframe_insert(data_path="hide_viewport")
                ob.keyframe_insert(data_path="hide_render")
    return

def load_scene():
    bpy.ops.wm.ply_import(filepath=scene_path, directory=os.path.dirname(scene_path), files=[{"name":os.path.basename(scene_path), "name":os.path.basename(scene_path)}])
    obj = bpy.context.selected_objects[0]
    bpy.ops.object.mode_set(mode='OBJECT')
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Vertex_Color_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    attribute = nodes.new(type='ShaderNodeAttribute')
    attribute.attribute_name = 'Col'
    mat.node_tree.links.new(bsdf.inputs['Base Color'], attribute.outputs['Color'])
    mat.node_tree.links.new(material_output.inputs['Surface'], bsdf.outputs['BSDF'])
    default_scene_collection = bpy.context.scene.collection
    default_scene_collection.objects.link(obj)
    bpy.context.view_layer.update()
    if dataset in ["random_scene_test", "shapenet_scene_test_sit", "shapenet_scene_test_lie"]:
        imported_model = bpy.context.selected_objects[0]
        mat = bpy.data.materials.new("scene")
        gray_color = (0.5, 0.5, 0.5, 1.0)
        mat.diffuse_color = gray_color
        imported_model.active_material = mat
    return obj

def set_render_path():
    render_path = os.path.join(project_dir, "save", "visualization_results", f"visualization_figures_split{visualize_split}", "humanise", sample_id, method)
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    bpy.context.scene.render.filepath = render_path + "/"

def render_image():
    bpy.ops.render.render(animation=True)

def delete():
    collection = bpy.data.collections.get("Collection")
    for obj in collection.objects:
        if obj.name in ["Camera", "Light"]:
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

def get_human_mesh_dir():
    dir_name = os.path.join(project_dir, "save", "results_for_eval", method_prefix, "humanise", sample_id, "rep0", pred_or_gt, "sample00_rep00_" + human_fmt)
    return dir_name

def get_scene_path():
    if render_quality == "low":
        scene_path = project_dir + "/../../dataset/ScanNetv2/scans/" + scene_name + "/" + scene_name + "_vh_clean_2.ply"
    else:
        scene_path = project_dir + "/../../dataset/ScanNetv2/scans/" + scene_name + "/" + scene_name + "_vh_clean.ply"
    return scene_path

def render_pipeline():
    #set light and camera
    set_light_and_camera()
    #load human mesh
    load_body_meshes()
    #load scene mesh
    load_scene()
    #set render path
    set_render_path()
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

if __name__ == "__main__":
    visualize_split = 1
    n_person_per_frame = 3
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = visualize_split - 1
    bpy.context.scene.render.fps = 1
    bpy.context.scene.render.film_transparent = True

    render_mode = "show"
    batch_mode = "single"
    render_quality = "low"
    dataset = "humanise"
    human_fmt = "obj"
    project_dir = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion"

    if batch_mode == "single":
        method = "ssomotion"
        sample_id = "sample0"

        if method == "ssomotion":
            method_prefix = "cmdm_action2motion_qkv"
            pred_or_gt = "pred"
        elif method == "gt":
            method_prefix = "cmdm_action2motion_qkv"
            pred_or_gt = "gt"
        else:
            raise NotImplementedError

        pickle_path = os.path.join(project_dir, "save", "results_for_eval", method_prefix, "humanise", sample_id, "rep0", pred_or_gt, "results.pkl")
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        scene_name = pickle_data["scene_id"]
        scene_path = get_scene_path()
        dir_name = get_human_mesh_dir()
        render_pipeline()
    elif batch_mode == "batch":
        sample_ids = glob.glob(os.path.join(project_dir, "save", "results_for_eval", "cmdm_action2motion_qkv", "humanise", "*", "render_config.json"))
        sample_ids = [sample_id.split("/")[-2] for sample_id in sample_ids]
        for sample_id in sample_ids:
            for method in ["ssomotion", "gt"]:
                if os.path.exists(os.path.join(project_dir, "save", "visualization_results", f"visualization_figures_split{visualize_split}", "humanise", sample_id, method, f"{visualize_split-1:0>4}.png")):
                    continue
                if method == "ssomotion":
                    method_prefix = "cmdm_action2motion_qkv"
                    pred_or_gt = "pred"
                elif method == "gt":
                    method_prefix = "cmdm_action2motion_qkv"
                    pred_or_gt = "gt"
                else:
                    raise NotImplementedError

                pickle_path = os.path.join(project_dir, "save", "results_for_eval", method_prefix, "humanise", sample_id, "rep0", pred_or_gt, "results.pkl")
                with open(pickle_path, "rb") as f:
                    pickle_data = pickle.load(f)
                scene_name = pickle_data["scene_id"]
                scene_path = get_scene_path()
                dir_name = get_human_mesh_dir()
                render_pipeline()
