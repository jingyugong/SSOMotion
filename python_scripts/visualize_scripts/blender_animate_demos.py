import bpy
import os
import glob
import json
from math import pi


render_quality = "low"
demo_id = "MPH16+sit-bed_walk_sit-chair+0"
dataset = "prox"
method = "ours"
human_fmt = "obj"
project_dir = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion"
scene_name = demo_id.split("+")[0]
demos_render_config_file = project_dir + "/assets/demos_render_config.json"
if method == "ours":
    dir_name = project_dir + "/save/trained_models/cmdm_action2motion_qkv/generated_motions/multi_round_motion_in_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt
elif method == "dimos":
    dir_name = project_dir + "/save/others_results/dimos_results/results_" + dataset + "/" +demo_id + "/meshes/body_meshes"
elif method == "omnicontrol":
    dir_name = project_dir + "/save/others_results/omnicontrol_results/results_" + dataset + "/" +demo_id + "/sample00_rep00_" + human_fmt

if dataset == "prox":
    if render_quality == "low":
        scene_path = project_dir + "/dataset/proxs/scenes_downsampled/" + scene_name + ".ply"
    else:
        scene_path = project_dir + "/dataset/proxs/scenes/" + scene_name + ".ply"
elif dataset == "replica":
    if render_quality == "low":
        scene_path = project_dir + "/dataset/replica/" + scene_name + "/mesh_downsampled.ply"
    else:
        scene_path = project_dir + "/dataset/replica/" + scene_name + "/mesh_woceil.ply"


def set_light_and_camera():
    #set light and camera
    with open(demos_render_config_file) as f:
        demos_render_config = json.load(f)
    light_position = demos_render_config[demo_id]["light"]
    cam_position = demos_render_config[demo_id]["cam_xyz"]
    cam_rotation = demos_render_config[demo_id]["cam_xyz_rot"]
    bpy.data.objects['Light'].location[0] = light_position[0]
    bpy.data.objects['Light'].location[1] = light_position[1]
    bpy.data.objects['Light'].location[2] = light_position[2]
    bpy.data.objects['Camera'].location[0] = cam_position[0]
    bpy.data.objects['Camera'].location[1] = cam_position[1]
    bpy.data.objects['Camera'].location[2] = cam_position[2]
    bpy.data.objects['Camera'].rotation_euler[0] = cam_rotation[0]*pi/180
    bpy.data.objects['Camera'].rotation_euler[1] = cam_rotation[1]*pi/180
    bpy.data.objects['Camera'].rotation_euler[2] = cam_rotation[2]*pi/180


def load_body_meshes():
    all_files = glob.glob(dir_name+"/*."+human_fmt)
    all_files = [file_name.split("/")[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    imported_models = []
    for file_name in all_files:
        if human_fmt == "obj":
            bpy.ops.wm.obj_import(filepath=dir_name+"/"+file_name, forward_axis='Y', up_axis='Z')
        elif human_fmt == "ply":
            bpy.ops.wm.ply_import(filepath=dir_name+"/"+file_name, directory=dir_name, files=[{"name":file_name, "name":file_name}])
        imported_model = bpy.context.selected_objects[0]
        for f in imported_model.data.polygons:
            f.use_smooth=True
        imported_models.append(imported_model)
    bpy.context.scene.frame_start=0
    bpy.context.scene.frame_end=n_frames-1
    bpy.context.scene.render.fps=30
    for frame in range(n_frames):
        bpy.context.scene.frame_set(frame)
        for i, ob in enumerate(imported_models):
            # If our iteration has reached our designated frame, mark it as visible
            if i == frame:
                ob.hide_viewport = ob.hide_render = False
                ob.keyframe_insert(data_path="hide_viewport")
                ob.keyframe_insert(data_path="hide_render")
            # Otherwise, set it to invisible
            else:
                ob.hide_viewport = ob.hide_render = True
                ob.keyframe_insert(data_path="hide_viewport")
                ob.keyframe_insert(data_path="hide_render")
    return imported_models


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
    return obj

if __name__ == "__main__":
    set_light_and_camera()
    #load human mesh
    load_body_meshes()
    #load scene mesh
    load_scene()
