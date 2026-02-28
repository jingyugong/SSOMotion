import bpy
import os
import glob
import json
from math import pi


render_quality = "medium"
demo_id = "MPH8+walk_sit-bed+0"
dataset = "prox"
method = "ssomotion"
human_fmt = "obj"


platform = os.name

project_dir = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion"
scene_name = demo_id.split("+")[0]
demos_render_config_file = os.path.join(project_dir, "assets", "demos_render_config.json")

ssomotion_save_prefix = "save/trained_models/cmdm_action2motion_qkv"
ssomotion_nosemantic_save_prefix = "save/trained_models/cmdm_action2motion_qkv_nosemantic"

method_to_dir = {
    "ssomotion" : os.path.join(project_dir, ssomotion_save_prefix, "generated_motions", "multi_round_motion_in_" + dataset),
    "ssomotion_nosemantic" : os.path.join(project_dir, ssomotion_nosemantic_save_prefix, "generated_motions", "multi_round_motion_in_" + dataset),
    "dimos" : os.path.join(project_dir, "save", "others_results", "dimos_results", "results_" + dataset),
    "omnicontrol" : os.path.join(project_dir, "save", "others_results", "omnicontrol_results","results_" + dataset)
}
dir_root = method_to_dir[method]
    

dataset_prefix = "dataset"
if dataset == "prox":
    if render_quality == "low":
        scene_path = os.path.join(project_dir, dataset_prefix, "proxs", "scenes_downsampled" , scene_name + ".ply")
    else:
        scene_path = os.path.join(project_dir, dataset_prefix, "proxs", "scenes", scene_name + ".ply")
elif dataset == "replica":
    if render_quality == "low":
        scene_path = os.path.join(project_dir, dataset_prefix, "replica", scene_name, "mesh_downsampled.ply")
    else:
        scene_path = os.path.join(project_dir, dataset_prefix, "replica", scene_name, "mesh_woceil.ply")
                
        
color_list = [
    [0xb0, 0x2c, 0x24], #dark red
    [0xec, 0x52, 0x4b], #red
    [0xe0, 0x8b, 0x33], #orange
    [0xaa, 0x86, 0x2f], #yellow
    [0x57, 0xbc, 0x8f], #light green
    [0x41, 0x91, 0x4d], #green
    [0x0f, 0x3d, 0xd1], #blue
    [0x41, 0x48, 0xa0], #dark blue
    [0x51, 0x1d, 0x82], #purple
    [0x3e, 0x1d, 0x5b], #dark purple
]


def demo_id_to_color(demo_id):
    color_idx = sum([ord(d) for d in demo_id]) % len(color_list)
    color = color_list[color_idx]
    color = [color[0]/255., color[1]/255., color[2]/255.]
    return color


def fetch_motion_seq_colors(n_frames):
    start_color = (1., 1., 1., 1.)
    color_3d = demo_id_to_color(demo_id)
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


def load_body_meshes(use_color = False):
    all_files = glob.glob(dir_name+os.sep+"*."+human_fmt)
    all_files = [file_name.split(os.sep)[-1] for file_name in all_files]
    all_files.sort()
    n_frames = len(all_files)
    human_body_colors = fetch_motion_seq_colors(n_frames)
    imported_models = []
    for i, file_name in enumerate(all_files):
        if human_fmt == "obj":
            bpy.ops.wm.obj_import(filepath=dir_name+os.sep+file_name, forward_axis='Y', up_axis='Z')
        elif human_fmt == "ply":
            bpy.ops.wm.ply_import(filepath=dir_name+os.sep+file_name, directory=dir_name, files=[{"name":file_name, "name":file_name}])
        imported_model = bpy.context.selected_objects[0]
        if use_color:
            mat = bpy.data.materials.new(f"{i:0>6}")
            mat.diffuse_color = human_body_colors[i]
            imported_model.active_material = mat
        for f in imported_model.data.polygons:
            f.use_smooth=True
        imported_models.append(imported_model)
    bpy.context.scene.frame_start=0
    bpy.context.scene.frame_end=n_frames-1
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
    return obj


def animate():
    animate_save_dir = os.path.join(project_dir, "save", "visualization_results", "visualization_videos", demo_id)
    os.makedirs(animate_save_dir, exist_ok=True)
    # 设置渲染路径和文件名
    bpy.context.scene.render.filepath = os.path.join(animate_save_dir, f"{method}.avi")
    
    # 设置渲染格式为AVI
    bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
    # 设置分辨率 & FPS
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 90
    bpy.context.scene.render.fps=30
    bpy.context.scene.render.use_file_extension = False
    # 渲染动画
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
    
    
if __name__ == "__main__":
    selected_methods = ["ssomotion"] 
    for cur_method in selected_methods:
        method = cur_method
        dir_root = method_to_dir[method]
        for cur_demo_id in os.listdir(os.path.join(dir_root)):
            demo_id = cur_demo_id
            scene_name = demo_id.split("+")[0]
            if method in ["ssomotion", "ssomotion_nosemantic"]:
                dir_name = os.path.join(dir_root, demo_id, "sample00_rep00_" + human_fmt)
            elif method == "dimos":
                dir_name = os.path.join(dir_root, demo_id, "meshes", "body_meshes")
            elif method == "omnicontrol":
                dir_name = os.path.join(dir_root, demo_id, "sample00_rep00_" + human_fmt)
            if dataset == "prox":
                if render_quality == "low":
                    scene_path = os.path.join(project_dir, dataset_prefix, "proxs", "scenes_downsampled", scene_name + ".ply")
                else:
                    scene_path = os.path.join(project_dir, dataset_prefix, "proxs", "scenes", scene_name + ".ply")
            elif dataset == "replica":
                if render_quality == "low":
                    scene_path = os.path.join(project_dir, dataset_prefix, "replica", scene_name, "mesh_downsampled.ply")
                else:
                    scene_path = os.path.join(project_dir, dataset_prefix, "replica", scene_name, "mesh_woceil.ply")
            set_light_and_camera()
            #load human mesh
            load_body_meshes(use_color=False)
            #load scene mesh
            load_scene()
            animate()
            delete()
