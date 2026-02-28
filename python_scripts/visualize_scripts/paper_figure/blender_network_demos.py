import bpy
import os
import glob
import json
import math
from pathlib import Path
from mathutils import Vector
import numpy as np

def load_scene(filename):
    scene_path = os.path.join(assets_dir, filename)
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

def load_body_mesh(body_mesh_file):
    filepath = os.path.join(assets_dir, body_mesh_file)
    human_body_color = (1., 1., 1., 1.)
    bpy.ops.wm.obj_import(filepath=filepath, forward_axis='Y', up_axis='Z')
    imported_model = bpy.context.selected_objects[0]
    mat = bpy.data.materials.new("human_in_world")
    mat.diffuse_color = human_body_color
    imported_model.active_material = mat
    for f in imported_model.data.polygons:
        f.use_smooth=True
    return

def fetch_motion_seq_colors(n_frames):
    start_color = (1., 1., 1., 1.)
    color_3d = [0xff, 0x40, 0x00]
    color_3d = [color_3d[0]/255., color_3d[1]/255., color_3d[2]/255.]
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

def load_body_meshes(demo_id):
    dir_name = assets_dir+"/"+demo_id+"/sample00_rep00_obj"
    visualize_split = 1
    n_person_per_frame = 4
    human_fmt = "obj"
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

def load_xyz_planes(images):
    locations = [
            (-2., 0, 0.), (2., 0, 0.), (0, -2., 0.), (0, 2., 0.), (0, 0, -0.94),
            (-2.1, 0, 0.), (2.1, 0, 0.), (0, -2.1, 0.), (0, 2.1, 0.), (0, 0, -1.04),
            (-2.2, 0, 0.), (2.2, 0, 0.), (0, -2.2, 0.), (0, 2.2, 0.), (0, 0, -1.14),
            ]
    rotations = [
            (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), 0, math.radians(180)), (math.radians(90), 0, math.radians(180)), (0, 0, 0),
            (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), 0, math.radians(180)), (math.radians(90), 0, math.radians(180)), (0, 0, 0),
            (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), 0, math.radians(180)), (math.radians(90), 0, math.radians(180)), (0, 0, 0),
            ]
    scales = [
            (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (4, 4, 0),
            (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (4, 4, 0),
            (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (-4.0, 3.2, 0), (4, 4, 0),
            ]
    for (image, location, rotation, scale) in zip(images, locations, rotations, scales):
        bpy.ops.mesh.primitive_plane_add(size=1.0)
        plane = bpy.context.object
        plane.location = location 
        plane.rotation_euler = rotation
        plane.scale = scale

        mat = bpy.data.materials.new(name="ImageMaterial")
        plane.data.materials.append(mat)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs[4].default_value = 0.5

        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(image)

        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
        mat.blend_method = 'BLEND'
    return

def load_rgbds_img():
    x_map1_rgb = os.path.join(assets_dir, "x_map1_rgb.png")
    x_map2_rgb = os.path.join(assets_dir, "x_map2_rgb.png")
    y_map1_rgb = os.path.join(assets_dir, "y_map1_rgb.png")
    y_map2_rgb = os.path.join(assets_dir, "y_map2_rgb.png")
    z_map1_rgb = os.path.join(assets_dir, "z_map1_rgb.png")
    x_map1_depth = os.path.join(assets_dir, "x_map1_depth.png")
    x_map2_depth = os.path.join(assets_dir, "x_map2_depth.png")
    y_map1_depth = os.path.join(assets_dir, "y_map1_depth.png")
    y_map2_depth = os.path.join(assets_dir, "y_map2_depth.png")
    z_map1_depth = os.path.join(assets_dir, "z_map1_depth.png")
    x_map1_sem = os.path.join(assets_dir, "x_map1_sem.png")
    x_map2_sem = os.path.join(assets_dir, "x_map2_sem.png")
    y_map1_sem = os.path.join(assets_dir, "y_map1_sem.png")
    y_map2_sem = os.path.join(assets_dir, "y_map2_sem.png")
    z_map1_sem = os.path.join(assets_dir, "z_map1_sem.png")
    images = [x_map1_rgb, x_map2_rgb, y_map1_rgb, y_map2_rgb, z_map1_rgb, x_map1_depth, x_map2_depth, y_map1_depth, y_map2_depth, z_map1_depth, x_map1_sem, x_map2_sem, y_map1_sem, y_map2_sem, z_map1_sem]
    load_xyz_planes(images)
    return

def set_triplane_light_and_camera():
    bpy.data.objects['Light'].location[0] = -2.
    bpy.data.objects['Light'].location[1] = 2.
    bpy.data.objects['Light'].location[2] = 5. 
    bpy.data.objects['Light'].data.energy = 5000
    #add new light
    bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(-5., 5., 4.), scale=(1, 1, 1))
    obj = bpy.context.selected_objects[0]
    obj.data.energy = 10000
    #set camera
    bpy.data.objects['Camera'].location = (-9.2, 9.2, 7.1)
    bpy.data.objects['Camera'].rotation_euler = (math.radians(60), 0, math.radians(-135))
    return

def set_motion_light_and_camera():
    bpy.data.objects['Light'].location[0] = 2.
    bpy.data.objects['Light'].location[1] = 2.
    bpy.data.objects['Light'].location[2] = 5. 
    bpy.data.objects['Light'].data.energy = 5000
    bpy.data.objects['Camera'].location = (2.5, 4.5, 2.8)
    bpy.data.objects['Camera'].rotation_euler = (math.radians(60), 0, math.radians(150))
    return

def set_direction_light_and_camera():
    bpy.data.objects['Light'].location[0] = 2.5
    bpy.data.objects['Light'].location[1] = 0.
    bpy.data.objects['Light'].location[2] = 5. 
    bpy.data.objects['Light'].data.energy = 5000
    bpy.data.objects['Camera'].location = (5., 2., 3.)
    bpy.data.objects['Camera'].rotation_euler = (math.radians(60), 0, math.radians(105))
    return

def set_render_path(filename):
    render_path = os.path.join(assets_dir, filename)
    bpy.context.scene.render.filepath = render_path + "/"
    return

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
    return

def render_image():
    bpy.ops.render.render(animation=True)

def triplane_map_render():
    #set light and camera
    set_triplane_light_and_camera()
    #load human mesh
    load_body_mesh("000000_local.obj")
    #load semantic image
    load_rgbds_img()
    #set render path
    set_render_path("rgbds_map")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def normal_motion_render():
    #set light and camera
    set_motion_light_and_camera()
    #load human mesh
    load_body_meshes("MPH16+walk_sit-bed+0")
    #load scene mesh
    load_scene("MPH16.ply")
    #set render path
    set_render_path("normal_motion")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def noisy_motion_render():
    #set light and camera
    set_motion_light_and_camera()
    #load human mesh
    load_body_meshes("MPH16+walk_sit-bed+0+noise")
    #load scene mesh
    load_scene("MPH16.ply")
    #set render path
    set_render_path("noisy_motion")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def create_material(color):
    mat = bpy.data.materials.new(name="RedMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    return mat

def draw_joints(joints):
    joint_radius = 0.03
    joint_color = (1., 0., 0., 1.)
    joint_mat = create_material(joint_color)
    joint_name_prefix = "joint"
    for i, (x, y, z) in enumerate(joints):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=joint_radius,
            location=(x, y, z),
            segments=32,
            ring_count=16
        )
        sphere = bpy.context.object
        sphere.name = f"{joint_name_prefix}_{i}"
        sphere.data.materials.append(joint_mat)
    return

def draw_directions(joints, directions):
    name = "direction"
    direction_color = (1., 0., 0., 1.)
    scale=0.05
    shaft_scale=(0.1 * scale, 0.1 * scale, 1.3 * scale)
    cone_scale=(0.3 * scale, 0.3 * scale, 0.3 * scale)
    direction_mat = create_material(direction_color)
    for joint, direction in zip(joints, directions):
        joint = np.array(joint)
        direction = np.array(direction)
        # Cylinder
        shaft_len = 2 * shaft_scale[2]
        shaft_end = joint + direction * shaft_len
        shaft_mid = (joint + shaft_end) / 2

        bpy.ops.mesh.primitive_cylinder_add(location=shaft_mid)
        shaft = bpy.context.object
        shaft.scale = shaft_scale
        shaft.name = f"{name}_shaft"
        shaft.data.materials.append(direction_mat)

        # Cone
        bpy.ops.mesh.primitive_cone_add(location=shaft_end)
        cone = bpy.context.object
        cone.scale = cone_scale
        cone.name = f"{name}_head"
        cone.data.materials.append(direction_mat)

        # Rotate
        up = Vector((0, 0, 1))
        vec = Vector(direction)
        axis = up.cross(vec)
        angle = up.angle(vec)

        for obj in [shaft, cone]:
            if axis.length > 1e-6:
                obj.rotation_mode = 'AXIS_ANGLE'
                obj.rotation_axis_angle = (angle, *axis)
    return

def load_joint_directions(filename):
    joint_direction_path = os.path.join(assets_dir, filename, "joints_direction.json")
    with open(joint_direction_path, 'r') as f:
        joint_direction = json.load(f)
    draw_joints(joint_direction["joints"])
    draw_directions(joint_direction["joints"],joint_direction["direction"])
    return

def joint_direction_render():
    #set light and camera
    set_direction_light_and_camera()
    #load human joint
    load_joint_directions("MPH16+walk_sit-bed+0")
    #load scene mesh
    load_scene("MPH16.ply")
    #set render path
    set_render_path("joint_direction")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()

if __name__ == "__main__":
    render_mode = "show"
    assets_dir = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/visualization_results/figs/network"
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0 
    bpy.context.scene.render.film_transparent = True
    triplane_map_render()
    normal_motion_render()
    noisy_motion_render()
    joint_direction_render()
