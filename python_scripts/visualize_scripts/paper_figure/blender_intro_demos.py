import bpy
import os
import math
from pathlib import Path

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

def set_bev_light_and_camera():
    bpy.data.objects['Light'].location[0] = 2.5
    bpy.data.objects['Light'].location[1] = 0. 
    bpy.data.objects['Light'].location[2] = 5. 
    bpy.data.objects['Light'].data.energy = 5000
    bpy.data.objects['Camera'].location = (-0.2, 0, 9)
    bpy.data.objects['Camera'].rotation_euler = (0, 0, math.radians(240))
    return

def set_occ_light_and_camera():
    bpy.data.objects['Light'].location[0] = -2.
    bpy.data.objects['Light'].location[1] = 2.
    bpy.data.objects['Light'].location[2] = 5. 
    bpy.data.objects['Light'].data.energy = 5000
    bpy.data.objects['Camera'].location = (-4., 4., 2.)
    bpy.data.objects['Camera'].rotation_euler = (math.radians(70), 0, math.radians(-135))
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

def bev_render():
    #set light and camera
    set_bev_light_and_camera()
    #load human mesh
    load_body_mesh("000000_world.obj")
    #load scene mesh
    load_scene("MPH16.ply")
    #set render path
    set_render_path("bev")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def load_xyz_planes(x_plane, y_plane, z_plane):
    images = [x_plane, y_plane, z_plane]
    locations = [(2., 0, 0.), (0, -2., 0.), (0, 0, -0.94)]
    rotations = [(math.radians(90), math.radians(0), math.radians(270)), (math.radians(90), 0, math.radians(180)), (0, 0, 0)]
    scales = [(-4.0, 3.2, 0), (-4.0, 3.2, 0), (4, 4, 0)]
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

        tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(image)

        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    return

def load_semantic_img():
    x_plane = os.path.join(assets_dir, "x_map2_sem.png")
    y_plane = os.path.join(assets_dir, "y_map1_sem.png")
    z_plane = os.path.join(assets_dir, "z_map1_sem.png")
    load_xyz_planes(x_plane, y_plane, z_plane)
    return

def load_rgb_img():
    x_plane = os.path.join(assets_dir, "x_map2_rgb.png")
    y_plane = os.path.join(assets_dir, "y_map1_rgb.png")
    z_plane = os.path.join(assets_dir, "z_map1_rgb.png")
    load_xyz_planes(x_plane, y_plane, z_plane)
    return

def load_depth_img():
    x_plane = os.path.join(assets_dir, "x_map2_depth.png")
    y_plane = os.path.join(assets_dir, "y_map1_depth.png")
    z_plane = os.path.join(assets_dir, "z_map1_depth.png")
    load_xyz_planes(x_plane, y_plane, z_plane)
    return

def occ_render():
    #set light and camera
    set_occ_light_and_camera()
    #load human mesh
    load_body_mesh("000000_local.obj")
    #load scene mesh
    load_scene("occ_local.ply")
    #set render path
    set_render_path("occ")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def sem_render():
    #set light and camera
    set_occ_light_and_camera()
    #load human mesh
    load_body_mesh("000000_local.obj")
    #load semantic image
    load_semantic_img()
    #set render path
    set_render_path("sem")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def rgb_render():
    #set light and camera
    set_occ_light_and_camera()
    #load human mesh
    load_body_mesh("000000_local.obj")
    #load semantic image
    load_rgb_img()
    #set render path
    set_render_path("rgb")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

def depth_render():
    #set light and camera
    set_occ_light_and_camera()
    #load human mesh
    load_body_mesh("000000_local.obj")
    #load semantic image
    load_depth_img()
    #set render path
    set_render_path("depth")
    if render_mode == "render":
        #render image
        render_image()
        #delete object except for camera and light
        delete()
    return

if __name__ == "__main__":
    render_mode = "show"
    assets_dir = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/visualization_results/figs/intro"
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0 
    bpy.context.scene.render.film_transparent = True
    bev_render()
    occ_render()
    sem_render()
    rgb_render()
    depth_render()
