import bpy
import glob
dir_name = "/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/trained_models/cmdm_action2motion/generated_motions/multi_round_motion_in_prox/MPH16+sit-bed_walk_sit-chair+0/sample00_rep00_obj"
all_files = glob.glob(dir_name+"/*.obj")
all_files = [file_name.split("/")[-1] for file_name in all_files]
all_files.sort()
n_frames = len(all_files)
imported_models = []
for file_name in all_files:
    bpy.ops.wm.obj_import(filepath=dir_name+"/"+file_name, forward_axis='Y', up_axis='Z')
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
