export ROUND=$1
case $ROUND in
    'single')
        #python3 -m python_scripts.generate_scripts.generate_single_round_in_scene --model_path /home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/trained_models/cmdm_action2motion/model000609390.pt --num_repetitions 1 --output_dir /home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/trained_models/cmdm_action2motion/generated_motions/single_round_motion_in_scenes
        echo "invalid round"
        exit 1
        ;;
    'multi')
        export DATASET='prox'
        export SHARED_DIR='/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/trained_models/cmdm_action2motion_qkv'
        for DEMO_ID in "MPH16+sit-bed_walk_sit-chair+0"
        do
            python3 -m python_scripts.generate_scripts.generate_multi_round_in_scene --model_path ${SHARED_DIR}/model001017526.pt --num_repetitions 1 --output_dir ${SHARED_DIR}/generated_motions/multi_round_motion_in_${DATASET} --max_lasting_frames 5 --multi_round_demo_id ${DEMO_ID} --use_scene_diffusion --inpainting
        done
        ;;
    *)
        echo "invalid round"
        exit 1
        ;;
esac
