export PROCESS_MODE=$1
case $PROCESS_MODE in
    'single')
        python -m python_scripts.visualize_scripts.extract_results_smplx_mesh --npy_path /home/gongjingyu/gcode/RGBD/code/OccupancyMotion/save/results_for_eval/cmdm_action2motion/shapenet_scene_test_sit/Armchairs/9faefdf6814aaa975510d59f3ab1ed64/path0/rep0/results.npy --sample_i 0 --rep_i 0
        ;;
    'batch')
        RESULTS_PATTERN=~/gcode/RGBD/code/guided-motion-diffusion/save/trained_models/mixed_action2motion_control/generated_motions_ortho/multi_round_motion_in_prox/*/results.npy
        for SCENE_RESULT in ${RESULTS_PATTERN} 
        do
            echo ${SCENE_RESULT}
            python -m python_scripts.visualize_scripts.extract_results_smplx_mesh --npy_path ${SCENE_RESULT} --sample_i 0 --rep_i 0
        done
        ;;
    *)
        echo "invalid process mode"
        exit
        ;;
esac
