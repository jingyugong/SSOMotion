export PROCESS_MODE=$1
case $PROCESS_MODE in
    'batch')
        RESULTS_PATTERN=/home/gongjingyu/gcode/RGBD/code/OccupancyMotion/dataset/shapenet/shapenet_real/Sofas/71fd7103997614db490ad276cd2af3a4/sit_before_lie/selected/optimization_after_get_body_21_sit.pkl
        for SCENE_RESULT in ${RESULTS_PATTERN} 
        do
            echo ${SCENE_RESULT}
            python -m python_scripts.visualize_scripts.extract_pose_smplx_mesh --pkl_path ${SCENE_RESULT}
        done
        ;;
    *)
        echo "invalid process mode"
        exit
        ;;
esac
