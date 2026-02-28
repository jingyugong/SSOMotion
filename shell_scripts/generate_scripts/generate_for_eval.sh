ACTION=$1
case $ACTION in
    'walk')
	export TEST_SCENE_TYPE=random_scene_test
        ;;
    'sit')
	export TEST_SCENE_TYPE=shapenet_scene_test_sit
        ;;
    'lie')
	export TEST_SCENE_TYPE=shapenet_scene_test_lie
        ;;
    *)
        echo "unknown action: $ACTION"
        exit 1
        ;;
esac
python3 -m python_scripts.generate_scripts.generate_for_eval --model_path ./save/trained_models/cmdm_action2motion_qkv/model001017526.pt --num_repetitions 10 --output_dir ./save/results_for_eval/cmdm_action2motion_qkv/${TEST_SCENE_TYPE} --use_scene_diffusion --inpainting --max_lasting_frames 5 --test_scene_type ${TEST_SCENE_TYPE}
