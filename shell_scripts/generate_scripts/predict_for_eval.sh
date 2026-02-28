export TEST_SCENE_TYPE=humanise
python3 -m python_scripts.generate_scripts.predict_for_eval --model_path ./save/trained_models/cmdm_action2motion_qkv/model001017526.pt --num_repetitions 1 --output_dir ./save/results_for_eval/cmdm_action2motion_qkv/${TEST_SCENE_TYPE} --inpainting --max_lasting_frames 5 --test_scene_type ${TEST_SCENE_TYPE}
