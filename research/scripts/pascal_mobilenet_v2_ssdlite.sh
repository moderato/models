cd ..
PIPELINE_CONFIG_PATH=/home/zhongyilin/Documents/models/research/object_detection/trained_models/pascal_ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_pascal.config
MODEL_DIR=/home/zhongyilin/Documents/models/research/object_detection/trained_models/pascal_ssdlite_mobilenet_v2/
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py --pipeline_config_path=${PIPELINE_CONFIG_PATH} --model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES --alsologtostderr
