cd ..
PIPELINE_CONFIG_PATH=$HOME/Documents/TrafficSignBench/Detection/models/research/object_detection/trained_models/bdd100k_ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_bdd100k.config
MODEL_DIR=$HOME/Documents/TrafficSignBench/Detection/models/research/object_detection/trained_models/bdd100k_ssdlite_mobilenet_v2/train_logs
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
# python object_detection/model_main.py \
# 	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
# 	--model_dir=${MODEL_DIR} --num_train_steps=${NUM_TRAIN_STEPS} \
# 	--sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
# 	--alsologtostderr

python object_detection/legacy/train.py \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
	--train_dir=${MODEL_DIR} \
	--logtostderr
