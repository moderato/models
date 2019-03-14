cd ..
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=$HOME/Documents/TrafficSignBench/Detection/models/research/object_detection/trained_models/gtsdb_ssd_mobilenet_v1/train_logs/pipeline.config
TRAINED_CKPT_PREFIX=$HOME/Documents/TrafficSignBench/Detection/models/research/object_detection/trained_models/gtsdb_ssd_mobilenet_v1/train_logs/model.ckpt-120004
# EXPORT_DIR=$OPENVINO_ROOTDIR/deployment_tools/model_optimizer_R3/gtsdb/tf/MobileNetV1
EXPORT_DIR=.
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
