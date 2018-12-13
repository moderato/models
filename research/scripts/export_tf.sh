cd ..
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/opt/intel/computer_vision_sdk_fpga_2018.4.420/deployment_tools/model_optimizer_R3/gtsdb/tf/MobileNetV2/pipeline.config
TRAINED_CKPT_PREFIX=/opt/intel/computer_vision_sdk_fpga_2018.4.420/deployment_tools/model_optimizer_R3/gtsdb/tf/MobileNetV2/model.ckpt-200000
EXPORT_DIR=/opt/intel/computer_vision_sdk_fpga_2018.4.420/deployment_tools/model_optimizer_R3/gtsdb/tf/MobileNetV2
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
