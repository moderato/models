cd ..
python object_detection/legacy/train.py \
    --train_dir=object_detection/trained_models/gtsdb_ssd_mobilenet_v1/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_mobilenet_v1/ssd_mobilenet_v1_gtsdb.config \
    --alsologtostderr

