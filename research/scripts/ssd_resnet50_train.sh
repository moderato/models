cd ..
python object_detection/legacy/train.py \
    --num_clones=1 \
    --train_dir=object_detection/trained_models/gtsdb_ssd_resnet50/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_resnet50/ssd_resnet50_gtsdb.config \
    --alsologtostderr
