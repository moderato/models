cd ..
python object_detection/legacy/train.py \
    --num_clones=1 \
    --train_dir=object_detection/trained_models/gtsdb_ssd_resnet18/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_resnet18/ssd_resnet18_gtsdb.config
    # --alsologtostderr
