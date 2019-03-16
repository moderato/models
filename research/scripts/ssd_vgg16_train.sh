cd ..
python object_detection/legacy/train.py \
    --num_clones=1 \
    --train_dir=object_detection/trained_models/gtsdb_ssd_vgg16/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_vgg16/ssd_vgg16_gtsdb.config
    # --alsologtostderr
