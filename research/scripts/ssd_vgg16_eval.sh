cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/gtsdb_ssd_vgg16/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_vgg16/ssd_vgg16_gtsdb.config  \
    --eval_dir=object_detection/trained_models/gtsdb_ssd_vgg16/eval_logs\
    --logtostderr
