cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/gtsdb_ssd_resnet18/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_resnet18/ssd_resnet18_gtsdb.config  \
    --eval_dir=object_detection/trained_models/gtsdb_ssd_resnet18/eval_logs\
    --logtostderr
