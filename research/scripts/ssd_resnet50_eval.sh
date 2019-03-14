cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/gtsdb_ssd_resnet50/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_resnet50/ssd_resnet50_gtsdb.config  \
    --eval_dir=object_detection/trained_models/gtsdb_ssd_resnet50/eval_logs\
    --logtostderr
