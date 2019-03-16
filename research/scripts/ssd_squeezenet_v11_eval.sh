cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/gtsdb_ssd_squeezenet_v11/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssd_squeezenet_v11/ssd_squeezenet_v11_gtsdb.config  \
    --eval_dir=object_detection/trained_models/gtsdb_ssd_squeezenet_v11/eval_logs\
    --logtostderr
