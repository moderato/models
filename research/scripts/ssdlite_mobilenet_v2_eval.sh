cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/gtsdb_ssdlite_mobilenet_v2/train_logs \
    --pipeline_config_path=object_detection/trained_models/gtsdb_ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_gtsdb.config  \
    --eval_dir=object_detection/trained_models/gtsdb_ssdlite_mobilenet_v2/eval_logs\
    --logtostderr
