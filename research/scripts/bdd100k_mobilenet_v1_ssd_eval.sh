cd ..
python object_detection/legacy/eval.py \
    --checkpoint_dir=object_detection/trained_models/bdd100k_ssd_mobilenet_v1/train_logs \
    --pipeline_config_path=object_detection/trained_models/bdd100k_ssd_mobilenet_v1/ssd_mobilenet_v1_bdd100k.config  \
    --eval_dir=object_detection/trained_models/bdd100k_ssd_mobilenet_v1/eval_logs\
    --logtostderr
