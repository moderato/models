# Assuming the dataset is extracted to $Home/Documents/data/GTSDBdevkit
cd ../../
python object_detection/dataset_tools/create_gtsdb_tf_records.py \
    --label_map_path=object_detection/data/gtsdb_label_map.pbtxt \
    --dataset_directory=$HOME/Documents/data/GTSDBdevkit/GTSDB/JPEGImages/ \
    --train_splits=trainval --validation_splits=test \
    --output_directory=$HOME/Documents/data/GTSDBdevkit/tfrecords
