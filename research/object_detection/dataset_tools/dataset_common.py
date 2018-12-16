# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

GTSDB_LABELS = {
    'none': (0, 'Background'),
    '20': (1, 'prohibitory'),
    '30': (2, 'prohibitory'),
    '50': (3, 'prohibitory'),
    '60': (4, 'prohibitory'),
    '70': (5, 'prohibitory'),
    '80': (6, 'prohibitory'),
    'ends 80': (7, 'other'),
    '100': (8, 'prohibitory'),
    '120': (9, 'prohibitory'),
    'no overtaking': (10, 'prohibitory'),
    'no overtaking (trucks)': (11, 'prohibitory'),
    'priority at next intersection': (12, 'danger'),
    'priority road': (13, 'other'),
    'give way': (14, 'other'),
    'stop': (15, 'other'),
    'no traffic both ways': (16, 'prohibitory'),
    'no trucks': (17, 'prohibitory'),
    'no entry': (18, 'other'),
    'danger': (19, 'danger'),
    'bend left': (20, 'danger'),
    'bend right': (21, 'danger'),
    'bend': (22, 'danger'),
    'uneven road': (23, 'danger'),
    'slippery road': (24, 'danger'),
    'road narrows': (25, 'danger'),
    'construction': (26, 'danger'),
    'traffic signal': (27, 'danger'),
    'pedestrian crossing': (28, 'danger'),
    'school crossing': (29, 'danger'),
    'cycles crossing': (30, 'danger'),
    'snow': (31, 'danger'),
    'animals': (32, 'danger'),
    'restriction ends': (33, 'other'),
    'go right': (34, 'mandatory'),
    'go left': (35, 'mandatory'),
    'go straight': (36, 'mandatory'),
    'go right or straight': (37, 'mandatory'),
    'go left or straight': (38, 'mandatory'),
    'keep right': (39, 'mandatory'),
    'keep left': (40, 'mandatory'),
    'roundabout': (41, 'mandatory'),
    'restriction ends (overtaking)': (42, 'other'),
    'restriction ends (overtaking (trucks))': (43, 'other')
}

GTSDB_LABELS_MAIN = {
    '0':(0,'Background'),
    '1':(1,'prohibitory'),
    '2':(1,'prohibitory'),
    '3':(1,'prohibitory'),
    '4':(1,'prohibitory'),
    '5':(1,'prohibitory'),
    '6':(1,'prohibitory'),
    '7':(4,'other'),
    '8':(1,'prohibitory'),
    '9':(1,'prohibitory'),
    '10':(1,'prohibitory'),
    '11':(1,'prohibitory'),
    '12':(2,'danger'),
    '13':(4,'other'),
    '14':(4,'other'),
    '15':(4,'other'),
    '16':(1,'prohibitory'),
    '17':(1,'prohibitory'),
    '18':(4,'other'),
    '19':(2,'danger'),
    '20':(2,'danger'),
    '21':(2,'danger'),
    '22':(2,'danger'),
    '23':(2,'danger'),
    '24':(2,'danger'),
    '25':(2,'danger'),
    '26':(2,'danger'),
    '27':(2,'danger'),
    '28':(2,'danger'),
    '29':(2,'danger'),
    '30':(2,'danger'),
    '31':(2,'danger'),
    '32':(2,'danger'),
    '33':(4,'other'),
    '34':(3,'mandatory'),
    '35':(3,'mandatory'),
    '36':(3,'mandatory'),
    '37':(3,'mandatory'),
    '38':(3,'mandatory'),
    '39':(3,'mandatory'),
    '40':(3,'mandatory'),
    '41':(3,'mandatory'),
    '42':(4,'other'),
    '43':(4,'other'),
}

GTSDB_LABELS_M = {
    '1':(1,'prohibitory'),
    '2':(2,'danger'),
    '3':(3,'mandatory'),
    '4':(4,'other')
}

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

COCO_LABELS = {
    "bench":  (14, 'outdoor') ,
    "skateboard":  (37, 'sports') ,
    "toothbrush":  (80, 'indoor') ,
    "person":  (1, 'person') ,
    "donut":  (55, 'food') ,
    "none":  (0, 'background') ,
    "refrigerator":  (73, 'appliance') ,
    "horse":  (18, 'animal') ,
    "elephant":  (21, 'animal') ,
    "book":  (74, 'indoor') ,
    "car":  (3, 'vehicle') ,
    "keyboard":  (67, 'electronic') ,
    "cow":  (20, 'animal') ,
    "microwave":  (69, 'appliance') ,
    "traffic light":  (10, 'outdoor') ,
    "tie":  (28, 'accessory') ,
    "dining table":  (61, 'furniture') ,
    "toaster":  (71, 'appliance') ,
    "baseball glove":  (36, 'sports') ,
    "giraffe":  (24, 'animal') ,
    "cake":  (56, 'food') ,
    "handbag":  (27, 'accessory') ,
    "scissors":  (77, 'indoor') ,
    "bowl":  (46, 'kitchen') ,
    "couch":  (58, 'furniture') ,
    "chair":  (57, 'furniture') ,
    "boat":  (9, 'vehicle') ,
    "hair drier":  (79, 'indoor') ,
    "airplane":  (5, 'vehicle') ,
    "pizza":  (54, 'food') ,
    "backpack":  (25, 'accessory') ,
    "kite":  (34, 'sports') ,
    "sheep":  (19, 'animal') ,
    "umbrella":  (26, 'accessory') ,
    "stop sign":  (12, 'outdoor') ,
    "truck":  (8, 'vehicle') ,
    "skis":  (31, 'sports') ,
    "sandwich":  (49, 'food') ,
    "broccoli":  (51, 'food') ,
    "wine glass":  (41, 'kitchen') ,
    "surfboard":  (38, 'sports') ,
    "sports ball":  (33, 'sports') ,
    "cell phone":  (68, 'electronic') ,
    "dog":  (17, 'animal') ,
    "bed":  (60, 'furniture') ,
    "toilet":  (62, 'furniture') ,
    "fire hydrant":  (11, 'outdoor') ,
    "oven":  (70, 'appliance') ,
    "zebra":  (23, 'animal') ,
    "tv":  (63, 'electronic') ,
    "potted plant":  (59, 'furniture') ,
    "parking meter":  (13, 'outdoor') ,
    "spoon":  (45, 'kitchen') ,
    "bus":  (6, 'vehicle') ,
    "laptop":  (64, 'electronic') ,
    "cup":  (42, 'kitchen') ,
    "bird":  (15, 'animal') ,
    "sink":  (72, 'appliance') ,
    "remote":  (66, 'electronic') ,
    "bicycle":  (2, 'vehicle') ,
    "tennis racket":  (39, 'sports') ,
    "baseball bat":  (35, 'sports') ,
    "cat":  (16, 'animal') ,
    "fork":  (43, 'kitchen') ,
    "suitcase":  (29, 'accessory') ,
    "snowboard":  (32, 'sports') ,
    "clock":  (75, 'indoor') ,
    "apple":  (48, 'food') ,
    "mouse":  (65, 'electronic') ,
    "bottle":  (40, 'kitchen') ,
    "frisbee":  (30, 'sports') ,
    "carrot":  (52, 'food') ,
    "bear":  (22, 'animal') ,
    "hot dog":  (53, 'food') ,
    "teddy bear":  (78, 'indoor') ,
    "knife":  (44, 'kitchen') ,
    "train":  (7, 'vehicle') ,
    "vase":  (76, 'indoor') ,
    "banana":  (47, 'food') ,
    "motorcycle":  (4, 'vehicle') ,
    "orange":  (50, 'food')
  }

# use dataset_inspect.py to get these summary
data_splits_num = {
    'train': 588,
    'val': 153,
}

def slim_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, image_preprocessing_fn, anchor_encoder, num_epochs=None, is_training=True):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      num_classes: total class numbers in dataset.
      batch_size: the size of each batch.
      split_name: 'train' of 'val'.
      file_pattern: The file pattern to use when matching the dataset sources (full path).
      num_readers: the max number of reader used for reading tfrecords.
      num_preprocessing_threads: the max number of threads used to run preprocessing function.
      image_preprocessing_fn: the function used to dataset augumentation.
      anchor_encoder: the function used to encoder all anchors.
      num_epochs: total epoches for iterate this dataset.
      is_training: whether we are in traing phase.

    Returns:
      A batch of [image, shape, loc_targets, cls_targets, match_scores].
    """
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = {}
    for name, pair in GTSDB_LABELS.items():
        labels_to_names[pair[0]] = name

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=data_splits_num[split_name],
                items_to_descriptions=None,
                num_classes=num_classes,
                labels_to_names=labels_to_names)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_readers,
            common_queue_capacity=32 * batch_size,
            common_queue_min=8 * batch_size,
            shuffle=is_training,
            num_epochs=num_epochs)

    [org_image, filename, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'filename', 'shape',
                                                                     'object/label',
                                                                     'object/bbox',
                                                                     'object/difficult'])

    if is_training:
        # if all is difficult, then keep the first one
        isdifficult_mask =tf.cond(tf.count_nonzero(isdifficult, dtype=tf.int32) < tf.shape(isdifficult)[0],
                                lambda : isdifficult < tf.ones_like(isdifficult),
                                lambda : tf.one_hot(0, tf.shape(isdifficult)[0], on_value=True, off_value=False, dtype=tf.bool))

        glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
        gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)

    # Pre-processing image, labels and bboxes.

    if is_training:
        image, glabels, gbboxes = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
    else:
        image = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
        glabels, gbboxes = glabels_raw, gbboxes_raw

    gt_targets, gt_labels, gt_scores = anchor_encoder(glabels, gbboxes)

    return tf.train.batch([image, filename, shape, gt_targets, gt_labels, gt_scores],
                    dynamic_pad=False,
                    batch_size=batch_size,
                    allow_smaller_final_batch=(not is_training),
                    num_threads=num_preprocessing_threads,
                    capacity=64 * batch_size)
