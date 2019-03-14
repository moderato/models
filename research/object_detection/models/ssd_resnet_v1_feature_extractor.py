"""SSDFeatureExtractor for MobilenetV1 features."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import resnet_v1

slim = tf.contrib.slim


class SSDResnetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using ResNetV1 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               resnet_base_fn,
               resnet_scope_name,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      resnet_base_fn: base resnet network to use.
      resnet_scope_name: scope name under which to construct resnet
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.

    Raises:
      ValueError: On supplying invalid arguments for unused arguments.
    """
    super(SSDResnetV1FeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    if self._depth_multiplier != 1.0:
      raise ValueError('Only depth 1.0 is supported, found: {}'.
                       format(self._depth_multiplier))
    if self._use_explicit_padding is True:
      raise ValueError('Explicit padding is not a valid option.')
    self._resnet_base_fn = resnet_base_fn
    self._resnet_scope_name = resnet_scope_name
    self._additional_layer_depth = additional_layer_depth

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      return resized_inputs - [[channel_means]]
    else:
      return resized_inputs

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block2', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]

    Raises:
      ValueError: depth multiplier is not supported.
    """
    if self._depth_multiplier != 1.0:
      raise ValueError('Depth multiplier not supported.')

    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': ['block2', 'block4', '', '', '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }

    with tf.variable_scope(
        self._resnet_scope_name, reuse=self._reuse_weights) as scope:
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = self._resnet_base_fn(
              inputs=ops.pad_to_multiple(preprocessed_inputs,
                                         self._pad_to_multiple),
              num_classes=None,
              is_training=None,
              global_pool=False,
              output_stride=None,
              store_non_strided_activations=True,
              scope=scope)
          image_features = self._filter_features(image_features)

      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

        # print("^^^^^")
        # print(feature_maps)
        # print(type(feature_maps))
        # print("^^^^^")
        
    return feature_maps.values()

class SSDResnet18V1FeatureExtractor(SSDResnetV1FeatureExtractor):
  """SSD Resnet18 V1 feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet18 V1 feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet18V1FeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_18,
        'resnet_v1_18',
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

class SSDResnet50V1FeatureExtractor(SSDResnetV1FeatureExtractor):
  """SSD Resnet50 V1 feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet50 V1 feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet50V1FeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_50,
        'resnet_v1_50',
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)