from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def squeezenet_arg_scope(weight_decay=0.0005):
  """Defines the squeezenet arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer(),
                      padding='SAME') as arg_sc:
    return arg_sc

def squeezenet_v11(inputs,
                   num_classes=1000,
                   is_training=True,
                   dropout_keep_prob=0.5,
                   scope='squeezenet_v11',
                   output_pred=False):
  """Squeezenet V1.1

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'squeezenet_v11', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):

      net = slim.conv2d(inputs, 64, [3, 3], stride=2, padding='SAME', scope='conv1')
      
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      # fire2
      net = slim.conv2d(net, 16, [1, 1], stride=1, padding='SAME', scope='fire2/squeeze')
      expand1 = slim.conv2d(net, 64, [1, 1], stride=1, padding='SAME', scope='fire2/expand1')
      expand3 = slim.conv2d(net, 64, [3, 3], stride=1, padding='VALID', scope='fire2/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire2/concat')

      # fire3
      net = slim.conv2d(net, 16, [1, 1], stride=1, padding='SAME', scope='fire3/squeeze')
      expand1 = slim.conv2d(net, 64, [1, 1], stride=1, padding='SAME', scope='fire3/expand1')
      expand3 = slim.conv2d(net, 64, [3, 3], stride=1, padding='VALID', scope='fire3/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire3/concat')

      net = slim.max_pool2d(net, [2, 2], scope='pool3')

      # fire4
      net = slim.conv2d(net, 32, [1, 1], stride=1, padding='SAME', scope='fire4/squeeze')
      expand1 = slim.conv2d(net, 128, [1, 1], stride=1, padding='SAME', scope='fire4/expand1')
      expand3 = slim.conv2d(net, 128, [3, 3], stride=1, padding='VALID', scope='fire4/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire4/concat')

      # fire5
      net = slim.conv2d(net, 32, [1, 1], stride=1, padding='SAME', scope='fire5/squeeze')
      expand1 = slim.conv2d(net, 128, [1, 1], stride=1, padding='SAME', scope='fire5/expand1')
      expand3 = slim.conv2d(net, 128, [3, 3], stride=1, padding='VALID', scope='fire5/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire5/concat')

      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # fire6
      net = slim.conv2d(net, 48, [1, 1], stride=1, padding='SAME', scope='fire6/squeeze')
      expand1 = slim.conv2d(net, 192, [1, 1], stride=1, padding='SAME', scope='fire6/expand1')
      expand3 = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID', scope='fire6/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire6/concat')

      # fire7
      net = slim.conv2d(net, 48, [1, 1], stride=1, padding='SAME', scope='fire7/squeeze')
      expand1 = slim.conv2d(net, 192, [1, 1], stride=1, padding='SAME', scope='fire7/expand1')
      expand3 = slim.conv2d(net, 192, [3, 3], stride=1, padding='VALID', scope='fire7/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire7/concat')

      # fire8
      net = slim.conv2d(net, 64, [1, 1], stride=1, padding='SAME', scope='fire8/squeeze')
      expand1 = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='fire8/expand1')
      expand3 = slim.conv2d(net, 256, [3, 3], stride=1, padding='VALID', scope='fire8/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire8/concat')

      # fire9
      net = slim.conv2d(net, 64, [1, 1], stride=1, padding='SAME', scope='fire9/squeeze')
      expand1 = slim.conv2d(net, 256, [1, 1], stride=1, padding='SAME', scope='fire9/expand1')
      expand3 = slim.conv2d(net, 256, [3, 3], stride=1, padding='VALID', scope='fire9/expand3')
      net = tf.concat(axis=3, [expand1, expand3], name='fire9/concat')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if output_pred:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout9')
        net = slim.conv2d(net, num_classes, [1, 1], stride=1, padding='SAME', scope='conv10')
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='pool10')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='pool10/squeezed')
        end_points['pool10'] = net
      return net, end_points
squeezenet_v11.default_image_size = 224
