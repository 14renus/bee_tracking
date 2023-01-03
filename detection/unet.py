import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import math
from utils.func import CLASSES
import numpy as np
from utils.func import clipped_sigmoid

def _create_conv_relu(inputs, name, filters, dropout_ratio, is_training, strides=[1,1], kernel_size=[3,3], padding="SAME", relu=True, random_seed=0):
    '''
    Create layer computation with Con2D, dropout, batch normalization, and leadyrelu.

    Args:
      input: placeholder inputs.
      name: prefix of layer names.
      filters: number of filters in conv2d layer, each filter transforms [input w, input h, channels] --> [image w, image h, 1]
        num filters equivalent to output dimension, output shape has [batch_size, input w, input h, filters]
      dropout_ratio: fraction rate of input units randomly set to 0.
      kernal_size: kernel h and w. 1 filter shape is [kernel h, kernel w, input channels]
      padding: "SAME" preserves input w and h. otherwise, naturally applying filters reduces input dimension by 2*kernel_dim//2.
      relu: whether to apply relu activation.
      random_seed: seed for droupout, to remove variance in training.
    '''
    net = tf.layers.conv2d(inputs=inputs, filters=filters, strides=strides, kernel_size=kernel_size, padding=padding, name="%s_conv" % name)
    if dropout_ratio > 0:
        if random_seed > 0:
            net = tf.layers.dropout(inputs=net, rate=dropout_ratio, seed=random_seed, training=is_training, name="%s_dropout" % name)
        else:
            net = tf.layers.dropout(inputs=net, rate=dropout_ratio, training=is_training, name="%s_dropout" % name)
    net = tf.layers.batch_normalization(net, center=True, scale=False, training=is_training, name="%s_bn" % name)
    if relu:
        net = tf.nn.relu(net) # leaky relu
    return net


def _create_pool(data, name, pool_size=[2,2], strides=[2,2]):
    '''
    Create max pool layer to downsample data.

    Default pool_size and strides means divide input h and w by 2.
    '''
    pool = tf.layers.max_pooling2d(inputs=data, pool_size=pool_size, strides=strides, padding='SAME', name=name)
    return pool


def _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training):
    '''
    Create contracting section of U-Net with num_layers number of steps.

    With each step, the input w and h dvides by 2. The num of channels (dim_out) doubles.
    Each step consists of 2 convolution blocks and then downsampling.
    '''
    interim = []

    dim_out = num_filters
    for i in range(num_layers):
        name = "c_%i" % i
        conv1 = _create_conv_relu(data, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        pool = _create_pool(conv2, name)
        data = pool

        dim_out *=2
        interim.append(conv2)

    return (interim, data)


def _expansive_path(data, interim, num_layers, dim_in, dropout_ratio, is_training):
    '''
    Create expansive path of U-Net with num_layers number of steps.

    With each step the input w and h doubles to reach the original input size. The number of channels (dim_out) divides by 2.
    Each step consists of upsampling, concatenation with the output of the corresponding output in the contracting path,
    and then 2 convolution blocks.
    '''
    dim_out = int(dim_in / 2)
    for i in range(num_layers):
        name = "e_%i" % i
        upconv = tf.layers.conv2d_transpose(data, filters=dim_out, kernel_size=2, strides=2, name="%s_upconv" % name)
        concat = tf.concat([interim[len(interim)-i-1], upconv], 3)
        conv1 = _create_conv_relu(concat, name + "_1", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        #suffix = "last" if (i == num_layers - 1) else suffix + "_2"
        conv2 = _create_conv_relu(conv1, name + "_2", dim_out, dropout_ratio=dropout_ratio, is_training=is_training)
        data = conv2
        dim_out = int(dim_out / 2)
    return data


def create_unet2(num_layers, num_filters, data, is_training, prev=None, dropout_ratio=0, set_random_seed=False, num_classes=3):
    '''
    Creates U-net architecture given architecture params and data placeholder.

    Args:
      num_layers: number of layer steps (made of 2 conv+relu pairs with same outout dim) in each of the contracting and expansive paths.
      num_filters: number of convolution filters (output dimension) of first layer step in contracting path.
      data: placeholder with shape [batch_size, image h, image w, num_channels].
      prev: relu output of the previous frame, if available.
      dropout_ratio: ratio of neurons to drop during dropout for each forward call.
      set_random_seed: remove variance from dropout layer between train iterations.
      num_classes: number of classes to predict.
    '''
    classes = num_classes if num_classes > 2 else 1

    (interim, contracting_data) = _contracting_path(data, num_layers, num_filters, dropout_ratio, is_training)

    middle_dim = num_filters * 2**num_layers
    middle_conv_1 = _create_conv_relu(contracting_data, "m_1", middle_dim, dropout_ratio=dropout_ratio, random_seed=set_random_seed*1, is_training=is_training)
    middle_conv_2 = _create_conv_relu(middle_conv_1, "m_2", middle_dim, dropout_ratio=dropout_ratio, random_seed=set_random_seed*2, is_training=is_training)
    middle_end = middle_conv_2

    expansive_path = _expansive_path(middle_end, interim, num_layers, middle_dim, dropout_ratio, is_training)
    last_relu = expansive_path

    if prev != None:
        expansive_path = tf.concat([prev, expansive_path], 3)

    conv_logits = _create_conv_relu(expansive_path, "conv_logits", num_filters, dropout_ratio=dropout_ratio, is_training=is_training)
    logits = _create_conv_relu(conv_logits, "logits", classes, dropout_ratio=dropout_ratio, is_training=is_training)

    conv_angle = _create_conv_relu(expansive_path, "conv_angle", num_filters, dropout_ratio=dropout_ratio, is_training=is_training, relu=False)
    angle_pred = _create_conv_relu(conv_angle, "angle_pred", 1, dropout_ratio=dropout_ratio, is_training=is_training, relu=False)
    return logits, last_relu, angle_pred


def loss(logits, labels, weight_map, num_classes=3):
    '''
    Calculates cross entropy loss of bee class predictions.

    :param logits: segmentation output of bee class head, no softmax applied.
    :param labels: same dimension of logits, integers corresponding to classes (starting with 0).
    :param weight_map: same dimension as the logits, weight to apply to each pixel prediction.
    :param numclasses: dimension of logits and labels.
    :return: weighted average cross entropy loss.
    '''
    if num_classes > 2:
        oh_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8), depth=num_classes, name="one_hot")
        # Compute softmax and loss across last axis, which is the class dimension.
        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=oh_labels)
    else:
        logits = tf.squeeze(logits)  # (BATCH_SIZE, DS, DS, 1) --> (BATCH_SIZE, DS, DS,)
        loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32))
    weighted_loss = tf.multiply(loss_map, weight_map)
    loss = tf.reduce_mean(weighted_loss, name="weighted_loss")
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')


def angle_loss(angle_pred, angle_labels, weight_map, ignore_bg=False, use_weights=True):
    '''

    :param angle_pred: model output
    :param angle_labels: radians/2pi if full bee, 1 if cell bee, -1 if background
    :param weight_map: same dimension as the predictions, weight to apply to each pixel prediction.
    :param ignore_bg: ignore background loss when computing angle_loss
    :return: sum of foreground and background loss, which are weighted averages of:
      foreground: mean squared error (regression loss)
      background: angle loss = sin^2 of the difference of angle pred and label.
    '''

    sh = tf.shape(angle_pred)
    angle_pred = tf.reshape(angle_pred, [sh[0],sh[1],sh[2]])  # [BATCH_SIZE, DS, DS]
    # Masks on pixels for background and foreground.
    if ignore_bg:
        bg_mask = tf.less(angle_labels, 0)
    else:
        bg_mask = tf.logical_or(tf.less(angle_pred, 0), tf.less(angle_labels, 0))
    fg_mask = tf.logical_not(bg_mask)

    num_bg, num_fg = tf.reduce_sum(tf.cast(bg_mask,dtype=tf.uint8)), tf.reduce_sum(tf.cast(fg_mask,dtype=tf.uint8))

    fg_loss = tf.square(tf.sin((tf.boolean_mask(angle_pred, fg_mask) - tf.boolean_mask(angle_labels, fg_mask)) * math.pi))
    bg_loss = tf.square(tf.boolean_mask(angle_pred, bg_mask) - tf.boolean_mask(angle_labels, bg_mask))

    if use_weights:
        fg_loss = tf.multiply(tf.boolean_mask(weight_map, fg_mask), fg_loss)
        bg_loss = tf.multiply(tf.boolean_mask(weight_map, bg_mask), bg_loss)

    bg_loss = tf.reduce_mean(bg_loss, name="bg_angle_loss")
    fg_loss = tf.reduce_mean(fg_loss, name="fg_angle_loss")

    # Avoid nan if no bg or fg pixels to compute loss over.
    bg_loss = tf.cond(num_bg > 0, lambda: bg_loss, lambda: 0.)
    fg_loss = tf.cond(num_fg > 0, lambda: fg_loss, lambda: 0.)

    if ignore_bg:
        loss = fg_loss
    else:
        loss = fg_loss + bg_loss
    #tf.add_to_collection('losses', loss)
    return loss #tf.add_n(tf.get_collection('losses'), name='total_loss')

def metrics(loss, logits, labels, angle_preds, angle_labels, loss_softmax, loss_angle, num_classes):
        '''
        Calculate metrics for a train or test step.

        :param loss: cross entropy + regression angle loss already calculated from tf model.
        :param logits: Raw class preds before softmax/sigmoid [BATCH_SIZE, DS, DS, num_classes]
        :param angle_preds: Regression preds for angle
        :return: Tuple of metrics
                 - Boolean to indicate train (0) or test (1)
                 - loss: passed from model
                 - bg: "background overlap" = (correct class = 0) / # background pixels
                 - fg: "foreground overlap" = (correct class !=0) / # foreground pixels
                 Just for foreground pixels:
                 - fg_error: "class error" = incorrect class / # foreground pixels
                 - angle_error: "angle error" = mean difference in angle
        '''

        if num_classes > 2:
            pred_class = np.argmax(logits, axis=3)
        else:
            pred_class = np.round(clipped_sigmoid(logits.squeeze()))
        pred_angle = angle_preds[:, :, :, 0]

        lb = labels
        angle = angle_labels
        is_bg = (lb == 0)
        is_fg = np.logical_not(is_bg)
        n_fg = np.sum(is_fg)
        # Background accuracy. Correct if pred class 0 and angle < 0.
        # bg = float(np.sum((pred_class[is_bg] == 0) & (pred_angle[is_bg] < 0)))/np.sum(is_bg)
        bg = float(np.sum(pred_class[is_bg] == 0))/np.sum(is_bg)
        fg = 0
        fg_err = 0
        angle_err = 0
        if n_fg > 0:
            # Foreground accuracy. Correct if pred class != 0.
            fg = float(np.sum(pred_class[is_fg] != 0))/n_fg
            # Foreground error. Incorrect if pred class != label class.
            fg_err = np.mean(lb[is_fg] != pred_class[is_fg])
            # Foreground angle error. Abs difference in angle pred and label
            angle_err = np.mean(np.abs(pred_angle[is_fg] - angle[is_fg]))
        return np.array([0, loss, bg, fg, fg_err, angle_err, loss_softmax, loss_angle])