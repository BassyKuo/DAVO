from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

def se_block(input_feature, name, ratio=8, mode='gp', activation=tf.nn.relu):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    Adapted from: https://github.com/kobiso/SENet-tensorflow-slim/blob/51b6d27b57498ce163d3b21acdb45292d1857619/nets/attention_module.py
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        if mode == 'gp':
            # Global average pooling 1x1
            squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
            print ("[SE_BLOCK] use global_pooling: ", squeeze)
        elif mode == 'gp2x2':
            # Global average pooling 2x2
            h, w = input_feature.shape.as_list()[1:3]
            half_h, half_w = h // 2, w // 2
            squeeze = tf.concat([
                tf.reduce_mean(input_feature[:, :half_h, :half_w, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, :half_h, half_w:, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, half_h:, :half_w, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, half_h:, half_w:, :], axis=[1,2], keepdims=True)
                ], axis=-1)
            print ("[SE_BLOCK] use global_pooling 2x2: ", squeeze)
        else:
            raise NameError("not support `%s' mode." % mode)
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel//ratio,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        print ("[SE_BLOCK] .... (fc1) bottleneck_fc: ", excitation)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        print ("[SE_BLOCK] .... (fc2) recover_fc: ", excitation)
        scale = input_feature * excitation
    return scale

def se(input_feature, name, layer_channels=[8,19], spp_size=[8,6,4], mode='gp', activation=tf.nn.relu):
    """Contains the implementation of Squeeze-and-Excitation block.
    As described in https://arxiv.org/abs/1709.01507.
    Adapted from: https://github.com/kobiso/SENet-tensorflow-slim/blob/51b6d27b57498ce163d3b21acdb45292d1857619/nets/attention_module.py
    """

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        if mode == 'gp':
            # Global average pooling 1x1
            pooling_vector = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
            print ("[SE] use global_pooling 1x1: ", pooling_vector)
        elif mode == 'gp2x2':
            # Global average pooling 2x2
            h, w = input_feature.shape.as_list()[1:3]
            half_h, half_w = h // 2, w // 2
            pooling_vector = tf.concat([
                tf.reduce_mean(input_feature[:, :half_h, :half_w, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, :half_h, half_w:, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, half_h:, :half_w, :], axis=[1,2], keepdims=True),
                tf.reduce_mean(input_feature[:, half_h:, half_w:, :], axis=[1,2], keepdims=True)
                ], axis=-1)
            print ("[SE] use global_pooling 2x2: ", pooling_vector)
        elif mode == 'spp':
            # Pyramid pooling
            pooling_vector = spatial_pyramid_pool(input_feature,
                                        int(input_feature.get_shape()[0]),
                                        [int(input_feature.get_shape()[1]), int(input_feature.get_shape()[2])],
                                        out_pool_size=spp_size)
            pooling_vector = tf.expand_dims(tf.expand_dims(pooling_vector, 1), 1)
            print ("[SE] use spp_pooling: ", pooling_vector)
        else:
            raise NameError("not support `%s' mode." % mode)
        excitation = tf.layers.dense(inputs=pooling_vector,
                                     units=layer_channels[0],
                                     activation=activation,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        print ("[SE] .... (fc1) bottleneck_fc: ", excitation)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=layer_channels[1],
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        print ("[SE] .... (fc2) recover_fc: ", excitation)
    return excitation

def se_spp_block(input_feature, name, ratio=8, spp_size=[2,1], activation=tf.nn.relu):

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        # Pyramid pooling
        #squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        spp_pooling = spatial_pyramid_pool(input_feature,
                                    int(input_feature.get_shape()[0]),
                                    [int(input_feature.get_shape()[1]), int(input_feature.get_shape()[2])],
                                    out_pool_size=spp_size)
        spp_pooling = tf.expand_dims(tf.expand_dims(spp_pooling, 1), 1)
        print ("[SE_SPP_BLOCK] use spp_pooling: ", spp_pooling)
        excitation = tf.layers.dense(inputs=spp_pooling,
                                     units=channel//ratio,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        scale = input_feature * excitation
    return scale

# Spatial Pyramid Pooling block
# https://arxiv.org/abs/1406.4729
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of avgerage pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling

    Adapted from: https://github.com/peace195/sppnet/blob/master/alexnet_spp.py#L109
    """
    for i in range(len(out_pool_size)):
        h_strd = h_size = math.ceil(float(previous_conv_size[0]) / out_pool_size[i])
        w_strd = w_size = math.ceil(float(previous_conv_size[1]) / out_pool_size[i])
        pad_h = int(out_pool_size[i] * h_size - previous_conv_size[0])
        pad_w = int(out_pool_size[i] * w_size - previous_conv_size[1])
        new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]]))
        #max_pool = tf.nn.max_pool(new_previous_conv,
                                #ksize=[1,h_size, h_size, 1],
                                #strides=[1,h_strd, w_strd,1],
                                #padding='SAME')
        avg_pool = tf.nn.avg_pool(new_previous_conv,
                                ksize=[1,h_size, h_size, 1],
                                strides=[1,h_strd, w_strd,1],
                                padding='SAME')
        if (i == 0):
            spp = tf.reshape(avg_pool, [num_sample, -1])
        else:
            spp = tf.concat(axis=1, values=[spp, tf.reshape(avg_pool, [num_sample, -1])])

    return spp

