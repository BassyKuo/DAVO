# origin: https://github.com/tinghuiz/SfMLearner/blob/master/nets.py
from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
from .attention_module import se_block

#TRANSLATION_SCALING = 1.      # should set to 50 to use the gt pose
#TRANSLATION_SCALING = 50.      # should set to 50 to use the gt pose

def couple_net_v0_dilation(tgt_image, src0_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=128):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 2

    inputs = tf.concat([tgt_image, src0_image, src1_image], axis=3)
    print (">>> [PoseNN] inputs : ", inputs)
    print (">>> [PoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [PoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], rate=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], rate=4, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], rate=8, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            with tf.variable_scope('pose'):
                # cnv6 = tf.layers.conv2d(cnv5, 128, [3, 3], dilation_rate=(2, 2), padding='same', activation=tf.nn.relu, name='cnv6')
                if se_attention is True:         # mode1
                    print (">>> [PoseNN] use se_attention (insert se_block between cnv5 & cnv6)")
                    cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                elif se_attention == 'se_skipadd':    # mode3
                    print (">>> [PoseNN] use cnv5 + dilated_cnv6_se_attention")
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                    se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                    cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                elif se_attention == 'se_replace':    # mode2
                    print (">>> [PoseNN] use se_attention replace with cnv6")
                    cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                else:
                    cnv6  = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                cnv7      = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg  = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant facilitates training.
                pose_final  = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6, cnv6)


def decouple_net_v0_dilation(tgt_image, src0_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=128):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 2

    inputs = tf.concat([tgt_image, src0_image, src1_image], axis=3)
    print (">>> [PoseNN] inputs : ", inputs)
    print (">>> [PoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [PoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], rate=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], rate=4, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], rate=8, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            cnv6s = {}
            poses_avg = {}
            with tf.variable_scope('pose'):
                for name in ['rotation', 'translation']:
                    with tf.variable_scope(name):
                        # cnv6 = tf.layers.conv2d(cnv5, 128, [3, 3], dilation_rate=(2, 2), padding='same', activation=tf.nn.relu, name='cnv6')
                        if se_attention is True:         # mode1
                            print (">>> [PoseNN][%s] use se_attention (insert se_block between cnv5 & cnv6)" % name)
                            cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                        elif se_attention == 'se_skipadd':    # mode3
                            print (">>> [PoseNN][%s] use cnv5 + dilated_cnv6_se_attention" % name)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                            se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                            cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                        elif se_attention == 'se_replace':    # mode2
                            print (">>> [PoseNN][%s] use se_attention replace with cnv6" % name)
                            cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                        else:
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                        pred = slim.conv2d(cnv7, 3*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                        avg = tf.reduce_mean(pred, [1, 2])
                        poses_avg[name] = avg
                        cnv6s[name] = cnv6
                # Empirically we found that scaling by a small constant facilitates training.
                rot_final   = tf.reshape(poses_avg['rotation'],    [-1, num_source, 3])
                trans_final = tf.reshape(poses_avg['translation'], [-1, num_source, 3])
                pose_final  = 0.01 * tf.concat([rot_final, trans_final], axis=-1)   # -V4 : 2019/08/05
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6s['rotation'], cnv6s['translation'])

def couple_sharednet_v0_dilation(tgt_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=128):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 1

    inputs = tf.concat([tgt_image, src1_image], axis=3)
    print (">>> [SharedPoseNN] inputs : ", inputs)
    print (">>> [SharedPoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [SharedPoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net', reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], rate=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], rate=4, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], rate=8, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            with tf.variable_scope('pose'):
                # cnv6 = tf.layers.conv2d(cnv5, 128, [3, 3], dilation_rate=(2, 2), padding='same', activation=tf.nn.relu, name='cnv6')
                if se_attention is True:         # mode1
                    print (">>> [SharedPoseNN] use se_attention (insert se_block between cnv5 & cnv6)")
                    cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                elif se_attention == 'se_skipadd':    # mode3
                    print (">>> [SharedPoseNN] use cnv5 + dilated_cnv6_se_attention")
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                    se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                    cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                elif se_attention == 'se_replace':    # mode2
                    print (">>> [SharedPoseNN] use se_attention replace with cnv6")
                    cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                else:
                    cnv6  = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                cnv7      = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg  = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant facilitates training.
                pose_final  = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6, cnv6)

def decouple_sharednet_v0_dilation(tgt_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=128):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 1

    inputs = tf.concat([tgt_image, src1_image], axis=3)
    print (">>> [SharedPoseNN] inputs : ", inputs)
    print (">>> [SharedPoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [SharedPoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net', reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], rate=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], rate=4, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], rate=8, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            cnv6s = {}
            poses_avg = {}
            with tf.variable_scope('pose'):
                for name in ['rotation', 'translation']:
                    with tf.variable_scope(name):
                        # cnv6 = tf.layers.conv2d(cnv5, 128, [3, 3], dilation_rate=(2, 2), padding='same', activation=tf.nn.relu, name='cnv6')
                        if se_attention is True:         # mode1
                            print (">>> [SharedPoseNN][%s] use se_attention (insert se_block between cnv5 & cnv6)" % name)
                            cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                        elif se_attention == 'se_skipadd':    # mode3
                            print (">>> [SharedPoseNN][%s] use cnv5 + dilated_cnv6_se_attention" % name)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                            se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                            cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                        elif se_attention == 'se_replace':    # mode2
                            print (">>> [SharedPoseNN][%s] use se_attention replace with cnv6" % name)
                            cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                        else:
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], rate=2, scope='cnv6')
                        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                        pred = slim.conv2d(cnv7, 3*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                        avg = tf.reduce_mean(pred, [1, 2])
                        print ("layer 7:", cnv7)
                        print ("layer 8:", pred)
                        print ("avg:", avg)
                        poses_avg[name] = avg
                        cnv6s[name] = cnv6
                # Empirically we found that scaling by a small constant facilitates training.
                rot_final   = tf.reshape(poses_avg['rotation'],    [-1, num_source, 3])
                trans_final = tf.reshape(poses_avg['translation'], [-1, num_source, 3])
                pose_final  = 0.01 * tf.concat([rot_final, trans_final], axis=-1)   # -V4 : 2019/08/05
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6s['rotation'], cnv6s['translation'])


def couple_net_v0(tgt_image, src0_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=256):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 2

    inputs = tf.concat([tgt_image, src0_image, src1_image], axis=3)
    print (">>> [PoseNN] inputs : ", inputs)
    print (">>> [PoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [PoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            with tf.variable_scope('pose'):
                if se_attention is True:         # mode1
                    print (">>> [PoseNN] use se_attention (insert se_block between cnv5 & cnv6)")
                    cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=2, scope='cnv6')
                elif se_attention == 'se_skipadd':    # mode3
                    print (">>> [PoseNN] use cnv5 + cnv6_se_attention (stride=1)")
                    cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=1, scope='cnv6')
                    se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                    cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                elif se_attention == 'se_replace':    # mode2
                    print (">>> [PoseNN] use se_attention replace with cnv6")
                    cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                else:
                    cnv6  = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=2, scope='cnv6')
                cnv7      = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg  = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant facilitates training.
                pose_final   = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
                print (">>> pose_scale = 0.01")
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6, cnv6)


def decouple_net_v0(tgt_image, src0_image, src1_image, dropout=False, is_training=True, se_attention=False, batch_norm=False, cnv6_num_outputs=256):
    """
    Input:
      flow_maps: centrelized.
    Return:
      pose_final = [rz,ry,rx,tx,ty,tz]
    """
    num_source = 2

    inputs = tf.concat([tgt_image, src0_image, src1_image], axis=3)
    print (">>> [PoseNN] inputs : ", inputs)
    print (">>> [PoseNN] use batch_norm : ", slim.batch_norm if batch_norm else None)
    print (">>> [PoseNN] cnv6_num_outputs = ", cnv6_num_outputs)

    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm if batch_norm else None,
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            if dropout:
                cnv5 = slim.dropout(cnv5, 0.7, is_training=is_training)
            # Pose specific layers
            cnv6s = {}
            poses_avg = {}
            with tf.variable_scope('pose'):
                for name in ['rotation', 'translation']:
                    with tf.variable_scope(name):
                        if se_attention is True:         # mode1
                            print (">>> [PoseNN][%s] use se_attention (insert se_block between cnv5 & cnv6)" % name)
                            cnv5 = se_block(cnv5, 'cnv5_se_attention', ratio=8)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=2, scope='cnv6')
                        elif se_attention == 'se_skipadd':    # mode3
                            print (">>> [PoseNN][%s] use cnv5 + cnv6_se_attention (stride=1)" % name)
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=1, scope='cnv6')
                            se_cnv6 = se_block(cnv6, 'cnv6_se_attention', ratio=8)
                            cnv6 = tf.nn.relu(cnv5 + se_cnv6, name="cnv6_se_attention/add_cnv5/relu")
                        elif se_attention == 'se_replace':    # mode2
                            print (">>> [PoseNN][%s] use se_attention replace with cnv6" % name)
                            cnv6 = se_block(cnv5, 'cnv6_se_attention', ratio=8)
                        else:
                            cnv6 = slim.conv2d(cnv5, cnv6_num_outputs, [3, 3], stride=2, scope='cnv6')
                        cnv7 = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                        pred = slim.conv2d(cnv7, 3*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                        avg = tf.reduce_mean(pred, [1, 2])
                        poses_avg[name] = avg
                        cnv6s[name] = cnv6
                # Empirically we found that scaling by a small constant facilitates training.
                #pose_avg = tf.concat([rot_avg, trans_avg], axis=-1)
                #pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])        # -V2 : WRONG WAY! QQQQQ
                rot_final   = tf.reshape(poses_avg['rotation'],    [-1, num_source, 3])
                trans_final = tf.reshape(poses_avg['translation'], [-1, num_source, 3])
                pose_final  = 0.01 * tf.concat([rot_final, trans_final], axis=-1)   # -V4 : 2019/08/05
                print (">>> pose_scale = 0.01")
            # Exp mask specific layers
            end_points = utils.convert_collection_to_dict(end_points_collection)
            #return pose_final, end_points
            return pose_final, (cnv6s['rotation'], cnv6s['translation'])

def build_seg_channel_weight(seg, prefix=""):
    """
    Input:
      seg: 1 channel segmentation train-id (0 ~ 18)
    """
    seg = tf.cast(seg, tf.int32)
    with tf.variable_scope('{}seg_channel_weight'.format(prefix), reuse=tf.AUTO_REUSE):
        seg = tf.one_hot(seg, depth=19, dtype=tf.float32)                   # shape=[B,H,W,1,19]
        weight = tf.get_variable('weight', shape=(19,), dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=0, stddev=0.05), trainable=True)
        weight = tf.sigmoid(weight)
        print (">>> [PoseNN] build segmentation channels weights : ", weight)
        mask = tf.multiply(seg, weight)
        mask = tf.reduce_sum(mask, axis=4)
        return mask, weight

