"""
Tianwei Shen, HKUST, 2018 - 2019.
DeepSlam class defines the training procedure and losses
"""
from __future__ import division
from functools import partial
import re
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets.posenn import *
from nets.depthnn import *
from nets.attention_module import se_block, se_spp_block, se
from nets.monodepth2_nets import Net as MonoNet
from utils.geo_utils import get_relative_pose, projective_inverse_warp, pose_vec2mat, mat2euler, \
    fundamental_matrix_from_rt, reprojection_error, get_inv_pose, warp_depth_by_flow
from utils.tools import disp_to_depth
from utils.flow_utils import flow_to_image
from utils.seg_utils.get_dataset_colormap import label_to_color_image

coef_trans = 1.
coef_rot   = 10.

num_scales = 1

class DAVO(object):
    def __init__(self, version=None, att_19=None):
        self.version = version
        self.att_19  = att_19

    def build_train_graph(self):
        '''[summary]
        build training graph

        Returns:
            data loader and batch sample for train() to initialize
            undefined placeholders
        '''

        opt = self.opt

        is_read_pose     = True
        is_read_flow     = True
        is_read_depth    = False
        is_read_seglabel = True
        self.dropout = "-dropout" in self.version
        print(">>> Read Flow: ", is_read_flow)
        print(">>> Read Depth: ", is_read_depth)
        print(">>> Read Seglabel: ", is_read_seglabel)
        print("[DATA AUGMENTATION] ", opt.data_aug)
        print("[DATA FLIPPING] ", opt.data_flip)
        print("[DROPOUT] ", self.dropout)
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            num_scales,
                            is_read_pose,
                            is_read_flow,
                            is_read_depth,
                            is_read_seglabel,
                            data_aug=opt.data_aug,
                            data_flip=opt.data_flip)
        with tf.name_scope("data_loading"):
            batch_sample = loader.load_train_batch()
            # give additional batch_size info since the input is undetermined placeholder
            inputs = batch_sample.get_next()
            tgt_image = inputs[0]           # frame_t1
            src_image_stack = inputs[1]     # frame_t0, frame_t2
            intrinsics = inputs[2]
            tgt_image.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])                         # [B,128,416,3]
            src_image_stack.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3*opt.num_source])    # [B,128,416,6]
            intrinsics.set_shape([opt.batch_size, num_scales, 3, 3])                                        # [B,4,3,3]
            if is_read_pose:
                poses = inputs[3]           # [row1_pose@*_cam.txt, row2_pose@*_cam.txt, row3_pose@*_cam.txt]
                poses.set_shape([opt.batch_size, 3, 6])
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)
            if is_read_flow:
                pred_flows = inputs[4]
                pred_flows.set_shape([opt.batch_size, opt.num_source*2, opt.img_height, opt.img_width, 2])
                pred_flows_inv = [
                                tf.zeros_like(pred_flows[:,2,...]), # tgt
                                pred_flows[:,2,...],                # tgt->src0
                                pred_flows[:,3,...]                 # tgt->src1
                               ]    # shape=[B,h,w,2]
                pred_flows = [
                                tf.zeros_like(pred_flows[:,0,...]), # tgt
                                pred_flows[:,0,...],                # src0->tgt
                                pred_flows[:,1,...]                 # src1->tgt
                               ]    # shape=[B,h,w,2]

                # flow color [visualization]
                pred_flow_color = [ flow_to_image(flow) for flow in pred_flows ]    # (range: 0~1). shape=[B, h, w, 3]
                self.pred_flow_color = [ self.convert_to_tf_image(flow_color) for flow_color in pred_flow_color[1:] ]

            if is_read_depth:
                pred_depths = inputs[5]
                pred_depths.set_shape([opt.batch_size, opt.num_source+1, opt.img_height, opt.img_width, 1])
                pred_depths = [
                                pred_depths[:,1,...],                # tgt
                                pred_depths[:,0,...],                # src0
                                pred_depths[:,2,...]                 # src1
                               ]    # shape=[B,h,w,1]
                self.pred_depths = pred_depths
            if is_read_seglabel:
                pred_seglabels = inputs[6]
                pred_seglabels.set_shape([opt.batch_size, opt.num_source+1, opt.img_height, opt.img_width, 1])
                pred_seglabels = [
                                pred_seglabels[:,1,...],                # tgt
                                pred_seglabels[:,0,...],                # src0
                                pred_seglabels[:,2,...]                 # src1
                               ]    # shape=[B,h,w,1]
                self.pred_seglabels_color = [label_to_color_image(seg) for seg in pred_seglabels]

        with tf.name_scope("pose_and_explainability_prediction"):
            # 0. Choose PoseNN se mode. (optional)
            if "-se_insert" in self.version:
                se_attention = True
            elif "-se_skipadd" in self.version:
                se_attention = "se_skipadd"
            elif "-se_replace" in self.version:
                se_attention = "se_replace"
            else:
                se_attention = False

            self.input_images  = [
                    tgt_image,   # src0->tgt for -sharedNNforward
                    src_image_stack[...,:3],
                    src_image_stack[...,3:],
                    tgt_image    # tgt->src1 for -sharedNNforward
                    ]

            # 1. Choose PoseNN type.
            if "-sharedNN" in self.version:
                if "-dilatedPoseNN" in self.version:
                    print (">>> choose dilated Shared CNN")
                    poseNet = decouple_sharednet_v0_dilation
                elif "-dilatedCouplePoseNN" in self.version:
                    print (">>> choose dilated Shared CouplePose CNN")
                    poseNet = couple_sharednet_v0_dilation
                elif "-couplePoseNN" in self.version:
                    raise NameError("not support `-sharedNN-couplePoseNN' mode.")
                else:
                    raise NameError("unknown PoseNN type.")
            elif "-dilatedPoseNN" in self.version:
                print (">>> choose dilated CNN")
                poseNet = decouple_net_v0_dilation
            elif "-dilatedCouplePoseNN" in self.version:
                print (">>> choose dilated CouplePose CNN")
                poseNet = couple_net_v0_dilation
            elif "-couplePoseNN" in self.version:
                print (">>> choose CouplePose CNN")
                poseNet = couple_net_v0
            else:
                print (">>> choose vanilla CNN")
                poseNet = decouple_net_v0

            # 1.1. adjust conv6 channels in PoseNN.
            cnv6_num_match = re.search("-cnv6_([0-9]+)", self.version)
            cnv6_num_outputs = 128 if cnv6_num_match is None else int(cnv6_num_match.group(1))
            print (">>> choose cnv6_num_outputs :", cnv6_num_outputs, cnv6_num_match)

            # 2. Concat additional information in inputs.
            Version = re.search("^(v[0-9.]+)", self.version)
            Version = "v0" if Version is None else Version.group(1)
            print (">>> Version: ", Version, re.search("^(v[0-9.]+)", self.version))
            pred_info = None
            if "v0" in Version:
                print (">>> [PoseNN] Only input RGB")
            elif "v1" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (src->tgt)")
                pred_info = pred_flows + [pred_flows[0]]    # src->tgt
            if "-seglabelid" in self.version:
                print (">>> [PoseNN] ...... + seglabelid (1 channel, id=0 ~ 18)")
                if pred_info is not None:
                    pred_info = [
                            tf.concat([info,seg], axis=3) for info,seg in zip(pred_info,pred_seglabels)
                            ]
                else:
                    pred_info = pred_seglabels

            # 3. Attention Part. (AttentionNN)
            # 3.1. Choose activation function in se_block layers.
            if "-fc_tanh" in self.version:
                activation_fn = tf.nn.tanh
                print (">>> [SE][activation_fn] tanh.")
            elif "-fc_lrelu" in self.version:
                activation_fn = tf.nn.leaky_relu
                print (">>> [SE][activation_fn] leaky_relu.")
            else:
                activation_fn = tf.nn.relu
                print (">>> [SE][activation_fn] relu.")

            # 3.2. Prepare normalized optical flows for se_block's inputs.
            se_input_flows = [f for f in pred_flows] + [pred_flows[0]]
            if "-norm_flow" in self.version:
                se_input_flows = [(f - 0.32140523) / 15.384229 for f in se_input_flows]
                print (">>> [SE][input] Normailed : (flow - flow.mean()) / flow.std()")

            # 3.3. Prepare positive value of optical flows for se_block's inputs. (h: horizontal , v: vertical)
            if "-abs_flow_h" in self.version:
                se_input_flows = [ tf.stack([tf.abs(f[...,0]), f[...,1]], axis=-1) for f in se_input_flows]
                print (">>> [SE][input] absolute value of horizontal se_input_flows : | flow[0] | ")
            elif "-abs_flow_v" in self.version:
                se_input_flows = [ tf.stack([f[...,0], tf.abs(f[...,1])], axis=-1) for f in se_input_flows]
                print (">>> [SE][input] absolute value of vertical se_input_flows : | flow[1] | ")
            elif "-abs_flow" in self.version:
                se_input_flows = [tf.abs(f) for f in se_input_flows]
                print (">>> [SE][input] absolute value of se_input_flows : | flow[:2] | ")

            self.se_input_flows_color = [ flow_to_image(f) for f in se_input_flows ]    # (range: 0~1). shape=[B, h, w, 3]
            self.se_input_flows_color = [ self.convert_to_tf_image(f) for f in self.se_input_flows_color ]

            # 3.4. Prepare normalized depth for se_block's inputs.
            if is_read_depth:
                se_input_depths = [d for d in pred_depths] + pred_depths[0]
                if "-norm_depth" in self.version:
                    se_input_depths = [d / 80. for d in se_input_depths]

            # 3.5. Choose se_block's inputs, and generate attentions for PoseNN inputs.
            with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                seg_19  = [ tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels ]
                attention_maps = []
                if "-se_flow_on_depthseg_sharedlayers" in self.version:
                    depth_thres_init = re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version)
                    depth_thres_init = 15. if depth_thres is None else float(depth_thres.group(1))
                    depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=depth_thres_init, stddev=0.1), trainable=True)
                    self.depth_thres = depth_thres
                    print (">>> [PoseNN][se_flow_on_depthseg_sharedlayers] depth threshold = ", depth_thres_init)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    seg_38 = [ tf.concat([seg_near,seg_far], axis=-1) for seg_near,seg_far in zip(depth_seg_19['near'],depth_seg_19['far']) ] # shape=[B,H,W,38]
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_38[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow multiply to near/far segs (i.e., seg_38) :", attention_map)
                elif "-se_flow_on_depthseg_seplayers" in self.version:
                    depth_thres = re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version)
                    depth_thres_init = 15. if depth_thres is None else float(depth_thres.group(1))
                    depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=depth_thres_init, stddev=0.1), trainable=True)
                    self.depth_thres = depth_thres
                    print (">>> [PoseNN][se_flow_on_depthseg_seplayers] depth threshold = ", depth_thres_init)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    for i in range(opt.seq_length):
                        attention_map = 0
                        for name in ['near', 'far']:
                            attention_weights = se(se_input_flows[i], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                            attention_map += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][i] * attention_weights, axis=-1), -1)
                        attention_maps.append(attention_map)
                    print (">>> [PoseNN] build two SE_flow multiply to near/far segs (i.e., seg_38) seperatly:", attention_map)
                elif "-se_flow_on_depthseg" in self.version:
                    raise NameError("please select `-se_flow_on_depthseg_seplayers' or `-se_flow_on_depthseg_sharedlayers'.")
                elif "-se_mixDepthFlow" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = \
                                tf.expand_dims(tf.reduce_sum(
                                                se_block(tf.concat([se_input_depths[i], se_input_flows[i]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), 
                                                axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_depth + flow_attention:", attention_map)
                elif "-se_mixDispFlow" in self.version:
                    se_input_depths = [1./d for d in pred_depths]
                    for i in range(opt.seq_length):
                        attention_map = \
                                tf.expand_dims(tf.reduce_sum(
                                                se_block(tf.concat([se_input_depths[i], se_input_flows[i]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), 
                                                axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_disp + flow_attention:", attention_map)
                elif "-se_flow" in self.version:
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow :", attention_map)
                elif "-se_gp2x2_flow_nobottle" in self.version:
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:19, fc2:19):", attention_map)
                elif "-se_gp2x2_flow" in self.version:
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:8, fc2:19):", attention_map)
                elif "-se_spp21_flow" in self.version:
                    for i in range(opt.seq_length):
                        attention_weights  = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow ([2,1]) (fc1:8, fc2:19):", attention_map)
                elif "-se_spp2_flow" in self.version:   # same to gp2x2
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow ([2]) (fc1:8, fc2:19):", attention_map)
                elif "-se_spp_flow" in self.version or "-se_spp864_flow" in self.version:
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow (fc1:8, fc2:19):", attention_map)
                elif "-se_depth_wo_tgt_to_seg" in self.version:
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_depths[i], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_depth_to_seg" in self.version:
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights  = se(se_input_depths[i], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation:", attention_map)
                elif "-se_depth_wo_tgt" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = \
                            tf.expand_dims(tf.reduce_sum(
                                            se_block(se_input_depths[i], "se_depth", ratio=1, activation=activation_fn), 
                                            axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_depth_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_depth" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[i], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_depth_attention :", attention_map)
                elif "-se_disp_wo_tgt_to_seg" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_disps[i], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_disp_to_seg" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_disps[i], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation:", attention_map)
                elif "-se_disp_wo_tgt" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_disps[i], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_disp_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_disp" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_disps[i], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_disp_attention :", attention_map)
                elif "-se_rgb_wo_tgt_to_seg" in self.version:
                    se_input_rgbs = self.input_images
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_rgbs[i], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_rgb_to_seg" in self.version:
                    se_input_rgbs = self.input_images
                    self.att_19s = []
                    for i in range(opt.seq_length):
                        attention_weights = se(se_input_rgbs[i], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation:", attention_map)
                elif "-se_rgb_wo_tgt" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[i], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_rgb_frames_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_rgb" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[i], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_rgb_frames_attention :", attention_map)
                elif "-se_seg_wo_tgt" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_segmentation_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation_attention :", attention_map)
                elif "-se_gp2x2_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation_attention with 2x2 global pooling:", attention_map)
                elif "-se_spp21_seg" in self.version or "-se_spp_seg_21" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2,1]):", attention_map)
                elif "-se_spp2_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2]):", attention_map)
                elif "-se_spp_seg" in self.version or "-se_spp864_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention :", attention_map)
                elif "-se_SegFlow_to_seg_8" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        se_inputs  = [
                                tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                                tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                                tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                                ]
                        attention_weights = se(se_inputs[i], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:8, fc2:19):", attention_map)
                elif "-se_SegFlow_to_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(opt.seq_length):
                        se_inputs  = [
                                tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                                tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                                tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                                ]
                        attention_weights = se(se_inputs[i], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:19, fc2:19):", attention_map)
                elif "-se_mixSegFlow" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[i], se_input_flows[i]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention:", attention_map)
                elif "-se_spp21_mixSegFlow" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[i], se_input_flows[i]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation + flow_attention ([2,1]):", attention_map)
                elif "-no_segmask" in self.version:
                    for i in range(opt.seq_length):
                        attention_map = tf.ones_like(seg_19[i])[..., 0:1]
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] No use attention module.")
                elif "-segmask_" in self.version and "-static" in self.version:
                    for i in range(opt.seq_length):
                        attention_map, self.segweights = build_seg_channel_weight(pred_seglabels[i], prefix="pose_exp_net/")
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build weighted segmentation_attention (only attention_map_src0, attention_map_src1):", attention_map)
                else:
                    for i in range(opt.seq_length):
                        attention_map, self.segweights = build_seg_channel_weight(pred_seglabels[i], prefix="pose_exp_net/")
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build weighted segmentation_attention :", attention_map)

            # 4. PoseNN Part.
            # 4.1. Generate masked inputs.
            use_se_flow = len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/se_flow'))
            attention_map_tgt  = attention_maps[0]
            attention_map_src0 = attention_maps[1]
            attention_map_src1 = attention_maps[2]
            if use_se_flow:
                print (">>> [PoseNN] se_flow mode.")
                print (">>> [PoseNN] attention_map ----> 1")
                attention_map_tgt     = tf.ones_like(attention_map_tgt)
                attention_map_tgtsrc1 = tf.ones_like(attention_map_tgt)
            else:
                attention_map_tgtsrc1 = attention_map_tgt
            if pred_info is not None:
                print (">>> [PoseNN] concat pred_info now")
                if "-segmask_" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    self.input_images[0] = tf.multiply(self.input_images[0],  attention_map_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1],  attention_map_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2],  attention_map_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3],  attention_map_tgtsrc1)
                    if "-segmask_all" in self.version and ".555" in Version:
                        print (">>> [PoseNN] use segmask_all dilation network (src0, src1 concating flows = 0)")
                        pred_info[0] = tf.multiply(pred_info[0],  attention_map_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  tf.ones_like(attention_map_src0))
                        pred_info[2] = tf.multiply(pred_info[2],  tf.ones_like(attention_map_src1))
                        pred_info[3] = tf.multiply(pred_info[3],  attention_map_tgtsrc1)
                    elif "-segmask_all" in self.version:
                        print (">>> [PoseNN] use segmask_all dilation network")
                        pred_info[0] = tf.multiply(pred_info[0],  attention_map_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  attention_map_src0)
                        pred_info[2] = tf.multiply(pred_info[2],  attention_map_src1)
                        pred_info[3] = tf.multiply(pred_info[3],  attention_map_tgtsrc1)
                    elif "-segmask_rgb" in self.version:
                        print (">>> [PoseNN] use segmask_rgb dilation network")
                else:
                    print (">>> ======> Segmentation Attention [OFF]")
                self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
                self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
                self.input_images[3] = tf.concat([self.input_images[3],  pred_info[3]], axis=3)
            else:
                if "-segmask" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    print (">>> [PoseNN] use segmask_rgb dilation network")
                    self.input_images[0] = tf.multiply(self.input_images[0], attention_map_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1], attention_map_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2], attention_map_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3], attention_map_tgtsrc1)

            # 4.2. Predict ego-motion.
            poseNet = partial(poseNet, dropout=self.dropout, is_training=True, se_attention=se_attention, batch_norm="-batch_norm" in self.version, cnv6_num_outputs=cnv6_num_outputs) # -2
            if "-sharedNN" in self.version:
                print (">>> ===== [ SharedNN PoseNN (%s) ] =====" % Version)
                pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1]) #tgt->src0
                pred_pose1, _ = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            else:
                pred_poses, _ = poseNet(self.input_images[0], self.input_images[1], self.input_images[2])
            print (">>> [PoseNN] pred_poses: ", pred_poses)

            # Visualization in tf.summary.
            self.attention_maps = [attention_map_tgt, attention_map_src0, attention_map_src1]
            self.masked_images  = [
                    self.input_images[0][..., :3],   # tgt_masked_images
                    self.input_images[1][..., :3],   # src0_masked_images
                    self.input_images[2][..., :3]    # src1_masked_images
                    ]
            if pred_info is not None:
                if pred_info[0].shape.as_list()[-1] > 1:
                    self.masked_flows  = [
                        self.convert_to_tf_image( flow_to_image( self.input_images[0][...,3:5])),  # tgt_masked_flows
                        self.convert_to_tf_image( flow_to_image( self.input_images[1][...,3:5])),  # src0_masked_flows
                        self.convert_to_tf_image( flow_to_image( self.input_images[2][...,3:5]))   # src1_masked_flows
                        ]
            self.masked_seglabels  = [
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[0], tf.float32), self.attention_maps[0])),   # tgt_masked_seglabels
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[1], tf.float32), self.attention_maps[1])),   # src0_masked_seglabels
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[2], tf.float32), self.attention_maps[2]))    # src1_masked_seglabels
                ]

        with tf.name_scope("compute_loss"):
            trans_training_loss = 0
            rot_training_loss = 0
            pose_training_loss = 0
            trans_loss = 0
            rot_loss = 0
            pose_loss = 0
            self.relative_pose_gt = []
            self.relative_trans_scale = []
            self.pred_trans_scale = []
            self.scaleErr = []

            for i in range(opt.num_source):
                # Inverse warp the source image to the target image frame
                relative_pose = get_relative_pose(poses[:,0,:], poses[:,i+1,:])
                relative_rot = tf.slice(relative_pose, [0, 0, 0], [-1, 3, 3])
                relative_rot_vec = mat2euler(relative_rot)
                relative_trans_vec = tf.slice(relative_pose, [0, 0, 3], [-1, 3, 1])
                relative_pose_vec = tf.squeeze(tf.concat([relative_rot_vec, relative_trans_vec], axis=1), axis=-1)
                self.relative_pose_gt.append(relative_pose_vec)

                # Relative pose error
                prior_trans_vec = relative_pose_vec[:,3:]
                prior_rot_vec = relative_pose_vec[:,:3]
                pred_trans_vec  = pred_poses[:, i, 3:]
                pred_rot_vec  = pred_poses[:, i, :3]

                prior_trans_vec_scale = tf.norm(prior_trans_vec, axis=1)
                pred_trans_vec_scale  = tf.norm(pred_trans_vec, axis=1)
                positionErr = tf.norm(prior_trans_vec - pred_trans_vec, axis=1)
                scaleErr    = prior_trans_vec_scale - pred_trans_vec_scale

                # Training Loss.
                trans_training_loss += tf.reduce_mean(self.compute_trans_loss(prior_trans_vec, pred_trans_vec))
                trans_training_loss += tf.reduce_mean(
                        (scaleErr) ** 2
                        )
                self.relative_trans_scale.append(prior_trans_vec_scale)
                self.pred_trans_scale.append(pred_trans_vec_scale)
                self.scaleErr.append(scaleErr)

                rot_training_loss += tf.reduce_mean(self.compute_rot_loss( prior_rot_vec, pred_rot_vec ))

                # Training Loss for tf.summary visualization.
                trans_loss += tf.reduce_mean(self.compute_trans_loss(prior_trans_vec, pred_trans_vec))
                trans_loss += tf.reduce_mean(
                        (scaleErr) ** 2
                        )
                rot_loss += tf.reduce_mean(self.compute_rot_loss( prior_rot_vec, pred_rot_vec ))

            # Training Loss.
            trans_vec_loss = opt.pose_weight * trans_training_loss * coef_trans
            rot_vec_loss = opt.pose_weight * rot_training_loss * coef_rot
            pose_training_loss += trans_vec_loss + rot_vec_loss
            # Summary visualization loss.
            pose_loss += 0.1 * trans_loss * coef_trans + \
                         0.1 * rot_loss * coef_rot

            print (" >> positionErr:", positionErr)
            print (" >> scaleErr:", scaleErr)
            print (" >> relative_trans_scale:", self.relative_trans_scale)
            print (" >> pred_trans_scale:", self.pred_trans_scale)

            self.scaleErr = tf.stack(self.scaleErr, axis=1)
            self.relative_trans_scale = tf.stack(self.relative_trans_scale, axis=1)
            self.pred_trans_scale = tf.stack(self.pred_trans_scale, axis=1)


        with tf.name_scope("train_op"):
            all_train_vars = [var for var in tf.trainable_variables()]
            pose_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net')
            self.pose_net_vars = pose_net_vars
            trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/pose/translation')
            rot_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/pose/rotation')
            train_vars = [var for var in all_train_vars if var not in trans_vars and var not in rot_vars]
            pose_without_rot_vars = [var for var in pose_net_vars if var not in rot_vars]
            print ("All train vars:")
            for i,v in enumerate(all_train_vars):
                print ("   ", i, v)
            print ("Model vars:")
            for i,v in enumerate(tf.model_variables()):
                print ("   ", i, v)
            print ("PoseNN vars:")
            for i,v in enumerate(pose_net_vars):
                print ("   ", i, v)
            print ("TransNN vars:")
            for i,v in enumerate(trans_vars):
                print ("   ", i, v)
            print ("RotNN vars:")
            for i,v in enumerate(rot_vars):
                print ("   ", i, v)
            print ("poseNN w/o rot vars:")
            for i,v in enumerate(pose_without_rot_vars):
                print ("   ", i, v)

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

            _ENABLE_SUMMARIES_GRADIENTS = False

            if "-decay0.95" in self.version:
                decay_rate = 0.95
            else:
                decay_rate = 0.9

            if "-decay100k" in self.version:
                thre = 100000
            elif "-decay50k" in self.version:
                thre = 50000
            else:
                thre = self.global_step-1

            if "-decay" in self.version and "-staircase" in self.version:
                learning_rate = tf.train.exponential_decay(opt.learning_rate, self.global_step, thre, decay_rate, staircase=True)
                print (">>>> use decay weight (staircase)")
            elif "-decay" in self.version:
                learning_rate = tf.train.exponential_decay(opt.learning_rate, self.global_step, thre, decay_rate, staircase=False)
                print (">>>> use decay weight")
            else:
                learning_rate = opt.learning_rate

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                adam_opt = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                self.pose_net_op = slim.learning.create_train_op(trans_vec_loss + rot_vec_loss, adam_opt,
                        variables_to_train=pose_net_vars,
                        colocate_gradients_with_ops=True,
                        summarize_gradients=_ENABLE_SUMMARIES_GRADIENTS,
                        global_step=None)
                self.train_all_op = self.pose_net_op

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.pose_loss   = tf.convert_to_tensor(pose_loss)
        self.trans_loss  = tf.convert_to_tensor(trans_loss)
        self.rot_loss    = tf.convert_to_tensor(rot_loss)
        self.pose_training_loss   = tf.convert_to_tensor(pose_training_loss)
        self.trans_training_loss  = tf.convert_to_tensor(trans_training_loss)
        self.rot_training_loss    = tf.convert_to_tensor(rot_training_loss)
        self.tgt_image = tgt_image
        self.src_image_stack = src_image_stack
        return loader, batch_sample


    def compute_pose_loss(self, prior_pose_vec, pred_pose_vec):
        rot_vec_err = self.compute_rot_loss(prior_pose_vec[:,:3], pred_pose_vec[:,:3])
        trans_err = self.compute_trans_loss(prior_pose_vec[:,3:], pred_pose_vec[:,3:])
        return rot_vec_err + trans_err

    def compute_trans_loss(self, prior_trans_vec, pred_trans_vec):
        trans_err = tf.norm(tf.nn.l2_normalize(
            prior_trans_vec, axis=1) - tf.nn.l2_normalize(pred_trans_vec, axis=1), axis=1)
        return trans_err

    def compute_rot_loss(self, prior_rot_vec, pred_rot_vec):
        rot_vec_err = tf.norm(prior_rot_vec - pred_rot_vec, axis=1)
        return rot_vec_err


    # reference: https://github.com/yzcjtr/GeoNet/blob/master/geonet_model.py
    # and https://arxiv.org/abs/1712.00175
    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keepdims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean


    def normalize_for_show(self, disp, thresh=90):
        dims = disp.shape.as_list() # shape=[B,i,j,1]
        disp_max = tf.contrib.distributions.percentile(disp, q=thresh, axis=[1,2])  # shape=[B,1] ; obtain min as q=0, max as q=100, median as q=50
        disp_max = tf.expand_dims(tf.expand_dims(disp_max, 1), 1)
        disp_max = tf.tile(disp_max, [1, dims[1], dims[2], 1])
        zeros = tf.zeros_like(disp_max)
        clip_disp = tf.clip_by_value(disp, zeros, disp_max)
        return clip_disp


    def collect_summaries(self):
        opt = self.opt
        for var in self.pose_net_vars:
            tf.summary.histogram(var.op.name, var)
        tf.summary.scalar("pose_loss", self.pose_loss)
        tf.summary.scalar("trans_loss", self.trans_loss)
        tf.summary.scalar("rot_loss", self.rot_loss)
        tf.summary.scalar("pose_training_loss", self.pose_training_loss)
        tf.summary.scalar("trans_training_loss", self.trans_training_loss)
        tf.summary.scalar("rot_training_loss", self.rot_training_loss)

        # ---[ export RGB image
        tf.summary.image('image_target', self.deprocess_image(self.tgt_image))   # target prediction
        for i in range(opt.num_source):
            tf.summary.image(
                'image_source%d' % (i),
                self.deprocess_image(self.src_image_stack[:, :, :, i*3:(i+1)*3]))

        # ---[ export input_flow
        if getattr(self, "pred_flow_color", None) is not None:
            for i,flow in enumerate(self.pred_flow_color):
                print (">>>> Summary flow", flow)
                tf.summary.image('flow_src%d' % (i), flow)

        # ---[ export pred_seglabels_color
        if getattr(self, "pred_seglabels_color", None) is not None:
            tf.summary.image("pred_seglabels_color_tgt", self.pred_seglabels_color[0])
            for i,_color in enumerate(self.pred_seglabels_color[1:]):
                tf.summary.image("pred_seglabels_color_src%d" % (i), _color)

        # ---[ export segweights (-segmask_*)
        if getattr(self, "segweights", None) is not None:
            for i in range(self.segweights.shape.as_list()[0]):
                tf.summary.scalar("segweights/%02d" % (i), self.segweights[i])

        # ---[ export segmask (-segmask_*)
        if getattr(self, "attention_maps", None) is not None:
            tf.summary.image('attention_map_tgt', self.attention_maps[0])
            for i,_color in enumerate(self.attention_maps[1:]):
                tf.summary.image('attention_map_src%d' % (i), _color)

        # ---[ export *_mask (-segmask_*)
        if getattr(self, "masked_images", None) is not None:
            for i,_color in enumerate(self.masked_images):
                if i == 0:
                    tf.summary.image("masked_images_tgt", _color)
                else:
                    tf.summary.image("masked_images_src%d" % (i-1), _color)
        if getattr(self, "masked_flows", None) is not None:
            for i,_color in enumerate(self.masked_flows):
                if i == 0:
                    tf.summary.image("masked_flows_tgt", _color)
                else:
                    tf.summary.image("masked_flows_src%d" % (i-1), _color)
        if getattr(self, "masked_deltadepths", None) is not None:
            for i,_color in enumerate(self.masked_deltadepths):
                if i == 0:
                    tf.summary.image("masked_deltadepths_tgt", _color)
                else:
                    tf.summary.image("masked_deltadepths_src%d" % (i-1), _color)
        if getattr(self, "masked_depths", None) is not None:
            for i,_color in enumerate(self.masked_depths):
                if i == 0:
                    tf.summary.image("masked_depths_tgt", _color)
                else:
                    tf.summary.image("masked_depths_src%d" % (i-1), _color)
        if getattr(self, "masked_seglabels", None) is not None:
            for i,_color in enumerate(self.masked_seglabels):
                if i == 0:
                    tf.summary.image("masked_seglabels_tgt", _color)
                else:
                    tf.summary.image("masked_seglabels_src%d" % (i-1), _color)

        # ---[ export se_input_flow_color
        if getattr(self, "se_input_flows_color", None) is not None:
            for i,flow in enumerate(self.se_input_flows_color):
                print (">>>> Summary se_input_flow", flow)
                if i == 0:
                    tf.summary.image('se_input_flow_tgt', flow)
                else:
                    tf.summary.image('se_input_flow_src%d' % (i-1), flow)


    def train(self, opt):
        self.opt = opt
        data_loader, batch_sample = self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver(tf.trainable_variables() + \
                                    [self.global_step],
                                     max_to_keep=None)

        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            data_loader.init_data_pipeline(sess, batch_sample)
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))

            if opt.continue_train:
                if opt.init_ckpt_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_ckpt_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()

            fetches = {
                       "global_step": self.global_step,
                       "incr_global_step": self.incr_global_step
                       }

            fetches['train_op'] = self.train_all_op
            fetches["summary"] = sv.summary_op
            print ("\nsummary")
            for su in tf.get_collection(tf.GraphKeys.SUMMARIES,""):
                print ("  > ", su)

            fetches["pose_loss"] = self.pose_loss
            fetches["trans_loss"] = self.trans_loss
            fetches["rot_loss"] = self.rot_loss
            fetches["pred_poses"] = self.pred_poses
            fetches["relative_pose_gt"] = self.relative_pose_gt
            if getattr(self, "scaleErr", []) != []:
                fetches["GroundTruth_trans_scale"] = self.relative_trans_scale    # shape=[b,num_source]
                fetches["Pred_trans_scale"] = self.pred_trans_scale               # shape=[b,num_source]
                fetches["scaleErr"] = self.scaleErr                               # shape=[b,num_source]
                print (">>> print scaleErr : ", fetches["scaleErr"])

            step = 0
            while step < opt.max_steps:
                results = sess.run(fetches)
                gs = results["global_step"]
                step = gs - 1

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f"
                          % (train_epoch, train_step, self.steps_per_epoch,
                             (time.time() - start_time)/opt.summary_freq))
                    print("pose/trans/rot loss: [%.3f/%.3f/%.3f]\n" % (
                        results["pose_loss"], results["trans_loss"], results["rot_loss"]))
                    start_time = time.time()

                # save model
                if step != 0 and step % opt.save_freq == 0:
                    self.save(sess, opt.checkpoint_dir, gs-1)


    def select_tensor_or_placeholder_input(self, input_uint8):
        if input_uint8 == None:
            input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                        self.img_height, self.img_width, 3], name='raw_input')
            self.inputs = input_uint8
        else:
            self.inputs = None
        input_mc = self.preprocess_image(input_uint8)
        return input_mc


    def build_depth_test_graph(self, input_uint8):
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        with tf.name_scope("depth_prediction"):
            pred_disp, depth_net_endpoints = disp_net_res50(input_mc, is_training=False)
            pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph_davo(self, input_uint8, input_flow, input_depth, input_seglabel):
        """
        input_flow.shape = [B, 2, h, w, 2]
        """
        assert self.version != None
        is_read_depth = True if "depth" in self.version or "disp" in self.version else False
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        image_stack = tf.concat([tgt_image, src_image_stack], axis=3)
        self.tgt_images  = [
                tgt_image,
                src_image_stack[...,:3],
                src_image_stack[...,3:]
                ]
        self.input_images  = [
                tgt_image,
                src_image_stack[...,:3],
                src_image_stack[...,3:]
                ]

        pred_flows = [
                        tf.zeros_like(input_flow[:,0,...]), # tgt
                        input_flow[:,0,...],                # tgt->src0
                        input_flow[:,1,...]                 # tgt->src1
                       ]    # shape=[B,h,w,2]
        pred_flows_inv = [
                        tf.zeros_like(input_flow[:,2,...]), # tgt
                        input_flow[:,2,...],                # tgt->src0
                        input_flow[:,3,...]                 # tgt->src1
                       ]    # shape=[B,h,w,2]
        pred_flow_color = [ flow_to_image(flow) for flow in pred_flows ]    # (range: 0~1). shape=[B, h, w, 3]
        self.pred_flow_color = [ self.convert_to_tf_image(flow_color) for flow_color in pred_flow_color[1:] ]

        if is_read_depth:
            pred_depths = [
                            input_depth[:,1,...],                # tgt
                            input_depth[:,0,...],                # src0
                            input_depth[:,2,...]                 # src1
                           ]    # shape=[B,h,w,1]

        if input_seglabel is not None:
            pred_seglabels = input_seglabel
            pred_seglabels = [
                            pred_seglabels[:,1,...],                # tgt
                            pred_seglabels[:,0,...],                # src0
                            pred_seglabels[:,2,...]                 # src1
                           ]    # shape=[B,h,w,1]
            self.pred_seglabels_color = [label_to_color_image(seg) for seg in pred_seglabels]

        # ===[ PoseNN
        with tf.name_scope("pose_and_explainability_prediction"):
            # 0. Choose PoseNN se mode. (optional)
            if "-se_insert" in self.version:
                se_attention = True
            elif "-se_skipadd" in self.version:
                se_attention = "se_skipadd"
            elif "-se_replace" in self.version:
                se_attention = "se_replace"
            else:
                se_attention = False

            self.input_images  = [
                    tgt_image,   # src0->tgt for -sharedNNforward
                    src_image_stack[...,:3],
                    src_image_stack[...,3:],
                    tgt_image    # tgt->src1 for -sharedNNforward
                    ]

            # 1. Choose PoseNN type.
            if "-sharedNN" in self.version:
                if "-dilatedPoseNN" in self.version:
                    print (">>> choose dilated Shared CNN")
                    poseNet = decouple_sharednet_v0_dilation
                elif "-dilatedCouplePoseNN" in self.version:
                    print (">>> choose dilated Shared CouplePose CNN")
                    poseNet = couple_sharednet_v0_dilation
                elif "-couplePoseNN" in self.version:
                    raise NameError("not support `-sharedNN-couplePoseNN' mode.")
                else:
                    raise NameError("unknown PoseNN type.")
            elif "-dilatedPoseNN" in self.version:
                print (">>> choose dilated CNN")
                poseNet = decouple_net_v0_dilation
            elif "-dilatedCouplePoseNN" in self.version:
                print (">>> choose dilated CouplePose CNN")
                poseNet = couple_net_v0_dilation
            elif "-couplePoseNN" in self.version:
                print (">>> choose CouplePose CNN")
                poseNet = couple_net_v0
            else:
                print (">>> choose vanilla CNN")
                poseNet = decouple_net_v0

            # 1.1. adjust conv6 channels in PoseNN.
            cnv6_num_match = re.search("-cnv6_([0-9]+)", self.version)
            cnv6_num_outputs = 128 if cnv6_num_match is None else int(cnv6_num_match.group(1))
            print (">>> choose cnv6_num_outputs :", cnv6_num_outputs, cnv6_num_match)

            # 2. Concat additional information in inputs.
            Version = re.search("^(v[0-9.]+)", self.version)
            Version = "v0" if Version is None else Version.group(1)
            print (">>> Version: ", Version, re.search("^(v[0-9.]+)", self.version))
            pred_info = None
            if "v0" in Version:
                print (">>> [PoseNN] Only input RGB")
            elif "v1" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (src->tgt)")
                pred_info = pred_flows + [pred_flows[0]]    # src->tgt
            if "-seglabelid" in self.version:
                print (">>> [PoseNN] ...... + seglabelid (1 channel, id=0 ~ 18)")
                if pred_info is not None:
                    pred_info = [
                            tf.concat([info,seg], axis=3) for info,seg in zip(pred_info,pred_seglabels)
                            ]
                else:
                    pred_info = pred_seglabels

            # 3. Attention Part. (AttentionNN)
            # 3.1. Choose activation function in se_block layers.
            if "-fc_tanh" in self.version:
                activation_fn = tf.nn.tanh
                print (">>> [SE][activation_fn] tanh.")
            elif "-fc_lrelu" in self.version:
                activation_fn = tf.nn.leaky_relu
                print (">>> [SE][activation_fn] leaky_relu.")
            else:
                activation_fn = tf.nn.relu
                print (">>> [SE][activation_fn] relu.")

            # 3.2. Prepare normalized optical flows for se_block's inputs.
            se_input_flows = [f for f in pred_flows] + [pred_flows[0]]
            if "-norm_flow" in self.version:
                se_input_flows = [(f - 0.32140523) / 15.384229 for f in se_input_flows]
                print (">>> [SE][input] Normailed : (flow - flow.mean()) / flow.std()")

            # 3.3. Prepare positive value of optical flows for se_block's inputs. (h: horizontal , v: vertical)
            if "-abs_flow_h" in self.version:
                se_input_flows = [ tf.stack([tf.abs(f[...,0]), f[...,1]], axis=-1) for f in se_input_flows]
                print (">>> [SE][input] absolute value of horizontal se_input_flows : | flow[0] | ")
            elif "-abs_flow_v" in self.version:
                se_input_flows = [ tf.stack([f[...,0], tf.abs(f[...,1])], axis=-1) for f in se_input_flows]
                print (">>> [SE][input] absolute value of vertical se_input_flows : | flow[1] | ")
            elif "-abs_flow" in self.version:
                se_input_flows = [tf.abs(f) for f in se_input_flows]
                print (">>> [SE][input] absolute value of se_input_flows : | flow[:2] | ")

            self.se_input_flows_color = [ flow_to_image(f) for f in se_input_flows ]    # (range: 0~1). shape=[B, h, w, 3]
            self.se_input_flows_color = [ self.convert_to_tf_image(f) for f in self.se_input_flows_color ]

            # 3.4. Prepare normalized depth for se_block's inputs.
            if is_read_depth:
                se_input_depths = [d for d in pred_depths] + pred_depths[0]
                if "-norm_depth" in self.version:
                    se_input_depths = [d / 80. for d in se_input_depths]

            # 3.5. Choose se_block's inputs, and generate attentions for PoseNN inputs.
            with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                seg_19  = [ tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels ]
                attention_maps = []
                if "-se_flow_on_depthseg_sharedlayers" in self.version:
                    depth_thres_init = re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version)
                    depth_thres_init = 15. if depth_thres is None else float(depth_thres.group(1))
                    depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=depth_thres_init, stddev=0.1), trainable=True)
                    self.depth_thres = depth_thres
                    print (">>> [PoseNN][se_flow_on_depthseg_sharedlayers] depth threshold = ", depth_thres_init)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    seg_38 = [ tf.concat([seg_near,seg_far], axis=-1) for seg_near,seg_far in zip(depth_seg_19['near'],depth_seg_19['far']) ] # shape=[B,H,W,38]
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_38[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow multiply to near/far segs (i.e., seg_38) :", attention_map)
                elif "-se_flow_on_depthseg_seplayers" in self.version:
                    depth_thres = re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version)
                    depth_thres_init = 15. if depth_thres is None else float(depth_thres.group(1))
                    depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(mean=depth_thres_init, stddev=0.1), trainable=True)
                    self.depth_thres = depth_thres
                    print (">>> [PoseNN][se_flow_on_depthseg_seplayers] depth threshold = ", depth_thres_init)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    for i in range(self.seq_length):
                        attention_map = 0
                        for name in ['near', 'far']:
                            attention_weights = se(se_input_flows[i], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                            attention_map += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][i] * attention_weights, axis=-1), -1)
                        attention_maps.append(attention_map)
                    print (">>> [PoseNN] build two SE_flow multiply to near/far segs (i.e., seg_38) seperatly:", attention_map)
                elif "-se_flow_on_depthseg" in self.version:
                    raise NameError("please select `-se_flow_on_depthseg_seplayers' or `-se_flow_on_depthseg_sharedlayers'.")
                elif "-se_mixDepthFlow" in self.version:
                    for i in range(self.seq_length):
                        attention_map = \
                                tf.expand_dims(tf.reduce_sum(
                                                se_block(tf.concat([se_input_depths[i], se_input_flows[i]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), 
                                                axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_depth + flow_attention:", attention_map)
                elif "-se_mixDispFlow" in self.version:
                    se_input_depths = [1./d for d in pred_depths]
                    for i in range(self.seq_length):
                        attention_map = \
                                tf.expand_dims(tf.reduce_sum(
                                                se_block(tf.concat([se_input_depths[i], se_input_flows[i]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), 
                                                axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_disp + flow_attention:", attention_map)
                elif "-se_flow" in self.version:
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow :", attention_map)
                elif "-se_gp2x2_flow_nobottle" in self.version:
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:19, fc2:19):", attention_map)
                elif "-se_gp2x2_flow" in self.version:
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:8, fc2:19):", attention_map)
                elif "-se_spp21_flow" in self.version:
                    for i in range(self.seq_length):
                        attention_weights  = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow ([2,1]) (fc1:8, fc2:19):", attention_map)
                elif "-se_spp2_flow" in self.version:   # same to gp2x2
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow ([2]) (fc1:8, fc2:19):", attention_map)
                elif "-se_spp_flow" in self.version or "-se_spp864_flow" in self.version:
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_flows[i], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_flow (fc1:8, fc2:19):", attention_map)
                elif "-se_depth_wo_tgt_to_seg" in self.version:
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_depths[i], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_depth_to_seg" in self.version:
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights  = se(se_input_depths[i], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation:", attention_map)
                elif "-se_depth_wo_tgt" in self.version:
                    for i in range(self.seq_length):
                        attention_map = \
                            tf.expand_dims(tf.reduce_sum(
                                            se_block(se_input_depths[i], "se_depth", ratio=1, activation=activation_fn), 
                                            axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_depth_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_depth" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[i], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_depth_attention :", attention_map)
                elif "-se_disp_wo_tgt_to_seg" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_disps[i], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_disp_to_seg" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_disps[i], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation:", attention_map)
                elif "-se_disp_wo_tgt" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_disps[i], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_disp_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_disp" in self.version:
                    se_input_disps = [1./d for d in pred_depths]
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(se_input_disps[i], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_disp_attention :", attention_map)
                elif "-se_rgb_wo_tgt_to_seg" in self.version:
                    se_input_rgbs = self.input_images
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_rgbs[i], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_rgb_to_seg" in self.version:
                    se_input_rgbs = self.input_images
                    self.att_19s = []
                    for i in range(self.seq_length):
                        attention_weights = se(se_input_rgbs[i], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                        self.att_19s.append( attention_weights )
                    print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation:", attention_map)
                elif "-se_rgb_wo_tgt" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[i], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_rgb_frames_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_rgb" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[i], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_rgb_frames_attention :", attention_map)
                elif "-se_seg_wo_tgt" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build SE_segmentation_attention (only attention_map_src0, attention_map_src1):", attention_map)
                elif "-se_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation_attention :", attention_map)
                elif "-se_gp2x2_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(seg_19[i], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation_attention with 2x2 global pooling:", attention_map)
                elif "-se_spp21_seg" in self.version or "-se_spp_seg_21" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2,1]):", attention_map)
                elif "-se_spp2_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2]):", attention_map)
                elif "-se_spp_seg" in self.version or "-se_spp864_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[i], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation_attention :", attention_map)
                elif "-se_SegFlow_to_seg_8" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        se_inputs  = [
                                tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                                tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                                tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                                ]
                        attention_weights = se(se_inputs[i], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:8, fc2:19):", attention_map)
                elif "-se_SegFlow_to_seg" in self.version:
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    for i in range(self.seq_length):
                        se_inputs  = [
                                tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                                tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                                tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                                ]
                        attention_weights = se(se_inputs[i], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                        attention_map = tf.expand_dims(tf.reduce_sum(seg_19[i] * attention_weights , axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:19, fc2:19):", attention_map)
                elif "-se_mixSegFlow" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[i], se_input_flows[i]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_segmentation + flow_attention:", attention_map)
                elif "-se_spp21_mixSegFlow" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[i], se_input_flows[i]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build SE_SPP_segmentation + flow_attention ([2,1]):", attention_map)
                elif "-no_segmask" in self.version:
                    for i in range(self.seq_length):
                        attention_map = tf.ones_like(seg_19[i])[..., 0:1]
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] No use attention module.")
                elif "-segmask_" in self.version and "-static" in self.version:
                    for i in range(self.seq_length):
                        attention_map, self.segweights = build_seg_channel_weight(pred_seglabels[i], prefix="pose_exp_net/")
                        attention_maps.append( attention_map )
                    attention_maps[0] = tf.ones_like(attention_maps[0])
                    print (">>> [PoseNN] build weighted segmentation_attention (only attention_map_src0, attention_map_src1):", attention_map)
                else:
                    for i in range(self.seq_length):
                        attention_map, self.segweights = build_seg_channel_weight(pred_seglabels[i], prefix="pose_exp_net/")
                        attention_maps.append( attention_map )
                    print (">>> [PoseNN] build weighted segmentation_attention :", attention_map)

            # 4. PoseNN Part.
            # 4.1. Generate masked inputs.
            use_se_flow = len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/se_flow'))
            attention_map_tgt  = attention_maps[0]
            attention_map_src0 = attention_maps[1]
            attention_map_src1 = attention_maps[2]
            if use_se_flow:
                print (">>> [PoseNN] se_flow mode.")
                print (">>> [PoseNN] attention_map ----> 1")
                attention_map_tgt     = tf.ones_like(attention_map_tgt)
                attention_map_tgtsrc1 = tf.ones_like(attention_map_tgt)
            else:
                attention_map_tgtsrc1 = attention_map_tgt
            if pred_info is not None:
                print (">>> [PoseNN] concat pred_info now")
                if "-segmask_" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    self.input_images[0] = tf.multiply(self.input_images[0],  attention_map_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1],  attention_map_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2],  attention_map_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3],  attention_map_tgtsrc1)
                    if "-segmask_all" in self.version and ".555" in Version:
                        print (">>> [PoseNN] use segmask_all dilation network (src0, src1 concating flows = 0)")
                        pred_info[0] = tf.multiply(pred_info[0],  attention_map_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  tf.ones_like(attention_map_src0))
                        pred_info[2] = tf.multiply(pred_info[2],  tf.ones_like(attention_map_src1))
                        pred_info[3] = tf.multiply(pred_info[3],  attention_map_tgtsrc1)
                    elif "-segmask_all" in self.version:
                        print (">>> [PoseNN] use segmask_all dilation network")
                        pred_info[0] = tf.multiply(pred_info[0],  attention_map_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  attention_map_src0)
                        pred_info[2] = tf.multiply(pred_info[2],  attention_map_src1)
                        pred_info[3] = tf.multiply(pred_info[3],  attention_map_tgtsrc1)
                    elif "-segmask_rgb" in self.version:
                        print (">>> [PoseNN] use segmask_rgb dilation network")
                else:
                    print (">>> ======> Segmentation Attention [OFF]")
                self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
                self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
                self.input_images[3] = tf.concat([self.input_images[3],  pred_info[3]], axis=3)
            else:
                if "-segmask" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    print (">>> [PoseNN] use segmask_rgb dilation network")
                    self.input_images[0] = tf.multiply(self.input_images[0], attention_map_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1], attention_map_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2], attention_map_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3], attention_map_tgtsrc1)

            # 4.2. Predict ego-motion.
            poseNet = partial(poseNet, dropout=False, is_training=False, se_attention=se_attention, batch_norm="-batch_norm" in self.version, cnv6_num_outputs=cnv6_num_outputs) # -2
            if "-sharedNN" in self.version:
                print (">>> ===== [ SharedNN PoseNN (%s) ] =====" % Version)
                pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1]) #tgt->src0
                pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            else:
                pred_poses, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[0], self.input_images[1], self.input_images[2])
            print (">>> [PoseNN] pred_poses: ", pred_poses)

            self.rot_cnv6 = tf.image.resize_bilinear(rot_cnv6, self.input_images[0].shape[1:3])
            self.trans_cnv6 = tf.image.resize_bilinear(trans_cnv6, self.input_images[0].shape[1:3])
            self.features = {'rot': self.rot_cnv6, 'trans': self.trans_cnv6}

            # Visualization in tf.summary.
            self.attention_maps = [attention_map_tgt, attention_map_src0, attention_map_src1]
            self.seg_19 = seg_19
            self.masked_images  = [
                    self.input_images[0][..., :3],   # tgt_masked_images
                    self.input_images[1][..., :3],   # src0_masked_images
                    self.input_images[2][..., :3]    # src1_masked_images
                    ]
            if pred_info is not None:
                if pred_info[0].shape.as_list()[-1] > 1:
                    self.masked_flows  = [
                        self.convert_to_tf_image( flow_to_image( self.input_images[0][...,3:5])),  # tgt_masked_flows
                        self.convert_to_tf_image( flow_to_image( self.input_images[1][...,3:5])),  # src0_masked_flows
                        self.convert_to_tf_image( flow_to_image( self.input_images[2][...,3:5]))   # src1_masked_flows
                        ]
            self.masked_seglabels  = [
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[0], tf.float32), self.attention_maps[0])),   # tgt_masked_seglabels
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[1], tf.float32), self.attention_maps[1])),   # src0_masked_seglabels
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[2], tf.float32), self.attention_maps[2]))    # src1_masked_seglabels
                ]

        self.pred_poses = pred_poses
        self.masks = {
                'image': self.masked_images,
                'flow': self.masked_flows,
                'seglabel': self.masked_seglabels,
                'attention': self.attention_maps,
                'att_19': attention_weights
                }





    def build_pose_test_graph(self, input_uint8):
        assert self.version != None
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)

        # Ver 1.0.0
        if "v0" in self.version:
            print (">>> choose v0")
            with tf.name_scope("pose_prediction"):
                pred_poses, _ = pose_net(tgt_image, src_image_stack, is_training=False)
                self.pred_poses = pred_poses

        else:
            raise NameError("version `%s' is not supported." % self.version)


    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.


    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def convert_to_tf_image(self, inputs):
        return tf.image.convert_image_dtype(inputs, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1,
                        input_img_uint8=None,
                        input_pose=None,
                        input_flow=None,
                        input_depth=None,
                        input_seglabel=None):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'davo':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph_davo(input_img_uint8, input_flow, input_depth, input_seglabel)

    def inference(self, sess, mode, inputs=None):
        fetches = {}
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        elif mode == 'feature':
            fetches['pose'] = self.pred_poses
            fetches['masks'] = self.masks
            fetches['features'] = self.features
            fetches['images'] = self.tgt_images
            fetches['flows'] = self.pred_flow_color
            fetches['segs'] = self.pred_seglabels_color
            fetches['seg_19'] = self.seg_19
        if inputs is None:
            results = sess.run(fetches)
        else:
            results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results


    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint step %d to %s..." % (step, checkpoint_dir))
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

