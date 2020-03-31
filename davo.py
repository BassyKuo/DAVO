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
        #self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        if "-noD" not in self.version:
            self.version = self.version + "-noD"
        is_read_pose     = True
        is_read_flow     = True
        is_read_depth    = True
        is_read_seglabel = True
        self.is_read_pose = is_read_pose
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
            #[bs, 128, 416, 3]
            tgt_image.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3])
            # [bs, 128, 416, 6]
            src_image_stack.set_shape([opt.batch_size, opt.img_height, opt.img_width, 3*opt.num_source])
            # [bs, 4, 3, 3]
            intrinsics.set_shape([opt.batch_size, num_scales, 3, 3])
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


#        if "-pretrained_monodepth2_depth" in self.version:
#            # i.e., read_depth = disparity, and transfer it into `depth'.
#            # use `delta depth' in the PoseNN.
#            print (">>>> [Delta Depth] use pretrained *-monodepth2_disp.npy")
#            pred_disp = [pred_depths[0]]        # tgt disp
#            pred_depths = [1. / d for d in pred_depths]       # pred_depths are disparity in "*-monodepth2_disp.npy"
#            pred_depth = [pred_depths[0]]       # tgt depth
#            pred_depth_map = pred_depth[0]
#
#        elif "-pretrained_monodepth2_disp" in self.version:
#            # i.e., read_depth = disparity, and use `delta disp' in the PoseNN.
#            print (">>>> [Delta Disp] use pretrained *-monodepth2_disp.npy")
#            pred_disp = [pred_depths[0]]        # tgt disp
#            pred_depth = [1. / pred_depths[0]]        # pred_depths are disparity in "*-monodepth2_disp.npy"
#            pred_depth_map = pred_depth[0]
#
#        if "-monodepth2NN" in self.version:
#            pose_scale = re.search("-pose_scale([0-9.]+)", self.version)
#            pose_scale = 0.01 if pose_scale is None else float(pose_scale.group(1))
#            print (">>> [MonoDepth2] pose_scale: ", pose_scale)
#            pred_depths = []
#            pred_disp_maps = []
#            net_builder = MonoNet(True, opt, pose_scale)
#            with tf.variable_scope('depth_net') as sc:
#                # tgt depth
#                res18_tc, skips_tc = net_builder.build_resnet18(tgt_image)
#                pred_disp = net_builder.build_disp_net(res18_tc,skips_tc)
#                pred_disp_rawscale = [tf.image.resize_bilinear(pred_disp[i], [opt.img_height, opt.img_width]) for i in range(num_scales)]
#                if opt.with_pose:
#                    pred_depth_rawscale = [1. / (0.01+preddisp) for preddisp in pred_disp_rawscale]
#                    pred_depth =  [1. / (0.01+preddisp) for preddisp in pred_disp[:num_scales+1]]    # -with_pose-2
#                else:
#                    pred_depth_rawscale = disp_to_depth(pred_disp_rawscale, 0.1, 100)
#                    pred_depth =  [disp_to_depth(pred_disp[i], 0.1, 100) for i in range(num_scales)]    # -with_pose-2
#
#                #pred_depth = pred_depth_rawscale
#                #pred_disp = pred_disp_rawscale
#                # pred_depth =  [disp_to_depth(pred_disp[i], 0.1, 100) for i in range(num_scales)]    # -with_pose
#                print (">>> [DepthNN] res18_tc:", res18_tc)
#                print (">>> [DepthNN] pred_depth:", pred_depth)
#                print (">>> [DepthNN] pred_disp:", pred_disp)
#                pred_depths.append(pred_depth[0])
#                pred_disp_maps.append(pred_disp[0])
#
#                #tgt_image_pyramid = [tf.image.resize_nearest_neighbor(tgt_image, [np.int(opt.img_height // (2 ** s)), np.int(opt.img_width // (2 ** s))]) for s in range(num_scales)]
#
#                # src depths
#                for src_image in tf.split(src_image_stack, opt.num_source, axis=3):
#                    src_res18_tc, skips_tc = net_builder.build_resnet18(src_image)
#                    src_pred_disp = net_builder.build_disp_net(res18_tc,skips_tc)
#                    src_pred_disp_rawscale = [tf.image.resize_bilinear(src_pred_disp[i], [opt.img_height, opt.img_width]) for i in range(num_scales)]
#                    if opt.with_pose:
#                        src_pred_depth_rawscale = [1. / (0.01+preddisp) for preddisp in src_pred_disp_rawscale]
#                        src_pred_depth =  [1. / (0.01+preddisp) for preddisp in src_pred_disp[:num_scales+1]]    # -with_pose-2
#                    else:
#                        src_pred_depth_rawscale = disp_to_depth(src_pred_disp_rawscale, 0.1, 100)
#                        src_pred_depth =  [disp_to_depth(src_pred_disp[i], 0.1, 100) for i in range(num_scales)]    # -with_pose-2
#
#                    #src_pred_depth = src_pred_depth_rawscale
#                    #src_pred_disp = src_pred_disp_rawscale
#                    # src_pred_depth =  [disp_to_depth(src_pred_disp[i], 0.1, 100) for i in range(num_scales)]     #-with_pose
#                    print (">>> [DepthNN] src_res18_tc:", src_res18_tc)
#                    print (">>> [DepthNN] src_pred_depth:", src_pred_depth)
#                    print (">>> [DepthNN] src_pred_disp:", src_pred_disp)
#                    pred_depths.append(src_pred_depth[0])
#                    pred_disp_maps.append(src_pred_disp[0])
#
#            if "-normalPoseType" not in self.version:
#                if '-seperatePoseType' in self.version:
#                    res18_ctp, _ = net_builder.build_resnet18(
#                        #tf.concat([src_image_stack[:, :, :, :3], tgt_image], axis=3),
#                        tf.concat([tgt_image, src_image_stack[:, :, :, :3]], axis=3),       # -2
#                        prefix='pose_exp_net/'
#                    )
#                    res18_ctn, _ = net_builder.build_resnet18(
#                        tf.concat([tgt_image, src_image_stack[:, :, :, 3:]], axis=3),
#                        prefix='pose_exp_net/'
#                    )
#                elif '-seperate2dFlowPoseType' in self.version:
#                    pred_flow_color = [ flow_to_image(flow) for flow in pred_flows ]    # tgt->src0, tgt->src1  (range: 0~1). shape=[B, h, w, 3]
#                    self.pred_flow_color = [ self.convert_to_tf_image(flow_color) for flow_color in pred_flow_color[1:] ]
#                    res18_ctp, _ = net_builder.build_resnet18(
#                        tf.concat([src_image_stack[:, :, :, :3], pred_flows[1]], axis=3),
#                        prefix='pose_exp_net/'
#                    )
#                    res18_ctn, _ = net_builder.build_resnet18(
#                        tf.concat([src_image_stack[:, :, :, 3:], pred_flows[2]], axis=3),
#                        prefix='pose_exp_net/'
#                    )
#                elif '-sharedPoseType' in self.version:
#                    res18_tp, _ = net_builder.build_resnet18(src_image_stack[:, :, :, :3], prefix='depth_net/')
#                    res18_tn, _ = net_builder.build_resnet18(src_image_stack[:, :, :, 3:], prefix='depth_net/')
#                    print (">>> [PoseNN] res18_tp:", res18_tp)
#                    print (">>> [PoseNN] res18_tn:", res18_tn)
#                    res18_ctp = tf.concat([res18_tp, res18_tc], axis=3)
#                    res18_ctn = tf.concat([res18_tc, res18_tn], axis=3)
#                else:
#                    raise NotImplementedError
#
#                with tf.variable_scope('pose_exp_net') as sc:
#                    if "-dcnet" in self.version:
#                        pred_pose_ctp = net_builder.build_pose_net3_dc(res18_ctp)
#                        pred_pose_ctn = net_builder.build_pose_net3_dc(res18_ctn)
#                    else:
#                        pred_pose_ctp = net_builder.build_pose_net2(res18_ctp)
#                        pred_pose_ctn = net_builder.build_pose_net2(res18_ctn)
#
#                    pred_poses = tf.concat([pred_pose_ctp, pred_pose_ctn], axis=1)
#                    print (">>> [PoseNN] res18_ctp:", res18_ctp)
#                    print (">>> [PoseNN] res18_ctn:", res18_ctn)
#                    print (">>> [PoseNN] pred_poses: ", pred_poses)


        #elif "-noD" not in self.version:
        if "-noD" not in self.version:
            with tf.name_scope("depth_prediction"):
                pred_depths = []
                # pred_disp_maps = []
                # pred_disp_for_smooth_loss = []
                # pred_depth_for_warpping = []
                with tf.device('/device:GPU:0'):
                    # Predict Disparity Map of Traget Image.
                    pred_disp, _ = disp_net_res50(tgt_image, is_training=True)
                    pred_depth = [1. / d for d in pred_disp]
                    print (" >> pred depths : ", pred_depth)
                    pred_depths.append(pred_depth[0])      # use depth pred1 : shape=[B,h,w,1]
                    #pred_depth_map = pred_depth[-1]      # use depth pred4 : shape=[B,h/8,w/8,1]
                    # input_pred_depth = [1. / self.spatial_normalize(d) for d in pred_disp]
                    # pred_depths.append(input_pred_depth[0])
                    # pred_disp_maps.append(pred_disp[0])
                    # pred_depth_for_warpping.append( pred_depth )
                    # pred_disp_for_smooth_loss.append( pred_disp )

                    # Predict Disparity Map of Source Image.
                    for src_image in tf.split(src_image_stack, opt.num_source, axis=3):
                        pred_src_disp, _ = disp_net_res50(src_image, is_training=True)
                        pred_src_depth = [1. / d for d in pred_src_disp]
                        pred_depths.append(pred_src_depth[0])      # use depth pred1 : shape=[B,h,w,1]
                        # input_pred_src_depth = [1. / self.spatial_normalize(d) for d in pred_src_disp]
                        # pred_depth_maps.append(input_pred_src_depth[0])      # use depth pred1 : shape=[B,h,w,1]
                        # pred_disp_maps.append(pred_src_disp[0])
                        # pred_depth_for_warpping.append( pred_src_depth )
                        # pred_disp_for_smooth_loss.append( pred_src_disp )
                    # (2) delta depth 
                    # xxxxx
                    #
                # self.pred_depth_maps = pred_depth_maps
                # self.pred_disp_maps = pred_disp_maps
                self.pred_depth = pred_depth
                self.pred_depths = pred_depths

#        try:
#            # Warped depth by predicted flow
#            warped_depths_and_masks = [
#                        warp_depth_by_flow(pred_depths[1], pred_flows[1]),    # tgt->src0
#                        warp_depth_by_flow(pred_depths[2], pred_flows[2]),    # tgt->src1
#                        ]    # shape=[B,h,w,1]
#
#            # Delt Depth by predicted flow
#            def warp_depth(src_d, f, tgt_d):
#                warp_src_d, mask = warp_depth_by_flow(src_d,f)
#                d_depth = warp_src_d - tgt_d
#                return tf.multiply(d_depth, mask)
#            delta_depths = [
#                            tf.zeros_like(pred_depths[0]),      # tgt->tgt
#                            warp_depth(pred_depths[1], pred_flows[1], pred_depths[0]),    # tgt->src0
#                            warp_depth(pred_depths[2], pred_flows[2], pred_depths[0]),    # tgt->src1
#                           ]    # shape=[B,h,w,1]
#            if "-ClipDeltaDepth" in self.version:
#                print (">>> Clip deltaDepth by value")
#                zeros = tf.zeros_like(delta_depths[0])
#                maximum = tf.reduce_max(delta_depths[1])
#                minimum = tf.reduce_min(delta_depths[2])
#                delta_depths = [
#                                delta_depths[0],      # tgt->tgt
#                                tf.clip_by_value(delta_depths[1], zeros, maximum + zeros),     # tgt->src0 (+)
#                                tf.clip_by_value(delta_depths[2], minimum + zeros, zeros),     # tgt->src1 (-)
#                               ]    # shape=[B,h,w,1]
#
#        except UnboundLocalError:
#            # local variable 'pred_depths' referenced before assignment
#            print (">>> [error] local variable 'pred_depths' referenced before assignment.")

        with tf.name_scope("pose_and_explainability_prediction"):
            # 1. Choose PoseNN se mode. (optional)
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
                # add at 2020/01/16
                if "-dilatedPoseNN" in self.version:
                    print (">>> choose dilated Shared CNN")
                    poseNet = decouple_sharednet_v0_dilation
                elif "-dilatedCouplePoseNN" in self.version:
                    print (">>> choose dilated Shared CouplePose CNN")
                    poseNet = couple_sharednet_v0_dilation
                elif "-couplePoseNN" in self.version:
                    #print (">>> choose Shared CouplePose CNN")
                    raise NameError("not support `-sharedNN-couplePoseNN' mode.")
                else:
                    raise NameError("unknown PoseNN type.")
                # -----------------------------------
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

            cnv6_num_match = re.search("-cnv6_([0-9]+)", self.version)
            cnv6_num_outputs = 128 if cnv6_num_match is None else int(cnv6_num_match.group(1))
            print (">>> choose cnv6_num_outputs :", cnv6_num_outputs, cnv6_num_match)

            # flow color
            pred_flow_color = [ flow_to_image(flow) for flow in pred_flows ]    # (range: 0~1). shape=[B, h, w, 3]
            self.pred_flow_color = [ self.convert_to_tf_image(flow_color) for flow_color in pred_flow_color[1:] ]

            # delta depth color
            try:
                delta_depth_color = [ flow_to_image( tf.concat([tf.zeros_like(delta_depth), delta_depth], -1) ) for delta_depth in delta_depths]    # tgt->src0, tgt->src1  (range: 0~1). shape=[B, h, w, 3]
                self.delta_depth_color = [ self.convert_to_tf_image(_color) for _color in delta_depth_color[1:] ]
            except:
                pass

            # 2. Concat additional information in inputs.
            Version = re.search("^(v[0-9.]+)", self.version)
            Version = "v0" if Version is None else Version.group(1)
            print (">>> Version: ", Version, re.search("^(v[0-9.]+)", self.version))
            pred_info = None
            if "v0" in Version:
                print (">>> [PoseNN] Only input RGB")
            elif "v1.555" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (tgt->src)")
                pred_info = [
                        pred_flows_inv[1],  # tgt->src0   for concating image_tgt (tgtsrc0)
                        pred_flows_inv[0],  # src0->src0  for concating image_src0
                        pred_flows_inv[0],  # src1->src1  for concating image_src1
                        pred_flows_inv[2]   # tgt->src1   for concating image_tgt (tgtsrc1)
                        ]
            elif "v1.55" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (tgt->src)")
                pred_info = pred_flows_inv + [pred_flows_inv[0]]    # tgt->src
            elif "v1" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (src->tgt)")
                pred_info = pred_flows + [pred_flows[0]]    # src->tgt
                # if `-sharedNN', 
                #               inputs = [tgt_image (+) zero_flow , src0_image (+) src0->tgt_flow] and 
                #                        [tgt_image (+) zero_flow , src1_image (+) src1->tgt_flow]
                # if `-sharedNNforward', 
                #               inputs = [src0_image (+) zero_flow , tgt_image (+) tgt->src0_flow] and 
                #                        [tgt_image (+) zero_flow , src1_image (+) src1->tgt_flow]
                if "-sharedNNforward" in self.version:
                    pred_info[0] = pred_flows_inv[1]        # tgt->src0
                    pred_info[1] = pred_flows_inv[0]        # zero
                    pred_info[2] = pred_flows[2]            # src1->tgt
                    pred_info[3] = pred_flows[0]            # zero
            elif "v2" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow + depth")
                pred_info = [
                        tf.concat([flo,dep], axis=3) for flo,dep in zip(pred_flows, pred_disp_maps)
                        ]
            elif "v3" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow + delta depth")
                pred_info = [
                        tf.concat([flo,deltad], axis=3) for flo,deltad in zip(pred_flows, delta_depths)
                        ]
            if "-seglabelid" in self.version:
                print (">>> [PoseNN] ...... + seglabelid (1 channel, id=0 ~ 18)")
                if pred_info is not None:
                    pred_info = [
                            tf.concat([info,seg], axis=3) for info,seg in zip(pred_info,pred_seglabels)
                            ]
                else:
                    pred_info = pred_seglabels

            # 3. Attention Part. (SE_BLOCK)
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
            if ".6" in self.version:
                se_input_flows[0] = pred_flows[1]    # src0->tgt
                se_input_flows[1] = pred_flows[2]    # src1->tgt
                se_input_flows[2] = pred_flows_inv[1]        # tgt->src0
                se_input_flows[3] = pred_flows_inv[2]        # tgt->src1
            elif ".555" in self.version:
                se_input_flows[1] = pred_flows_inv[1]    # tgt->src0
                se_input_flows[2] = pred_flows_inv[2]    # tgt->src1
            if "-sharedNNforward" in self.version:
                se_input_flows[0] = pred_flows_inv[1]    # tgt->src0
                se_input_flows[1] = pred_flows_inv[0]    # zero
                se_input_flows[2] = pred_flows[2]        # src1->tgt
                se_input_flows[3] = pred_flows[0]        # zero
            if "-norm_flow" in self.version:
                se_input_flows = [(f - 0.32140523) / 15.384229 for f in se_input_flows]
                print (">>> [SE][input] Normailed : (flow - flow.mean()) / flow.std()")

            # 3.3. Prepare absolute value of optical flows for se_block's inputs. (h: horizontal , v: vertical)
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
            se_input_depths = [d for d in pred_depths] + pred_depths[0]
            if "-norm_depth" in self.version:
                se_input_depths = [d / 80. for d in se_input_depths]

            # 3.5. Choose se_block's inputs, and generate attentions for PoseNN inputs.
            if "-se_flow_on_depthseg_sharedlayers" in self.version:
                # updated at 2020/01/31
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    try:
                        depth_thres = float(re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version).group(1))
                    except AttributeError:
                        depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=15., stddev=0.1), trainable=True)
                        self.depth_thres = depth_thres
                        #depth_thres = 20.
                    print (">>> [PoseNN][se_flow_on_depthseg_sharedlayers] depth threshold = ", depth_thres)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    seg_38 = [ tf.concat([seg_near,seg_far], axis=-1) for seg_near,seg_far in zip(depth_seg_19['near'],depth_seg_19['far']) ] # shape=[B,H,W,38]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_38[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_38[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_38[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow multiply to near/far segs (i.e., seg_38) :", segmask_tgt)
            elif "-se_flow_on_depthseg_seplayers" in self.version:
                # updated at 2020/01/31
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    try:
                        depth_thres = float(re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version).group(1))
                    except AttributeError:
                        depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=15, stddev=0.1), trainable=True)
                        self.depth_thres = depth_thres
                        #depth_thres = 20.
                    print (">>> [PoseNN][se_flow_on_depthseg_seplayers] depth threshold = ", depth_thres)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    segmask_tgt, segmask_src0, segmask_src1 = 0,0,0
                    for name in ['near', 'far']:
                        flowatt_tgt  = se(se_input_flows[0], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src0 = se(se_input_flows[1], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src1 = se(se_input_flows[2], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        segmask_tgt  += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][0] * flowatt_tgt , axis=-1), -1)
                        segmask_src0 += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][2] * flowatt_src1, axis=-1), -1)
                    #segmask_tgt =  tf.ones_like(segmask_tgt)
                print (">>> [PoseNN] build two SE_flow multiply to near/far segs (i.e., seg_38) seperatly:", segmask_tgt)
            elif "-se_flow_on_depthseg" in self.version:
                raise NameError("please select `-se_flow_on_depthseg_seplayers' or `-se_flow_on_depthseg_sharedlayers'.")
            elif "-se_mixDepthFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[0], se_input_flows[0]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[1], se_input_flows[1]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[2], se_input_flows[2]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_depth + flow_attention:", segmask_tgt)
            elif "-se_mixDispFlow" in self.version:
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[0], se_input_flows[0]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[1], se_input_flows[1]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[2], se_input_flows[2]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_disp + flow_attention:", segmask_tgt)
            elif "-se_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    if ".6" in Version:
                        flowatt_src0 = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # src0->tgt flow
                        flowatt_src1 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # src1->tgt flow
                        flowatt_tgtsrc0  = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)  # tgt->src0 flow
                        flowatt_tgtsrc1  = se(se_input_flows[3], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)  # tgt->src1 flow
                    elif ".55" in Version:
                        flowatt_tgtsrc0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # tgt->src0 flow
                        flowatt_tgtsrc1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # tgt->src1 flow
                    else:
                        flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    if ".65" in Version or ".55" in Version:
                        # (tgt, src0, tgt->src0)
                        print (">>> [PoseNN] segmask_tgt,src0 : use flow_tgt->src0")
                        print (">>> [PoseNN] segmask_tgt,src1 : use flow_tgt->src1")
                        segmask_tgtsrc0 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgtsrc0 , axis=-1), -1)     # for (tgt,src0) pair
                        segmask_tgtsrc1 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgtsrc1 , axis=-1), -1)     # for (tgt,src1) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_tgtsrc0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_tgtsrc1, axis=-1), -1)
                        segmask_tgt = segmask_tgtsrc0
                    elif ".6" in Version:
                        # (tgt, src0, src0->tgt)
                        print (">>> [PoseNN] segmask_tgt,src0 : use flow_src0->tgt")
                        print (">>> [PoseNN] segmask_tgt,src1 : use flow_src1->tgt")
                        segmask_tgtsrc0 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_src0 , axis=-1), -1)     # for (tgt,src0) pair
                        segmask_tgtsrc1 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_src1 , axis=-1), -1)     # for (tgt,src1) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                        segmask_tgt = segmask_tgtsrc0
                    else:
                        segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)         # for (tgt,src0) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow :", segmask_tgt)
            elif "-se_gp2x2_flow_nobottle" in self.version:
                # add at 2020/01/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:19, fc2:19):", segmask_tgt)
            elif "-se_gp2x2_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp21_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow ([2,1]) (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp2_flow" in self.version:   # same to gp2x2
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow ([2]) (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp_flow" in self.version or "-se_spp864_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_depth_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_depths[0], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_depths[1], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_depths[2], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_depth_to_seg" in self.version:
                # added at 2020/02/25
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_depths[0], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_depths[1], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_depths[2], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_depth_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_depth_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_depth" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[0], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_depth_attention :", segmask_tgt)
            elif "-se_disp_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                se_input_disps = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_disps[0], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_disps[1], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_disps[2], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_disp_to_seg" in self.version:
                # added at 2020/02/25
                se_input_disps = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_disps[0], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_disps[1], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_disps[2], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_disp_wo_tgt" in self.version:
                # added at 2020/02/23
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_disp_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_disp" in self.version:
                # added at 2020/02/23
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[0], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_disp_attention :", segmask_tgt)
            elif "-se_rgb_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                se_input_rgbs = self.input_images
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_rgbs[0], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_rgbs[1], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_rgbs[2], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_rgb_to_seg" in self.version:
                # added at 2020/02/25
                se_input_rgbs = self.input_images
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_rgbs[0], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_rgbs[1], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_rgbs[2], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_rgb_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[1], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[2], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_rgb_frames_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_rgb" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[0], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[1], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[2], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_rgb_frames_attention :", segmask_tgt)
            elif "-se_seg_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_segmentation_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation_attention :", segmask_tgt)
            elif "-se_gp2x2_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation_attention with 2x2 global pooling:", segmask_tgt)
            elif "-se_spp21_seg" in self.version or "-se_spp_seg_21" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2,1]):", segmask_tgt)
            elif "-se_spp2_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2]):", segmask_tgt)
            elif "-se_spp_seg" in self.version or "-se_spp864_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention :", segmask_tgt)
            elif "-se_SegFlow_to_seg_8" in self.version:
                # add at 2020/01/16
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    se_inputs  = [
                            tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                            tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                            tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                            ]
                    segflowatt_tgt  = se(se_inputs[0], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segflowatt_src0 = se(se_inputs[1], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segflowatt_src1 = se(se_inputs[2], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * segflowatt_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * segflowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * segflowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_SegFlow_to_seg" in self.version:
                # add at 2020/01/16
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    se_inputs  = [
                            tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                            tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                            tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                            ]
                    segflowatt_tgt  = se(se_inputs[0], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segflowatt_src0 = se(se_inputs[1], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segflowatt_src1 = se(se_inputs[2], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * segflowatt_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * segflowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * segflowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:19, fc2:19):", segmask_tgt)
            elif "-se_mixSegFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[0], se_input_flows[0]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[1], se_input_flows[1]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[2], se_input_flows[2]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention:", segmask_tgt)
            elif "-se_spp21_mixSegFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[0], se_input_flows[0]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[1], se_input_flows[1]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[2], se_input_flows[2]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation + flow_attention ([2,1]):", segmask_tgt)
            elif "-no_segmask" in self.version:
                seg_19  = [
                        tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                        ]
                segmask_tgt   = tf.ones_like(seg_19[0])[..., 0:1]
                segmask_src0  = tf.ones_like(seg_19[1])[..., 0:1]
                segmask_src1  = tf.ones_like(seg_19[2])[..., 0:1]
                print (">>> [PoseNN] No use attention module.")
            elif "-segmask_" in self.version and "-static" in self.version:
                segmask_src0, self.segweights = build_seg_channel_weight(pred_seglabels[1], prefix="pose_exp_net/")
                segmask_src1, _ = build_seg_channel_weight(pred_seglabels[2], prefix="pose_exp_net/")
                segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build weighted segmentation_attention (only segmask_src0, segmask_src1):", segmask_src0)
            else:
                segmask_tgt, self.segweights  = build_seg_channel_weight(pred_seglabels[0], prefix="pose_exp_net/")
                segmask_src0, _ = build_seg_channel_weight(pred_seglabels[1], prefix="pose_exp_net/")
                segmask_src1, _ = build_seg_channel_weight(pred_seglabels[2], prefix="pose_exp_net/")
                print (">>> [PoseNN] build weighted segmentation_attention :", segmask_tgt)

            # 4. PoseNN Part.
            # 4.1. Generate masked inputs.
            use_se_flow = len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/se_flow'))
            if ".55" in Version:
                print (">>> [PoseNN] vx.55 Version : segmask_tgt != 1")
            else:
                if use_se_flow:
                    print (">>> [PoseNN] se_flow mode.")
                    if "-sharedNNforward" in self.version:
                        print (">>> [PoseNN] segmask_src0 ----> 1")
                        segmask_src0    = tf.ones_like(segmask_src0)
                        segmask_tgtsrc1 = tf.ones_like(segmask_tgt)
                    else:
                        print (">>> [PoseNN] segmask_tgt ----> 1")
                        segmask_tgt     = tf.ones_like(segmask_tgt)
                        segmask_tgtsrc1 = tf.ones_like(segmask_tgt)
                else:
                    segmask_tgtsrc1 = segmask_tgt
            if pred_info is not None:
                print (">>> [PoseNN] concat pred_info now")
                if "-segmask_" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    self.input_images[0] = tf.multiply(self.input_images[0],  segmask_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1],  segmask_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2],  segmask_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3],  segmask_tgtsrc1)
                    if "-segmask_all" in self.version and ".555" in Version:
                        print (">>> [PoseNN] use segmask_all dilation network (src0, src1 concating flows = 0)")
                        pred_info[0] = tf.multiply(pred_info[0],  segmask_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  tf.ones_like(segmask_src0))
                        pred_info[2] = tf.multiply(pred_info[2],  tf.ones_like(segmask_src1))
                        pred_info[3] = tf.multiply(pred_info[3],  segmask_tgtsrc1)
                    elif "-segmask_all" in self.version:
                        print (">>> [PoseNN] use segmask_all dilation network")
                        pred_info[0] = tf.multiply(pred_info[0],  segmask_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  segmask_src0)
                        pred_info[2] = tf.multiply(pred_info[2],  segmask_src1)
                        pred_info[3] = tf.multiply(pred_info[3],  segmask_tgtsrc1)
                    elif "-segmask_rgb" in self.version:
                        print (">>> [PoseNN] use segmask_rgb dilation network")
                else:
                    print (">>> ======> Segmentation Attention [OFF]")
                self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
                self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
                self.input_images[3] = tf.concat([self.input_images[3],  pred_info[3]], axis=3)

                #if "-segmask_all" in self.version:
                    #print (">>> ======> Segmentation Attention [ON]")
                    #print (">>> [PoseNN] use segmask_all dilation network")
                    ## Only concat pred_info with source images (i.e., without target image) if Version is "vXXX.5"   (add at 2020/01/31)
                    #self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3) if ".5" not in Version else self.input_images[0]
                    #self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                    #self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
                    #self.input_images[0] = tf.multiply(self.input_images[0],  segmask_tgt)
                    #self.input_images[1] = tf.multiply(self.input_images[1],  segmask_src0)
                    #self.input_images[2] = tf.multiply(self.input_images[2],  segmask_src1)
                #elif "-segmask_rgb" in self.version:
                    #print (">>> ======> Segmentation Attention [ON]")
                    #print (">>> [PoseNN] use segmask_rgb dilation network")
                    #self.input_images[0] = tf.multiply(self.input_images[0], segmask_tgt)
                    #self.input_images[1] = tf.multiply(self.input_images[1], segmask_src0)
                    #self.input_images[2] = tf.multiply(self.input_images[2], segmask_src1)
                    ## Only concat pred_info with source images (i.e., without target image) if Version is "vXXX.5"   (add at 2020/01/31)
                    #self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3) if ".5" not in Version else self.input_images[0]
                    #self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                    #self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
                #else:
                    #print (">>> ======> Segmentation Attention [OFF]")
                    ## Only concat pred_info with source images (i.e., without target image) if Version is "vXXX.5"   (add at 2020/01/31)
                    #self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3) if ".5" not in Version else self.input_images[0]
                    #self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
                    #self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
            else:
                if "-segmask" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    print (">>> [PoseNN] use segmask_rgb dilation network")
                    self.input_images[0] = tf.multiply(self.input_images[0], segmask_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1], segmask_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2], segmask_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3], segmask_tgtsrc1)

            # 4.2. Predict ego-motion.
            poseNet = partial(poseNet, dropout=self.dropout, is_training=True, se_attention=se_attention, batch_norm="-batch_norm" in self.version, cnv6_num_outputs=cnv6_num_outputs) # -2
            if "-sharedNNinv" in self.version: # inverse pose
                # add at 2020/01/16
                print (">>> ===== [ SharedNNinv PoseNN (%s) ] =====" % Version)
                #pred_pose0, _ = poseNet(self.input_images[1], self.input_images[0])
                #pred_pose1, _ = poseNet(self.input_images[0], self.input_images[2])
                #pred_poses = tf.concat([get_inv_pose(pred_pose0), pred_pose1], axis=-2)
                raise NameError("`-sharedNNinv' is unavailable.")
            elif "-sharedNNforward" in self.version: # all forwarding poses
                # add at 2020/01/22
                print (">>> ===== [ SharedNNforward PoseNN (%s) ] =====" % Version)
                if ".5" in Version or ".6" in Version:
                    pred_pose0, _ = poseNet(self.input_images[1][...,0:3], self.input_images[0]) #src0->tgt
                    pred_pose1, _ = poseNet(self.input_images[3][...,0:3], self.input_images[2]) #tgt->src1
                else:
                    pred_pose0, _ = poseNet(self.input_images[1], self.input_images[0]) #src0->tgt
                    pred_pose1, _ = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            elif "-sharedNN" in self.version:
                # add at 2020/01/16
                print (">>> ===== [ SharedNN PoseNN (%s) ] =====" % Version)
                if ".555" in Version:
                    pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1][...,0:3]) #tgt->src0
                    pred_pose1, _ = poseNet(self.input_images[3], self.input_images[2][...,0:3]) #tgt->src1
                elif ".5" in Version or ".6" in Version:
                    pred_pose0, _ = poseNet(self.input_images[0][...,0:3], self.input_images[1]) #tgt->src0
                    pred_pose1, _ = poseNet(self.input_images[3][...,0:3], self.input_images[2]) #tgt->src1
                else:
                    pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1]) #tgt->src0
                    pred_pose1, _ = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            else:
                pred_poses, _ = poseNet(self.input_images[0], self.input_images[1], self.input_images[2])
            print (">>> [PoseNN] pred_poses: ", pred_poses)

            # Visualization in tf.summary.
            self.segmask = [segmask_tgt, segmask_src0, segmask_src1]
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
                        self.convert_to_tf_image( flow_to_image( self.input_images[2][...,3:5]))  # src1_masked_flows 
                        ]
                    if pred_info[0].shape.as_list()[-1] > 2:
                        if "v3" in self.version:
                            zeros = tf.zeros_like(pred_info[0][...,2:3])
                            self.masked_deltadepths  = [
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[0][...,5:6]], axis=-1))),   # tgt_masked_deltadepths   
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[1][...,5:6]], axis=-1))),   # src0_masked_deltadepths 
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[2][...,5:6]], axis=-1)))   # src1_masked_deltadepths 
                                ]
                        else:
                            self.masked_depths  = [
                                    self.input_images[0][...,5:6],  # tgt_masked_depths   
                                    self.input_images[1][...,5:6],  # src0_masked_depths 
                                    self.input_images[2][...,5:6]   # src1_masked_depths 
                                ]
            self.masked_seglabels  = [
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[0], tf.float32), self.segmask[0])),   # tgt_masked_seglabels   
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[1], tf.float32), self.segmask[1])),   # src0_masked_seglabels 
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[2], tf.float32), self.segmask[2]))   # src1_masked_seglabels 
                ]
            print (">>> [PoseNN] pred_poses: ", pred_poses)

        with tf.name_scope("compute_loss"):
            pixel_loss = 0
            smooth_loss = 0
            trans_training_loss = 0
            rot_training_loss = 0
            pose_training_loss = 0
            trans_loss = 0
            rot_loss = 0
            pose_loss = 0
            ssim_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            mask_stack_all = []
            proj_image_stack_all = []
            proj_error_stack_all = []
            self.relative_pose_gt = []
            for s in range(num_scales):
                # Scale the source and target images for computing loss at the according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])                
                curr_src_image_stack = tf.image.resize_area(src_image_stack, 
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                print (" >> s%d  curr_tgt_image: " % s, curr_tgt_image)

                if "-noD" in self.version or "-fixpretrainD" in self.version:
                    print (">>> No training depth")

                    # Prepare images for tensorboard summaries
                    tgt_image_all.append(curr_tgt_image)
                    src_image_stack_all.append(curr_src_image_stack)

                    for i in range(opt.num_source):     # num_source = seq_length - 1
                        # Inverse warp the source image to the target image frame

                        if "-sharedNNforward" in self.version and i == 0:
                            # add at 2020/01/22
                            relative_pose = get_relative_pose(poses[:,i+1,:], poses[:,0,:])
                        else:
                            relative_pose = get_relative_pose(poses[:,0,:], poses[:,i+1,:])
                        relative_rot = tf.slice(relative_pose, [0, 0, 0], [-1, 3, 3])
                        relative_rot_vec = mat2euler(relative_rot)
                        relative_trans_vec = tf.slice(relative_pose, [0, 0, 3], [-1, 3, 1])
                        relative_pose_vec = tf.squeeze(tf.concat([relative_rot_vec, relative_trans_vec], axis=1), axis=-1)
                        self.relative_pose_gt.append(relative_pose_vec)

                else:

                    for i in range(opt.num_source):     # num_source = seq_length - 1
                        # Inverse warp the source image to the target image frame

                        print (">>> Use version V2")
                        relative_pose = get_relative_pose(poses[:,0,:], poses[:,i+1,:])
                        relative_rot = tf.slice(relative_pose, [0, 0, 0], [-1, 3, 3])
                        relative_rot_vec = mat2euler(relative_rot)
                        relative_trans_vec = tf.slice(relative_pose, [0, 0, 3], [-1, 3, 1])
                        relative_pose_vec = tf.squeeze(tf.concat([relative_rot_vec, relative_trans_vec], axis=1), axis=-1)
                        self.relative_pose_gt.append(relative_pose_vec)

                        warp_pose = pred_poses[:,i,:]
                        pose_is_vec = True

                        print (">>> Warp Loss")
                        curr_proj_image, mask = projective_inverse_warp(
                            curr_src_image_stack[:,:,:,3*i:3*(i+1)], 
                            tf.squeeze(pred_depth[s], axis=3), 
                            warp_pose, intrinsics[:,s,:,:], is_vec=pose_is_vec)
                        curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                        curr_proj_error = tf.multiply(curr_proj_error, mask)

                        # below-threshold mask
                        dims = curr_proj_error.shape.as_list()
                        perct_thresh = tf.contrib.distributions.percentile(curr_proj_error, q=99, axis=[1,2])   # shape[B,3]
                        perct_thresh = tf.expand_dims(tf.expand_dims(perct_thresh, 1), 1)                   # shape=[B,1,1,3]
                        perct_thresh = tf.tile(perct_thresh, [1] + dims[1:-1] + [1])                        # shape=[B,i,j,3]
                        zeros = tf.zeros_like(curr_proj_error)
                        curr_proj_error = tf.clip_by_value(curr_proj_error, zeros, perct_thresh)
                        above_perct_thresh_region = tf.reduce_max(tf.cast(tf.equal(curr_proj_error, perct_thresh), 'float32'), axis=3)
                        above_perct_thresh_region = tf.greater_equal(above_perct_thresh_region, 1.0)
                        suppresion_mask = tf.expand_dims(1.0 - tf.cast(above_perct_thresh_region, 'float32'), axis=3)
                        curr_proj_error = tf.multiply(curr_proj_error, suppresion_mask)
                        mask = tf.multiply(mask, suppresion_mask)
                        # (1) segmentation mask
                        # xxxxxx
                        pixel_loss += tf.reduce_mean(curr_proj_error) 

                        # SSIM loss
                        if opt.ssim_weight > 0:
                            ssim_mask = slim.avg_pool2d(mask, 3, 1, 'VALID')
                            ssim_loss += tf.reduce_mean(
                                ssim_mask * self.compute_ssim_loss(curr_proj_image, curr_tgt_image))

                        # Prepare images for tensorboard summaries
                        if i == 0:
                            proj_image_stack = curr_proj_image
                            mask_stack = mask
                            proj_error_stack = curr_proj_error
                        else:
                            proj_image_stack = tf.concat([proj_image_stack, curr_proj_image], axis=3)
                            mask_stack = tf.concat([mask_stack, mask], axis=3)
                            proj_error_stack = tf.concat([proj_error_stack, curr_proj_error], axis=3)

                    tgt_image_all.append(curr_tgt_image)
                    src_image_stack_all.append(curr_src_image_stack)
                    proj_image_stack_all.append(proj_image_stack)
                    mask_stack_all.append(mask_stack)
                    proj_error_stack_all.append(proj_error_stack)

                    if opt.smooth_weight > 0:
                        smooth_loss += opt.smooth_weight/(2**s) * \
                            self.compute_smooth_loss(pred_disp[s], curr_tgt_image)


                # Pose Loss
                self.positionErr = []
                self.scaleErr = []
                self.relative_trans_scale = []
                self.pred_trans_scale = []

                for i in range(opt.num_source):     # num_source = seq_length - 1
                    # Relative pose error
                    relative_pose_vec = self.relative_pose_gt[i]                                # forgot to add this line QQQQQ @ 2019/07/24

                    prior_trans_vec = relative_pose_vec[:,3:]
                    prior_rot_vec = relative_pose_vec[:,:3]
                    pred_trans_vec  = pred_poses[:, i, 3:]
                    pred_rot_vec  = pred_poses[:, i, :3]

                    prior_trans_vec_scale = tf.norm(prior_trans_vec, axis=1)
                    pred_trans_vec_scale  = tf.norm(pred_trans_vec, axis=1)
                    positionErr = tf.norm(prior_trans_vec - pred_trans_vec, axis=1)
                    scaleErr    = prior_trans_vec_scale - pred_trans_vec_scale

                    if opt.pose_weight > 0 and s == 0:  # only do it for highest resolution

                        # Training Loss.
                        if "-PoseLossWeak" in self.version:
                            print (">>> TransLoss : ", self.version, "[Weak]")
                            trans_training_loss += tf.reduce_mean(self.compute_trans_loss(prior_trans_vec, pred_trans_vec))
                        else:
                        #if "-PoseLossStrong" in self.version:
                            print (">>> TransLoss : ", self.version, "[Strong]")
                            trans_training_loss += tf.reduce_mean(self.compute_trans_loss(prior_trans_vec, pred_trans_vec))
                            trans_training_loss += tf.reduce_mean(
                                    (scaleErr) ** 2
                                    )
                            self.relative_trans_scale.append(prior_trans_vec_scale)
                            self.pred_trans_scale.append(pred_trans_vec_scale)
                            self.scaleErr.append(scaleErr)

                        rot_training_loss += tf.reduce_mean(self.compute_rot_loss( prior_rot_vec, pred_rot_vec ))

                    # Summary visualization loss.
                    trans_loss += tf.reduce_mean(self.compute_trans_loss(prior_trans_vec, pred_trans_vec))
                    trans_loss += tf.reduce_mean(
                            (scaleErr) ** 2
                            )
                    rot_loss += tf.reduce_mean(self.compute_rot_loss( prior_rot_vec, pred_rot_vec ))

                    if "-noD" not in self.version and "-fixpretrainD" not in self.version:
                        pred_depth = [pred_depths[0]]


            unsv_loss = opt.ssim_weight * ssim_loss + \
                (1 - opt.ssim_weight) * pixel_loss + \
                smooth_loss
            # Training Loss.
            trans_vec_loss = opt.pose_weight * trans_training_loss * coef_trans
            rot_vec_loss = opt.pose_weight * rot_training_loss * coef_rot
            pose_training_loss += trans_vec_loss + rot_vec_loss
            # Summary visualization loss.
            pose_loss += 0.1 * trans_loss * coef_trans + \
                         0.1 * rot_loss * coef_rot

            print (" >> positionErr:", self.positionErr)
            print (" >> scaleErr:", self.scaleErr)
            print (" >> relative_trans_scale:", self.relative_trans_scale)
            print (" >> pred_trans_scale:", self.pred_trans_scale)

            if self.scaleErr != []:
                self.scaleErr = tf.stack(self.scaleErr, axis=1)
                self.relative_trans_scale = tf.stack(self.relative_trans_scale, axis=1)
                self.pred_trans_scale = tf.stack(self.pred_trans_scale, axis=1)
            if self.positionErr != []:
                self.positionErr = tf.stack(self.positionErr, axis=1)


        with tf.name_scope("train_op"):
            all_train_vars = [var for var in tf.trainable_variables()]
            depth_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='depth_')
            pose_net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net')
            self.pose_net_vars = pose_net_vars
            trans_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/pose/translation')
            rot_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/pose/rotation')
            train_vars = [var for var in all_train_vars if var not in trans_vars and var not in rot_vars]
            pose_without_rot_vars = [var for var in pose_net_vars if var not in rot_vars]
            print ("All train vars:")
            for i,v in enumerate(all_train_vars):
                print ("   ", i, v)
            #print ("Train_OP vars (-ALL):")
            #from tensorflow.python.ops import variables as tf_variables
            #for i,v in enumerate(tf_variables.trainable_variables()):
                #print ("   ", i, v)
            print ("Model vars:")
            for i,v in enumerate(tf.model_variables()):
                print ("   ", i, v)
            print ("DepthNN vars:")
            for i,v in enumerate(depth_net_vars):
                print ("   ", i, v)
            print ("DepthNN vars in model_variables:")
            for i,v in enumerate([var for var in tf.model_variables() if "depth_net" in var.op.name]):
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

            # self.grads_and_vars = optim.compute_gradients(unsv_loss, 
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)

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

            all_loss = unsv_loss + trans_vec_loss + rot_vec_loss

            if "-WARPLOSS" in self.version:
                print (">>>> Use warp_loss + ssim_loss + pose_loss update poseNN")
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    adam_opt = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                    self.pose_net_op = slim.learning.create_train_op(trans_vec_loss + rot_vec_loss + opt.ssim_weight * ssim_loss + (1-opt.ssim_weight) * pixel_loss, adam_opt, 
                            variables_to_train=pose_net_vars, 
                            colocate_gradients_with_ops=True,
                            summarize_gradients=_ENABLE_SUMMARIES_GRADIENTS,
                            global_step=None)
                    self.train_all_op = self.pose_net_op

            elif '-ALL' in self.version:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                    print (">>>> Use Trainall scheme")
                    self.train_all_op = slim.learning.create_train_op(all_loss, optim, 
                            colocate_gradients_with_ops=True,
                            global_step=None,
                            summarize_gradients=_ENABLE_SUMMARIES_GRADIENTS)

            elif "-noD" in self.version:
                print (">>>> Update poseNN (v1)")
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    adam_opt = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                    self.pose_net_op = slim.learning.create_train_op(trans_vec_loss + rot_vec_loss, adam_opt, 
                            variables_to_train=pose_net_vars, 
                            colocate_gradients_with_ops=True,
                            summarize_gradients=_ENABLE_SUMMARIES_GRADIENTS,
                            global_step=None)
                    self.train_all_op = self.pose_net_op

            else:
            #if '-ALL' in self.version:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
                    print (">>>> Use Trainall")
                    self.train_all_op = slim.learning.create_train_op(all_loss, optim, 
                            colocate_gradients_with_ops=True,
                            global_step=None,
                            summarize_gradients=_ENABLE_SUMMARIES_GRADIENTS)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.unsv_loss  = tf.convert_to_tensor(unsv_loss)
        self.pixel_loss  = tf.convert_to_tensor(pixel_loss)
        self.pose_loss   = tf.convert_to_tensor(pose_loss)
        self.trans_loss  = tf.convert_to_tensor(trans_loss)
        self.rot_loss    = tf.convert_to_tensor(rot_loss)
        self.pose_training_loss   = tf.convert_to_tensor(pose_training_loss)
        self.trans_training_loss  = tf.convert_to_tensor(trans_training_loss)
        self.rot_training_loss    = tf.convert_to_tensor(rot_training_loss)
        self.smooth_loss = tf.convert_to_tensor(smooth_loss)
        self.ssim_loss   = tf.convert_to_tensor(ssim_loss)
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        if "-noD" not in self.version:
            self.pred_depth = pred_depth
            self.proj_image_stack_all = proj_image_stack_all
            self.mask_stack_all = mask_stack_all
            self.proj_error_stack_all = proj_error_stack_all
        return loader, batch_sample


    def compute_smooth_loss(self, disp, img):
        def _gradient(pred):
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            return D_dx, D_dy

        disp_gradients_x, disp_gradients_y = _gradient(disp)
        image_gradients_x, image_gradients_y = _gradient(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keepdims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))
    

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


    # reference https://github.com/tensorflow/models/tree/master/research/vid2depth/model.py
    def compute_ssim_loss(self, x, y):
        """Computes a differentiable structured image similarity measure."""
        c1 = 0.01**2
        c2 = 0.03**2
        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')
        sigma_x = slim.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
        sigma_y = slim.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        ssim = ssim_n / ssim_d
        return tf.clip_by_value((1 - ssim) / 2, 0, 1)

    
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
        tf.summary.scalar("unsv_loss", self.unsv_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("ssim_loss", self.ssim_loss)

        tf.summary.scalar("pose_loss", self.pose_loss)
        tf.summary.scalar("trans_loss", self.trans_loss)
        tf.summary.scalar("rot_loss", self.rot_loss)
        if opt.pose_weight > 0:
            tf.summary.scalar("pose_training_loss", self.pose_training_loss)
            tf.summary.scalar("trans_training_loss", self.trans_training_loss)
            tf.summary.scalar("rot_training_loss", self.rot_training_loss)
            for i in range(opt.num_source):
                if getattr(self, "relative_trans_scale", []) != []:
                    tf.summary.scalar("GroundTruth_trans_scale[%d]" % (i), tf.reduce_mean(self.relative_trans_scale[i]))
                    tf.summary.scalar("Pred_trans_scale[%d]" % (i), tf.reduce_mean(self.pred_trans_scale[i]))
                    tf.summary.scalar("scaleErr[%d]" % (i), tf.reduce_mean(self.scaleErr[i]))
                if getattr(self, "positionErr", []) != []:
                    tf.summary.scalar("positionErr_n[%d]" % (i), tf.reduce_mean(self.positionErr[i]))
            if getattr(self, "stddev", None) is not None:
                tf.summary.scalar("SoftTransLossStddev", self.stddev)

        s = 0   # only show the error images of the highest resolution (scale 0)

        # ---[ export depthNN output
        #if getattr(self, "pred_depth", None) is not None:
            #scale_idx = 0
            #tf.summary.histogram("scale%d_tgt_depth" % s, self.pred_depth[scale_idx])                # target prediction
            #shown_disparity_image = self.normalize_for_show(1./self.pred_depth[scale_idx])              # target prediction
            #tf.summary.image('scale%d_tgt_disparity_image' % s, shown_disparity_image)               # target prediction @batch_idx=0~3
            #print(">>>> Summary scale%d_tgt_disparity_image: " % s, shown_disparity_image)
        # ---[ export input_depths
#        if getattr(self, "pred_depths", None) is not None:
#            tf.summary.histogram("tgt_depth",  self.pred_depths[0])           # target 
#            tf.summary.histogram("src0_depth", self.pred_depths[1])
#            tf.summary.histogram("src1_depth", self.pred_depths[2])
#            tf.summary.image('image_target/disp',  self.normalize_for_show(1./self.pred_depths[0]))         # shown_disparity_image
#            tf.summary.image('image_source0/disp', self.normalize_for_show(1./self.pred_depths[1]))         # shown_disparity_image
#            tf.summary.image('image_source1/disp', self.normalize_for_show(1./self.pred_depths[2]))         # shown_disparity_image
#        # ---[ export near_regions
#        if getattr(self, "depth_seg_19", None) is not None:
#            tf.summary.image('image_target/near',  tf.cast(tf.reduce_sum(self.depth_seg_19['near'][0], -1, True), tf.uint8) * self.pred_seglabels_color[0])
#            tf.summary.image('image_source0/near', tf.cast(tf.reduce_sum(self.depth_seg_19['near'][1], -1, True), tf.uint8) * self.pred_seglabels_color[1])
#            tf.summary.image('image_source1/near', tf.cast(tf.reduce_sum(self.depth_seg_19['near'][2], -1, True), tf.uint8) * self.pred_seglabels_color[2])
#            tf.summary.image('image_target/far',   tf.cast(tf.reduce_sum(self.depth_seg_19['far'][0], -1, True), tf.uint8) * self.pred_seglabels_color[0])
#            tf.summary.image('image_source0/far',  tf.cast(tf.reduce_sum(self.depth_seg_19['far'][1], -1, True), tf.uint8) * self.pred_seglabels_color[1])
#            tf.summary.image('image_source1/far',  tf.cast(tf.reduce_sum(self.depth_seg_19['far'][2], -1, True), tf.uint8) * self.pred_seglabels_color[2])
#        if getattr(self, "depth_thres", None) is not None:
#            tf.summary.scalar("depth_threshold", self.depth_thres)
#
#        # ---[ export RGB image
#        tf.summary.image('image_target', self.deprocess_image(self.tgt_image_all[0]))   # target prediction
#        for i in range(opt.num_source):
#            tf.summary.image(
#                'image_source%d' % (i),
#                self.deprocess_image(self.src_image_stack_all[0][:, :, :, i*3:(i+1)*3]))
#            try:
#                proj_images = self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3])
#                mask_images = self.mask_stack_all[s][:, :, :, i:i+1]
#                proj_error_images = self.deprocess_image(tf.clip_by_value(
#                    self.proj_error_stack_all[s][:, :, :, i*3:(i+1)*3] - 1, -1, 1))
#                tf.summary.image('scale%d_projected_image_%d' % (s, i), proj_images)
#                tf.summary.image('scale%d_proj_error_%d' % (s, i), proj_error_images)
#                tf.summary.image('scale%d_mask_%d' % (s, i), mask_images)
#            except:
#                pass
#
#        # ---[ export input_flow
#        if getattr(self, "pred_flow_color", None) is not None:
#            for i,flow in enumerate(self.pred_flow_color):
#                print (">>>> Summary flow", flow)
#                tf.summary.image('flow_src%d' % (i), flow)
#        # ---[ export delta_depth
#        if getattr(self, "delta_depth_color", None) is not None:
#            for i,_color in enumerate(self.delta_depth_color):
#                print (">>>> Summary _color", _color)
#                tf.summary.image('delta_depth_src%d' % (i), _color)
#
#        # ---[ export pred_seglabels_color
#        if getattr(self, "pred_seglabels_color", None) is not None:
#            tf.summary.image("pred_seglabels_color_tgt", self.pred_seglabels_color[0])
#            for i,_color in enumerate(self.pred_seglabels_color[1:]):
#                tf.summary.image("pred_seglabels_color_src%d" % (i), _color)
#
#        # ---[ export segweights (-segmask_*)
#        if getattr(self, "segweights", None) is not None:
#            for i in range(self.segweights.shape.as_list()[0]):
#                tf.summary.scalar("segweights/%02d" % (i), self.segweights[i])
#
#        # ---[ export segmask (-segmask_*)
#        if getattr(self, "segmask", None) is not None:
#            tf.summary.image('segmask_tgt', self.segmask[0])
#            for i,_color in enumerate(self.segmask[1:]):
#                tf.summary.image('segmask_src%d' % (i), _color)
#
#        # ---[ export *_mask (-segmask_*)
#        if getattr(self, "masked_images", None) is not None:
#            for i,_color in enumerate(self.masked_images):
#                if i == 0:
#                    tf.summary.image("masked_images_tgt", _color)
#                else:
#                    tf.summary.image("masked_images_src%d" % (i-1), _color)
#        if getattr(self, "masked_flows", None) is not None:
#            for i,_color in enumerate(self.masked_flows):
#                if i == 0:
#                    tf.summary.image("masked_flows_tgt", _color)
#                else:
#                    tf.summary.image("masked_flows_src%d" % (i-1), _color)
#        if getattr(self, "masked_deltadepths", None) is not None:
#            for i,_color in enumerate(self.masked_deltadepths):
#                if i == 0:
#                    tf.summary.image("masked_deltadepths_tgt", _color)
#                else:
#                    tf.summary.image("masked_deltadepths_src%d" % (i-1), _color)
#        if getattr(self, "masked_depths", None) is not None:
#            for i,_color in enumerate(self.masked_depths):
#                if i == 0:
#                    tf.summary.image("masked_depths_tgt", _color)
#                else:
#                    tf.summary.image("masked_depths_src%d" % (i-1), _color)
#        if getattr(self, "masked_seglabels", None) is not None:
#            for i,_color in enumerate(self.masked_seglabels):
#                if i == 0:
#                    tf.summary.image("masked_seglabels_tgt", _color)
#                else:
#                    tf.summary.image("masked_seglabels_src%d" % (i-1), _color)
#
#        # ---[ export se_input_flow_color
#        if getattr(self, "se_input_flows_color", None) is not None:
#            for i,flow in enumerate(self.se_input_flows_color):
#                print (">>>> Summary se_input_flow", flow)
#                if i == 0:
#                    tf.summary.image('se_input_flow_tgt', flow)
#                else:
#                    tf.summary.image('se_input_flow_src%d' % (i-1), flow)
#

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

        depth_vars = [var for var in tf.model_variables() if "depth_net" in var.op.name] 
        if len(depth_vars) > 0:
            self.depth_saver = tf.train.Saver(depth_vars,
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

            # ===[ Load pre-trained depthNN
            if "-pretrainD" in self.version:
                checkpoint = "pretrain-ckpt/model-250000"
                print("Resume depthNN from previous checkpoint: %s" % checkpoint)
                self.depth_saver.restore(sess, checkpoint)

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
            fetches["unsv_loss"] = tf.convert_to_tensor(self.unsv_loss)
            fetches["pixel_loss"] = tf.convert_to_tensor(self.pixel_loss)
            fetches["smooth_loss"] = tf.convert_to_tensor(self.smooth_loss)
            fetches["summary"] = sv.summary_op
            print ("\nsummary")
            for su in tf.get_collection(tf.GraphKeys.SUMMARIES,""):
                print ("  > ", su)
            if getattr(self, "pred_depth", None) is not None:
                fetches["pred_depth"] = self.pred_depth

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
            if getattr(self, "positionErr", []) != []:
                fetches["positionErr"] = self.positionErr                         # shape=[b,num_source]
                print (">>> print positionErr : ", fetches["positionErr"])

            if getattr(self, "pred_flow_color", None) is not None:
                fetches['colorflow'] = self.pred_flow_color
            if getattr(self, "delta_depth_color", None) is not None:
                fetches['colorddepth'] = self.delta_depth_color


            step = 0
            while step < opt.max_steps:
                results = sess.run(fetches)
                if getattr(self, "sv_train_all_op", None) is not None:
                    if step % 1000 < 100:
                        print ("[%5d] unsv. + sv. update" % step)
                        sess.run(self.sv_train_all_op)
                gs = results["global_step"]
                step = gs - 1

#                if step % opt.summary_freq == 0:
#                    print("Step: [%5d][%d] " % (step, gs))
#                    if getattr(self, "pred_depth", None) is not None:
#                        for s,d in enumerate(results["pred_depth"]):
#                            print(" | pred_depth[%d].max/mean/min = %.6f/%.6f/%.6f | " % (s, 
#                                np.max(d), np.mean(d), np.min(d)))
#
#                    if getattr(self, "pred_flow_color", None) is not None:
#                        for s,d in enumerate(results["colorflow"]):
#                            print(" | input_flow_color[src%d].max/mean/min = %.6f/%.6f/%.6f | " % (s, 
#                                np.max(d), np.mean(d), np.min(d)))
#                    if getattr(self, "delta_depth_color", None) is not None:
#                        for s,d in enumerate(results["colorddepth"]):
#                            print(" | input_delta_depth_color[src%d].max/mean/min = %.6f/%.6f/%.6f | " % (s, 
#                                np.max(d), np.mean(d), np.min(d)))
#
#
#
#                    for b in range(opt.batch_size):
#                        for n in range(opt.num_source):
#                            pp = results["pred_poses"][b][n]
#                            gp = results["relative_pose_gt"][n][b]
#                            print(" | [batch%.2d] predict_poses[%d].rz/ry/rx/tx/ty/tz = (%.6f/%.6f/%.6f) (%.6f/%.6f/%.6f)" % (b,n,
#                                pp[0], pp[1], pp[2], pp[3], pp[4], pp[5]))
#                            print(" | [batch%.2d] relv_gt_poses[%d].rz/ry/rx/tx/ty/tz = (%.6f/%.6f/%.6f) (%.6f/%.6f/%.6f)" % (b,n,
#                                gp[0], gp[1], gp[2], gp[3], gp[4], gp[5]))
#
#                    if "scaleErr" in results:
#                        for b,(gTS, pTS, SE) in enumerate(zip(results["GroundTruth_trans_scale"], results["Pred_trans_scale"], results["scaleErr"])):
#                            for n in range(opt.num_source):
#                                print (" > [batch%.2d] Pred_trans_scale[%d]        : %.6f" % (b,n,pTS[n]))
#                                print (" > [batch%.2d] GroundTruth_trans_scale[%d] : %.6f" % (b,n,gTS[n]))
#                                print (" > [batch%.2d] scaleErr[%d] = %.6f" % (b,n,SE[n]))
#                    if "positionErr" in results:
#                        for b,PE in enumerate(results["positionErr"]):
#                            for n in range(opt.num_source):
#                                print (" > [batch%.2d] positionErr[%d] = %.6f" % (b,n,PE[n]))

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f"
                          % (train_epoch, train_step, self.steps_per_epoch,
                             (time.time() - start_time)/opt.summary_freq))
                    print("total/pixel/smooth loss: [%.3f/%.3f/%.3f]\n" % (
                        results["unsv_loss"], results["pixel_loss"], results["smooth_loss"]))
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
        input_mc = self.select_tensor_or_placeholder_input(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        # src0_image  = src_image_stack[...,:3]
        # src1_image  = src_image_stack[...,3:]
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

        image_stack = tf.concat([tgt_image, src_image_stack], axis=3)

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
            # 1. Choose PoseNN se mode. (optional)
            if "-se_insert" in self.version:
                se_attention = True
            elif "-se_skipadd" in self.version:
                se_attention = "se_skipadd"
            elif "-se_replace" in self.version:
                se_attention = "se_replace"
            else:
                se_attention = False

            self.input_images  = [
                    tgt_image,
                    src_image_stack[...,:3],
                    src_image_stack[...,3:],
                    tgt_image    # tgt->src1 for -sharedNNforward
                    ]

            # 1. Choose PoseNN type.
            if "-sharedNN" in self.version:
                # add at 2020/01/16
                if "-dilatedPoseNN" in self.version:
                    print (">>> choose dilated Shared CNN")
                    poseNet = decouple_sharednet_v0_dilation
                elif "-dilatedCouplePoseNN" in self.version:
                    print (">>> choose dilated Shared CouplePose CNN")
                    poseNet = couple_sharednet_v0_dilation
                elif "-couplePoseNN" in self.version:
                    #print (">>> choose Shared CouplePose CNN")
                    raise NameError("not support `-sharedNN-couplePoseNN' mode.")
                else:
                    raise NameError("unknown PoseNN type.")
                # -----------------------------------
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

            cnv6_num_match = re.search("-cnv6_([0-9]+)", self.version)
            cnv6_num_outputs = 128 if cnv6_num_match is None else int(cnv6_num_match.group(1))
            print (">>> choose cnv6_num_outputs :", cnv6_num_outputs, cnv6_num_match)

            # flow color
            pred_flow_color = [ flow_to_image(flow) for flow in pred_flows ]    # (range: 0~1). shape=[B, h, w, 3]
            self.pred_flow_color = [ self.convert_to_tf_image(flow_color) for flow_color in pred_flow_color[1:] ]

            # delta depth color
            try:
                delta_depth_color = [ flow_to_image( tf.concat([tf.zeros_like(delta_depth), delta_depth], -1) ) for delta_depth in delta_depths]    # tgt->src0, tgt->src1  (range: 0~1). shape=[B, h, w, 3]
                self.delta_depth_color = [ self.convert_to_tf_image(_color) for _color in delta_depth_color[1:] ]
            except:
                pass

            # 2. Concat additional information in inputs.
            Version = re.search("^(v[0-9.]+)", self.version)
            Version = "v0" if Version is None else Version.group(1)
            print (">>> Version: ", Version, re.search("^(v[0-9.]+)", self.version))
            pred_info = None
            if "v0" in self.version:
                print (">>> [PoseNN] Only input RGB")
            elif "v1.555" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (tgt->src)")
                pred_info = [
                        pred_flows_inv[1],  # tgt->src0   for concating image_tgt (tgtsrc0)
                        pred_flows_inv[0],  # src0->src0  for concating image_src0
                        pred_flows_inv[0],  # src1->src1  for concating image_src1
                        pred_flows_inv[2]   # tgt->src1   for concating image_tgt (tgtsrc1)
                        ]
            elif "v1.55" in Version:
                print (">>> [PoseNN] ...... ready to concat : flow (tgt->src)")
                pred_info = pred_flows_inv + [pred_flows_inv[0]]    # tgt->src
            elif "v1" in self.version:
                print (">>> [PoseNN] ...... ready to concat : flow")
                pred_info = pred_flows + [pred_flows[0]]    # src->tgt
                # if `-sharedNN', 
                #               inputs = [tgt_image (+) zero_flow , src0_image (+) src0->tgt_flow] and 
                #                        [tgt_image (+) zero_flow , src1_image (+) src1->tgt_flow]
                # if `-sharedNNforward', 
                #               inputs = [src0_image (+) zero_flow , tgt_image (+) tgt->src0_flow] and 
                #                        [tgt_image (+) zero_flow , src1_image (+) src1->tgt_flow]
                if "-sharedNNforward" in self.version:
                    pred_info[0] = pred_flows_inv[1]        # tgt->src0
                    pred_info[1] = pred_flows_inv[0]        # zero
                    pred_info[2] = pred_flows[2]            # src1->tgt
                    pred_info[3] = pred_flows[0]            # zero
            elif "v2" in self.version:
                print (">>> [PoseNN] ...... ready to concat : flow + depth")
                pred_info = [
                        tf.concat([flo,dep], axis=3) for flo,dep in zip(pred_flows, pred_disp_maps)
                        ]
            elif "v3" in self.version:
                print (">>> [PoseNN] ...... ready to concat : flow + delta depth")
                pred_info = [
                        tf.concat([flo,deltad], axis=3) for flo,deltad in zip(pred_flows, delta_depths)
                        ]
            if "-seglabelid" in self.version:
                print (">>> [PoseNN] ...... + seglabelid (1 channel, id=0 ~ 18)")
                if pred_info is not None:
                    pred_info = [
                            tf.concat([info,seg], axis=3) for info,seg in zip(pred_info,pred_seglabels)
                            ]
                else:
                    pred_info = pred_seglabels

            # 3. Attention Part. (SE_BLOCK)
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
            if ".6" in self.version:
                se_input_flows[0] = pred_flows[1]    # src0->tgt
                se_input_flows[1] = pred_flows[2]    # src1->tgt
                se_input_flows[2] = pred_flows_inv[1]        # tgt->src0
                se_input_flows[3] = pred_flows_inv[2]        # tgt->src1
            elif ".555" in self.version:
                se_input_flows[1] = pred_flows_inv[1]    # tgt->src0
                se_input_flows[2] = pred_flows_inv[2]    # tgt->src1
            if "-sharedNNforward" in self.version:
                se_input_flows[0] = pred_flows_inv[1]    # tgt->src0
                se_input_flows[1] = pred_flows_inv[0]    # zero
                se_input_flows[2] = pred_flows[2]        # src1->tgt
                se_input_flows[3] = pred_flows[0]        # zero
            if "-norm_flow" in self.version:
                se_input_flows = [(f - 0.32140523) / 15.384229 for f in se_input_flows]
                print (">>> [SE][input] Normailed : (flow - flow.mean()) / flow.std()")

            # 3.3. Prepare absolute value of optical flows for se_block's inputs. (h: horizontal , v: vertical)
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
            se_input_depths = [d for d in pred_depths]
            if "-norm_depth" in self.version:
                se_input_depths = [d / 80. for d in se_input_depths]

            # 3.5. Choose se_block's inputs, and generate attentions for PoseNN inputs.
            if "-se_flow_on_depthseg_sharedlayers" in self.version:
                # added at 2020/01/31
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    try:
                        depth_thres = float(re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version).group(1))
                    except AttributeError:
                        depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=15., stddev=0.1), trainable=True)
                        self.depth_thres = depth_thres
                        #depth_thres = 20.
                    print (">>> [PoseNN][se_flow_on_depthseg_sharedlayers] depth threshold = ", depth_thres)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    seg_38 = [ tf.concat([seg_near,seg_far], axis=-1) for seg_near,seg_far in zip(depth_seg_19['near'],depth_seg_19['far']) ] # shape=[B,H,W,38]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,38], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_38[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_38[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_38[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow multiply to near/far segs (i.e., seg_38) :", segmask_tgt)
            elif "-se_flow_on_depthseg_seplayers" in self.version:
                # added at 2020/01/31
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    try:
                        depth_thres = float(re.search("-se_flow_on_depthseg_.*layers_([0-9.]+)", self.version).group(1))
                    except AttributeError:
                        depth_thres = tf.get_variable('se_flow/depth_threshold', shape=(), dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=15., stddev=0.1), trainable=True)
                        self.depth_thres = depth_thres
                        #depth_thres = 20.
                    print (">>> [PoseNN][se_flow_on_depthseg_seplayers] depth threshold = ", depth_thres)
                    near_regions = [ tf.where(_depth < depth_thres, tf.ones_like(_depth), tf.zeros_like(_depth)) for _depth in pred_depths ]    # nearest: depth=5. ; farest: depth=75.
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    depth_seg_19 = {
                            'near': [ _seg*_near for _seg,_near in zip(seg_19,near_regions) ],
                            'far':  [ _seg*(_near-1.)*(-1.) for _seg,_near in zip(seg_19,near_regions) ]
                            }
                    self.depth_seg_19 = depth_seg_19
                    segmask_tgt, segmask_src0, segmask_src1 = 0,0,0
                    for name in ['near', 'far']:
                        flowatt_tgt  = se(se_input_flows[0], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src0 = se(se_input_flows[1], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src1 = se(se_input_flows[2], "se_flow_%s" % name, layer_channels=[8,19], mode='gp', activation=activation_fn)
                        segmask_tgt  += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][0] * flowatt_tgt , axis=-1), -1)
                        segmask_src0 += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 += tf.expand_dims(tf.reduce_sum(depth_seg_19[name][2] * flowatt_src1, axis=-1), -1)
                    #segmask_tgt =  tf.ones_like(segmask_tgt)
                print (">>> [PoseNN] build two SE_flow multiply to near/far segs (i.e., seg_38) seperatly:", segmask_tgt)
            elif "-se_flow_on_depthseg" in self.version:
                raise NameError("please select `-se_flow_on_depthseg_seplayers' or `-se_flow_on_depthseg_sharedlayers'.")
            elif "-se_mixDepthFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[0], se_input_flows[0]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[1], se_input_flows[1]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[2], se_input_flows[2]], axis=-1), "se_depthflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_depth + flow_attention:", segmask_tgt)
            elif "-se_mixDispFlow" in self.version:
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[0], se_input_flows[0]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[1], se_input_flows[1]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([se_input_depths[2], se_input_flows[2]], axis=-1), "se_dispflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_disp + flow_attention:", segmask_tgt)
            elif "-se_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    if ".6" in Version:
                        flowatt_src0 = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # src0->tgt flow
                        flowatt_src1 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # src1->tgt flow
                        flowatt_tgtsrc0  = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)  # tgt->src0 flow
                        flowatt_tgtsrc1  = se(se_input_flows[3], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)  # tgt->src1 flow
                    elif ".55" in Version:
                        flowatt_tgtsrc0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # tgt->src0 flow
                        flowatt_tgtsrc1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)      # tgt->src1 flow
                    else:
                        flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                        flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    if ".65" in Version or ".55" in Version:
                        # (tgt, src0, tgt->src0)
                        print (">>> [PoseNN] segmask_tgt,src0 : use flow_tgt->src0")
                        print (">>> [PoseNN] segmask_tgt,src1 : use flow_tgt->src1")
                        segmask_tgtsrc0 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgtsrc0 , axis=-1), -1)     # for (tgt,src0) pair
                        segmask_tgtsrc1 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgtsrc1 , axis=-1), -1)     # for (tgt,src1) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_tgtsrc0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_tgtsrc1, axis=-1), -1)
                        segmask_tgt = segmask_tgtsrc0
                    elif ".6" in Version:
                        # (tgt, src0, src0->tgt)
                        print (">>> [PoseNN] segmask_tgt,src0 : use flow_src0->tgt")
                        print (">>> [PoseNN] segmask_tgt,src1 : use flow_src1->tgt")
                        segmask_tgtsrc0 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_src0 , axis=-1), -1)     # for (tgt,src0) pair
                        segmask_tgtsrc1 = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_src1 , axis=-1), -1)     # for (tgt,src1) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                        segmask_tgt = segmask_tgtsrc0
                    else:
                        segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)         # for (tgt,src0) pair
                        segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                        segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow :", segmask_tgt)
                if getattr(self, "att_19", None) is not None:
                    flowatt_tgt  = tf.constant(self.att_19[0])
                    flowatt_src0 = tf.constant(self.att_19[1])
                    flowatt_src1 = tf.constant(self.att_19[2])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)         # for (tgt,src0) pair
                    segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Use static attention. >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            elif "-se_gp2x2_flow_nobottle" in self.version:
                # add at 2020/01/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[19,19], mode='gp2x2', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:19, fc2:19):", segmask_tgt)
            elif "-se_gp2x2_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='gp2x2', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_flow with 2x2 global pooling (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp21_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2,1], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow ([2,1]) (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp2_flow" in self.version:   # same to gp2x2
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[2], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow ([2]) (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_spp_flow" in self.version or "-se_spp864_flow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    flowatt_tgt  = se(se_input_flows[0], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    flowatt_src0 = se(se_input_flows[1], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    flowatt_src1 = se(se_input_flows[2], "se_flow", layer_channels=[8,19], mode='spp', spp_size=[8,6,4], activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * flowatt_tgt , axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_tgt)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * flowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * flowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_flow (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_depth_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_depths[0], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_depths[1], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_depths[2], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_depth_to_seg" in self.version:
                # added at 2020/02/25
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_depths[0], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_depths[1], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_depths[2], "se_depth", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_depth_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_depth_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_depth_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_depth" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[0], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_depth", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_depth_attention :", segmask_tgt)
            elif "-se_disp_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                se_input_disps = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_disps[0], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_disps[1], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_disps[2], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_disp_to_seg" in self.version:
                # added at 2020/02/25
                se_input_disps = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_disps[0], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_disps[1], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_disps[2], "se_disp", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_disp_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_disp_wo_tgt" in self.version:
                # added at 2020/02/23
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_disp_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_disp" in self.version:
                # added at 2020/02/23
                se_input_depths = [1./d for d in pred_depths]
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[0], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[1], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(se_input_depths[2], "se_disp", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_disp_attention :", segmask_tgt)
            elif "-se_rgb_wo_tgt_to_seg" in self.version:
                # added at 2020/02/25
                se_input_rgbs = self.input_images
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_rgbs[0], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_rgbs[1], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_rgbs[2], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_rgb_to_seg" in self.version:
                # added at 2020/02/25
                se_input_rgbs = self.input_images
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    att_19_tgt  = se(se_input_rgbs[0], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src0 = se(se_input_rgbs[1], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    att_19_src1 = se(se_input_rgbs[2], "se_rgb", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * att_19_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * att_19_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * att_19_src1, axis=-1), -1)
                    #segmask_tgt  = tf.ones_like(segmask_src0)
                    self.att_19s = [att_19_tgt, att_19_src0, att_19_src1]
                print (">>> [PoseNN] build SE_rgb_attention, and multiply to segmentation:", segmask_tgt)
            elif "-se_rgb_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[1], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[2], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_rgb_frames_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_rgb" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[0], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[1], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(self.input_images[2], "se_rgb", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_rgb_frames_attention :", segmask_tgt)
            elif "-se_seg_wo_tgt" in self.version:
                # added at 2020/02/23
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    #segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build SE_segmentation_attention (only segmask_src0, segmask_src1):", segmask_tgt)
            elif "-se_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation_attention :", segmask_tgt)
            elif "-se_gp2x2_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_block(seg_19[0], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[1], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(seg_19[2], "se_seg", ratio=1, mode='gp2x2', activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation_attention with 2x2 global pooling:", segmask_tgt)
            elif "-se_spp21_seg" in self.version or "-se_spp_seg_21" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2,1]):", segmask_tgt)
            elif "-se_spp2_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[2], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention ([2]):", segmask_tgt)
            elif "-se_spp_seg" in self.version or "-se_spp864_seg" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[0], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[1], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(seg_19[2], "se_spp_seg", ratio=1, spp_size=[8,6,4], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation_attention :", segmask_tgt)
            elif "-se_SegFlow_to_seg_8" in self.version:
                # add at 2020/01/16
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    se_inputs  = [
                            tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                            tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                            tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                            ]
                    segflowatt_tgt  = se(se_inputs[0], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segflowatt_src0 = se(se_inputs[1], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segflowatt_src1 = se(se_inputs[2], "se_segflow", layer_channels=[8,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * segflowatt_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * segflowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * segflowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:8, fc2:19):", segmask_tgt)
            elif "-se_SegFlow_to_seg" in self.version:
                # add at 2020/01/16
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    se_inputs  = [
                            tf.concat([seg_19[0], se_input_flows[0]], axis=-1),
                            tf.concat([seg_19[1], se_input_flows[1]], axis=-1),
                            tf.concat([seg_19[2], se_input_flows[2]], axis=-1)
                            ]
                    segflowatt_tgt  = se(se_inputs[0], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segflowatt_src0 = se(se_inputs[1], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segflowatt_src1 = se(se_inputs[2], "se_segflow", layer_channels=[19,19], mode='gp', activation=activation_fn)
                    segmask_tgt  = tf.expand_dims(tf.reduce_sum(seg_19[0] * segflowatt_tgt , axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(seg_19[1] * segflowatt_src0, axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(seg_19[2] * segflowatt_src1, axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention, and multiply to segmentation (fc1:19, fc2:19):", segmask_tgt)
            elif "-se_mixSegFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[0], se_input_flows[0]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[1], se_input_flows[1]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_block(tf.concat([seg_19[2], se_input_flows[2]], axis=-1), "se_segflow", ratio=1, activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_segmentation + flow_attention:", segmask_tgt)
            elif "-se_spp21_mixSegFlow" in self.version:
                with tf.variable_scope("pose_exp_net", reuse=tf.AUTO_REUSE) as sc:
                    seg_19  = [
                            tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                            ]
                    print (">>> [PoseNN] Transfer pred_seglabels into 19 channels", seg_19[0])
                    segmask_tgt =  tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[0], se_input_flows[0]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src0 = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[1], se_input_flows[1]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                    segmask_src1 = tf.expand_dims(tf.reduce_sum(se_spp_block(tf.concat([seg_19[2], se_input_flows[2]], axis=-1), "se_spp_segflow", ratio=1, spp_size=[2,1], activation=activation_fn), axis=-1), -1)
                print (">>> [PoseNN] build SE_SPP_segmentation + flow_attention ([2,1]):", segmask_tgt)
            elif "-no_segmask" in self.version:
                seg_19  = [
                        tf.squeeze(tf.one_hot(tf.cast(pred_seg, dtype=tf.int32), depth=19, dtype=tf.float32), -2) for pred_seg in pred_seglabels                   # shape=[B,H,W,1,19]
                        ]
                segmask_tgt   = tf.ones_like(seg_19[0])[..., 0:1]
                segmask_src0  = tf.ones_like(seg_19[1])[..., 0:1]
                segmask_src1  = tf.ones_like(seg_19[2])[..., 0:1]
                print (">>> [PoseNN] No use attention module.")
            elif "-segmask_" in self.version and "-static" in self.version:
                segmask_src0, self.segweights = build_seg_channel_weight(pred_seglabels[1], prefix="pose_exp_net/")
                segmask_src1, _ = build_seg_channel_weight(pred_seglabels[2], prefix="pose_exp_net/")
                segmask_tgt = tf.ones_like(segmask_src0)
                print (">>> [PoseNN] build weighted segmentation_attention (only segmask_src0, segmask_src1):", segmask_src0)
            else:
                segmask_tgt, self.segweights  = build_seg_channel_weight(pred_seglabels[0], prefix="pose_exp_net/")
                segmask_src0, _ = build_seg_channel_weight(pred_seglabels[1], prefix="pose_exp_net/")
                segmask_src1, _ = build_seg_channel_weight(pred_seglabels[2], prefix="pose_exp_net/")
                print (">>> [PoseNN] build weighted segmentation_attention :", segmask_tgt)

            # 4. PoseNN Part.
            # 4.1. Generate masked inputs.
            use_se_flow = len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pose_exp_net/se_flow'))
            if ".55" in Version:
                print (">>> [PoseNN] vx.55 Version : segmask_tgt != 1")
            else:
                if use_se_flow:
                    print (">>> [PoseNN] se_flow mode.")
                    if "-sharedNNforward" in self.version:
                        print (">>> [PoseNN] segmask_src0 ----> 1")
                        segmask_src0    = tf.ones_like(segmask_src0)
                        segmask_tgtsrc1 = tf.ones_like(segmask_tgt)
                    else:
                        print (">>> [PoseNN] segmask_tgt ----> 1")
                        segmask_tgt     = tf.ones_like(segmask_tgt)
                        segmask_tgtsrc1 = tf.ones_like(segmask_tgt)
                else:
                    segmask_tgtsrc1 = segmask_tgt
            if pred_info is not None:
                print (">>> [PoseNN] concat pred_info now")
                if "-segmask_" in self.version:
                    print (">>> ======> Segmentation Attention [ON]")
                    self.input_images[0] = tf.multiply(self.input_images[0],  segmask_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1],  segmask_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2],  segmask_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3],  segmask_tgtsrc1)
                    if "-segmask_all" in self.version and ".555" in Version:
                        print (">>> [PoseNN] use segmask_all dilation network (src0, src1 concating flows = 0)")
                        pred_info[0] = tf.multiply(pred_info[0],  segmask_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  tf.ones_like(segmask_src0))
                        pred_info[2] = tf.multiply(pred_info[2],  tf.ones_like(segmask_src1))
                        pred_info[3] = tf.multiply(pred_info[3],  segmask_tgtsrc1)
                    elif "-segmask_all" in self.version:
                        print (">>> [PoseNN] use segmask_all dilation network")
                        pred_info[0] = tf.multiply(pred_info[0],  segmask_tgt)
                        pred_info[1] = tf.multiply(pred_info[1],  segmask_src0)
                        pred_info[2] = tf.multiply(pred_info[2],  segmask_src1)
                        pred_info[3] = tf.multiply(pred_info[3],  segmask_tgtsrc1)
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
                    self.input_images[0] = tf.multiply(self.input_images[0], segmask_tgt)
                    self.input_images[1] = tf.multiply(self.input_images[1], segmask_src0)
                    self.input_images[2] = tf.multiply(self.input_images[2], segmask_src1)
                    self.input_images[3] = tf.multiply(self.input_images[3], segmask_tgtsrc1)
            #if pred_info is not None:
            #    print (">>> [PoseNN] concat pred_info now")
            #    if "-segmask_all" in self.version:
            #        print (">>> ======> Segmentation Attention [ON]")
            #        print (">>> [PoseNN] use segmask_all dilation network")
            #        self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
            #        self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
            #        self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
            #        self.input_images[0] = tf.multiply(self.input_images[0],  segmask_tgt)
            #        self.input_images[1] = tf.multiply(self.input_images[1],  segmask_src0)
            #        self.input_images[2] = tf.multiply(self.input_images[2],  segmask_src1)
            #    elif "-segmask_rgb" in self.version:
            #        print (">>> ======> Segmentation Attention [ON]")
            #        print (">>> [PoseNN] use segmask_rgb dilation network")
            #        self.input_images[0] = tf.multiply(self.input_images[0], segmask_tgt)
            #        self.input_images[1] = tf.multiply(self.input_images[1], segmask_src0)
            #        self.input_images[2] = tf.multiply(self.input_images[2], segmask_src1)
            #        self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
            #        self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
            #        self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
            #    else:
            #        print (">>> ======> Segmentation Attention [OFF]")
            #        self.input_images[0] = tf.concat([self.input_images[0],  pred_info[0]], axis=3)
            #        self.input_images[1] = tf.concat([self.input_images[1],  pred_info[1]], axis=3)
            #        self.input_images[2] = tf.concat([self.input_images[2],  pred_info[2]], axis=3)
            #else:
            #    if "-segmask" in self.version:
            #        print (">>> ======> Segmentation Attention [ON]")
            #        print (">>> [PoseNN] use segmask_rgb dilation network")
            #        self.input_images[0] = tf.multiply(self.input_images[0], segmask_tgt)
            #        self.input_images[1] = tf.multiply(self.input_images[1], segmask_src0)
            #        self.input_images[2] = tf.multiply(self.input_images[2], segmask_src1)

            # 4.2. Predict ego-motion.
            poseNet = partial(poseNet, dropout=False, is_training=False, se_attention=se_attention, batch_norm="-batch_norm" in self.version, cnv6_num_outputs=cnv6_num_outputs) # -2
            if "-sharedNNinv" in self.version: # inverse pose
                # add at 2020/01/16
                print (">>> ===== [ SharedNNinv PoseNN (%s) ] =====" % Version)
                #pred_pose0, _ = poseNet(self.input_images[1], self.input_images[0])
                #pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[0], self.input_images[2])
                #pred_poses = tf.concat([get_inv_pose(pred_pose0), pred_pose1], axis=-2)
                raise NameError("`-sharedNNinv' is unavailable.")
            elif "-sharedNNforward" in self.version: # all forwarding poses
                # add at 2020/01/22
                print (">>> ===== [ SharedNNforward PoseNN (%s) ] =====" % Version)
                if ".5" in Version or ".6" in Version:
                    pred_pose0, _ = poseNet(self.input_images[1][...,0:3], self.input_images[0]) #src0->tgt
                    pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3][...,0:3], self.input_images[2]) #tgt->src1
                else:
                    pred_pose0, _ = poseNet(self.input_images[1], self.input_images[0]) #src0->tgt
                    pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            elif "-sharedNN" in self.version:
                # add at 2020/01/16
                print (">>> ===== [ SharedNN PoseNN (%s) ] =====" % Version)
                if ".555" in Version:
                    pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1][...,0:3]) #tgt->src0
                    pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3], self.input_images[2][...,0:3]) #tgt->src1
                elif ".5" in Version or ".6" in Version:
                    pred_pose0, _ = poseNet(self.input_images[0][...,0:3], self.input_images[1]) #tgt->src0
                    pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3][...,0:3], self.input_images[2]) #tgt->src1
                else:
                    pred_pose0, _ = poseNet(self.input_images[0], self.input_images[1]) #tgt->src0
                    pred_pose1, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[3], self.input_images[2]) #tgt->src1
                pred_poses = tf.concat([pred_pose0, pred_pose1], axis=-2)
            else:
                pred_poses, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[0], self.input_images[1], self.input_images[2])
            print (">>> [PoseNN] pred_poses: ", pred_poses)

            self.rot_cnv6 = tf.image.resize_bilinear(rot_cnv6, self.input_images[0].shape[1:3])
            self.trans_cnv6 = tf.image.resize_bilinear(trans_cnv6, self.input_images[0].shape[1:3])
            self.features = {'rot': self.rot_cnv6, 'trans': self.trans_cnv6}
            #if "-dilatedPoseNN" in self.version:
                #pred_poses, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[0], self.input_images[1], self.input_images[2], is_training=False, se_attention=se_attention) # -2
                #self.rot_cnv6 = tf.image.resize_bilinear(rot_cnv6, self.input_images[0].shape[1:3])
                #self.trans_cnv6 = tf.image.resize_bilinear(trans_cnv6, self.input_images[0].shape[1:3])
                #self.features = {'rot': self.rot_cnv6, 'trans': self.trans_cnv6}
            #else:
                #pred_poses, (rot_cnv6, trans_cnv6) = poseNet(self.input_images[0], self.input_images[1], self.input_images[2], is_training=False, se_attention=se_attention) # -2
                #self.rot_cnv6 = tf.image.resize_bilinear(rot_cnv6, self.input_images[0].shape[1:3])
                #self.trans_cnv6 = tf.image.resize_bilinear(trans_cnv6, self.input_images[0].shape[1:3])
                #self.features = {'rot': self.rot_cnv6, 'trans': self.trans_cnv6}


            # Visualization in tf.summary.
            self.segmask = [segmask_tgt, segmask_src0, segmask_src1]
            if "-segmask" in self.version and "flow" in self.version:
                try:
                    self.att_19 = [flowatt_tgt, flowatt_src0, flowatt_src1]
                except:
                    if ".55" in Version:
                        self.att_19 =  [flowatt_tgtsrc0, flowatt_tgtsrc0, flowatt_tgtsrc1, flowatt_tgtsrc1]
                    else:
                        self.att_19 =  [flowatt_tgtsrc0, flowatt_src0, flowatt_src1, flowatt_tgtsrc1]
            elif "-segmask" in self.version and getattr(self, "segweights", False) is not False:
                att_19 = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.segweights, 0), 0), 0)
                self.att_19 = [att_19, att_19, att_19]
            elif "-segmask" in self.version and getattr(self, "att_19s", False) is not False:
                self.att_19 = self.att_19s
            else:
                self.att_19 = []
            try:
                self.seg_19 = seg_19
            except:
                self.seg_19 = []
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
                        self.convert_to_tf_image( flow_to_image( self.input_images[2][...,3:5]))  # src1_masked_flows 
                        ]
                    if pred_info[0].shape.as_list()[-1] > 2:
                        if "v3" in self.version:
                            zeros = tf.zeros_like(pred_info[0][...,2:3])
                            self.masked_deltadepths  = [
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[0][...,5:6]], axis=-1))),   # tgt_masked_deltadepths   
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[1][...,5:6]], axis=-1))),   # src0_masked_deltadepths 
                                    self.convert_to_tf_image( flow_to_image( tf.concat([zeros, self.input_images[2][...,5:6]], axis=-1)))   # src1_masked_deltadepths 
                                ]
                        else:
                            self.masked_depths  = [
                                    self.input_images[0][...,5:6],  # tgt_masked_depths   
                                    self.input_images[1][...,5:6],  # src0_masked_depths 
                                    self.input_images[2][...,5:6]   # src1_masked_depths 
                                ]
            self.masked_seglabels  = [
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[0], tf.float32), self.segmask[0])),   # tgt_masked_seglabels   
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[1], tf.float32), self.segmask[1])),   # src0_masked_seglabels 
                self.convert_to_tf_image( tf.multiply(tf.cast(self.pred_seglabels_color[2], tf.float32), self.segmask[2]))   # src1_masked_seglabels 
                ]
            print (">>> [PoseNN] pred_poses: ", pred_poses)

        self.pred_poses = pred_poses
        self.masks = {
                'image': self.masked_images,
                'flow': self.masked_flows,
                'delta_depth': getattr(self, "masked_deltadepths", self.masked_images),
                'seglabel': self.masked_seglabels,
                'attention': self.segmask,
                'att_19': self.att_19
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
        if self.mode == 'depth':
            self.build_depth_test_graph(input_img_uint8)
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph(input_img_uint8)
        if self.mode == 'davo':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph_davo(input_img_uint8, input_flow, input_depth, input_seglabel)

        #if self.mode == 'decouplevo_pose':
            #self.seq_length = seq_length
            #self.num_source = seq_length - 1
            #self.build_pose_test_graph(input_img_uint8)


    def inference(self, sess, mode, inputs=None):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        if mode == 'segatten':
            fetches['pose'] = self.pred_poses
            fetches['masks'] = self.masks
        if mode == 'feature':
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

