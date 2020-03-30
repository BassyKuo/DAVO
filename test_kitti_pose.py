from __future__ import division
import os
import re
import math
import cv2
import scipy.misc
import tensorflow as tf
import numpy as np
from utils import geo_utils
from glob import glob
from dcvo import DCVO
from data_loader import DataLoader
from utils.common_utils import complete_batch_size, is_valid_sample
from utils.seg_utils.labels import seg_labels

_OUTPUT_FEATUREMAP = False
_OUTPUT_NEW_MERGE  = False
_OUTPUT_ATTENTION  = False

_USE_FLIPPING_IMAGES = False

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("dataset_dir", None, "Raw odometry dataset directory")
flags.DEFINE_string("concat_img_dir", None, "Preprocess image dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("version", "v1", "version")
FLAGS = flags.FLAGS

np.set_printoptions(precision=4, suppress=True)

def load_kitti_image_sequence_names(dataset_dir, frames, seq_length, load_pose=False, load_flow=False, load_depth=False, load_seglabel=False):
    image_sequence_names = []
    target_inds = []
    image_sequence_poses = []
    image_sequence_flows = []
    image_sequence_depths = []
    image_sequence_seglabels = []
    frame_num = len(frames)
    for tgt_idx in range(frame_num):
        if not is_valid_sample(frames, tgt_idx, seq_length):
            continue
        curr_drive, curr_frame_id = frames[tgt_idx].split(' ')
        img_filename = os.path.join(dataset_dir, '%s/%s.jpg' % (curr_drive, curr_frame_id))
        img_posename = os.path.join(dataset_dir, '%s/%s_cam.txt' % (curr_drive, curr_frame_id))
        #img_flowname = os.path.join(dataset_dir, '%s/%s-flow.npy' % (curr_drive, curr_frame_id))
        img_flowname = os.path.join(dataset_dir, '%s/%s-flownet2.npy' % (curr_drive, curr_frame_id)) # shape=(4,h,w,2)
        #img_flowname = os.path.join(dataset_dir, '%s/%s-pwcflow.npy' % (curr_drive, curr_frame_id)) # shape=(4,h,w,2)
        #img_flowname = os.path.join(dataset_dir, '%s/%s-pwcflow_dump.npy' % (curr_drive, curr_frame_id)) # shape=(4,h,w,2)
        #img_depthname = os.path.join(dataset_dir, '%s/%s-depth.npy' % (curr_drive, curr_frame_id))
        #img_depthname = os.path.join(dataset_dir, '%s/%s-monodepth2_disp.npy' % (curr_drive, curr_frame_id))
        img_depthname = os.path.join(dataset_dir, '%s/%s-monodepth2_depth.npy' % (curr_drive, curr_frame_id))
        img_seglabelname = os.path.join(dataset_dir, '%s/%s-seglabel.npy' % (curr_drive, curr_frame_id))
        image_sequence_names.append(img_filename)
        image_sequence_poses.append(img_posename)
        image_sequence_flows.append(img_flowname)
        image_sequence_depths.append(img_depthname)
        image_sequence_seglabels.append(img_seglabelname)
        target_inds.append(tgt_idx)

    if load_seglabel:
        seglabel = image_sequence_seglabels
    else:
        seglabel = image_sequence_names
    if load_depth:
        depth = image_sequence_depths
    else:
        depth = image_sequence_names
    if load_flow:
        flow = image_sequence_flows
    else:
        flow = image_sequence_names
    if load_pose:
        return image_sequence_names, target_inds, image_sequence_poses, flow, depth, seglabel
    else:
        return image_sequence_names, target_inds, image_sequence_names, flow, depth, seglabel


def main():
    # get input images
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
    max_src_offset = int((FLAGS.seq_length - 1)/2)
    N = len(glob(concat_img_dir + '/*.jpg')) + 2*max_src_offset
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]

    with open(FLAGS.dataset_dir + 'sequences/%.2d/times.txt' % FLAGS.test_seq, 'r') as f:
        times = f.readlines()
    times = np.array([float(s[:-1]) for s in times])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # setup input tensor
        read_flow = True
        read_depth = True
        read_seglabel = True
        loader = DataLoader(FLAGS.concat_img_dir, FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width, FLAGS.seq_length-1, read_flow=read_flow, read_depth=read_depth, read_seglabel=read_seglabel)
        image_sequence_names, tgt_inds, image_sequence_poses, image_sequence_flows, image_sequence_depths, image_sequence_seglabels = \
                load_kitti_image_sequence_names(FLAGS.concat_img_dir, test_frames, FLAGS.seq_length, load_pose=True, load_flow=read_flow, load_depth=read_depth, load_seglabel=read_seglabel)
        image_sequence_names = complete_batch_size(image_sequence_names, FLAGS.batch_size)
        image_sequence_poses = complete_batch_size(image_sequence_poses, FLAGS.batch_size)
        image_sequence_flows = complete_batch_size(image_sequence_flows, FLAGS.batch_size)
        image_sequence_depths = complete_batch_size(image_sequence_depths, FLAGS.batch_size)
        image_sequence_seglabels = complete_batch_size(image_sequence_seglabels, FLAGS.batch_size)
        tgt_inds = complete_batch_size(tgt_inds, FLAGS.batch_size)
        assert len(tgt_inds) == len(image_sequence_names)
        batch_sample = loader.load_test_batch_flow(image_sequence_names, image_sequence_poses, image_sequence_flows, image_sequence_depths, image_sequence_seglabels)
        sess.run(batch_sample.initializer)
        inputs_batch = batch_sample.get_next()
        input_batch = inputs_batch[0]
        input_pose  = inputs_batch[1]
        input_flow  = inputs_batch[2]
        input_depth = inputs_batch[3]
        input_seglabel = inputs_batch[4]
        input_batch.set_shape([FLAGS.batch_size, FLAGS.img_height, FLAGS.img_width * FLAGS.seq_length, 3])
        input_pose.set_shape([FLAGS.batch_size, FLAGS.seq_length, 6])
        #input_flow.set_shape([FLAGS.batch_size, FLAGS.seq_length-1, FLAGS.img_height, FLAGS.img_width, 2])
        input_flow.set_shape([FLAGS.batch_size, (FLAGS.seq_length-1)*2, FLAGS.img_height, FLAGS.img_width, 2])
        input_depth.set_shape([FLAGS.batch_size, FLAGS.seq_length, FLAGS.img_height, FLAGS.img_width, 1])
        input_seglabel.set_shape([FLAGS.batch_size, FLAGS.seq_length, FLAGS.img_height, FLAGS.img_width, 1])

        # Data flip.
        if _USE_FLIPPING_IMAGES:
            input_batch    = loader.img_flip(input_batch)
            input_pose     = loader.pose_flip(input_pose)
            input_flow     = loader.npy_flip(input_flow)
            input_depth    = loader.npy_flip(input_depth)
            input_seglabel = loader.npy_flip(input_seglabel)

        name = os.path.basename( os.path.dirname(FLAGS.output_dir + "/") )
        featuremap_root = os.path.dirname(os.path.dirname(FLAGS.output_dir + "/")) + "/featuremaps"
        if _USE_FLIPPING_IMAGES:
            featuremap_root = featuremap_root + "-flip"
        featuremap_output_dir = "%s/%s/%02d-featuremaps/" % (featuremap_root, name, int(FLAGS.test_seq))
        featuremap_merge_output_dir = "%s/%s/%02d-merged-featuremaps/" % (featuremap_root, name, int(FLAGS.test_seq))
        img_output_dir = "%s/%s/%02d-segmasks/" % (featuremap_root, name, int(FLAGS.test_seq))
        att_root = os.path.dirname(os.path.dirname(FLAGS.output_dir + "/")) + "/attentions"
        att_output_dir = "%s/%s/%02d-attentions/" % (att_root, name, int(FLAGS.test_seq))
        new_merge_root = os.path.dirname(os.path.dirname(FLAGS.output_dir + "/")) + "/new_merge"
        new_merge_output_dir = "%s/%s/%02d-tgtsrc1/" % (new_merge_root, name, int(FLAGS.test_seq))
        if _OUTPUT_FEATUREMAP:
            if not os.path.isdir(featuremap_root):
                os.makedirs(featuremap_root)
            if not os.path.isdir(img_output_dir):
                os.makedirs(img_output_dir)
            if not os.path.isdir(featuremap_output_dir):
                os.makedirs(featuremap_output_dir)
            if not os.path.isdir(featuremap_merge_output_dir):
                os.makedirs(featuremap_merge_output_dir)
        if _OUTPUT_NEW_MERGE:
            if not os.path.isdir(new_merge_output_dir):
                os.makedirs(new_merge_output_dir)
        if _OUTPUT_ATTENTION:
            if not os.path.isdir(att_output_dir):
                os.makedirs(att_output_dir)
            _file1 = open("%s/%s/%02d-tgtsrc0.txt" % (att_root, name, int(FLAGS.test_seq)), "w")
            _file2 = open("%s/%s/%02d-tgtsrc1.txt" % (att_root, name, int(FLAGS.test_seq)), "w")
            _file1.write("id,%s\n" % ",".join(seg_labels)) # title
            _file2.write("id,%s\n" % ",".join(seg_labels)) # title


        kitti_out_file = os.path.join(FLAGS.output_dir, '%.2d-pred_kitti_pose.txt' % FLAGS.test_seq)
        kitti_out_file_r = os.path.join(FLAGS.output_dir, '%.2d-pred_kitti_rel_pose.txt' % FLAGS.test_seq)
        if os.path.isfile(kitti_out_file):
            os.remove(kitti_out_file)
        prev_pose = np.eye(4).astype(float)
        recover_pose = [prev_pose]
        pred_pose_list = []
        pose_vec_ph = tf.placeholder(tf.float32, [3,6])
        pose_mat_tensor = geo_utils.pose_vec2mat(pose_vec_ph)

        # init system
        system = DCVO(version=FLAGS.version)
        system.setup_inference(FLAGS.img_height, FLAGS.img_width,
                               "dcvo", FLAGS.seq_length, FLAGS.batch_size, input_batch, input_flow=input_flow, input_depth=input_depth, input_seglabel=input_seglabel)
        saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
        #sess.run(tf.global_variables_initializer())

        #saver = tf.train.Saver([var for var in tf.model_variables()]) 

        saver.restore(sess, FLAGS.ckpt_file)

        round_num = len(image_sequence_names) // FLAGS.batch_size
        for i in range(round_num):
            #pred = system.inference(sess, mode='pose')      # len(pred['pose']) = B
            #pred = system.inference(sess, mode='segatten')      # len(pred['pose']) = B
            pred = system.inference(sess, mode='feature')      # len(pred['pose']) = B
            for j in range(FLAGS.batch_size):
                tgt_idx = tgt_inds[i * FLAGS.batch_size + j]
                pred_poses = pred['pose'][j]                # pred['pose'].shape=[B,6]. pred_poses.shape=[2,6]
                if pred['masks']['att_19'] is not []:
                    att_19 = pred['masks']['att_19']            # shape=(B,1,1,19)
                    if _OUTPUT_ATTENTION:
                        #np.save( att_output_dir + "/%.6d-attention_19.npy" % (tgt_idx - max_src_offset+1), np.stack([att[j] for att in att_19], 0) ) # tgt, src0, src1
                        _file1.write("%06d,%s\n" % ((tgt_idx - max_src_offset+1), ",".join([str(att) for att in att_19[1][0,0,0]])))     # tgtsrc0
                        _file2.write("%06d,%s\n" % ((tgt_idx - max_src_offset+1), ",".join([str(att) for att in att_19[2][0,0,0]])))     # tgtsrc1

                if 'features' in pred and (_OUTPUT_FEATUREMAP or _OUTPUT_NEW_MERGE) and ( \
                        #(((tgt_idx - max_src_offset+1) > 35 and (tgt_idx - max_src_offset) < 131) and (int(FLAGS.test_seq) == 0)) or \
                        #(((tgt_idx - max_src_offset+1) == 512 or (tgt_idx - max_src_offset) == 1585) and (int(FLAGS.test_seq) == 2)) or \
                        ((tgt_idx - max_src_offset+1) >= 2300 and (tgt_idx - max_src_offset+1) <= 2500 and (int(FLAGS.test_seq) == 5)) or \
                        #((tgt_idx - max_src_offset+1) >= 350 and (int(FLAGS.test_seq) == 7)) or \
                        ((tgt_idx - max_src_offset+1) >= 519 and (tgt_idx - max_src_offset+1) < 521 and (int(FLAGS.test_seq) == 7)) or \
                        ((tgt_idx - max_src_offset+1) // 1000 % 2 == 1 and (int(FLAGS.test_seq) == 10)) or \
                        (((tgt_idx - max_src_offset+1) // 1000 % 2 == 0 and (int(FLAGS.test_seq) == 8 or int(FLAGS.test_seq) == 3 or int(FLAGS.test_seq) == 6)) or \
                        (int(FLAGS.test_seq) == 22 or int(FLAGS.test_seq) == 16 or int(FLAGS.test_seq) == 11 or int(FLAGS.test_seq) == 15)) \
                        ):
                    features = pred['features']                # rot_cnv6, trans_cnv6
                    rot_cnv6 = features['rot'][j]                 # [h,w,256]
                    trans_cnv6 = features['trans'][j]             # [h,w,256]
                    images = [img[j] for img in pred['images']]   # r,g,b
                    def color_map(feature, maximum=None):
                        assert len(feature.shape) == 3 and feature.shape[-1] == 1
                        #print (feature.shape, feature.max(), feature.min(), feature.mean())
                        if maximum is not None:
                            feature = feature / maximum * 255
                            #print (">>>>", feature.shape, feature.max(), feature.min(), feature.mean())
                        feature_img = feature.astype(np.uint8)
                        colormap = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
                        return colormap

                    def overlap(img, img2, alpha=0.3):
                        assert img.shape[2] == 3 and img2.shape[2] == 3
                        if img.min() < 0:
                            img = (img + 1) / 2 * 255
                            img = img.astype(np.uint8)
                        if img2.min() < 0:
                            img2 = (img2 + 1) / 2 * 255
                            img2 = img2.astype(np.uint8)
                        beta = 1-alpha
                        gamma = 0
                        img_add = cv2.addWeighted(img, alpha, img2, beta, gamma)
                        return img_add

                    def generate_mask(img, colormap, alpha=0.3):
                        assert img.shape[2] == 3
                        assert colormap.shape[2] == 3
                        if img.min() < 0:
                            img = (img + 1) / 2 * 255
                            img = img.astype(np.uint8)
                        img = img[:,:,::-1] # r,g,b -> b,g,r
                        beta = 1-alpha
                        gamma = 0
                        img_add = cv2.addWeighted(img, alpha, colormap, beta, gamma)
                        output = np.vstack((colormap, img, img_add))
                        return output

                    # Merge src + tgt images.
                    tgt_merge_src0 = overlap(images[1], images[0], 0.8)
                    tgt_merge_src1 = overlap(images[2], images[0], 0.8)

                    # RotNN feature
                    r_feature_color = color_map(np.mean(rot_cnv6, axis=-1)[..., np.newaxis], rot_cnv6.max())
                    if _OUTPUT_FEATUREMAP:
                        cv2.imwrite(featuremap_output_dir + '%.6d-rot_feature-avg.png' % (tgt_idx - max_src_offset + 1), r_feature_color)
                    r_feature_color = color_map(np.sum(rot_cnv6, axis=-1)[..., np.newaxis], np.sum(rot_cnv6, axis=-1).max())
                    if _OUTPUT_FEATUREMAP:
                        cv2.imwrite(featuremap_output_dir + '%.6d-rot_feature-sum.png' % (tgt_idx - max_src_offset + 1), r_feature_color)
                    rot_merge = generate_mask(images[1], r_feature_color)
                    src0_rot_merge = generate_mask(tgt_merge_src0, r_feature_color)
                    src1_rot_merge = generate_mask(tgt_merge_src1, r_feature_color)

                    # TransNN feature
                    t_feature_color = color_map(np.mean(trans_cnv6, axis=-1)[..., np.newaxis], trans_cnv6.max())
                    if _OUTPUT_FEATUREMAP:
                        cv2.imwrite(featuremap_output_dir + '%.6d-trans_feature-avg.png' % (tgt_idx - max_src_offset + 1), t_feature_color)
                    t_feature_color = color_map(np.sum(trans_cnv6, axis=-1)[..., np.newaxis], np.sum(trans_cnv6, axis=-1).max())
                    if _OUTPUT_FEATUREMAP:
                        cv2.imwrite(featuremap_output_dir + '%.6d-trans_feature-sum.png' % (tgt_idx - max_src_offset + 1), t_feature_color)
                    trans_merge = generate_mask(images[1], t_feature_color)
                    src0_trans_merge = generate_mask(tgt_merge_src0, t_feature_color)
                    src1_trans_merge = generate_mask(tgt_merge_src1, t_feature_color)

                    # Merge.
                    orig_merge = np.concatenate([trans_merge, rot_merge], axis=1)
                    src0_merge = np.concatenate([src0_trans_merge, src0_rot_merge], axis=1)
                    src1_merge = np.concatenate([src1_trans_merge, src1_rot_merge], axis=1)

                    # Masked images
                    masks = pred['masks']                # 'image', 'flow', 'delta_depth', 'attention'
                    mask_img = [img[j] for img in masks['image']]
                    if _OUTPUT_FEATUREMAP:
                        scipy.misc.imsave(img_output_dir + '%.6d-masked_images.jpg' % (tgt_idx - max_src_offset + 1),       np.concatenate([masks['image'][idx] for idx in [1,0,2]], axis=2)[j])
                        scipy.misc.imsave(img_output_dir + '%.6d-masked_flows.jpg' % (tgt_idx - max_src_offset + 1),        np.concatenate([masks['flow'][idx] for idx in [1,0,2]], axis=2)[j])
                        scipy.misc.imsave(img_output_dir + '%.6d-masked_seglabels.jpg' % (tgt_idx - max_src_offset + 1),    np.concatenate([masks['seglabel'][idx] for idx in [1,0,2]], axis=2)[j])
                    #if i == 0:
                        #scale = int(0.4 / masks['attention'][1].mean())
                        #if masks['attention'][1].max() > 0.01 and masks['attention'][1].mean() > 0.01:
                            #scale = 10
                        #else:
                            #scale = 150

                    try:
                        attention255 = [np.minimum( 255, (255 * scale * masks['attention'][idx])).astype(np.uint8) for idx in [0,1,2]]
                    except:
                        scale = int(0.4 / masks['attention'][2].mean())
                        attention255 = [np.minimum( 255, (255 * scale * masks['attention'][idx])).astype(np.uint8) for idx in [0,1,2]]
                    # print (attention255[0].shape) # 1,128,416,1
                    print (i, scale, masks['attention'][2].max(), masks['attention'][2].min(), masks['attention'][2].mean(), "\t\t", attention255[2].max(), attention255[2].min(), attention255[2].mean())
                    #assert attention255[1].max() <= 255 and attention255[1].min() >= 0
                    if _OUTPUT_FEATUREMAP:
                        cv2.imwrite(img_output_dir + '%.6d-attentions.png' % (tgt_idx - max_src_offset + 1),                np.concatenate([attention255[idx] for idx in [1,0,2]], axis=2)[j])

                    #mask_src0 = (images[1] + 1) / 2.        # 0 ~ 1
                    #mask_src1 = (images[2] + 1) / 2.        # 0 ~ 1
                    #mask_src0 = mask_src0 * np.log( attention255[1][j] ) / np.log(255)
                    #mask_src1 = mask_src1 * np.log( attention255[2][j] ) / np.log(255)
                    #mask_src0 = mask_src0 * 2. - 1.
                    #mask_src1 = mask_src1 * 2. - 1.
                    src0_att = np.tanh(((attention255[1][j]-attention255[1].min()) / np.max(attention255[1]-attention255[1].min()) - 0.5) * (np.pi * 2.5)) / 2. + 0.5
                    src1_att = np.tanh(((attention255[2][j]-attention255[2].min()) / np.max(attention255[2]-attention255[2].min()) - 0.5) * (np.pi * 2.5)) / 2. + 0.5
                    src0_att *= 255
                    src1_att *= 255
                    mask_src0 = generate_mask(images[1], np.tile(src0_att.astype(np.uint8), [1,1,3]))
                    mask_src1 = generate_mask(images[2], np.tile(src1_att.astype(np.uint8), [1,1,3]))
                    h,w,c = mask_src0.shape
                    mask_src0 = mask_src0[h//3*2:]
                    mask_src1 = mask_src1[h//3*2:]
                    #print (attention255[1].max(), attention255[1].min(), src0_att.max(), src0_att.min(), mask_src0.shape, mask_src0.max(), mask_src0.min())
                    #print (attention255[2].max(), attention255[2].min(), src1_att.max(), src1_att.min(), mask_src1.shape, mask_src1.max(), mask_src1.min())
                    mask_src0_rot_merge = generate_mask(mask_src0, r_feature_color, 0.5)
                    mask_src1_rot_merge = generate_mask(mask_src1, r_feature_color, 0.5)
                    mask_src0_trans_merge = generate_mask(mask_src0, t_feature_color, 0.5)
                    mask_src1_trans_merge = generate_mask(mask_src1, t_feature_color, 0.5)
                    mask_src0_merge = np.concatenate([mask_src0_trans_merge, mask_src0_rot_merge], axis=1)
                    mask_src1_merge = np.concatenate([mask_src1_trans_merge, mask_src1_rot_merge], axis=1)

                    #if "-se_flow" in FLAGS.version or "-se_spp_flow" in FLAGS.version or "-se_mixSegFlow" in FLAGS.version:
                    #if re.search("-se_.*[Ff]low", FLAGS.version):
                    if True:
                        flows = [f[j] for f in pred['flows']]
                        masked_flows = masks['flow'][1:]
                        src0_attention = np.tile(attention255[1][j], [1,2,3])
                        src1_attention = np.tile(attention255[2][j], [1,2,3])
                        src0_flow = np.tile(flows[0][:,:,::-1], [1,2,1])        # RGB -> BGR
                        src1_flow = np.tile(flows[1][:,:,::-1], [1,2,1])
                        src0_masked_flow = np.tile(masked_flows[0][j][:,:,::-1], [1,2,1])
                        src1_masked_flow = np.tile(masked_flows[1][j][:,:,::-1], [1,2,1])
                        src0_merge_all = np.concatenate([src0_merge, src0_attention, src0_flow, src0_masked_flow], axis=0)
                        src1_merge_all = np.concatenate([src1_merge, src1_attention, src1_flow, src1_masked_flow], axis=0)
                        vsplit = np.zeros([src0_merge_all.shape[0],10,3])
                        merge = np.hstack([src0_merge_all, vsplit, src1_merge_all])
                        if _OUTPUT_FEATUREMAP:
                            cv2.imwrite(featuremap_merge_output_dir + 'MERGE--%.6d-transL_rotR_feature.png' % (tgt_idx - max_src_offset + 1), merge)
                    else:
                        tgt_attention = np.tile(attention255[0][j], [1,2,3])
                        merge = np.concatenate([orig_merge, tgt_attention], axis=0)
                        cv2.imwrite(featuremap_merge_output_dir + 'MERGE--%.6d-transL_rotR_feature.png' % (tgt_idx - max_src_offset + 1), merge)

                    if _OUTPUT_NEW_MERGE:
                        segs = pred['segs']
                        tgt_seg, src0_seg, src1_seg = [ np.tile(seg[j][:,:,::-1], [1,2,1]) for seg in segs ]        # RGB->BGR
                        h = src1_merge.shape[0]
                        w = src1_merge.shape[1]
                        hsplit = np.zeros([5,src1_merge.shape[1],3])
                        src1_merge_all = np.concatenate([src1_merge[h//3:h//3*2], src1_flow, src1_seg, src1_attention], axis=0)
                        #src1_merge_all = np.concatenate([src1_merge[h//3:h//3*2], src1_seg, src1_flow, hsplit, src1_attention, src1_merge[h//3*2:]], axis=0)
                        if int(FLAGS.test_seq) == 2 or int(FLAGS.test_seq) == 0:
                            out_fig = np.concatenate([orig_merge[h//3:h//3*2], src0_seg, src0_flow, src0_attention], axis=0)
                            out_fig = out_fig[:,:out_fig.shape[1]//2,:]
                            cv2.imwrite(new_merge_output_dir + 'tgtsrc0-%.6d-figure.png' % (tgt_idx - max_src_offset + 1), out_fig)
                        src1_one_col = src1_merge_all[:,:src1_merge_all.shape[1]//2,:]
                        h = src1_seg.shape[0]
                        #cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-figure.png' % (tgt_idx - max_src_offset + 1), src1_one_col)
                        # cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-figure-rgb.png' % (tgt_idx - max_src_offset + 1), src1_one_col[:h])
                        # cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-figure-flow.png' % (tgt_idx - max_src_offset + 1), src1_one_col[h:h*2])
                        # cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-figure-seg.png' % (tgt_idx - max_src_offset + 1), src1_one_col[h*2:h*3])
                        # cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-figure-attention.png' % (tgt_idx - max_src_offset + 1), src1_one_col[h*3:])

                        # Output Segmentation.
                        tgt_seg_19, src0_seg_19, src1_seg_19 = [seg[j] for seg in pred['seg_19']]
                        for labelid in range(19):
                            seg = src1_seg_19[...,labelid]
                            cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-seg%02d.png' % (tgt_idx - max_src_offset + 1, labelid), np.tile(seg[...,np.newaxis] * 255, [1,1,3]))

                        # Output Mask image / flow.
                        mask_img = masks['image'][2][j][...,::-1]
                        cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-maskimg.png' % (tgt_idx - max_src_offset + 1), mask_img / mask_img.max() * 255)
                        cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-maskflow.png' % (tgt_idx - max_src_offset + 1), masks['flow'][2][j][...,::-1])

                        # Ourpur Row.
                        #row = np.concatenate([src1_merge[h//3:h//3*2,:w//2,:], src1_flow[:,:w//2,:], mask_src1_merge[h//3*2:,:,:]], axis=1)
                        ##row = np.concatenate([mask_src1, src1_flow[:,:w//2,:], src1_attention[:,:w//2,:], mask_src1_merge[h//3*2:,:,:]], axis=1)
                        #cv2.imwrite(new_merge_output_dir + 'tgtsrc1-%.6d-row.png' % (tgt_idx - max_src_offset + 1), row)

                # Insert the target pose [0, 0, 0, 0, 0, 0] to the middle
                pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)     # pred_poses.shape=[3,6]
                curr_times = times[tgt_idx-max_src_offset : tgt_idx+max_src_offset+1]
                out_file = os.path.join(FLAGS.output_dir, '%.6d.txt' % (tgt_idx - max_src_offset))

                # Output 6DoF file
                #with open(kitti_out_file_r, "a") as f:
                    #p = pred_poses[-1,:]    # tgt->src1
                    #f.write("%06d %s\n" % ((tgt_idx - max_src_offset), " ".join(str(float(_p)) for _p in p)))
                ##out_file = os.path.join(FLAGS.output_dir, '%.6d-6DoF.txt' % (tgt_idx - max_src_offset))
                ##with open(out_file, "w") as f:
                    ##for p in pred_poses:
                        ##f.write(" ".join(str(float(_p)) for _p in p) + "\n")

                pose_mat = sess.run(pose_mat_tensor, feed_dict={pose_vec_ph : pred_poses})
                # before 2019-12-05
                #pred_pose_list.append( pose_mat[0] )  # tgt-> src0 = F(src->tgt)
                # 2019-12-05 update
                if i == 0:
                    pred_pose_list.append( pose_mat[0] )  # tgt-> src0 = F(src->tgt)
                pred_pose_list.append( np.linalg.inv(pose_mat[2]) )  # tgt-> src1 = F(tgt->src1_

        # before 2019-12-05
        #pred_pose_list.append( np.linalg.inv(pose_mat[2]) )  # src1 -> tgt = F(tgt->src1)
        # 2019-12-05 update

        # for VERSION2 (-v2)
        for p in pred_pose_list:
            prev_pose = np.dot(prev_pose, p)    # correct!
            recover_pose.append(prev_pose)
        # for VERSION3 (-v3)
        #for p in pred_pose_list:
            #p = np.linalg.inv(p)
            #prev_pose = np.dot(prev_pose, p)    # correct!
            #recover_pose.append(prev_pose)

        with open(kitti_out_file, 'w') as kitti_f:
            for p in recover_pose:
                s = ' '.join([str(float(x)) for x in p[:3,:].reshape((12))])
                kitti_f.write('%s\n' % s)
        print ("Done. Please check %s" % kitti_out_file)

if __name__ == '__main__':
    main()
