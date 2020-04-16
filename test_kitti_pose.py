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
from davo import DAVO
from data_loader import DataLoader
from utils.common_utils import complete_batch_size, is_valid_sample
from utils.seg_utils.labels import seg_labels

_OUTPUT_FEATUREMAP = False
_OUTPUT_NEW_MERGE  = False
_OUTPUT_ATTENTION  = False

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 1, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("test_seq", 9, "Sequence id to test")
flags.DEFINE_string("concat_img_dir", None, "Preprocess image dataset directory")
flags.DEFINE_string("output_dir", None, "Output directory")
flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("version", "v1", "version")

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
        img_flowname = os.path.join(dataset_dir, '%s/%s-flownet2.npy' % (curr_drive, curr_frame_id)) # shape=(4,h,w,2)
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
        depth = image_sequence_seglabels
    if load_flow:
        flow = image_sequence_flows
    else:
        flow = image_sequence_names
    if load_pose:
        return image_sequence_names, target_inds, image_sequence_poses, flow, depth, seglabel
    else:
        return image_sequence_names, target_inds, image_sequence_names, flow, depth, seglabel


def main(_):
    # get input images
    FLAGS = flags.FLAGS
    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    concat_img_dir = os.path.join(FLAGS.concat_img_dir, '%.2d' % FLAGS.test_seq)
    max_src_offset = int((FLAGS.seq_length - 1)/2)
    N = len(glob(concat_img_dir + '/*.jpg')) + 2*max_src_offset
    test_frames = ['%.2d %.6d' % (FLAGS.test_seq, n) for n in range(N)]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # setup input tensor
        read_flow = True
        read_seglabel = True
        read_depth = True if "depth" in FLAGS.version else False
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
        input_flow.set_shape([FLAGS.batch_size, (FLAGS.seq_length-1)*2, FLAGS.img_height, FLAGS.img_width, 2])
        input_depth.set_shape([FLAGS.batch_size, FLAGS.seq_length, FLAGS.img_height, FLAGS.img_width, 1])
        input_seglabel.set_shape([FLAGS.batch_size, FLAGS.seq_length, FLAGS.img_height, FLAGS.img_width, 1])

        kitti_out_file = os.path.join(FLAGS.output_dir, '%.2d-pred_kitti_pose.txt' % FLAGS.test_seq)
        if os.path.isfile(kitti_out_file):
            os.remove(kitti_out_file)
        prev_pose = np.eye(4).astype(float)
        recover_pose = [prev_pose]
        pred_pose_list = []
        pose_vec_ph = tf.placeholder(tf.float32, [3,6])
        pose_mat_tensor = geo_utils.pose_vec2mat(pose_vec_ph)

        # init system
        system = DAVO(version=FLAGS.version)
        system.setup_inference(FLAGS.img_height, FLAGS.img_width,
                               "davo", FLAGS.seq_length, FLAGS.batch_size, input_batch, input_flow=input_flow, input_depth=input_depth, input_seglabel=input_seglabel)
        saver = tf.train.Saver([var for var in tf.trainable_variables()]) 

        saver.restore(sess, FLAGS.ckpt_file)

        round_num = len(image_sequence_names) // FLAGS.batch_size
        for i in range(round_num):
            pred = system.inference(sess, mode='pose')
            for j in range(FLAGS.batch_size):
                tgt_idx = tgt_inds[i * FLAGS.batch_size + j]
                pred_poses = pred['pose'][j]                         # pred['pose'].shape=[B,6]. pred_poses.shape=[2,6]

                # Insert the target pose [0, 0, 0, 0, 0, 0] to the middle
                pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=0)     # pred_poses.shape=[3,6]
                pose_mat = sess.run(pose_mat_tensor, feed_dict={pose_vec_ph : pred_poses})
                if i == 0:
                    pred_pose_list.append( pose_mat[0] )             # tgt->src0
                pred_pose_list.append( np.linalg.inv(pose_mat[2]) )  # inv(tgt->src1)

        for p in pred_pose_list:
            prev_pose = np.dot(prev_pose, p)
            recover_pose.append(prev_pose)
        with open(kitti_out_file, 'w') as kitti_f:
            for p in recover_pose:
                s = ' '.join([str(float(x)) for x in p[:3,:].reshape((12))])
                kitti_f.write('%s\n' % s)
        print ("Done. Please check %s" % kitti_out_file)

if __name__ == '__main__':
    tf.app.run()
