from __future__ import division
import os
import random
import tensorflow as tf
import numpy as np

class DataLoader(object):
    def __init__(self, 
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None, 
                 img_width=None, 
                 num_source=None, 
                 num_scales=None,
                 read_pose=False,
                 read_flow=False,
                 read_depth=False,
                 read_seglabel=False,
                 data_aug=True,
                 data_flip=True):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales
        self.read_pose = read_pose
        self.read_flow = read_flow
        self.read_depth = read_depth
        self.read_seglabel = read_seglabel
        self.data_aug = data_aug
        self.data_flip = data_flip

    def load_train_batch(self):
        """
        Load a batch of training instances using the new tensorflow
        Dataset api.
        """
        def _parse_train_img(img_path):
            with tf.device('/cpu:0'):
                img_buffer = tf.read_file(img_path)
                image_decoded = tf.image.decode_jpeg(img_buffer)
                tgt_image, src_image_stack = \
                    self.unpack_image_sequence(
                        image_decoded, self.img_height, self.img_width, self.num_source)
            return tgt_image, src_image_stack   # tgt = frame_t1, src0 = frame_t0, src1 = frame_t1

        def _parse_input_flow(filename):
            flow = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            flow.set_shape([self.num_source*2, self.img_height, self.img_width, 2])
            return flow

        def _parse_input_depth(filename):
            depth = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            depth.set_shape([self.num_source+1, self.img_height, self.img_width, 1])
            return depth

        def _parse_input_seglabel(filename):
            seglabel = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            seglabel.set_shape([self.num_source+1, self.img_height, self.img_width, 1])
            return seglabel

        def _load_npy(filename):
            with open(filename, 'rb') as f:
                return np.load(f, allow_pickle=True)

        def _batch_preprocessing(stack_images, intrinsics, pose_data, flow_data, depth_data, seglabel_data):
            intrinsics = tf.cast(intrinsics, tf.float32)
            intrinsics = self.get_multi_scale_intrinsics(intrinsics, self.num_scales)   # shape : [B, num_scales, 3, 3]

            image_all = tf.concat([stack_images[0], stack_images[1]], axis=3)
            # Data Augmentation : brightness, contrast, saturation, hue.
            if self.data_aug:
                image_all, image_all_aug = self.data_augmentation(image_all)
                do_aug = tf.random_uniform([], 0, 1)
                image_all = tf.cond(do_aug < 0.5, lambda: image_all, lambda: image_all_aug)
            # Data Augmentation : flip.
            if self.data_flip:
                do_flip = tf.random_uniform([], 0, 1)
                image_all     = tf.cond(do_flip < 0.5, lambda: image_all, lambda: self.img_flip(image_all))
                pose_data     = tf.cond(do_flip < 0.5, lambda: pose_data, lambda: self.pose_flip(pose_data))
                flow_data     = tf.cond(do_flip < 0.5, lambda: flow_data, lambda: self.npy_flip(flow_data))
                depth_data    = tf.cond(do_flip < 0.5, lambda: depth_data, lambda: self.npy_flip(depth_data))
                seglabel_data = tf.cond(do_flip < 0.5, lambda: seglabel_data, lambda: self.npy_flip(seglabel_data))

            tgt_image = image_all[:, :, :, :3]
            src_image_stack = image_all[:, :, :, 3:]
            return tgt_image, src_image_stack, intrinsics, pose_data, flow_data, depth_data, seglabel_data

        file_list = self.format_file_list(self.dataset_dir, 'train')
        self.steps_per_epoch = int(len(file_list['image_file_list'])//self.batch_size)

        input_image_names_ph = tf.placeholder(tf.string, shape=[None], name='input_image_names_ph')
        image_dataset = tf.data.Dataset.from_tensor_slices(
            input_image_names_ph).map(_parse_train_img, num_parallel_calls=4)

        cam_intrinsics_ph = tf.placeholder(tf.float32, [None, 3, 3], name='cam_intrinsics_ph')
        intrinsics_dataset = tf.data.Dataset.from_tensor_slices(cam_intrinsics_ph)

        datasets = (image_dataset, intrinsics_dataset)

        if self.read_pose:
            poses_ph = tf.placeholder(tf.float32, [None, self.num_source+1, 6], name='poses_ph')
            pose_dataset = tf.data.Dataset.from_tensor_slices(poses_ph) # pose = [src0, tgt, src1] (absolute pose)
        else:
            pose_dataset = datasets[1]

        if self.read_flow:
            print ("Read Flow")
            flow_names_ph = tf.placeholder(tf.string, shape=[None], name='flow_names_ph')
            flow_dataset = tf.data.Dataset.from_tensor_slices( flow_names_ph ).map(_parse_input_flow, num_parallel_calls=4)
        else:
            flow_dataset = datasets[1]

        if self.read_seglabel:
            print ("Read SegLabel")
            seglabel_names_ph = tf.placeholder(tf.string, shape=[None], name='seglabel_names_ph')
            seglabel_dataset = tf.data.Dataset.from_tensor_slices( seglabel_names_ph ).map(_parse_input_seglabel, num_parallel_calls=4)
        else:
            seglabel_dataset = datasets[1]

        if self.read_depth:
            print ("Read Depth")
            depth_names_ph = tf.placeholder(tf.string, shape=[None], name='depth_names_ph')
            depth_dataset = tf.data.Dataset.from_tensor_slices( depth_names_ph ).map(_parse_input_depth, num_parallel_calls=4)
        else:
            depth_dataset = seglabel_dataset

        datasets = (datasets[0], datasets[1], pose_dataset, flow_dataset, depth_dataset, seglabel_dataset)


        all_dataset = tf.data.Dataset.zip(datasets)
        all_dataset = all_dataset.batch(self.batch_size).repeat().prefetch(self.batch_size*8)
        all_dataset = all_dataset.map(_batch_preprocessing, num_parallel_calls=4)
        iterator = all_dataset.make_initializable_iterator()
        return iterator

    # Adapted from https://github.com/FangGet/tf-monodepth2
    # edit at 05/26 by Frank
    # add random brightness, contrast, saturation and hue to all source image
    # [H, W, (num_source + 1) * 3]
    def img_flip(self, im):
        """
        - im : tf.concat([tgt, src0, src1], axis=-1)
          im.shape : [B,H,W,3*3]
        """
        num_img = np.int(im.get_shape().as_list()[-1] // 3)
        flip_im_list = []
        for i in range(num_img):
            flip_im_list.append( tf.image.flip_left_right(im[...,3*i: 3*(i+1)]) )
        im = tf.concat(flip_im_list, axis=-1)
        return im

    def npy_flip(self, inputs):
        """
        - inputs :
            1) flow : src0->tgt, src1->tgt
               flow.shape : [B,2,H,W,2]
            2) depth : src0, tgt, src1
               depth.shape : [B,3,H,W,1]
            3) seglabel : src0, tgt, src1
               seglabel.shape : [B,3,H,W,1]
        """
        num_img = np.int(inputs.get_shape().as_list()[1])
        flip_im_list = []
        for i in range(num_img):
            flip_im_list.append( tf.image.flip_left_right(inputs[:,i,:,:,:]) )
        flips = tf.stack(flip_im_list, axis=1)
        return flips

    def pose_flip(self, pose):
        """Flip the x-axis.
        - pose : (rz,ry,rx,tx,ty,tz)
          pose.shape : [B,3,6]
          pose[:,0,:] : tgt
          pose[:,1,:] : src0
          pose[:,2,:] : src1
        """
        pose = tf.stack([
            pose[:,:,0],
            tf.where(pose[:,:,1] > tf.zeros_like(pose[:,:,1]), np.pi - pose[:,:,1], - np.pi - pose[:,:,1]), 
            pose[:,:,2],
            pose[:,:,3] * (-1), 
            pose[:,:,4], 
            pose[:,:,5]
            ], axis=-1)
        return pose

    def data_augmentation(self, im):
        def random_flip(im):
            def flip_one(sim):
                do_flip = tf.random_uniform([], 0, 1)
                return tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(sim), lambda: sim)

            im = tf.map_fn(lambda sim: flip_one(sim), im)
            #im = tf.cond(do_flip > 0.5, lambda: tf.map_fn(lambda sim: tf.image.flip_left_right(sim),im), lambda : im)
            return im

        def augment_image_properties(im):
            # im.shape : (128, 416, 9)
            # random brightness
            brightness_seed = random.randint(0, 2**31 - 1)
            im = tf.image.random_brightness(im, 0.2, brightness_seed)

            contrast_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_contrast(im, 0.8, 1.2, contrast_seed)

            num_img = np.int(im.get_shape().as_list()[-1] // 3)

            # saturation_seed = random.randint(0, 2**31 - 1)
            saturation_im_list = []
            saturation_factor = random.uniform(0.8,1.2) #tf.random_ops.random_uniform([], 0.8, 1.2, seed=saturation_seed)
            for i in range(num_img):
                saturation_im_list.append(tf.image.adjust_saturation(im[:,:, 3*i: 3*(i+1)],saturation_factor))
                # tf.image.random_saturation(im[:,:, 3*i: 3*(i+1)], 0.8, 1.2, seed=saturation_seed))
            im = tf.concat(saturation_im_list, axis=2)

            #hue_seed = random.randint(0, 2 ** 31 - 1)
            hue_im_list = []
            hue_delta = random.uniform(-0.1,0.1) #tf.random_ops.random_uniform([], -0.1, 0.1, seed=hue_seed)
            for i in range(num_img):
                hue_im_list.append(tf.image.adjust_hue(im[:, :, 3 * i: 3 * (i + 1)],hue_delta))
                 #  tf.image.random_hue(im[:, :, 3 * i: 3 * (i + 1)], 0.1, seed=hue_seed))
            im = tf.concat(hue_im_list, axis=2)
            return im

        def random_augmentation(im):
            def augmentation_one(sim):
                do_aug = tf.random_uniform([], 0, 1)
                return tf.cond(do_aug > 0.5, lambda: augment_image_properties(sim), lambda : sim)
            im = tf.map_fn(lambda sim: augmentation_one(sim), im)
            #im = tf.cond(do_aug > 0.5, lambda: tf.map_fn(lambda sim: augment_image_properties(sim), im), lambda: im)
            return im

        #im = random_flip(im)
        im_aug = random_augmentation(im)
        return im, im_aug



    def load_test_batch_flow(self, image_sequence_names, image_sequence_poses, image_sequence_flows, image_sequence_depths, image_sequence_seglabels):
        """load a batch of test images for inference"""
        def _parse_test_img(img_path):
            with tf.device('/cpu:0'):
                img_buffer = tf.read_file(img_path)
                image_decoded = tf.image.decode_jpeg(img_buffer)
            return image_decoded
        def collect_all_poses(cam_filelist):
            all_poses = []
            for filename in cam_filelist:
                with open(filename) as f:
                    lines = f.readlines()
                    one_sample_pose = []
                    for i in range(1, len(lines)):
                        pose = [float(num) for num in lines[i].split(',')]
                        pose_vec = np.reshape(pose, [6])
                        one_sample_pose.append(pose_vec)
                    one_sample_pose = np.stack(one_sample_pose, axis=0)
                all_poses.append(one_sample_pose)
            for p in all_poses:
                if p.shape != (3,6):
                    print (p.shape)
            all_poses = np.stack(all_poses, axis=0)
            return all_poses

        def _parse_input_flow(filename):
            flow = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            print (flow)
            return flow

        def _parse_input_depth(filename):
            depth = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            print (depth)
            return depth

        def _parse_input_seglabel(filename):
            seglabel = tf.py_func(_load_npy, [filename], [tf.float32])[0]
            return seglabel

        def _load_npy(filename):
            with open(filename, 'rb') as f:
                return np.load(f, allow_pickle=True)


        #def _batch_preprocessing(stack_images, pose_data):
            #pose_data = tf.cast(pose_data, tf.float32)
            #image_all = tf.concat([stack_images[0], stack_images[1]], axis=3)
            #tgt_image = image_all[:, :, :, :3]
            #src_image_stack = image_all[:, :, :, 3:]
            #return tgt_image, src_image_stack, pose_data
        def _batch_preprocessing(stack_images, pose_data, flow_data, depth_data, seglabel_data):
            image_all = stack_images
            image_all     = self.img_flip(image_all)
            pose_data     = self.pose_flip(pose_data)
            flow_data     = self.npy_flip(flow_data)
            depth_data    = self.npy_flip(depth_data)
            seglabel_data = self.npy_flip(seglabel_data)

            src1_image = image_all[:, :, :, :3]
            tgt_image  = image_all[:, :, :, 3:6]
            src0_image = image_all[:, :, :, 6:]
            image_all = tf.concat([src0_image, tgt_image, src1_image], axis=3)
            return image_all, pose_data, flow_data, depth_data, seglabel_data

        self.read_pose = True

        image_dataset = tf.data.Dataset.from_tensor_slices(image_sequence_names).map(_parse_test_img, num_parallel_calls=4)

        all_poses = collect_all_poses(image_sequence_poses)
        pose_dataset = tf.data.Dataset.from_tensor_slices(all_poses)

        print ("Read Flow")
        flow_dataset = tf.data.Dataset.from_tensor_slices( image_sequence_flows ).map(_parse_input_flow, num_parallel_calls=4)
        print ("Read Depth")
        depth_dataset = tf.data.Dataset.from_tensor_slices( image_sequence_depths ).map(_parse_input_depth, num_parallel_calls=4)
        print ("Read SegLabel")
        seglabel_dataset = tf.data.Dataset.from_tensor_slices( image_sequence_seglabels ).map(_parse_input_seglabel, num_parallel_calls=4)

        datasets = (image_dataset, pose_dataset, flow_dataset, depth_dataset, seglabel_dataset)

        all_dataset = tf.data.Dataset.zip(datasets)
        all_dataset = all_dataset.batch(self.batch_size).prefetch(self.batch_size*8)
        #all_dataset = all_dataset.map(_batch_preprocessing, num_parallel_calls=4)
        iterator = all_dataset.make_initializable_iterator()
        return iterator


    def load_test_batch(self, image_sequence_names, image_sequence_poses):
        """load a batch of test images for inference"""
        def _parse_test_img(img_path):
            with tf.device('/cpu:0'):
                img_buffer = tf.read_file(img_path)
                image_decoded = tf.image.decode_jpeg(img_buffer)
            return image_decoded
        def collect_all_poses(cam_filelist):
            all_poses = []
            for filename in cam_filelist:
                with open(filename) as f:
                    lines = f.readlines()
                    one_sample_pose = []
                    for i in range(1, len(lines)):
                        pose = [float(num) for num in lines[i].split(',')]
                        pose_vec = np.reshape(pose, [6])
                        one_sample_pose.append(pose_vec)
                    one_sample_pose = np.stack(one_sample_pose, axis=0)
                all_poses.append(one_sample_pose)
            all_poses = np.stack(all_poses, axis=0)
            return all_poses

        #def _batch_preprocessing(stack_images, pose_data):
            #pose_data = tf.cast(pose_data, tf.float32)
            #image_all = tf.concat([stack_images[0], stack_images[1]], axis=3)
            #tgt_image = image_all[:, :, :, :3]
            #src_image_stack = image_all[:, :, :, 3:]
            #return tgt_image, src_image_stack, pose_data

        self.read_pose = True

        image_dataset = tf.data.Dataset.from_tensor_slices(image_sequence_names).map(_parse_test_img, num_parallel_calls=4)

        all_poses = collect_all_poses(image_sequence_poses)
        pose_dataset = tf.data.Dataset.from_tensor_slices(all_poses)

        datasets = (image_dataset, pose_dataset)

        all_dataset = tf.data.Dataset.zip(datasets)
        all_dataset = all_dataset.batch(self.batch_size).prefetch(self.batch_size*8)
        #all_dataset = all_dataset.map(_batch_preprocessing, num_parallel_calls=4)
        iterator = all_dataset.make_initializable_iterator()
        return iterator


    def init_data_pipeline(self, sess, batch_sample):
        def _load_cam_intrinsics(cam_filelist):
            all_cam_intrinsics = []
            for filename in cam_filelist:
                with open(filename) as f:
                    line = f.readlines()
                    cam_intri_vec = [float(num) for num in line[0].split(',')]
                    cam_intrinsics = np.reshape(cam_intri_vec, [3, 3])
                    all_cam_intrinsics.append(cam_intrinsics)
            all_cam_intrinsics = np.stack(all_cam_intrinsics, axis=0)
            return all_cam_intrinsics
        
        def _load_poses_6dof(cam_filelist):
            all_poses = []
            for filename in cam_filelist:
                with open(filename) as f:
                    lines = f.readlines()
                    one_sample_pose = []                                    # one_sample_pose = [row1_pose, row2_pose, row3_pose]
                    for i in range(1, len(lines)):
                        pose = [float(num) for num in lines[i].split(',')]  # row1(frame_t-1), row2(frame_t), row3(frame_t+1) in *_cam.txt
                        pose_vec = np.reshape(pose, [6])
                        one_sample_pose.append(pose_vec)
                    one_sample_pose = np.stack(one_sample_pose, axis=0)     # shape=[3,6]
                all_poses.append(one_sample_pose)
            all_poses = np.stack(all_poses, axis=0)
            return all_poses    # tf.stack([batch1_3x6_pose, batch2_3x6_pose, ...]). shape = [batch_size, 3, 6]

        file_list = self.format_file_list(self.dataset_dir, 'train')
        input_dict = {}

        # load input_flow using native python
        if self.read_flow:
            print('load flow data...')
            all_flows_file = file_list['flow_file_list']
            input_dict['data_loading/flow_names_ph:0'] = all_flows_file[:self.batch_size*self.steps_per_epoch]

        # load input_depth using native python
        if self.read_depth:
            print('load depth data...')
            all_depths_file = file_list['depth_file_list']
            input_dict['data_loading/depth_names_ph:0'] = all_depths_file[:self.batch_size*self.steps_per_epoch]

        # load input_seglabel using native python
        if self.read_seglabel:
            print('load seglabel data...')
            all_seglabels_file = file_list['seglabel_file_list']
            input_dict['data_loading/seglabel_names_ph:0'] = all_seglabels_file[:self.batch_size*self.steps_per_epoch]


        # load cam_intrinsics using native python
        print('load camera intrinsics...')
        cam_intrinsics = _load_cam_intrinsics(file_list['cam_file_list'])

        input_dict.update({'data_loading/input_image_names_ph:0':
                      file_list['image_file_list'][:self.batch_size *
                                                   self.steps_per_epoch],
                      'data_loading/cam_intrinsics_ph:0':
                      cam_intrinsics[:self.batch_size*self.steps_per_epoch]})
        if self.read_pose:
            print('load pose data...')
            all_poses = _load_poses_6dof(file_list['cam_file_list'])
            input_dict['data_loading/poses_ph:0'] = all_poses[:self.batch_size*self.steps_per_epoch]

        sess.run(batch_sample.initializer, feed_dict=input_dict)


    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = tf.shape(fx)[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics


#    def data_augmentation(self, im, intrinsics, out_h, out_w):
#        # Random scaling
#        def random_scaling(im, intrinsics):
#            _, in_h, in_w, _ = im.get_shape().as_list()
#            scaling = tf.random_uniform([2], 1, 1.15)
#            x_scaling = scaling[0]
#            y_scaling = scaling[1]
#            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
#            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
#            im = tf.image.resize_area(im, [out_h, out_w])
#            fx = intrinsics[:,0,0] * x_scaling
#            fy = intrinsics[:,1,1] * y_scaling
#            cx = intrinsics[:,0,2] * x_scaling
#            cy = intrinsics[:,1,2] * y_scaling
#            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
#            return im, intrinsics
#
#        # Random cropping
#        def random_cropping(im, intrinsics, out_h, out_w):
#            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
#            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
#            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
#            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
#            im = tf.image.crop_to_bounding_box(
#                im, offset_y, offset_x, out_h, out_w)
#            fx = intrinsics[:,0,0]
#            fy = intrinsics[:,1,1]
#            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
#            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
#            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
#            return im, intrinsics
#        im, intrinsics = random_scaling(im, intrinsics)
#        im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
#        im = tf.cast(im, dtype=tf.uint8)
#        return im, intrinsics


    def format_file_list(self, data_root, split):
        all_list = {}
        #with open(data_root + '/%s.txt' % split, 'r') as f:
        with open(data_root + '/%s_all.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        flow_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '-flownet2.npy') for i in range(len(frames))]
        depth_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '-monodepth2_depth.npy') for i in range(len(frames))]
        seglabel_file_list = [os.path.join(data_root, subfolders[i], 
            frame_ids[i] + '-seglabel.npy') for i in range(len(frames))]
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        all_list['flow_file_list'] = flow_file_list
        all_list['depth_file_list'] = depth_file_list
        all_list['seglabel_file_list'] = seglabel_file_list
        return all_list


    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, img_width, num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack


    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, 0, tgt_start_idx, 0], 
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0, 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, 0, int(tgt_start_idx + img_width), 0], 
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, 0, i*img_width, 0], 
                                    [-1, -1, img_width, -1]) 
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack


    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
