from __future__ import division
import argparse
import scipy.misc
import numpy as np
from absl import logging
from glob import glob
import os,sys
import datetime

import itertools
import multiprocessing

CURDIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(CURDIR, '..')))
#from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM

# Process data in chunks for reporting progress.
NUM_CHUNKS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
parser.add_argument("--img_height", type=int, default=128, help="image height")
parser.add_argument("--img_width", type=int, default=416, help="image width")
parser.add_argument("--num_threads", type=int, default=8, help="number of threads to use")
parser.add_argument("--skip_image", type=bool, default=False, help="do not generate images")
parser.add_argument("--generate_test", type=bool, default=False, help="generate test images")
args = parser.parse_args()

def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


def dump_example(n, is_training):
    if is_training:
        frame_num = data_loader.num_train
        example = data_loader.get_train_example_with_idx(n)
    else:
        frame_num = data_loader.num_test
        example = data_loader.get_test_example_with_idx(n)

    if example == False:
        return
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    try:
        os.makedirs(dump_dir)
    except OSError:
        if not os.path.isdir(dump_dir):
            raise

    #if n % 2000 == 0:
        #print('Progress %d/%d....' % (n, frame_num))

    # save image file
    if not args.skip_image:
        image_seq = concat_image_seq(example['image_seq'])
        dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
        scipy.misc.imsave(dump_img_file, image_seq.astype(np.uint8))
        print ("img: ", dump_img_file)

    # save camera info
    if is_training:
        intrinsics = example['intrinsics']
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
        dump_eval_pose_dir = './kitti_eval/tum_pose_data/ground_truth/seq%d/%s/' % (args.seq_length, example['folder_name'])
        dump_eval_pose_file = os.path.join(dump_eval_pose_dir, '%.6d.txt' % (int(example['file_name']) - 1))
        with open(dump_cam_file, 'w') as f:
            f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.\n' % (fx, cx, fy, cy))
            if 'pose' in example:
                try:
                    os.makedirs(dump_eval_pose_dir)
                except OSError:
                    if not os.path.isdir(dump_eval_pose_dir):
                        raise
                poses = example['pose'] # (rz,ry,rx,tx,ty,tz)
                for each_pose in poses:
                    # Save in ./${dump_root}/${sequnce_name}/${frame_idx}_cam.txt
                    f.write(','.join([str(num) for num in each_pose])+'\n')
                    print("%s : " % dump_cam_file, ','.join([str(num) for num in each_pose])+'\n')
                # Save in ./kitti_eval for evaluation
                with open(os.path.join(args.dataset_dir, 'sequences/%s/times.txt' % example['folder_name']), 'r') as time_f:
                    times = time_f.readlines()
                times = np.array([float(s[:-1]) for s in times])
                half_offset = int((args.seq_length - 1)/2)
                target_pose_idx = int(example['file_name'])
                curr_times = times[target_pose_idx-half_offset : target_pose_idx+half_offset+1]
                #dump_pose_seq_TUM(dump_eval_pose_file, poses, curr_times, False)

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)

    global data_loader
    if args.dataset_name == 'kitti_odom':
        from kitti.kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti.kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length)

    if args.dataset_name == 'cityscapes':
        from cityscapes.cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        split='train',
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    is_training = not(args.generate_test)
    print ("is traing: ", is_training)
    all_frames = range(data_loader.num_train) if is_training else range(data_loader.num_test)
    frame_chunks = np.array_split(all_frames, NUM_CHUNKS) # len(frame_chunks)=NUM_CHUNKS

    num_cores = multiprocessing.cpu_count()
    num_threads = num_cores if args.num_threads is None else args.num_threads
    pool = multiprocessing.Pool(num_threads)

    if args.generate_test:
        print ("Generate Test")
        logging.info('Generating test data...')
        for index, frame_chunk in enumerate(frame_chunks):
            pool.map(_dump_example_star,
                     zip(frame_chunk, itertools.repeat(is_training)))
            logging.info('Chunk %d/%d: saving entries...', index + 1, NUM_CHUNKS)
    else:
        print ("Generate Train/Val")
        # Split into train/val
        np.random.seed(8964)
        train_file_txt = os.path.join(args.dump_root, 'train.txt')
        val_file_txt = os.path.join(args.dump_root, 'val.txt')
        for _file in [train_file_txt, val_file_txt]:
            if os.path.exists(_file):
                file_t = os.path.getmtime(_file)
                file_suffix = datetime.datetime.fromtimestamp(file_t).strftime("%Y%m%d")
                os.rename(_file, _file+".bak-"+file_suffix)
        subfolders = ['%.2d' % s for s in data_loader.train_seqs]
        with open(train_file_txt, 'w') as tf:
            with open(val_file_txt, 'w') as vf:
                logging.info('Generating train/val data...')
                for index, frame_chunk in enumerate(frame_chunks):
                    pool.map(_dump_example_star,
                             zip(frame_chunk, itertools.repeat(is_training)))
                    logging.info('Chunk %d/%d: saving entries...', index + 1, NUM_CHUNKS)
                    for s in subfolders:
                        if not os.path.isdir(args.dump_root + '/%s' % s):
                            continue
                        imfiles = glob(os.path.join(args.dump_root, '%s' % s, '*.jpg'))
                        frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                        for frame in frame_ids:
                            if np.random.random() < 0.1:
                                vf.write('%s %s\n' % (s, frame))
                            else:
                                tf.write('%s %s\n' % (s, frame))

        logging.info('Generating train/val data...')
        for index, frame_chunk in enumerate(frame_chunks):
            pool.map(_dump_example_star,
                     zip(frame_chunk, itertools.repeat(is_training)))
            logging.info('Chunk %d/%d: saving entries...', index + 1, NUM_CHUNKS)

        pool.close()
        pool.join()

def _dump_example_star(params):
    return dump_example(*params)

if __name__ == '__main__':
    main()
