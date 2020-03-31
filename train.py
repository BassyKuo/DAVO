from __future__ import division
import os
import random
import pprint
import numpy as np
import tensorflow as tf
from davo import DAVO

flags = tf.app.flags
flags.DEFINE_string("version", "v1", "version")
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_ckpt_file", None, "Specific checkpoint file to initialize from")

flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.1, "Weight for smoothness")
flags.DEFINE_float("pose_weight", 0.1, "Weight for pose loss")
flags.DEFINE_float("ssim_weight", 0.85, "Weight for SSIM loss")

flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 310000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 200, "Logging every log_freq iterations")
flags.DEFINE_integer("save_freq", 25000, "Save the model every save_freq iterations (overwrites the previous latest model)")

flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("data_aug", False, "Add augemented data (random brightness/contrast/saturation/hue)")
flags.DEFINE_boolean("data_flip", False, "Add flipping data")

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    flags.DEFINE_integer("num_source", flags.FLAGS.seq_length - 1, "source images (seq_length-1)")
    FLAGS = flags.FLAGS

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    system = DAVO(FLAGS.version)
    system.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
