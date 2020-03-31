# Dynamic Attention-based Visual Odometry

![os](https://img.shields.io/badge/Ubuntu-16.04-orange) ![python](https://img.shields.io/badge/Python-3.6-orange) ![tensorflow](https://img.shields.io/badge/Tensorflow-1.13.1-orange) ![cuda](https://img.shields.io/badge/Cuda-10.0-orange) ![license](https://img.shields.io/badge/License-MIT-blue)

Implementation of DAVO architecture.

## Environment
This codebase is tested on Ubuntu 16.04 with Tensorflow 1.13.1 and CUDA 10.0 (w/ cuDNN 7.5).

### Installation

#### 1. Check the python version

Make sure your python version is 3.6.

```bash
python -V    # should be python 3.6
```

#### 2. Requirements

Install packages from the requirements file.

```bash
pip install -r requirements.txt
```

#### (optional) 3. Use customized cuda path

Reset CUDA_HOME to other path you used.

```bash
export CUDA_HOME="<your_cuda_path>"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

## Quickstart

Use the following scripts to help you while you already setup the environment and dataset.

#### Training
```bash
# Train model w/ flipping images.
./run_training.sh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh

# Train model w/ augmented images (including flipping, brightness, contrast, saturation, hue).
./run_training.sh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh --data_aug
```

#### Inference
```bash
export ckpt_dir="ckpt_dir/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh/"
export version="v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh"
export seq_name="03"
export ckpt_step="1500000"
export output_root="test_DAVO"

# Run script to generate predicted poses of sequence ${seq_name}.
./run_estimation.sh ${ckpt_dir} ${version} ${seq_name} ${ckpt_step} ${output_root}

# The result would be saved in ${output_root}/${version}--model-${ckpt_step}
```

#### Evaluation
```bash
cd kitti_benchmark/

export test_output_dir="../${output_root}/${version}--model-${ckpt_step}"
export save_name="${version}--model-${ckpt_step}"

# Use pose_kitti_eval.sh to run the KITTI Benchmark.
./pose_kitti_eval.sh ${test_output_dir} ${save_name}
```

---

## Prepare Traing/Testing Data

There are three types of inputs used in DAVO:

1. [RGB frames](./doc/preprocessing.md#rgb-frames)
2. [Optical flows](./doc/preprocessing.md#optical-flows)
3. [Semantic segmentations](./doc/preprocessing.md#semantic-segmentations)

Please check [here](doc/preprocessing.md) to see how to prepare them for DAVO.


## Training
Once the data are formatted following the above instructions, you are able to train the model with the following command:

```bash
python train.py \
    --dataset_dir=$kitti_odom_dump \
    --img_width=416 \
    --img_height=128 \
    --batch_size=4 \
    --seq_length=3 \
    --max_steps=310000 \
    --save_freq=25000 \
    --learning_rate=0.001 \
    --pose_weight=0.1 \
    --checkpoint_dir=./ckpt/${version} \
    --version=${version}
```

#### Example:

```bash
python train.py \
    --dataset_dir=./kitti_odom-dump/ --img_width=416 --img_height=128 --batch_size=4 \
    --seq_length=3 --max_steps=310000 --save_freq=25000 --learning_rate=0.001  --pose_weight=0.1 \
    --checkpoint_dir=./ckpt/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh \
    --version=v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh
```

Note that `${version}` can be selected from the following list (optional options can be used by concating the architecture version):

### Architecture version:

<img src="./doc/figures/DAVO-arch.png" width="680">

| Version                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh | DAVO |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-no_segmask | Ours w/o attention |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-static | Ours w/ static_attention | 
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-no_segmask-se_insert | Ours w/ feature attention |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_seg_wo_tgt-fc_tanh | DAVO (segmentation source) |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_depth_wo_tgt_to_seg-fc_tanh | DAVO (depth source) |
| v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_rgb_wo_tgt_to_seg-fc_tanh | DAVO (rgb source) |



#### Choose learning rate decay:

* -decay50k
* -decay100k
* -decay50k-staircase
* -decay100k-staircase

#### Choose PoseNN: 

* -dilatedPoseNN
* -dilatedCouplePoseNN
* -sharedNN-dilatedPoseNN
* -sharedNN-dilatedCouplePoseNN

    #### [adjust conv6 channels in PoseNN]
    * -cnv6_<num_channels>
  
    #### [adjust se_block between conv5 and conv6 in PoseNN]
    * -se_insert
    * -se_replace
    * -se_skipadd
    
#### Choose Attention mode:

* -segmask_all
* -segmask_rgb
* -no_segmask

#### Choose AttentionNN source and layer types:

* -se_flow
* -se_spp_flow
* -se_seg
* -se_spp_seg
* -se_mixSegFlow
* -se_spp_mixSegFlow
* -se_SegFlow_to_seg
* -se_app_mixSegFlow
* -se_seg_wo_tgt
* -se_depth_wo_tgt_to_seg
* -se_rgb_wo_tgt_to_seg

    #### [adjust the activated function in AttentionNN]
    * -fc_relu
    * -fc_tanh
    * -fc_lrelu


## Evaluation
To evaluate the pose estimation performance in the paper, use the following command to generate esitimated poses:

```bash
python test_kitti_pose.py \
    --test_seq=$seq \
    --concat_img_dir=./kitti_odom-dump/ \
    --ckpt_file=${ckpt_file} \
    --version=${version} \
    --output_dir=${output_dir}
```

#### Example:
```bash
python test_kitti_pose.py \
    --test_seq=3 \
    --concat_img_dir=./kitti_odom-dump/ \
    --ckpt_file=./ckpt/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh/model-10 \
    --version=v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh
    --output_dir=./test-DAVO/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh--model-10
```


## Visualization

In this part, you can use [evo](https://github.com/MichaelGrupp/evo) tool to visualize trajetories with references sequences 00-10:

```bash
for seq in {00..10..1} ; do 
    evo_traj kitti \
        --ref kitti_benchmark/data/odometry/poses/${seq}.txt \
        ./test-DAVO/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh--model-10/${seq}-pred_kitti_pose.txt \
        -p --plot_mode xz --save_plot plots/${figure_name}.png
done
```

Because of the lack of ground truth poses in the testing sequence 11-21, you can use trajetories generated from ORBSLAM2-S ([ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) stereo version) to compare our prediction:
```bash
for seq in {11..21..1} ; do 
    evo_traj kitti \
        --ref kitti_benchmark/data/odometry/poses_from_ORBSLAM2-S/${seq}-ORB-SLAM2-S.txt \
        ./test-DAVO/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh--model-10/${seq}-pred_kitti_pose.txt \
        -p --plot_mode xz --save_plot plots/${figure_name}.png
done
```

If you use the `--save_plot` argument to save as png file, please change the format first:

```
evo_config set plot_export_format png
```

Or you can also change trajetory colors:

```
evo_config set plot_seaborn_palette Dark2
```

Please use `evo_config show` to see more configures.


# Contact
Please feel free to contact us if you have any questions :smile:

# Acknowledgements
We appreciate the great works/repos along this direction, such as [SfMLearner](https://github.com/tinghuiz/SfMLearner), [GeoNet](https://github.com/yzcjtr/GeoNet), [DeepMatchVO](https://github.com/hlzz/DeepMatchVO) and also the evaluation tools such as [KITTI VO/SLAM devkit](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [evo](https://github.com/MichaelGrupp/evo) for KITTI full sequence evaluation.
