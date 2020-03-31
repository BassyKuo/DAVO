# Dynamic Attention-based Visual Odometry

![os](https://img.shields.io/badge/Ubuntu-16.04-orange) ![python](https://img.shields.io/badge/Python-3.6-orange) ![tensorflow](https://img.shields.io/badge/Tensorflow-1.13.1-orange) ![cuda](https://img.shields.io/badge/Cuda-10.0-orange) ![license](https://img.shields.io/badge/License-MIT-blue)

Implementation of DAVO architecture.

## Environment
This codebase is tested on Ubuntu 16.04 with Tensorflow 1.13.1 and CUDA 10.0.

## Quickstart

### Requirements

```bash
pip install -r requirements.txt
```

## Traing/Testing Data Preprocessing

### KITTI Odometry
For [KITTI](http://www.cvlibs.net/datasets/kitti), you need to download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) first, 
and unzip them into the `$kitti_raw_odom` folder. Then run the following command:

```bash
# for KITTI odometry dataset
python data/preprocess.py --dataset_dir=$kitti_raw_odom --dataset_name='kitti_odom' --dump_root=$kitti_odom_dump --seq_length=3 --img_width=416 --img_height=128 --num_threads=8
```
###### [NOTE]: Add argument `--generate_test` in your comand to genrate testing data.

### (optional) Cityscapes Sequence

For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip` (324GB), 2) `camera_trainvaltest.zip` (1.9MB), and unzip them into the `$cityscapes_raw` folder.
Then run the following command:

```bash
# for Cityscapes dataset
python data/preprocess.py --dataset_dir=$cityscapes_raw --dataset_name='cityscapes' --dump_root=$cityscapes_odom_dump --seq_length=3 --img_width=416 --img_height=128 --num_threads=8
```

Notice that for Cityscapes the img_height is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

###### [NOTE]: Add argument `--generate_test` in your comand to genrate testing data.

## Training
Once the data are formatted following the above instructions, you are able to train the model by running the following command:

```bash
python train.py --dataset_dir=$kitti_odom_dump --img_width=416 --img_height=128 --batch_size=4 --seq_length=3 --max_steps=310000 --save_freq=25000 --learning_rate=0.001 --num_scales=1 --pose_weight=0.1 --checkpoint_dir=./ckpt/${version} --version=${version}
```

For example:

```bash
python train.py --dataset_dir=./kitti_odom-dump/ --img_width=416 --img_height=128 --batch_size=4 --seq_length=3 --max_steps=310000 --save_freq=25000 --learning_rate=0.001 --num_scales=1 --pose_weight=0.1 --checkpoint_dir=./ckpt/v1-upperflow-ALL-PoseLossWeak-decay100k-staircase --version=v1-upperflow-ALL-PoseLossWeak-decay100k-staircase
```

where `${version}` can be selected from the following list (optional options can be used by concating the architecture version):

### Architecture version:

<img src="./_github/figures/arch-overall.png" width="680">

| Version                   | Description                                              |
| ------------------------- | -------------------------------------------------------- |
| v0                        | baseline (DeepMatchV0)                                   |
| v0-decouple               | baseline (DeepMatchV0) w/ decoupled PoseNN               |
| v1                        | DAVO w/ whole optical flow                               |
| v1-upperflow              | DAVO w/ upper optical flow                               |
| v2                        | DAVO w/ whole optical flow and whole inverse depth map   |
| v2-upperflow              | DAVO w/ upper optical flow and whole inverse depth map   |
| v2-upperflow-lowerdepth   | DAVO w/ upper optical flow and lower inverse depth map   |
| v2-lowerdepth             | DAVO w/ whole optical flow and lower inverse depth map   |


### (optional) Learning rate mode:

* -decay50k
* -decay100k
* -decay50k-staircase
* -decay100k-staircase


### (optional) Networks training scheme:

<img src="./_github/figures/training_mode.png" width="500">

* -ALL [default]
* -SHARED
* -SEP
* -pretrainD (no training DepthNN) : only for v2 version
    > you need to download pretrain DepthNN model from [here](https://drive.google.com/file/d/1xWNm9MclJHD729uS6U6k2Oopn--Vnban/view?usp=sharing), and unzip them into the `pretrain-ckpt/` folder under the root.
* -noD (no DepthNN model) : only for v1 version

###### [NOTE]: v1 version defaultly uses `-noD`.


### (optional) PossLoss mode:

* -PoseLossStrong [default]
* -PoseLossWeak


## Evaluation
To evaluate the pose estimation performance in the paper, use the following command to generate esitimated poses:

```bash
./run_estimation.sh ./ckpt/${checkpoint_folder} ${version} ${checkpoint_step}
```

For example:

```bash
./run_estimation.sh ./ckpt/v1-upperflow-ALL-PoseLossWeak-decay100k-staircase v1-upperflow-ALL-PoseLossWeak-decay100k-staircase 300000
```

Then run the evaluation routine:

```bash
./evaluation_routine.sh <name_table.txt>
```

###### [NOTE]: Please follow the `name_table.txt.example` to create your own file.

After this step, you can check the trajetory evaluation errors in the `kitti_benchmark/*.csv`, or use the following command to quickly get results:

```bash
while IFS=' ' read -r name path; do printf "%s,%s\n" "$name.csv-h-RMSE" "`cat kitti_benchmark/$name.csv*-RMSE`" ; done < name_table.txt
```

## Testing and Visualization

In this part, you can use `evo_traj` command to visualize trajetories with references sequences 00-10:

```bash
for seq in {00..10..1} ; do 
    evo_traj kitti --ref kitti_benchmark/data/odometry/poses/${seq}.txt kitti_benchmark/results/v1-upperflow-ALL-PoseLossWeak-decay100k-staircase--model-300000/poses/${seq}-pred_kitti_pose.txt <other_version_poses> \
        -p --plot_mode xz -s --save_plot plots/${figure_name}.png
done
```

Because of the lack of ground truth poses in the testing sequence 11-21, we can use trajetories generated from ORBSLAM2-S ([ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) stereo version w/ Loop Closing) to compare our prediction:
```bash
for seq in {11..21..1} ; do 
    evo_traj kitti --ref kitti_benchmark/data/odometry/poses_from_ORBSLAM2-S/${seq}-pred_kitti_pose.txt v1-upperflow-ALL-PoseLossWeak-decay100k-staircase--model-300000/poses/${seq}-pred_kitti_pose.txt <other_version_poses> \
        -p --plot_mode xz -s --save_plot plots/${figure_name}.png
done
```

If you use the `--save_plot` argument to save as png file, please change the format first:

```bash
evo_config set plot_export_format png
```

Or you can also change trajetory colors:

```bash
evo_config set plot_seaborn_palette Dark2
```

Please use `evo_config show` to see more configures.


# Contact
Please feel free to contact us if you have any questions :smile:

# Acknowledgements
We appreciate the great works/repos along this direction, such as [SfMLearner](https://github.com/tinghuiz/SfMLearner), [GeoNet](https://github.com/yzcjtr/GeoNet), [DeepMatchVO](https://github.com/hlzz/DeepMatchVO) and also the evaluation tools such as [KITTI VO/SLAM devkit](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [evo](https://github.com/MichaelGrupp/evo) for KITTI full sequence evaluation.
