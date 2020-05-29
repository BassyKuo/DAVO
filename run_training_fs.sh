#!/bin/bash

version="$1"
name="$version"-fs
with_data_flip=--data_flip
with_data_aug="$2"
i=2

if [ "$#" -le 0 ] || [ "$#" -gt 2 ] ; then 
    echo "ERROR: illegal number of parameters."
    echo ""
    echo "[Example]"
    echo "    ./run_training.sh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh"
    echo "or"
    echo "    ./run_training.sh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh --data_aug"
fi

while : ; do 
    test -d "./ckpt/${name}${with_data_aug}${with_data_flip}" || break
    name=${version}-$i
    i=$(($i+1))
done

echo "ckpt_dir: ./ckpt/${name}${with_data_aug}${with_data_flip}"
python train.py \
    --dataset_dir=./kitti_odom_fsdump/ \
    --img_width=416 \
    --img_height=128 \
    --batch_size=4 \
    --seq_length 3 \
    --max_steps 1600000  \
    --save_freq 100000 \
    --summary_freq 25000 \
    --learning_rate 0.001 \
    --pose_weight 0.1 \
    --checkpoint_dir ./ckpt_fs/${name}${with_data_aug}${with_data_flip} ${with_data_aug} ${with_data_flip} \
    --version ${version} #> ${name}${with_data_aug}${with_data_flip}.log
