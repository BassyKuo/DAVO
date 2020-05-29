#!/bin/bash
#================================================================

CKPT_FOLDER="$1"                        # ckpt dir name                          [./ckpt/XXXX/]
version="$2"                            # model version                          [v1-xxxxx]
seq="$3"                                # sequence name                          [00]
step="$4"                               # ckpt step                              [1500000]
TEST_FOLDER=${5:-"test-DAVO"}           # directory where prediction files save  [./test-DAVO/]

#================================================================

if [ "$#" -le 0 ] || [ "$#" -gt 6 ] ; then 
    echo -e "\033[0;31m[ERROR]\033[0m illegal number of parameters."
    echo "" 
    echo "[Command]"
    echo "    ./run_inference.sh <ckpt_dir> <version> <seq_name> <model_step> [<output_folder>] "
    echo ""
    echo "[Example]"
    echo "    ./run_inference.sh ckpt/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh 05 1500000"
    echo "or"
    echo "    ./run_inference.sh ckpt/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh 05 1500000 test-DAVO/"
    exit 1
fi

#DATASET_DIR="./kitti_odom-dump/"
#DATASET_DIR="./malaga_odom-dump/10fps/"
DATASET_DIR="./Malaga_odom/malaga_odom-dump/10fps/"
#DATASET_DIR="./Malaga_odom/malaga_odom-dump/5fps/"

CKPT_FOLDER=`echo $CKPT_FOLDER | sed 's#\/$##g'`
checkpoint=`head -n1 $CKPT_FOLDER/checkpoint | sed 's/^.*\"\([^"]\+\)\"/\1/g' | sed 's/^[^0-9]*\([0-9]\+\)/\1/g'`
model_name=model-${step:-$checkpoint}
save_model_name=`echo $CKPT_FOLDER | sed 's#^.*\/\(v[^/]*\)/*#\1#g'`
#output_subdir=`echo $CKPT_FOLDER | sed 's#.*ckpt\/\([^/]\+\)\/.*#\1#g'`

ckpt_file="$CKPT_FOLDER/$model_name"
output_dir="./$TEST_FOLDER/${save_model_name}--$model_name"

if ! test -f ${ckpt_file}.index ; then 
    echo -e "\033[0;31m[ERROR]\033[0m Cannot find chekcpoint file: ${ckpt_file}"
    exit 1
fi

echo "Call $model_name >>>"
echo "Processing ... $seq"
python test_kitti_pose.py \
    --concat_img_dir ${DATASET_DIR} \
    --test_seq $seq \
    --output_dir ${output_dir}/ \
    --ckpt_file ${ckpt_file} \
    --seq_length 3  \
    --batch_size 1 \
    --version ${version}

echo "Done."
echo "Please check ${output_dir}"
