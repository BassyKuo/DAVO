#!/bin/bash
#================================================================

CKPT_FOLDER="$1"                        # ckpt dir name                          [./ckpt/XXXX/]
version="$2"                            # model version                          [v1-xxxxx]
seq="$3"                                # sequence name                          [00]
step="$4"                               # ckpt step                              [1500000]
TEST_FOLDER=${5:-"test-DAVO"}           # directory where prediction files save  [./test-DAVO/]

#================================================================

if [ "$#" -le 0 ] || [ "$#" -gt 6 ] ; then 
    echo "ERROR: illegal number of parameters."
    echo "[Command]"
    echo "    ./run_estimation.sh <ckpt_dir> <version> <seq_name> <model_step> [<output_folder>] "
    echo ""
    echo "[Example]"
    echo "    ./run_estimation.sh ckpt/data_aug/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh 05 1500000"
    echo "or"
    echo "    ./run_estimation.sh ckpt/data_aug/v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh 05 1500000 test-DAVO/"
fi

CKPT_FOLDER=`echo $CKPT_FOLDER | sed 's#\/$##g'`
checkpoint=`head -n1 $CKPT_FOLDER/checkpoint | sed 's/^.*\"\([^"]\+\)\"/\1/g' | sed 's/^[^0-9]*\([0-9]\+\)/\1/g'`
model_name=model-${step:-$checkpoint}
save_model_name=`echo $CKPT_FOLDER | sed 's#^.*\/\(v[^/]*\)/*#\1#g'`
output_subdir=`echo $CKPT_FOLDER | sed 's#.*ckpt\/\([^/]\+\)\/.*#\1#g'`

ckpt_file="$CKPT_FOLDER/$model_name"
output_dir="./$TEST_FOLDER/$output_subdir/${save_model_name}--$model_name"

test -f ${ckpt_file}.index || exit 1

echo "Call $model_name >>>"
echo "Processing ... $seq"
python test_kitti_pose.py    --test_seq $seq \
    --output_dir ${output_dir}/    --ckpt_file ${ckpt_file}  --seq_length 3  --concat_img_dir ./kitti_odom-dump/   --batch_size 1 --version ${version}

echo "Done."
echo "Please check ${output_dir}"

#test $seq -eq 8 && python generate_eval_results.py 08 ${output_dir}
