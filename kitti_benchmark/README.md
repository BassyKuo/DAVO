# KITTI Evaultaion Toolkit

## How to use

1. run test_kitti_pose.py

    ```
    cd $DAVO_ROOT
    CKPT_FILE="./pretrain-ckpt/model-1500000"
    TEST_OUTPUT_DIR="./outputs/"
    MODEL_NAME="model-1500000"
    VERSION="v1-decay100k-sharedNN-dilatedPoseNN-cnv6_128-segmask_all-se_flow-abs_flow-fc_tanh"
    for seq in {00..10..1} ; do 
        python test_kitti_pose.py \
            --test_seq=$seq \
            --output_dir=./$TEST_OUTPUT_DIR/$MODEL_NAME/ \
            --ckpt_file=$CKPT_FILE \
            --concat_img_dir=./kitti_odom-dump/ \
            --version=$VERSION
    done
    ```

2. Copy poses to the specific folder for the KITTI evaluation toolkit

    ```
    est_dir="./$TEST_OUTPUT_DIR/$MODEL_NAME"
    kitti_result_dir="./results/${MODEL_NAME}/"
    kitti_result_datadir="${kitti_result_dir}/data/"

    for seq in {00..10..1} ; do 
        # Copy est_file to kitti_benchmark directory
        est_file=${est_dir}/${seq}-pred_kitti_pose.txt
        cp ${est_file} ${kitti_result_datadir}/${seq}.txt
    done
    ```

3. (optional) generate KITTI evaluation tool: test_odometry_all

    ```
    cd $DAVO_ROOT/kitti_benchmark
    make
    ```

4. run testing

    ```
    ./test_odometry_all $MODEL_NAME 
    ```

5. run show_errors.py to show translational and rotational error for each sequence

    ```
    # Show all sequence results
    python show_errors.py results/$MODEL_NAME/errors/
    # Show specitic sequence results
    python show_errors.py results/$MODEL_NAME/errors/ -seq 0 1 2 10
    # Save results into file
    python show_errors.py results/$MODEL_NAME/errors/ -seq 0 1 2 10
    # Save all sequence results in ${MODEL_NAME}.csv file
    python show_errors.py -o results/$MODEL_NAME/errors/
    ```



## Source

1. odometry benchmark: http://www.cvlibs.net/datasets/kitti/eval_odometry.php \[[downlaod toolkit](https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip)\]
2. pose ground truth: http://www.cvlibs.net/download.php?file=data_odometry_poses.zip



## Other tools

1. [python tools](https://github.com/utiasSTARS/pykitti)
2. [matlab tools](https://github.com/ambarpal/3d-hough/tree/master/code/kitti/devkit)
