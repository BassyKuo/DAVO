# KITTI Evaultaion Toolkit

## How to use

1. run test_kitti_pose.py

    cd $DCVO_ROOT
    TEST_OUTPUT_DIR=outputs/
    MODEL_NAME=pretrain-model-258000
    CKPT_FILE=./pretrain-ckpt/model-258000
    for seq in {00..10..1} ; do 
        python test_kitti_pose.py --test_seq $seq --dataset_dir ./kitti_odom/ --output_dir ./$TEST_OUTPUT_DIR/$MODEL_NAME/$seq/ \
            --ckpt_file $CKPT_FILE --seq_length 3 --concat_img_dir ./kitti_odom-dump/ --batch_size 1
    done

2. use (evaluation_converter.sh)[../evaluation_converter.sh] convert outputs of sequence 00~10 from test_kitti_pose.py

    cd $DCVO_ROOT
    TEST_OUTPUT_DIR=outputs/
    MODEL_NAME=pretrain-model-258000
    #./evaluation_converter.sh $seq kitti_odom/poses/${seq}.txt test-pose_model_output_poses/pretrain-model-258000/$seq/${seq}-pred_kitti_pose.txt pretrain-model-258000
    ./evaluation_converter.sh $seq kitti_odom/poses/${seq}.txt $TEST_OUTPUT_DIR/$MODEL_NAME/$seq/${seq}-pred_kitti_pose.txt $MODEL_NAME
    done

3. run make

    cd $DCVO_ROOT/kitti_benchmark
    make

4. run testing

    ./test_odometry $MODEL_NAME 

5. run show_errors.py to show rotation error and translation error for each sequence

    # Show all sequence results
    python show_errors.py results/$MODEL_NAME/errors/
    # Show specitic sequence results
    python show_errors.py results/$MODEL_NAME/errors/ -seq 0 1 2 10
    # Save results into file
    python show_errors.py results/$MODEL_NAME/errors/ -seq 0 1 2 10



## Source

1. odometry benchmark: http://www.cvlibs.net/datasets/kitti/eval_odometry.php \[(downlaod)[https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_odometry.zip]\]
2. pose ground truth: http://www.cvlibs.net/download.php?file=data_odometry_poses.zip



## Other tools

1. (python tools)[https://github.com/utiasSTARS/pykitti]
2. (matlab tools)[https://github.com/ambarpal/3d-hough/tree/master/code/kitti/devkit]
