#!/bin/bash

# Usage:
#    ./pose_kitti_eval.sh <prediction_dir> <save_dir_name>

if [ $# -lt 2 ] ; then
    echo -e "\033[1;31m[ERROR]\033[0m Need at least 2 arguments."
    echo -e "\033[1;33m[Usage]\033[0m "
    echo "    ./pose_kitti_eval.sh <prediction_dir> <save_dir_name>"
    exit 1
fi

est_dir="$1"                        # [example] test/baseline/
save_name="$2"                      # [example] baseline_dir
# -------------------------------------------------------------------------------------------------------------------------------------------------

if [ ! -f ${est_dir}/00-pred_kitti_pose.txt ] ; then 
    echo -e "\033[1;31m[ERROR]\033[0m Cannot find ${est_dir}/*-pred_kitti_pose.txt. Please run 'test_kitti_pose.py' first."
    exit 1
fi

# ===[ Main
# Fixed variables
kitti_result_dir="./results/${save_name}/"
kitti_result_datadir="${kitti_result_dir}/data/"
test -d ${kitti_result_datadir} || mkdir -p ${kitti_result_datadir}

for seq_name in `seq -w 00 10` ; do 
    # Copy est_file to kitti_benchmark directory
    est_file=${est_dir}/${seq_name}-pred_kitti_pose.txt
    cp ${est_file} ${kitti_result_datadir}/${seq_name}.txt
done

# Run KITTI benchmark.
./test_odometry_all ${save_name}
# Get and show RMSE of translation and rotation.
python show_errors.py -o results/${save_name}/errors/
printf "%s,%s\n" "${save_name}" "$(cat ${save_name}.csv)"
