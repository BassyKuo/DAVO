#!/bin/bash

# Usage:
#    ./pose_kitti_eval.sh <prediction_dir> <save_dir_name>

# < 2019/11/7 by bass >

if [ $# -lt 2 ] ; then
    echo -e "\033[1;31m[ERROR]\033[0m Need at least 2 arguments."
    echo -e "\033[1;33m[Usage]\033[0m "
    echo "    ./pose_kitti_eval.sh <prediction_dir> <save_dir_name>"
    exit 1
fi

est_dir="$1"                        # [example] test/baseline/
save_name="$2"                      # [example] baseline_dir
#seq_name="${3:-00}"                 # [example] 03
#ref_file="${4:-kitti_benchmark/data/odometry/poses/${seq_name}.txt}"                       # [example] kitti_benchmark/data/odometry/poses/03.txt
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
for seq_name in `seq -w 00 10` ; do
   filename=${kitti_result_dir}/${seq_name}-stats.txt
   python -c "t,r = open('"${filename}"').read()[:-1].split(' ') ; print('%.6f %.6f' % (float(t) * 100. , float(r) * 57.3 * 100.))"       # trans unit: [0.01m/m] ; rot unit: [deg/100m]
done
echo "[== Avg. Error ==]"
filename=${kitti_result_dir}/avg-stats.txt
python -c "t,r = open('"${filename}"').read()[:-1].split(' ') ; print('%.6f %.6f' % (float(t) * 100. , float(r) * 57.3 * 100.))"       # trans unit: [0.01m/m] ; rot unit: [deg/100m]
