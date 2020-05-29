#!/bin/bash

list="list4"
plot_dir="malaga-plots-4"

# inf
while IFS=" " read -r name step alias ; do for step in {10..16} ; do for i in {00..10} ; do 
    ./run_inference.sh ckpt/$name/ $name $i ${step}00000 test_DAVO/KITTI-seq/ ; done ; done ; done < ${list}

#while IFS=" " read -r name step alias ; do for step in {10..16} ; do for i in 03 04 09 ; do 
#    ./run_inference_malaga.sh ckpt/$name/ $name $i ${step}00000 test_DAVO/Malaga-seq-10fps/ ; done ; done ; done < ${list}

while IFS=" " read -r name step alias ; do for step in {10..16} ; do for i in 03 04 09 ; do 
   ./run_inference_malaga.sh ckpt/$name/ $name $i ${step}00000 test_DAVO/Malaga_odom-10fps/ ; done ; done ; done < ${list}

# eval
cd kitti_benchmark
while IFS=" " read -r name step alias ; do for step in {10..16} ; do ./pose_kitti_eval.sh ../test_DAVO/KITTI-seq/${name}--model-${step}00000/ ${name}--model-${step}00000 ; done ; done < ../${list}
while IFS=" " read -r name step alias ; do for step in {10..16} ; do save_name="${name}--model-${step}00000" ; printf "%s,%s\n" "${save_name}" "$(cat ${save_name}.csv)"  ; done ; done < ../${list}

# evo_traj
while IFS=" " read -r name step alias ; do for step in {10..16} ; do 
    save_name="${name}--model-${step}00000" ;  mkdir -p results/$save_name/Malaga_odom-10fps-poses ; for i in 03 04 09; do 
    cp  ../test_DAVO/Malaga_odom-10fps/$save_name/$i-pred_kitti_pose.txt results/$save_name/Malaga_odom-10fps-poses/$i-${alias}-${step}00k-odom-10fps.txt ; done ; done ; done < ../${list}
mkdir ${plot_dir}
while IFS=" " read -r name step alias ; do for step in {10..16} ; do for i in 03 04 09 ; do 
    save_name="${name}--model-${step}00000" ;  
    evo_traj kitti --ref /warehouse/bass/SLAM/Monocular/Malaga_odom/malaga_odom-dump/10fps/$i-malaga-10fps.txt results/$save_name/Malaga_odom-10fps-poses/$i-${alias}-${step}00k-odom-10fps.txt \
        --plot_mode xz --save_plot ${plot_dir}/${alias}-00to10-${i}-${step}00k-10fps.png & done ; done ; done < ../${list}
