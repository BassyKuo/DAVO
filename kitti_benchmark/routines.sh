#!/bin/bash

#input="../list.txt"
input="$1"
suffix="$2"

while IFS=' ' read -r name ; do 
    folder=$(ls --color=no -td ../test-DCVO/data_aug/${name}*${suffix} | head -n1)
    save_name=$( echo $folder | cut -d'/' -f4 )
    ./pose_kitti_eval.sh $folder $save_name
    python show_errors.py -o results/$save_name/errors/
    printf "%s,%s\n" "$save_name" "$(cat ${save_name}.csv)" >> summary.csv
done < $input

