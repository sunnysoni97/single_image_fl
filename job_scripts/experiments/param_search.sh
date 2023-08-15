#!/bin/bash

echo "Queuing jobs for your hyper-parameter tuning experiments..."

seeds='42 84'
# server_lr='0.1 0.05 0.01 0.005 0.001'
# client_lr='0.01'
cluster_nums='5 10 50 100 1000'
# balancing_factor='0.01 0.1 0.5 1.0'
# entropy_thresholds='0.01 0.1 0.25 0.5 0.75' 


for cluster_num in $cluster_nums
do
    for seed in $seeds
    do
        sbatch job_scripts/jobfile_feddf.job -s $seed -k $cluster_num
    done
done
    