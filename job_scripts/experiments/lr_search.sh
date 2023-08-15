#!/bin/bash

echo "Queuing jobs for your hyper-parameter tuning experiments..."

seeds='42 84'
server_lr='0.1 0.05 0.01 0.005 0.001'
client_lr='0.01'


for slr in $server_lr
do
    for llr in $client_lr
    do
        for seed in $seeds
        do
            sbatch job_scripts/jobfile_feddf.job -l $llr -g $slr -s $seed
        done
    done
done
    