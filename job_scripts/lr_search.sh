#!/bin/bash

echo "Queuing jobs for your hyper-parameter tuning experiments..."

server_lr='0.003 0.008'
client_lr='0.3 0.8'

for slr in $server_lr
do
    for llr in $client_lr
    do
        sbatch job_scripts/jobfile_feddf.job -l $llr -g $slr
    done
done
    