#!/bin/bash

echo "Queuing experiment with 3 seeds"

seeds='42 127 1227'

for seed in $seeds
do
    sbatch job_scripts/jobfile_feddf.job -s $seed
done
