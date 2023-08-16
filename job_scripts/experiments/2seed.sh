#!/bin/bash

echo "Queuing experiment with 2 seeds"

seeds='42 84'

for seed in $seeds
do
    sbatch job_scripts/jobfile_feddf.job -s $seed
done
