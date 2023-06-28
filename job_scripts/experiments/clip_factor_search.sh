#!/bin/bash

echo "Queuing jobs for clipping factor experiment..."

clip_factor='0.5 1.0 1.5 2.0'

for cf in $clip_factor
do
    sbatch job_scripts/jobfile_feddf.job -c $cf
done
