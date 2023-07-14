#!/bin/bash

echo "Queuing jobs for clipping factor experiment..."

clip_factor='1.5 2.0 2.5 3.0 3.5'

for cf in $clip_factor
do
    sbatch job_scripts/jobfile_feddf.job -c $cf
done
