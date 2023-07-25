#!/bin/bash

echo "Queuing jobs for confidence threshold experiment..."

conf_thresholds='0.5 0.6 0.7 0.8 0.9'

for ct in $conf_thresholds
do
    sbatch job_scripts/jobfile_feddf.job -t $ct
done
