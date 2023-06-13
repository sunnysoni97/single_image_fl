#!/bin/bash

#SBATCH --time=00:59:00
#SBATCH --partition=staging
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --job-name=sunny_test_job

# module purge
# module load 2022
# module load Anaconda3/2022.05

# srun python test_file.py

DATA_DIR=$TMPDIR'/'`date +"%s_%N"`

echo $DATA_DIR