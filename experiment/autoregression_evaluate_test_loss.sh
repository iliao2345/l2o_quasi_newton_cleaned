#!/bin/bash

#SBATCH -a 0-1
#SBATCH -c 20

source /etc/profile
module load anaconda/2021a
python experiment/autoregression_evaluate_test_loss.py -r $SLURM_ARRAY_TASK_ID -d experiment/autoregression_mnist/
