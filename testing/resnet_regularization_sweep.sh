#!/bin/bash

#SBATCH -a 0-4
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2021a
export TF_CUDNN_USE_AUTOTUNE=1

python testing/resnet_regularization_sweep.py -r $SLURM_ARRAY_TASK_ID
