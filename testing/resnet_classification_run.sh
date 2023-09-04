#!/bin/bash

#SBATCH -c 40
#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2

source /etc/profile
module load anaconda/2021a
export TF_CUDNN_USE_AUTOTUNE=1

python testing/resnet_classification_run.py
