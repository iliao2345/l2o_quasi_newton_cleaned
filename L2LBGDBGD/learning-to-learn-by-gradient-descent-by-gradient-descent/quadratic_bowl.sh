#!/bin/bash

#SBATCH --exclusive
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40

source /etc/profile
module load anaconda/2021a
export TF_CUDNN_USE_AUTOTUNE=1
python quadratic_bowl.py

