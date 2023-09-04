#!/bin/bash

#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/2022a
export TF_CUDNN_USE_AUTOTUNE=1

python train.py --model resnet18 --epoch 18
