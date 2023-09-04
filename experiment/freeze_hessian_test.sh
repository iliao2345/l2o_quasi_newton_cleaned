#!/bin/bash
#SBATCH -a 0-2
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
source /etc/profile
module load anaconda/2021a
export TF_CUDNN_USE_AUTOTUNE=1
python freeze_hessian_test.py -x 0.270 0.00096 0.195 -s 100000 -t 1 -r $SLURM_ARRAY_TASK_ID -R 3 -d noisy_quadratic_bowl/training2/ -T NoisyQuadraticBowl -v 1
