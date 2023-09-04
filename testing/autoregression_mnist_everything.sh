#!/bin/bash

#SBATCH -c 6

source /etc/profile
module load anaconda/2021a

python testing/autoregression_mnist_everything.py

