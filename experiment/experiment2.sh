#!/bin/bash

#SBATCH -c 6

source /etc/profile
module load anaconda/2021a

#python experiment/experiment2.py -T Rosenbrock
#python experiment/experiment2.py -T Autoregression
#python experiment/experiment2.py -T ClassificationHead
python experiment/experiment2.py -T ResnetCIFAR10Classification
