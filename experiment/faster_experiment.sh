#!/bin/bash

#SBATCH -c 6

source /etc/profile
module load anaconda/2021a

python experiment/faster_experiment.py
