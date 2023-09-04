# This file contains code which evaluates the test loss for the autoregression task, with a model read from a file.

import argparse
import numpy as np
import tensorflow as tf
import pickle
import os

import context
from tasks import autoregression_mnist

# Get all the tuning task parameters from the training_routine_parent.py through argparse
parser = argparse.ArgumentParser(description='Evaluate autoregression test loss')
parser.add_argument('-r', type=int, dest='run_number', required=True)
parser.add_argument('-d', type=str, dest='save_directory', required=True)
parsed_args = parser.parse_args()

run_number = parsed_args.run_number  # this number differentiates between copies of jobs running this program
save_directory = parsed_args.save_directory  # directory to save the models and learning curves in

experiment_directory = "experiment/autoregression_mnist/"
training_directory = experiment_directory + "training2/"
models_directory = training_directory + "models/"
log_fname = experiment_directory + "test_loss_log.txt"


# Set up the training task
task = autoregression_mnist.AutoregressionTask()

optimizer_names = [
#    "Momentum",
#    "RMSprop",
#    "Adam",
#    "LARS",
#    "Yogi",
#    "LODO",
#    "LODO-Diagonal",
#    "LODO-Global",
#    "LODO-Residuals"
    "LODO-No-Momentum",
#    "LODO-SGD"
]

i = 0
for name in optimizer_names:
    for milestone_value, milestone_unit in [(300000, "steps"), (50000, "seconds")]:
        if i == run_number:
            fname = name + "_1_parameters_at_" + str(int(milestone_value)) + "_" + milestone_unit
            with open(os.path.join(models_directory, fname), "rb") as f:
                weights = pickle.load(f)
            loss = task.evaluate_test_loss(weights)
            with open(log_fname, "a") as f:
                f.write("\n" + name + " test loss after training up to " + str(milestone_value) + " " + milestone_unit + " for a randomly chosen training run: " + str(float(loss)) + "\n")
        i += 1
