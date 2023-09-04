# This file is a basic test run of the Resnet classification task.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import subprocess

import context
from tasks import resnet_classification
from training import minimization_procedure
from training import learning_curve_tools2
from training import hyperparameter_setup
from training import training_routine_parent2

# Create all necessary directories and a file for printouts
experiment_directory = "testing/resnet_classification/"
os.makedirs(experiment_directory, exist_ok=True)
for subdirectory in ("tuning/", "training/", "generated_images/", "training/learning_curve_data", "models/"):
    os.makedirs(experiment_directory + subdirectory, exist_ok=True)
tuning_directory = experiment_directory + "tuning/"
training_directory = experiment_directory + "training/"
models_directory = training_directory + "models/"
log_fname = experiment_directory + "log.txt"
log_file = open(log_fname, "w")
log_file.close()
log_file = open(log_fname, "a")

# Run the training
#step_milestone = 400  # 1.33 steps per second or so for Adam, batch size 256
#step_milestone = 3  # 0.14 steps per second or so for LODO, batch size 256
#step_milestone = 4500  # 0.36 steps per second or so for Adam, batch size 2048
#step_milestone = 3  # 0.125 steps per second or so for LODO, batch size 2048
#time_milestone = 36000

time_milestone = 1000
step_milestone = 200

command = ["python", "training/training_routine_child2.py"]
#command = command + ["-n", "Adam"]
command = command + ["-n", "LODO"]
#command = command + ["-x"] + list(map(lambda x: str(float(x)), hyperparameter_setup.resnet_defaults["Adam"]))
command = command + ["-x"] + list(map(lambda x: str(float(x)), hyperparameter_setup.resnet_defaults["LODO"]))
command = command + ["-s", str(step_milestone)]
command = command + ["-t", str(time_milestone)]
command = command + ["-r", "0"]
command = command + ["-R", "1"]
command = command + ["-d", training_directory]
command = command + ["-T", "ResnetCIFAR10Classification"]
command = command + ["-v", "24"]
print("start")
#print(subprocess.Popen(command, stdout = subprocess.PIPE).communicate()[0].decode("utf-8"))
print("finish")

# Read all the files containing learning curve data
main_optimizer_names = hyperparameter_setup.names
def read_main_test_learning_curves():
    """
    Read all the learning curve data files from training.
    """
#    fname = training_directory + "learning_curve_data/Adam_0"
    fname = training_directory + "learning_curve_data/LODO_0"
#    learning_curves = [learning_curve_tools2.LearningCurveFromFile(fname, name="Adam")]
    learning_curves = [learning_curve_tools2.LearningCurveFromFile(fname, name="LODO")]
    learning_curve_data = {"resnet_classification_run": learning_curves}
    return learning_curve_data

learning_curve_data = read_main_test_learning_curves()
print(learning_curve_data['resnet_classification_run'][0].learning_curve.shape)
print(learning_curve_data['resnet_classification_run'][0].tracked_quantities.shape)
print(learning_curve_data['resnet_classification_run'][0].tracked_quantities)

# Plot all the optimizer performances by step and time
reasonable_loss_range = (0, 5)
for mode in ("step", "time"):
    included_learning_curve_data = dict(learning_curve_data)
    learning_curve_tools2.draw_learning_curve_lists(
            included_learning_curve_data.values(),
            experiment_directory + "learning_curves_by_" + mode + ".pdf",
            loss_range=reasonable_loss_range,
#            x_max = step_milestone if mode == "step" else time_milestone,
            x_max = 200 if mode == "step" else 1000,
            x_axis=mode
            )
learning_curve_tools2.draw_metric_lists(
#        [value for key, value in included_learning_curve_data.items() if "LODO" in key],
        [value for key, value in included_learning_curve_data.items()],
        experiment_directory + "validation_learning_curves_by_" + mode + ".pdf",
        metric_name="validation loss",
        loss_range=reasonable_loss_range,
#        x_max=step_milestone,
        x_max=200,
        )

log_file.close()
