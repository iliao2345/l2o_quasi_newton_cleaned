# This file is a basic test run of the Resnet classification task.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import subprocess
import sys

import context
from tasks import resnet_classification
from training import minimization_procedure
from training import learning_curve_tools2
from training import hyperparameter_setup
from training import training_routine_parent2
import argparse

parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
parser.add_argument('-r', type=int, dest='run_number', required=True)
parsed_args = parser.parse_args()

# Create all necessary directories and a file for printouts
experiment_directory = "testing/resnet_classification/"
os.makedirs(experiment_directory, exist_ok=True)
for subdirectory in ("tuning/", "training/", "generated_images/", "training/learning_curve_data", "models/", "sweeping/"):
    os.makedirs(experiment_directory + subdirectory, exist_ok=True)
tuning_directory = experiment_directory + "tuning/"
training_directory = experiment_directory + "training/"
models_directory = training_directory + "models/"

# Run the training
#step_milestone = 400  # 1.33 steps per second or so for Adam, batch size 256
#step_milestone = 3  # 0.14 steps per second or so for LODO, batch size 256
#step_milestone = 4500  # 0.36 steps per second or so for Adam, batch size 2048
#step_milestone = 3  # 0.125 steps per second or so for LODO, batch size 2048
#time_milestone = 36000

time_milestone = 1
step_milestone = 3000
#step_milestone = 7000

#name = "LODO"
#hyperparameters = list(map(lambda x: float(x), hyperparameter_setup.resnet_defaults["LODO"]))
#name = "Adam"
#hyperparameters = list(map(lambda x: float(x), hyperparameter_setup.resnet_defaults["Adam"]))
name = "Momentum"
hyperparameters = [0.1, 0.9]
record_steps = [step_milestone]
record_times = [time_milestone]
run_number = parsed_args.run_number
n_runs = 5
save_directory = training_directory
task_name = "ResnetCIFAR10Classification"
validation_loss_evaluation_interval = 24

all_regularizations = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
regularization = all_regularizations[run_number]
minimizer = hyperparameter_setup.minimizer_setup_fns[name](*hyperparameters)
task = resnet_classification.ResnetClassificationTask(regularization)
initialization = task.get_initialization()
print("A")
sys.stdout.flush()

#tracked_quantity_names = ["validation loss", "validation accuracy", "test loss", "test accuracy"]
#tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x), lambda x: task.evaluate_test_loss(x), lambda x: task.evaluate_test_accuracy(x)]
tracked_quantity_names = ["validation loss", "validation accuracy"]
tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x)]
training_iteration_handler = learning_curve_tools2.TrainingIterationHandlerWithParameterSaving(
        task,
        time_limit=record_times[-1]+60,
        max_iteration=record_steps[-1]+2,
        break_condition="both",
        save_at_steps=record_steps,
        save_at_times=record_times,
        name = name + "_" + str(run_number),
        save_directory=save_directory,
        tracked_quantity_evaluation_interval=validation_loss_evaluation_interval,
        tracked_quantity_names = tracked_quantity_names,
        tracked_quantity_fns = tracked_quantity_fns,
)
print("B")
sys.stdout.flush()

# Perform the training task and save the learning curve
minimum = minimizer(training_iteration_handler, initialization)
print("C")
sys.stdout.flush()
training_iteration_handler.save_learning_curve(save_directory + "learning_curve_data/" + name + "_" + str(run_number))


learning_curve_data = []
for i in range(5):
    fname = training_directory + "learning_curve_data/" + name + "_" + str(i)
    learning_curve_data.append(learning_curve_tools2.LearningCurveFromFile(fname, name=name).learning_curve)

fig, ax = plt.subplots()
for learning_curve, regularization in zip(learning_curve_data, all_regularizations):
    ax.plot(np.arange(learning_curve.shape[0]), np.clip(learning_curve[:,2], 0, 5), label=str(regularization))
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend(loc="center left", bbox_to_anchor=(1.00, 0.5))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='y', which='minor', bottom=False)
plt.grid(b=True, which='major', color='0.6')
plt.grid(b=True, which='minor', color='0.8')
plt.savefig(experiment_directory + "sweeping/regularization_sweep.pdf", format="pdf", bbox_inches="tight")
plt.close()

learning_curve_data = []
for i in range(5):
    fname = training_directory + "learning_curve_data/" + name + "_" + str(i)
    learning_curve_data.append(learning_curve_tools2.LearningCurveFromFile(fname, name=name).tracked_quantities)

fig, ax = plt.subplots()
for learning_curve, regularization in zip(learning_curve_data, all_regularizations):
    ax.plot(np.arange(learning_curve.shape[0]), np.clip(learning_curve[:,3], 0, 1), label=str(regularization))
ax.set_xlabel("step")
ax.set_ylabel("accuracy")
ax.legend(loc="center left", bbox_to_anchor=(1.00, 0.5))
ax.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='y', which='minor', bottom=False)
plt.grid(b=True, which='major', color='0.6')
plt.grid(b=True, which='minor', color='0.8')
plt.savefig(experiment_directory + "sweeping/regularization_sweep_acc.pdf", format="pdf", bbox_inches="tight")
plt.close()
