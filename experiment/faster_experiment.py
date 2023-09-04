import numpy as np
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import pickle
import argparse

import context
from tasks import resnet_classification
from training import minimization_procedure
from training import learning_curve_tools2
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from tuning import hyperparameter_tuning_parent2
from training import training_routine_parent2

# Get all the tuning task parameters from the training_routine_parent2.py through argparse
parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
parsed_args = parser.parse_args()

# Create a file for printouts
experiment_directory = "experiment/resnet_classification/"
os.makedirs(experiment_directory, exist_ok=True)
for subdirectory in ("tuning2/", "training2/", "visualizations2/"):
    os.makedirs(experiment_directory + subdirectory, exist_ok=True)
tuning_directory = experiment_directory + "tuning2/"
training_directory = experiment_directory + "training2/"
models_directory = training_directory + "models/"
log_fname = experiment_directory + "log.txt"
log_file = open(log_fname, "w")
log_file.close()
log_file = open(log_fname, "a")

# Set up the training task
task = resnet_classification.ResnetClassificationTask()
reasonable_loss_range = (0, 5)

# Tune all the optimizers
default_hyperparameters = hyperparameter_setup.resnet_defaults
optimizer_names = [
    "Momentum",
    "RMSprop",
    "Adam",
    "Yogi",
#    "L-BFGS-no-line-search",
#    "O-LBFGS",
    "LODO",
]
#log_file.write("\nHyperparameter names are shown below:\n" + str(optimizer_names) + "\n")
#log_file.write("\nDefault hyperparameters are shown below:\n" + str({name : default_hyperparameters[name] for name in optimizer_names}) + "\n")
#tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
#        (3, 50, 16),
#        (3, 50, 16),
#        (3, 50, 16),
#        (3, 50, 16),
#        (2, 50, 16),
#        (1.7, 60, 16),
#        (1.4, 100, 16),
#        (1.2, 200, 16),
#        (0.9, 300, 16),
#        (0.6, 500, 16),
#]
#def tune_all_hyperparameters():
#    """
#    Tune all of the hyperparameters which we want to run training with.
#    """
#    tuner = hyperparameter_tuning_parent2.HyperparameterTuningMethod(
#            tuning_directory,
#            "ResnetCIFAR10Classification",
#            optimizer_names,
#            tuning_schedule=tuning_schedule,
#            plot_loss_range=reasonable_loss_range,
#            update_interval=60
#    )
#    best_hyperparameters = tuner.tune_all()
#    log_file.write("\nBest hyperparameters are shown below:\n" + str(best_hyperparameters) + "\n")
#    return best_hyperparameters
#
#best_hyperparameters = tune_all_hyperparameters()
best_hyperparameters = {name: default_hyperparameters[name] for name in optimizer_names}
best_hyperparameters["LODO"] = (0.08459, 7.946e-6, 0.9343)

# Do the main training run
step_milestone = 2000
time_milestone = 1
metric_evaluation_interval = 100
def train_with_all_optimizers():
    """
    Train on the task using multiple copies of each of the optimizers. Results are saved in training_directory.
    """
    all_hyperparameters = best_hyperparameters
    all_optimizer_names = all_hyperparameters.keys()
    trainer = training_routine_parent2.TrainingRoutine(
            training_directory,
            "ResnetCIFAR10Classification",
            all_optimizer_names,
            all_hyperparameters,
            save_iterations=[step_milestone],
            save_times=[time_milestone],
            population_size=4,
            plot_loss_range=reasonable_loss_range,
            update_interval=60,
            metric_evaluation_interval=metric_evaluation_interval
    )
    trainer.run_all()

train_with_all_optimizers()  # Submits training/training_routine_child jobs which write to output files

# Read all the files containing learning curve data, including fixed ablations but not parameterized ablations
def read_main_test_learning_curves():
    """
    Read all the learning curve data files from training.
    """
    learning_curve_data = dict()
    learning_curves_directory = training_directory + "learning_curve_data/"
    filenames = [f for f in os.listdir(learning_curves_directory) if os.path.isfile(os.path.join(learning_curves_directory, f))]
    for i, name in enumerate(optimizer_names):
        learning_curves = []
        match_pattern = "^" + name + "[0-9_]+$"
        for fname in filenames:
            if re.match(match_pattern, fname):
                learning_curve = learning_curve_tools2.LearningCurveFromFile(os.path.join(learning_curves_directory, fname), name=name)
                learning_curves.append(learning_curve)
        learning_curve_data[name] = learning_curves
    return learning_curve_data

learning_curve_data = read_main_test_learning_curves()

# Plot all the optimizer performances by step and time
def plot_learning_curves():
    learning_curve_data_no_ablations = {name:learning_curve_data[name] for name in optimizer_names if name in learning_curve_data}
    learning_curve_data_no_ablations = {name:li for name, li in learning_curve_data_no_ablations.items() if len(li)>=2}
    for mode in ("step", "time"):
        learning_curve_tools2.draw_learning_curve_lists(
                learning_curve_data_no_ablations.values(),
                experiment_directory + "learning_curves_by_" + mode + ".pdf",
                loss_range=reasonable_loss_range,
                x_max = step_milestone if mode == "step" else time_milestone,
                x_axis=mode
        )
        learning_curve_tools2.draw_metric_lists(
                learning_curve_data_no_ablations.values(),
                experiment_directory + "validation_learning_curves_by_step.pdf",
                metric_name="validation loss",
                loss_range=reasonable_loss_range,
                x_max=step_milestone,
        )
    learning_curve_tools2.draw_metric_lists(
            learning_curve_data_no_ablations.values(),
            experiment_directory + "validation_accuracy_by_step.pdf",
            metric_name="validation accuracy",
            loss_range=reasonable_loss_range,
            x_max=step_milestone,
    )
    learning_curve_tools2.draw_metric_lists(
            learning_curve_data_no_ablations.values(),
            experiment_directory + "test_learning_curves_by_step.pdf",
            metric_name="test loss",
            loss_range=reasonable_loss_range,
            x_max=step_milestone,
    )
    learning_curve_tools2.draw_metric_lists(
            learning_curve_data_no_ablations.values(),
            experiment_directory + "test_accuracy_by_step.pdf",
            metric_name="test accuracy",
            loss_range=reasonable_loss_range,
            x_max=step_milestone,
    )

plot_learning_curves()

# Record all the training set performances at specified step and time
def print_training_performances():
    """
    Write to the log file the means and stddevs of final loss over all the runs of identical optimizers.
    """
    for mode in ("step", "time"):
        x_max = step_milestone if mode == "step" else time_milestone
        log_file.write("\nTraining loss performance during final 90% of training up to " + mode + " " + str(x_max) + ":\n")
        x_axis_index = 0 if mode == "step" else 1
        for name in learning_curve_data.keys():
            means = []
            n_diverged = 0
            for learning_curve in learning_curve_data[name]:
                samples = learning_curve.learning_curve[np.logical_and(learning_curve.learning_curve[:,x_axis_index] < x_max, learning_curve.learning_curve[:,x_axis_index] > 0.9*x_max)][:,2]
                mean = np.mean(samples)
                if not np.isfinite(mean):
                    n_diverged += 1
                    continue
                means.append(mean)
            mean_over_runs = float(np.mean(means))
            stddev_over_runs = float(np.std(means))
            log_file.write(name + ": " + str(mean_over_runs) + " +/- " + str(stddev_over_runs) + ", (" + str(n_diverged) + "/" + str(len(learning_curve_data[name])) + " diverged)\n")

print_training_performances()

# Record how many steps each optimizer takes per second
def print_training_rates():
    """
    Write to the log file the number of steps each optimizer took per second, averaged over all the runs of identical optimizers.
    """
    log_file.write("\nStep/second training speed during training time for all training runs:\n")
    for name in learning_curve_data.keys():
        ratios = []
        for learning_curve in learning_curve_data[name]:
            ratios.append(np.max(learning_curve.learning_curve[:,0])/np.max(learning_curve.learning_curve[:,1]))
        log_file.write("Average for " + name + ": " + str(float(np.mean(ratios))) + " +/- " + str(float(np.std(ratios))) + "\n")
    log_file.write("\nStep/second training speed during training+validation time for all training runs:\n")
    for name in learning_curve_data.keys():
        ratios = []
        for learning_curve in learning_curve_data[name]:
            ratios.append(np.max(learning_curve.learning_curve[:,0])/learning_curve.time_since_init)
        log_file.write("Average for " + name + ": " + str(float(np.mean(ratios))) + " +/- " + str(float(np.std(ratios))) + "\n")

print_training_rates()

log_file.close()
