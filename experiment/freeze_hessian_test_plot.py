# This file contains code which runs training with and saves models at various points during training.

import argparse
import numpy as np
import tensorflow as tf
import pickle
import sys
import os

import context
from training import minimization_procedure
from training import learning_curve_tools2
from training import lodo
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from tasks import rosenbrock
from tasks import autoregression_mnist
from tasks import classification_head
from tasks import noisy_quadratic_bowl
from tasks import resnet_classification

# Get all the tuning task parameters from the training_routine_parent.py through argparse
parser = argparse.ArgumentParser(description='Learn using an optimizer and save models at various points during training.')
parser.add_argument('-s', type=float, nargs="*", dest='record_steps', required=True)
parser.add_argument('-R', type=int, dest='n_runs', required=True)
parsed_args = parser.parse_args()

record_steps = list(map(int, sorted(parsed_args.record_steps)))  # list of step milestones at which to save a model
n_runs = parsed_args.n_runs  # total number of copies of this program that are running
training_directory = "noisy_quadratic_bowl/training2/"
save_directory = "noisy_quadratic_bowl/visualizations2/"

reasonable_loss_range = (0, 40)
step_milestone = record_steps[-1]

# Read all the files containing learning curve data, including fixed ablations but not parameterized ablations
def read_main_test_learning_curves():
    """
    Read all the learning curve data files from training.
    """
    learning_curves_directory = training_directory + "learning_curve_data/"
    filenames_on_the_fly = [learning_curves_directory + "LODO_" + str(i) + "_pretrain_before_freeze" for i in range(n_runs)]
    filenames_frozen = [learning_curves_directory + "LODO_" + str(i) + "_learned_frozen_hessian" for i in range(n_runs)]
    filenames = filenames_on_the_fly + filenames_frozen
    learning_curves_on_the_fly = []
    for fname in filenames_on_the_fly:
        if os.path.exists(fname):
            learning_curve = learning_curve_tools2.LearningCurveFromFile(fname, name="LODO on_the_fly")
            learning_curves_on_the_fly.append(learning_curve)
    learning_curves_frozen = []
    for fname in filenames_frozen:
        if os.path.exists(fname):
            learning_curve = learning_curve_tools2.LearningCurveFromFile(fname, name="pretrained LODO")
            learning_curves_frozen.append(learning_curve)
    return {"LODO_on_the_fly": learning_curves_on_the_fly, "LODO_frozen": learning_curves_frozen}

learning_curve_data = read_main_test_learning_curves()

# Plot all the optimizer performances by step and time
def plot_learning_curves():
    learning_curve_tools2.draw_learning_curve_lists(
            learning_curve_data.values(),
            save_directory + "learning_curves_otf_vs_frozen.pdf",
            loss_range=reasonable_loss_range,
            x_max = step_milestone,
            x_axis="step"
    )
    print(list(learning_curve_data.values())[0][0].learning_curve.shape)
    print(list(learning_curve_data.values())[0][0].tracked_quantities.shape)
    print([list(learning_curve_data.values())[i][j].tracked_quantities.shape for i in range(2) for j in range(n_runs)])
    print(list(learning_curve_data.values())[0][0].learning_curve[:10, :])
    print(list(learning_curve_data.values())[1][0].learning_curve[:10, :])
    print('\n'.join(list(map(str, list(learning_curve_data.values())[1][0].learning_curve[:10, 2].tolist()))))
    print('\n'.join(list(map(str, list(learning_curve_data.values())[1][1].learning_curve[:10, 2].tolist()))))
    print('\n'.join(list(map(str, list(learning_curve_data.values())[1][2].learning_curve[:10, 2].tolist()))))
    print(list(learning_curve_data.values())[0][0].tracked_quantities[-10:, :])
    print(list(learning_curve_data.values())[1][0].tracked_quantities[-10:, :])
    learning_curve_tools2.draw_metric_lists(
            learning_curve_data.values(),
            save_directory + "hessian_approx_error_otf_vs_frozen.pdf",
            metric_name="hessian approximation error",
            loss_range=(0, np.inf),
            x_max=step_milestone,
    )

plot_learning_curves()

# Record all the training set performances at specified step and time
def print_training_performances():
    """
    Write to the log file the means and stddevs of final loss over all the runs of identical optimizers.
    """
    x_max = step_milestone
    print("\nTraining loss performance during final 90% of training up to step " + str(x_max) + ":\n")
    x_axis_index = 0
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
        print(name + ": " + str(mean_over_runs) + " +/- " + str(stddev_over_runs) + ", (" + str(n_diverged) + "/" + str(len(learning_curve_data[name])) + " diverged)\n")

print_training_performances()
