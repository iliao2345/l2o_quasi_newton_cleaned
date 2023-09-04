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

# Create a file for printouts
experiment_directory = "experiment/resnet_classification/"
training_directory = experiment_directory + "training2/"
models_directory = training_directory + "models/"

# Set up the training task
task = resnet_classification.ResnetClassificationTask()
reasonable_loss_range = (0, 5)

default_hyperparameters = hyperparameter_setup.resnet_defaults
optimizer_names = [
    "Momentum",
    "RMSprop",
    "Adam",
    "Yogi",
    "L-BFGS-no-line-search",
    "O-LBFGS",
    "LODO",
]
step_milestone = 3000
time_milestone = 24000
metric_evaluation_interval = 20

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

# Generate a graph of parameterized LODO ablations of vs the final loss
def generate_parameterized_ablations_graph():
    """
    For LODO with modified beta, depth, and block_size, make a plot of how the beta, depth, or block size affects the final loss.
    """
    learning_curves_directory = training_directory + "learning_curve_data/"
    filenames = [f for f in os.listdir(learning_curves_directory) if os.path.isfile(os.path.join(learning_curves_directory, f))]  # get all the learning curves
    for ablation_mode in ("beta", "depth", "block_size"):
        fig, ax = plt.subplots()
        ablation_parameters = []  # collect the ablation parameters, mean and stddev losses for relevant learning curves
        means = []
        stddevs = []
        fname_prefix = "LODO_modified_" + ablation_mode + "_"
        match_pattern = "^" + fname_prefix + "[0-9\.]+_[0-9]+$"
        for fname in filenames:
            if re.match(match_pattern, fname):  # filter learning curves by relevancy to this ablation experiment
                learning_curve = learning_curve_tools2.LearningCurveFromFile(os.path.join(learning_curves_directory, fname), name="")
                samples = learning_curve.learning_curve[np.logical_and(learning_curve.learning_curve[:,0] < step_milestone, learning_curve.learning_curve[:,0] > 0.9*step_milestone)][:,2]
                ablation_parameter = float(fname[len(fname_prefix):len(fname_prefix)+fname[len(fname_prefix):].find("_")])  # obtain ablation parameter from filename
                ablation_parameters.append(ablation_parameter)
                means.append(float(np.mean(samples)))
                stddevs.append(float(np.std(samples) / (samples.shape[0]-1)))

        # sort data
        ablation_parameters = np.array(ablation_parameters)
        means = np.array(means)
        stddevs = np.array(stddevs)
        order = np.argsort(ablation_parameters)
        ablation_parameters, means, stddevs = [arr[order] for arr in [ablation_parameters, means, stddevs]]

        # plot ablation parameter vs loss
        ax.plot(ablation_parameters, np.clip(means, reasonable_loss_range[0], reasonable_loss_range[1]), color="k")
        ax.fill_between(ablation_parameters, np.clip(means-stddevs, reasonable_loss_range[0], reasonable_loss_range[1]), np.clip(means+stddevs, reasonable_loss_range[0], reasonable_loss_range[1]), facecolor=(0.5, 0.5, 0.5))
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        ax.set_xlabel(ablation_mode)
        ax.set_ylabel("loss")
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', bottom=False)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.75)
        fig.set_size_inches((12, 8))
        plt.grid(b=True, which='major', color='0.6')
        plt.grid(b=True, which='minor', color='0.8')
        plt.savefig(experiment_directory + ablation_mode + "_variation.pdf")
        plt.close()

#generate_parameterized_ablations_graph()

# Generate some MNIST images
def visualize_all():
    filenames = [f for f in os.listdir(models_directory) if os.path.isfile(os.path.join(models_directory, f))]
    for name in optimizer_names:
        for milestone_value, milestone_unit in [(step_milestone, "steps"), (time_milestone, "seconds")]:
            fname = name + "_1_parameters_at_" + str(int(milestone_value)) + "_" + milestone_unit
            with open(os.path.join(models_directory, fname), "rb") as f:
                weights = pickle.load(f)
            task.visualize(weights, experiment_directory + "visualizations/" + name + "_at_" + str(int(milestone_value)) + "_" + milestone_unit + ".pdf")

# Generate example of preprocessed input images
def generate_sample_input():
    task.sample_new_batch()
    images = tf.gather_nd(autoregression_mnist.training_set_mnist_images, task.batch_indices[:,np.newaxis])
    queried_points = np.random.randint(autoregression_mnist.training_set_mnist_images.shape[1], size=(task.batch_size, 2))
    CNN_input = task.preprocess(images, queried_points)
    CNN_input = tf.reshape(tf.transpose(CNN_input[:5], [0, 3, 1, 2]), [25, 28, 28])
    CNN_input = tf.cast(127.5*(CNN_input+1), tf.int32)
    task.save_image_grid(CNN_input, experiment_directory + "visualizations/sample_input.pdf")

#visualize_all()
#generate_sample_input()
