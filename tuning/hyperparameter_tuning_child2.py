# This file contains code which runs training with a single set of argparsed hyperparams and saves the result.

import argparse
import numpy as np
import pickle

import context
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from tasks import rosenbrock
from tasks import autoregression_mnist
from tasks import classification_head
from tasks import noisy_quadratic_bowl
from tasks import resnet_classification
from training import minimization_procedure
from training import learning_curve_tools2

# Get all the tuning task parameters from the hyperparameter_tuning_parent.py through argparse
parser = argparse.ArgumentParser(description='Learn using an optimizer with specified hyperparameters for hyperparameter tuning purposes.')
parser.add_argument('-n', type=str, dest='name', required=True)
parser.add_argument('-x', type=float, nargs="*", dest='unnormalized_hyperparameter_means', required=True)
parser.add_argument('-s', type=float, dest='stddev', required=True)
parser.add_argument('-t', type=int, dest='max_iteration', required=True)
parser.add_argument('-g', type=int, dest='generation', required=True)
parser.add_argument('-b', type=str, dest='base_directory', required=True)
parser.add_argument('-T', type=str, dest='task_name', required=True)  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
parsed_args = parser.parse_args()

name = parsed_args.name  # name of optimizer, one of "LODO_modified_beta", "LODO_modified_depth", "LODO_modified_block_size", or something in hyperparameter_setup.names
unnormalized_means = parsed_args.unnormalized_hyperparameter_means  # mean of distributions of hyperparameters of optimizers
stddev = parsed_args.stddev  # stddev of distributions of hyperparameters of optimizers
max_iteration = parsed_args.max_iteration  # number of steps to learn for
generation = parsed_args.generation  # which generation of tuning this job is running for
base_directory = parsed_args.base_directory  # directory to save the models and learning curves in
task_name = parsed_args.task_name  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"

# Figure out which directory the result of the training should go
directory = base_directory + "generation_" + str(generation) + "/" + name + "/learning_curve_data/"

# Mutate the hyperparameters away from the generation mean using the given stddev
normalized_means = [normalizing_fn(mean) for normalizing_fn, mean in zip(hyperparameter_normalization.normalizing_fns[name], unnormalized_means)]
normalized_hyperparameters = [np.random.normal(mean, stddev) for mean in normalized_means]
unnormalized_hyperparameters = [unnormalizing_fn(hyperparameter) for unnormalizing_fn, hyperparameter in zip(hyperparameter_normalization.unnormalizing_fns[name], normalized_hyperparameters)]

# Set up the optimizer
if name == "LARS" and task_name == "Rosenbrock":  # LARS only works when the task has "layers" like a neural network
    hyperparameter_setup.minimizer_setup_fns[name] = lambda lr=0.001, momentum=0.9, weight_decay=0.0005: lambda handler, initialization: minimization_procedure.LARSMinimizationProcedure(lr=lr, weight_decay=weight_decay, momentum=momentum)(handler, initialization, [0, 1])
minimizer = hyperparameter_setup.minimizer_setup_fns[name](*unnormalized_hyperparameters)

# Set up the training task
if task_name == "Rosenbrock":
    task = rosenbrock.RosenbrockTask()
elif task_name == "Autoregression":
    task = autoregression_mnist.AutoregressionTask()
elif task_name == "ClassificationHead":
    task = classification_head.ClassificationHeadTask()
elif task_name == "NoisyQuadraticBowl":
    task = noisy_quadratic_bowl.NoisyQuadraticBowlTask()
elif task_name == "ResnetCIFAR10Classification":
    task = resnet_classification.ResnetClassificationTask()
training_iteration_handler = learning_curve_tools2.TrainingIterationHandler(task, max_iteration=max_iteration, record_trajectory=True, name=name, tracked_quantity_names=[], tracked_quantity_fns=[])

# Perform the training task and save the result
minimum = minimizer(training_iteration_handler, task.get_initialization())
training_iteration_handler.save_learning_curve(directory + "curve_" + "_".join(list(map(str, unnormalized_hyperparameters))))
