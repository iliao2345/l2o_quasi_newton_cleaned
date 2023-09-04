# This file contains code which runs training with and saves models at various points during training.

import argparse
import numpy as np
import tensorflow as tf
import pickle

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
parser.add_argument('-n', type=str, dest='name', required=True)
parser.add_argument('-x', type=float, nargs="*", dest='hyperparameters', required=True)
parser.add_argument('-s', type=float, nargs="*", dest='record_steps', required=True)
parser.add_argument('-t', type=float, nargs="*", dest='record_times', required=True)
parser.add_argument('-r', type=int, dest='run_number', required=True)
parser.add_argument('-R', type=int, dest='n_runs', required=True)
parser.add_argument('-d', type=str, dest='save_directory', required=True)
parser.add_argument('-T', type=str, dest='task_name', required=True)
parser.add_argument('-v', type=float, dest='validation_loss_evaluation_interval', required=False, default=1000)
parsed_args = parser.parse_args()

name = parsed_args.name  # name of optimizer, one of "LODO_modified_beta", "LODO_modified_depth", "LODO_modified_block_size", or something in hyperparameter_setup.names
hyperparameters = parsed_args.hyperparameters  # hyperparameters of optimizer
record_steps = list(map(int, sorted(parsed_args.record_steps)))  # list of step milestones at which to save a model
record_times = list(map(int, sorted(parsed_args.record_times)))  # list of time milestones at which to save a model
run_number = parsed_args.run_number  # this number differentiates between copies of jobs running this program
n_runs = parsed_args.n_runs  # total number of copies of this program that are running
save_directory = parsed_args.save_directory  # directory to save the models and learning curves in
task_name = parsed_args.task_name  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
validation_loss_evaluation_interval = parsed_args.validation_loss_evaluation_interval  # how often to evaluate the validation loss

# Set up the optimizer
if name == "LARS" and task_name == "Rosenbrock":
    hyperparameter_setup.minimizer_setup_fns[name] = lambda lr=0.001, momentum=0.9, weight_decay=0.0005: lambda handler, initialization: minimization_procedure.LARSMinimizationProcedure(lr=lr, weight_decay=weight_decay, momentum=momentum)(handler, initialization, [0, 1])
if name == "LODO-No-Momentum":
    minimizer = hyperparameter_setup.minimizer_setup_fns["LODO"](0.08458583790757619, 7.945595706216676e-06, 0.0)
elif name in hyperparameter_setup.minimizer_setup_fns:
    minimizer = hyperparameter_setup.minimizer_setup_fns[name](*hyperparameters)
elif name == "LODO_modified_beta":
    beta = 1-(1-hyperparameters[2])**(run_number/(n_runs-1))  # beta=0 recovers LODO without momentum
    minimizer = lodo.LODOMinimizationProcedure(
            initial_lr=hyperparameters[0],
            meta_lr=hyperparameters[1],
            beta=beta,
            n_layers=16,
            block_size=4
    )
    name = name + "_" + str(round(beta, 4))
elif name == "LODO_modified_depth":
    depth = 4*run_number  # depth=0 recovers the Momentum algorithm
    minimizer = lodo.LODOMinimizationProcedure(
            initial_lr=hyperparameters[0],
            meta_lr=hyperparameters[1],
            beta=hyperparameters[2],
            n_layers=depth,
            block_size=4
    )
    name = name + "_" + str(depth)
elif name == "LODO_modified_block_size":
    block_size = run_number+2  # block_size=1 nearly recovers LODO-Diagonal
    minimizer = lodo.LODOMinimizationProcedure(
            initial_lr=hyperparameters[0],
            meta_lr=hyperparameters[1],
            beta=hyperparameters[2],
            n_layers=16*4//block_size,
            block_size=block_size
    )
    name = name + "_" + str(block_size)

# Set up the training task
if task_name == "Rosenbrock":
    task = rosenbrock.RosenbrockTask()
    validation_loss_evaluation_interval = 10
elif task_name == "Autoregression":
    task = autoregression_mnist.AutoregressionTask()
elif task_name == "ClassificationHead":
    task = classification_head.ClassificationHeadTask()
elif task_name == "NoisyQuadraticBowl":
    task = noisy_quadratic_bowl.NoisyQuadraticBowl()
elif task_name == "ResnetCIFAR10Classification":
    task = resnet_classification.ResnetClassificationTask()
initialization = task.get_initialization()

# Function to measure the hessian approximation error metric ||I - h_theta H||_F/sqrt(n) for LODO versions
def hessian_error_fn(LODO_weights):
    if task_name == "Autoregression":  # reduce batch size so that the coming double GradientTape does not use too much memory
        task.batch_size = 16
    residuals = []
    for i in range(100):
        vect = np.random.normal(0, 1, size=(initialization.shape[0],))
        vect = vect / tf.math.sqrt(tf.math.reduce_sum(vect**2))
        with tf.GradientTape() as tape1:
            tape1.watch(LODO_weights)
            with tf.GradientTape() as tape2:
                tape2.watch(LODO_weights)
                if task_name == "NoisyQuadraticBowl":  # Do not add noise while evaluating the Hessian of the noisy quadratic bowl task
                    loss2 = task(LODO_weights, new_batch=False)
                else:
                    loss2 = task(LODO_weights, new_batch=True)
            grad = tape2.gradient(loss2, [LODO_weights])[0]
            loss1 = tf.math.reduce_sum(grad*vect)
        H_dot_vect = tape1.gradient(loss1, [LODO_weights])[0]
        LODO_dot_H_dot_vect = minimizer.predict_step(minimizer.weights, H_dot_vect)
        residuals.append(vect + LODO_dot_H_dot_vect)
    residuals = tf.stack(residuals, axis=0)
    frobenius_norm_error = tf.math.sqrt(tf.math.reduce_sum(residuals**2)/residuals.shape[0])
    if task_name == "Autoregression":  # increase batch size back to normal
        task.batch_size = 256
    return frobenius_norm_error
if name == "LODO":
    if task_name == "NoisyQuadraticBowl":
        tracked_quantity_names = ["hessian approximation error"]
        tracked_quantity_fns = [hessian_error_fn]
    elif task_name == "ClassificationHead":
        tracked_quantity_names = ["validation loss", "validation accuracy", "test loss", "test accuracy"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x), lambda x: task.evaluate_test_loss(x), lambda x: task.evaluate_test_accuracy(x)]
    elif task_name == "ResnetCIFAR10Classification":
        tracked_quantity_names = ["validation loss", "validation accuracy", "test loss", "test accuracy"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x), lambda x: task.evaluate_test_loss(x), lambda x: task.evaluate_test_accuracy(x)]
    else:
        tracked_quantity_names = ["validation loss"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x)]
else:
    if task_name == "NoisyQuadraticBowl":
        tracked_quantity_names = []
        tracked_quantity_fns = []
    elif task_name == "ClassificationHead":
        tracked_quantity_names = ["validation loss", "validation accuracy", "test loss", "test accuracy"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x), lambda x: task.evaluate_test_loss(x), lambda x: task.evaluate_test_accuracy(x)]
    elif task_name == "ResnetCIFAR10Classification":
        tracked_quantity_names = ["validation loss", "validation accuracy", "test loss", "test accuracy"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x), lambda x: task.evaluate_validation_accuracy(x), lambda x: task.evaluate_test_loss(x), lambda x: task.evaluate_test_accuracy(x)]
    else:
        tracked_quantity_names = ["validation loss"]
        tracked_quantity_fns = [lambda x: task.evaluate_validation_loss(x)]
training_iteration_handler = learning_curve_tools2.TrainingIterationHandlerWithParameterSaving(
        task,
        time_limit=record_times[-1]+60,
        max_iteration=record_steps[-1]+2,
        break_condition="both",
        save_at_steps=record_steps,
        save_at_times=record_times,
        name = name + "_" + str(run_number),
        save_directory=parsed_args.save_directory,
        tracked_quantity_evaluation_interval=validation_loss_evaluation_interval,
        tracked_quantity_names = tracked_quantity_names,
        tracked_quantity_fns = tracked_quantity_fns,
)

# Perform the training task and save the learning curve
minimum = minimizer(training_iteration_handler, initialization)
training_iteration_handler.save_learning_curve(parsed_args.save_directory + "learning_curve_data/" + name + "_" + str(run_number))
