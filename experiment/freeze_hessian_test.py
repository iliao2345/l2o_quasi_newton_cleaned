# This file contains code which runs training with and saves models at various points during training.

import argparse
import numpy as np
import tensorflow as tf
import pickle
import sys

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
parser.add_argument('-x', type=float, nargs="*", dest='hyperparameters', required=True)
parser.add_argument('-s', type=float, nargs="*", dest='record_steps', required=True)
parser.add_argument('-t', type=float, nargs="*", dest='record_times', required=True)
parser.add_argument('-r', type=int, dest='run_number', required=True)
parser.add_argument('-R', type=int, dest='n_runs', required=True)
parser.add_argument('-d', type=str, dest='save_directory', required=True)
parser.add_argument('-T', type=str, dest='task_name', required=True)
parser.add_argument('-v', type=float, dest='validation_loss_evaluation_interval', required=False, default=1000)
parsed_args = parser.parse_args()

hyperparameters = parsed_args.hyperparameters  # hyperparameters of optimizer
record_steps = list(map(int, sorted(parsed_args.record_steps)))  # list of step milestones at which to save a model
record_times = list(map(int, sorted(parsed_args.record_times)))  # list of time milestones at which to save a model
run_number = parsed_args.run_number  # this number differentiates between copies of jobs running this program
n_runs = parsed_args.n_runs  # total number of copies of this program that are running
save_directory = parsed_args.save_directory  # directory to save the models and learning curves in
task_name = parsed_args.task_name  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
validation_loss_evaluation_interval = parsed_args.validation_loss_evaluation_interval  # how often to evaluate the validation loss

# Set up the optimizer
minimizer = hyperparameter_setup.minimizer_setup_fns["LODO"](*hyperparameters)

# Set up the training task
if task_name == "Rosenbrock":
    task = rosenbrock.RosenbrockTask()
    validation_loss_evaluation_interval = 10
elif task_name == "Autoregression":
    task = autoregression_mnist.AutoregressionTask()
elif task_name == "ClassificationHead":
    task = classification_head.ClassificationHeadTask()
elif task_name == "NoisyQuadraticBowl":
    task = noisy_quadratic_bowl.NoisyQuadraticBowlTask()
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

training_iteration_handler = learning_curve_tools2.TrainingIterationHandlerWithParameterSaving(
        task,
        time_limit=record_times[-1]+60,
        max_iteration=record_steps[-1]+2,
        break_condition="both",
        save_at_steps=record_steps,
        save_at_times=record_times,
        name = "LODO_" + str(run_number) + "_pretrain_before_freeze",
        save_directory=parsed_args.save_directory,
        tracked_quantity_evaluation_interval=validation_loss_evaluation_interval,
        tracked_quantity_names = tracked_quantity_names,
        tracked_quantity_fns = tracked_quantity_fns,
)

print("Y")
sys.stdout.flush()
# Perform the training task and save the learning curve
minimum = minimizer(training_iteration_handler, initialization)
print("Z")
sys.stdout.flush()
training_iteration_handler.save_learning_curve(parsed_args.save_directory + "learning_curve_data/LODO_" + str(run_number) + "_pretrain_before_freeze")


class LODOWithCopiedHessianMinimizationProcedure(minimization_procedure.MinimizationProcedureClass):
    """
    This class allows us to minimize using the LODO algorithm.
    """

    def __init__(self, weights, permutations, initial_lr=1.0, meta_lr=0.001, beta=0.9, n_layers=16, block_size=4):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter
        self.n_layers = n_layers  # The number of layers in the neural network inside LODO (not including the transposed portion)
        self.block_size = block_size  # The block size of the block diagonal matrices contained in the neural network
        self.weights, self.permutations = weights, permutations

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.hidden_dimension = 2*self.task_dimension
        self.hidden_dimension = self.hidden_dimension + self.block_size - (self.hidden_dimension % self.block_size)

    def predict_step(self, weights, gradient):
        """
        Use a neural network to choose the step given the gradient. This neural network has no bias nodes nor activations,
        and its transpose is applied afterwards, such that we guarantee that the whole system is represented by a negative-
        semidefinite symmetric matrix.
        """

        # First apply the regular portion
        x = tf.concat([gradient, tf.zeros([self.hidden_dimension-self.task_dimension], dtype=tf.float64)], axis=0)
        for permutation, weight in zip(self.permutations[:len(self.permutations)//2], weights):
            x = tf.gather_nd(x, permutation)
            x = tf.reshape(tf.einsum('ni,nio->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
        # Then apply its transpose
        for permutation, weight in zip(self.permutations[len(self.permutations)//2:], reversed(weights)):
            x = tf.reshape(tf.einsum('ni,noi->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
            x = tf.gather_nd(x, permutation)
        # And finally multiply by some constants
        return -self.initial_lr*x[:self.task_dimension]

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using LODO.
        """

        self.get_ready_to_minimize(x)

        m = tf.zeros([self.task_dimension], dtype=tf.float64)

        while True:
            step = self.predict_step(self.weights, m)
            x = x + tf.cast(step, x.dtype)
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = training_iteration_handler.sample_training_loss(x)
            g = tape.gradient(loss, [x])[0]
            m = self.beta*m + (1-self.beta)*tf.cast(g, tf.float64)

            if training_iteration_handler.stopping_condition():
                break

print("A")
sys.stdout.flush()

task = type(task)()  # make a new task but save the optimizer
initialization = task.get_initialization()
training_iteration_handler = learning_curve_tools2.TrainingIterationHandlerWithParameterSaving(
        task,
        time_limit=record_times[-1]+60,
        max_iteration=record_steps[-1]+2,
        break_condition="both",
        save_at_steps=record_steps,
        save_at_times=record_times,
        name = "LODO_" + str(run_number) + "_learned_frozen_hessian",
        save_directory=parsed_args.save_directory,
        tracked_quantity_evaluation_interval=validation_loss_evaluation_interval,
        tracked_quantity_names = tracked_quantity_names,
        tracked_quantity_fns = tracked_quantity_fns,
)

print("B")
sys.stdout.flush()
# Perform the training task and save the learning curve
minimizer = LODOWithCopiedHessianMinimizationProcedure(minimizer.weights, minimizer.permutations, *hyperparameters)
print("C")
sys.stdout.flush()
minimum = minimizer(training_iteration_handler, initialization)
print("D")
sys.stdout.flush()
training_iteration_handler.save_learning_curve(parsed_args.save_directory + "learning_curve_data/LODO_" + str(run_number) + '_learned_frozen_hessian')
