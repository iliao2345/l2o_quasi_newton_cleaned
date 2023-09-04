# This file tests training/minimization_procedure.py and training/training_iteration_wrapper.py.
# It trains using Adam and L-BFGS saves and loads the learning curves, and then plots them.
# This test presumes that rosenbrock_grad_and_plotting.py works properly already.

import numpy as np
import tensorflow as tf
import os

import context
from tasks import rosenbrock
from training import minimization_procedure
from training import learning_curve_tools

os.makedirs("testing/rosenbrock_optimization/learning_curve_data/", exist_ok=True)

# Set up the Rosenbrock training task
task = rosenbrock.RosenbrockTask()

# List some optimizers
names = [
#        "Adam",
#        "RMSprop",
#        "Momentum",
#        "L-BFGS",
#        "L-BFGS-no-line-search",
#        "O-LBFGS",
#        "LARS",
#        "AdaHessian",
#        "Yogi",
        "Levenberg-Marquardt",
        "BFGS",
        ]

# Train with every optimizer in the list of names
training_iteration_handlers = []
for name in names:
    training_iteration_handler = learning_curve_tools.TrainingIterationHandler(task, time_limit=10, record_trajectory=True, name=name, validation_loss_evaluation_interval=50)

    if name == "Adam":
        minimizer = minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.Adam(lr=0.1))
    elif name == "RMSprop":
        minimizer = minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.RMSprop(lr=0.1))
    elif name == "Momentum":
        minimizer = minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.SGD(lr=0.1, momentum=0.9))
    elif name == "LARS":
        minimizer = lambda handler, initialization: minimization_procedure.LARSMinimizationProcedure(lr=0.1)(handler, initialization, [0, 1])
    elif name == "Yogi":
        minimizer = minimization_procedure.YogiMinimizationProcedure(lr=0.1)
    elif name == "L-BFGS":
        minimizer = minimization_procedure.LBFGSMinimizationProcedure()
    elif name == "L-BFGS-no-line-search":
        minimizer = minimization_procedure.LBFGSNoLineSearchMinimizationProcedure()
    elif name == "O-LBFGS":
        minimizer = minimization_procedure.OLBFGSMinimizationProcedure()
    elif name == "AdaHessian":
        minimizer = minimization_procedure.AdaHessianMinimizationProcedure(lr=0.3)
    elif name == "Levenberg-Marquardt":
        minimizer = minimization_procedure.LevenbergMarquardtMinimizationProcedure(lam=1.0)
    elif name == "BFGS":
        minimizer = minimization_procedure.BFGSMinimizationProcedure(learning_rate=0.01, H0_scale=1.0)

    minimum = minimizer(training_iteration_handler, task.get_initialization())
    training_iteration_handlers.append(training_iteration_handler)

# Visualize the paths that the optimizers take
task.visualize([handler.trajectory for handler in training_iteration_handlers],
               names,
               "testing/rosenbrock_optimization/rosenbrock_optimization_visualization.png")

# Save the learning curves
for handler in training_iteration_handlers:
    handler.save_learning_curve("testing/rosenbrock_optimization/learning_curve_data/rosenbrock_optimization_" + handler.name)

# Reload the learning curves
loaded_learning_curve_datas = []
for name in names:
    loaded_learning_curve_datas.append(learning_curve_tools.LearningCurveFromFile("testing/rosenbrock_optimization/learning_curve_data/rosenbrock_optimization_" + name, name))

# Plot the loaded learning curves by step and time
learning_curve_tools.draw_learning_curves(
        loaded_learning_curve_datas,
        "testing/rosenbrock_optimization/rosenbrock_optimization_learning_curves_step.jpg",
        x_axis="step",
        loss_range=(-np.inf, 0.1))
learning_curve_tools.draw_learning_curves(
        loaded_learning_curve_datas,
        "testing/rosenbrock_optimization/rosenbrock_optimization_learning_curves_time.jpg",
        x_axis="time (s)",
        loss_range=(-np.inf, 0.1))
learning_curve_tools.draw_validation_learning_curve_lists(
        [[curve, curve] for curve in loaded_learning_curve_datas],
        "testing/rosenbrock_optimization/rosenbrock_optimization_validation_learning_curves_time.jpg",
        loss_range=(-np.inf, 0.1))
