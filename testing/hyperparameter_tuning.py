# This file tests tuning/hyperparameter_tuning_parent.py and tuning/hyperparameter_tuning_child.py.
# This test presumes that rosenbrock_optimization.py works properly already.

import numpy as np
import tensorflow as tf

import context
from tuning import hyperparameter_tuning_parent
from training import hyperparameter_setup

tuning_schedule = [(3, 100, 32), (2, 100, 32), (1, 1000, 32), (0.5, 4000, 32), (0.1, 10000, 32)]
tuner = hyperparameter_tuning_parent.HyperparameterTuningMethod("testing/hyperparameter_tuning/", "Rosenbrock", hyperparameter_setup.names, tuning_schedule=tuning_schedule, plot_loss_range=(-np.inf, 0.2), update_interval=1)

best_hyperparameters = tuner.tune_all()
print(best_hyperparameters)
