# This file tests training/training_routine_parent.py and training/training_routine_child.py.
# This test presumes that rosenbrock_optimization.py works properly already.

import numpy as np
import tensorflow as tf

import context
from training import hyperparameter_setup
from training import training_routine_parent2

# Do the main training run
step_milestone = 100
time_milestone = 10
all_hyperparameters = dict(hyperparameter_setup.rosenbrock_defaults, **{
    "LODO_modified_beta" : hyperparameter_setup.rosenbrock_defaults["LODO"],
    "LODO_modified_depth" : hyperparameter_setup.rosenbrock_defaults["LODO"],
    "LODO_modified_block_size" : hyperparameter_setup.rosenbrock_defaults["LODO"]
})
all_optimizer_names = all_hyperparameters.keys()
trainer = training_routine_parent2.TrainingRoutine(
        "testing/rosenbrock_parallelism/",
        "Rosenbrock",
        all_optimizer_names,
        all_hyperparameters,
        save_iterations=[step_milestone],
        save_times=[time_milestone],
        population_size=8,
        plot_loss_range=(-np.inf, 0.2),
        update_interval=1,
        metric_evaluation_interval=10
)
trainer.run_all()
