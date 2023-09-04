# Learning to Optimize and Quasi-Newton Methods

This repository serves to run the tests required for the paper exploring the intersection of L2O and quasi-Newton methods.
A part of the code base is described below, with newer parts and experiments (eg. resnet classification, some hessian freezing experiments, other experiments of various sorts) not documented below.

The structure of the codebase is as follows:
 - Training tasks such as the Rosenbrock function are described in the tasks folder. Functionality such as task visualization are included here. All tasks inherit from a base task class.
 - Tools for anything to do with learning curves are in training/learning_curve_tools2.py. Notably, this includes a class which envelops a training task in a stateful learning curve recorder which is written to whenever the task is queried. This recorder also computes the logic for whether to break out of the training loop. All classes in this file inherit from a base class meant to hold and save learning curves.
 - Training loops are defined in training/minimization_procedure.py. All classes in this file inherit from a base class, to allow for the use of both tfp lbfgs and tf keras optimizers in the same framework.
 - A program for testing training using a distribution of hyperparameters is defined in tuning/hyperparameter_tuning_child2.py. Given population parameters which define a distribution of hyperparameters to sample from, it performs one training run and saves the learning curve.
 - A program for automatically tuning hyperparameters is defined in tuning/hyperparameter_tuning_parent2.py. It iteratively refines a multivariate normal distribution over hyperparameter space by testing with tuning/hyperparameter_tuning_child2.py. This program creates a folder system for saving its work in case portions of its work fail. It runs constantly and automatically writes new .sh files which call on tuning/hyperparameter_tuning_child2.py with generated hardcoded arguments. It automatically submits these .sh files to the job scheduler. tuning/hyperparameter_tuning_child2.py saves results to its own generated file, and the parent program monitors the file locations for the generated files to detect if they finish. It generates and runs the next job array when all the necessary files appear. It also checks progress with LLstat every so often to determine when the tuning/hyperparameter_tuning_child2.py instances finish unexpectedly/crash.
 - A program for training using specific hyperparameters is defined in training/training_procedure_child2.py. It performs one training run and saves the learning curve.
 - A program for automatically performing parallel training runs with tuned hyperparameters is defined in training/training_procedure_parent2.py. It runs multiple instances of training using training/training_routine_child2.py. This program creates a folder system for the children to save their work. It automatically writes .sh files which call on tuning/hyperparameter_tuning_child2.py with generated hardcoded arguments. It automatically submits these .sh files to the job scheduler. training/training_procedure_child2.py saves results to its own generated file, and the parent program monitors the file locations for the generated files to detect if they finish and blocks until they do. It also checks progress with LLstat every so often to determine when the training/training_procedure_child2.py instances finish unexpectedly/crash.

 - Tests of various functionalities of this code base are found in the testing/ folder. They help to debug the codebase and verify that the codebase does what is intended. Tests do various things such as sampling the gradient, performing full training runs, run parallel hyperparameter tuning and training with automatic job submission, plot learning curves, and generate visualizations.

 - Full experiments which have been run for the paper are found in the experiment/ folder. The experiments include one for the noisy quadratic bowl, another for Rosenbrock function minimization task, another one for MNIST autoregression, and another for resnet classification. For each experiment, numerous optimizers are tuned in parallel, and then a number of large training runs is performed in parallel. The results of tuning (original default hyperparameters, tuned hyperparameters) are logged to a file. The results of training (mean and stddev training losses at specific times and numbers of steps, training speed in steps per second, number of diverged optimization runs) are also logged to a file. Learning curves are plotted, and visualizations of the task are generated. For the MNIST autoregression task, ablation versions of LODO are also trained and information about them is logged. For parameterized ablations, the loss is plotted against the ablation parameter. Sometimes due to a race condition, a learning curve file may not be written, in which case all the learning curve files of the same optimizer (and generation, if tuning) should be deleted and the experiment resubmitted. It will resume progress.
