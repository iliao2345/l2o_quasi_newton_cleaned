# This file is a test run of hyperparameter tuning followed by training time followed by test time and then visualization, for all optimizers.
# A log file is created for all the printouts necessary for the paper, and images are generated of results:
#  - default hyperparameters,
#  - tuned hyperparameters,
#  - training time and steps, mean and stddev performances at specific times and numbers of steps for all optimizers
#  - test set means and stddev performances at specific times and numbers of steps for all optimizers
#  - images of training curve means and standard deviations of all optimizers
#  - visualization of training curves for select few optimizers
# This test presumes that testing/hyperparameter_tuning.py and testing/rosenbrock_parallelism.py works properly already.

import numpy as np
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt

import context
from tasks import rosenbrock
from training import minimization_procedure
from training import learning_curve_tools2
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from tuning import hyperparameter_tuning_parent
from training import training_routine_parent2

# Create a file for printouts
experiment_directory = "testing/rosenbrock_everything/"
os.makedirs(experiment_directory, exist_ok=True)
for subdirectory in ("tuning/", "training/"):
    os.makedirs(experiment_directory + subdirectory, exist_ok=True)
training_directory = experiment_directory + "training/"
tuning_directory = experiment_directory + "tuning/"
log_fname = experiment_directory + "log.txt"
log_file = open(log_fname, "w")
log_file.close()
log_file = open(log_fname, "a")

# Set up the Rosenbrock training task
task = rosenbrock.RosenbrockTask()
reasonable_loss_range = (0, 0.2)

# Tune all the optimizers
log_file.write("\nHyperparameter names are shown below:\n" + str(hyperparameter_setup.names) + "\n")
log_file.write("\nDefault hyperparameters are shown below:\n" + str(hyperparameter_setup.rosenbrock_defaults) + "\n")
tuning_schedule = [
        (3, 20, 12),
        (1, 20, 12),
]
def tune_all_hyperparameters():
    """
    Tune all of the hyperparameters which we want to run training with.
    """
    tuner = hyperparameter_tuning_parent.HyperparameterTuningMethod(
            tuning_directory,
            "Rosenbrock",
            hyperparameter_setup.names,
            tuning_schedule=tuning_schedule,
            plot_loss_range=reasonable_loss_range,
            update_interval=1
    )
    best_hyperparameters = tuner.tune_all()
    log_file.write("\nBest hyperparameters are shown below:\n" + str(best_hyperparameters) + "\n")
    return best_hyperparameters

best_hyperparameters = tune_all_hyperparameters()

# Do the main training run
step_milestone = 50
time_milestone = 1
def train_with_all_optimizers():
    """
    Train on the task using multiple copies of each of the optimizers. Results are saved in training_directory.
    """
    all_hyperparameters = dict(best_hyperparameters, **{
        "LODO_modified_beta" : best_hyperparameters["LODO"],
        "LODO_modified_depth" : best_hyperparameters["LODO"],
        "LODO_modified_block_size" : best_hyperparameters["LODO"]
    })
    all_optimizer_names = all_hyperparameters.keys()
    trainer = training_routine_parent2.TrainingRoutine(
            training_directory,
            "Rosenbrock",
            all_optimizer_names,
            all_hyperparameters,
            save_iterations=[step_milestone],
            save_times=[time_milestone],
            population_size=8,
            plot_loss_range=reasonable_loss_range,
            update_interval=10,
            metric_evaluation_interval=10
    )
    trainer.run_all()

train_with_all_optimizers()  # Submits training/training_routine_child jobs which write to output files

# Read all the files containing learning curve data, including fixed ablations but not parameterized ablations
main_optimizer_names = [
    "Adam",
    "RMSprop",
    "Momentum",
    "L-BFGS-no-line-search",
    "O-LBFGS",
#    "AdaHessian",
    "LODO",
    "LODO-Diagonal",
    "LODO-Global",
    "LODO-Residuals"
]
def read_main_test_learning_curves():
    """
    Read all the learning curve data files from training.
    """
    learning_curve_data = dict()
    learning_curves_directory = training_directory + "learning_curve_data/"
    filenames = [f for f in os.listdir(learning_curves_directory) if os.path.isfile(os.path.join(learning_curves_directory, f))]
    for i, name in enumerate(main_optimizer_names):
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
for mode in ("step", "time"):
    learning_curve_tools2.draw_learning_curve_lists(
            learning_curve_data.values(),
            experiment_directory + "learning_curves_by_" + mode + ".png",
            loss_range=reasonable_loss_range,
            x_max = step_milestone if mode == "step" else time_milestone,
            x_axis=mode
            )
learning_curve_tools2.draw_metric_lists(
        learning_curve_data.values(),
        experiment_directory + "validation_learning_curves_by_" + mode + ".png",
        metric_name="validation loss",
        loss_range=reasonable_loss_range,
        x_max=step_milestone,
        )
learning_curve_tools2.draw_metric_lists(
        [value for key, value in learning_curve_data.items() if "LODO" in key],
        experiment_directory + "hessian_approx_error_by_step.png",
        metric_name="hessian approximation error",
        loss_range=(0, 1.5),
        x_max=step_milestone,
        )

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
    log_file.write("\nStep/second training speed for all training runs:\n")
    for name in learning_curve_data.keys():
        ratios = []
        for learning_curve in learning_curve_data[name]:
            ratios.append(np.max(learning_curve.learning_curve[:,0])/np.max(learning_curve.learning_curve[:,1]))
        log_file.write("Average for " + name + ": " + str(float(np.mean(ratios))) + " +/- " + str(float(np.std(ratios))) + "\n")

print_training_rates()

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
        plt.savefig(experiment_directory + ablation_mode + "_variation.png")
        plt.close()

generate_parameterized_ablations_graph()

# Generate visualization of trajectory on Rosenbrock optimization task
def visualize_all():
    task = rosenbrock.RosenbrockTask()
    training_iteration_handlers = []
    for name in main_optimizer_names:
        print(name)
        training_iteration_handler = learning_curve_tools2.TrainingIterationHandler(task, max_iteration=100, time_limit=10, break_condition="both", record_trajectory=True, name=name)
        minimizer = hyperparameter_setup.minimizer_setup_fns[name](*best_hyperparameters[name])
        minimum = minimizer(training_iteration_handler, task.get_initialization())
        training_iteration_handlers.append(training_iteration_handler)
    task.visualize([handler.trajectory for handler in training_iteration_handlers],
                   main_optimizer_names,
                   experiment_directory + "visualization.png")

#visualize_all()

log_file.close()
