# This file runs all of the experiments which are in the paper.
# A log file is created for all the printouts necessary for the paper, and images are generated of results:
# For Rosenbrock and autoregression and classification head and noisy quadratic bowl tasks,
#  - default hyperparameters,
#  - tuned hyperparameters,
#  - training time and steps, mean and stddev performances at specific times and numbers of steps for all optimizers, for training and validation sets
#  - images of training curve means and standard deviations of all optimizers, with training and validation sets
#  - images of learning curves for ablated versions of LODO
# For Rosenbrock task:
#  - plot of trajectory of optimizers through loss landscape
# For autoregression task:
#  - generated MNIST images using the model
#  - images of preprocessed input into the CNN
# For classification head task:
#  - nothing extra
# For resnet task:
#  - nothing extra

import numpy as np
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import pickle
import argparse

import context
from tasks import rosenbrock
from tasks import autoregression_mnist
from tasks import classification_head
from tasks import noisy_quadratic_bowl
from tasks import resnet_classification
from training import minimization_procedure
from training import learning_curve_tools2
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from tuning import hyperparameter_tuning_parent2
from training import training_routine_parent2

# Get all the tuning task parameters from the training_routine_parent2.py through argparse
parser = argparse.ArgumentParser(description='Run all experiments on either Autoregression on MNIST task or Rosenbrock minimization task.')
parser.add_argument('-T', type=str, dest='task_name', required=True)
parsed_args = parser.parse_args()
task_name = parsed_args.task_name  # either "Rosenbrock" or "Autoregression" or "NoisyQuadraticBowl"
assert task_name == "Rosenbrock" or task_name == "Autoregression" or task_name == "ClassificationHead" or task_name == "NoisyQuadraticBowl" or task_name == "ResnetCIFAR10Classification"

# Create a file for printouts
if task_name == "Rosenbrock":
    experiment_directory = "experiment/rosenbrock/"
elif task_name == "Autoregression":
    experiment_directory = "experiment/autoregression_mnist/"
elif task_name == "ClassificationHead":
    experiment_directory = "experiment/classification_head/"
elif task_name == "NoisyQuadraticBowl":
    experiment_directory = "experiment/noisy_quadratic_bowl/"
elif task_name == "ResnetCIFAR10Classification":
    experiment_directory = "experiment/resnet_classification/"
os.makedirs(experiment_directory, exist_ok=True)
for subdirectory in ("tuning2/", "training2/", "visualizations2/"):
    os.makedirs(experiment_directory + subdirectory, exist_ok=True)
tuning_directory = experiment_directory + "tuning2/"
training_directory = experiment_directory + "training2/"
if task_name == "Autoregression" or task_name == "ClassificationHead" or task_name == "ResnetCIFAR10Classification":
    models_directory = training_directory + "models/"
log_fname = experiment_directory + "log.txt"
log_file = open(log_fname, "w")
log_file.close()
log_file = open(log_fname, "a")

# Set up the training task
if task_name == "Rosenbrock":
    task = rosenbrock.RosenbrockTask()
    reasonable_loss_range = (0, 0.1)
elif task_name == "Autoregression":
    task = autoregression_mnist.AutoregressionTask()
    reasonable_loss_range = (0.65, 0.95)
elif task_name == "ClassificationHead":
    task = classification_head.ClassificationHeadTask()
    reasonable_loss_range = (0, 4.7)
elif task_name == "NoisyQuadraticBowl":
    task = noisy_quadratic_bowl.NoisyQuadraticBowlTask()
    reasonable_loss_range = (0, 40)
elif task_name == "ResnetCIFAR10Classification":
    task = resnet_classification.ResnetClassificationTask()
    reasonable_loss_range = (0, 5)

# Tune all the optimizers
if task_name == "Rosenbrock":
    default_hyperparameters = hyperparameter_setup.rosenbrock_defaults
    optimizer_names = [
        "Momentum",
        "RMSprop",
        "Adam",
        "Yogi",
        "LODO"
    ]
elif task_name == "Autoregression":
    default_hyperparameters = hyperparameter_setup.autoregression_defaults
    optimizer_names = [
        "Momentum",
        "RMSprop",
        "Adam",
        "LARS",
        "Yogi",
        "LODO",
        "LODO-Diagonal",
        "LODO-Global",
        "LODO-Residuals",
        "LODO-No-Momentum",
        "LODO-SGD",
        "LNDO"
    ]
elif task_name == "ClassificationHead":
    default_hyperparameters = hyperparameter_setup.classification_head_defaults
    optimizer_names = [
        "Momentum",
        "RMSprop",
        "Adam",
        "Yogi",  ##################################################################################### Add other optimizers
        "L-BFGS-no-line-search",
        "O-LBFGS",
        "LODO",
        "LODO-Diagonal",
        "LODO-Global",
        "LODO-Residuals"
    ]
elif task_name == "NoisyQuadraticBowl":
    default_hyperparameters = hyperparameter_setup.autoregression_defaults
    optimizer_names = [
        "Momentum",
        "RMSprop",
        "Adam",
        "Yogi",
        "LODO",
#        "LODO-Diagonal",
#        "LODO-Global",
#        "LODO-Residuals",
        "L-BFGS-no-line-search",
        "O-LBFGS",
        "Levenberg-Marquardt",
        "BFGS",
    ]
elif task_name == "ResnetCIFAR10Classification":
    default_hyperparameters = hyperparameter_setup.resnet_defaults
    optimizer_names = [
        "Momentum",
        "RMSprop",
        "Adam",
        "LARS",
        "Yogi",
        "L-BFGS-no-line-search",
        "O-LBFGS",
        "LODO",
    ]
log_file.write("\nHyperparameter names are shown below:\n" + str(optimizer_names) + "\n")
log_file.write("\nDefault hyperparameters are shown below:\n" + str({name : default_hyperparameters[name] for name in optimizer_names}) + "\n")
if task_name == "Rosenbrock":
    tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
            (3, 200, 32),
            (3, 200, 32),
            (3, 200, 32),
            (2.5, 200, 32),
            (2, 200, 32),
            (1.5, 200, 32),
            (1, 200, 32),
            (0.75, 200, 32),
            (0.5, 200, 32),
            (0.3, 200, 32)
    ]
elif task_name == "Autoregression":
    tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
            (3, 1000, 32),
            (3, 1000, 32),
            (3, 1000, 32),
            (3, 1000, 32),
            (2, 1500, 32),
            (1.7, 1500, 32),
            (1.4, 2000, 32),
            (1.2, 3000, 32),
            (0.9, 5000, 32),
            (0.6, 8000, 32),
    ]
elif task_name == "ClassificationHead":
    tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
            (3, 100, 32),
            (3, 100, 32),
            (3, 100, 32),
            (3, 200, 32),
            (2, 300, 32),
            (1.7, 400, 32),
            (1.4, 600, 32),
            (1.2, 800, 32),
            (0.9, 1000, 32),
            (0.6, 1000, 32),
    ]
elif task_name == "NoisyQuadraticBowl":
    tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
            (3, 100, 32),
            (3, 300, 32),
            (3, 1000, 32),
            (3, 10000, 32),
            (2, 10000, 32),
            (1.7, 10000, 32),
            (1.4, 10000, 32),
            (1.2, 10000, 32),
            (0.9, 10000, 32),
            (0.6, 10000, 32),
    ]
elif task_name == "ResnetCIFAR10Classification":
    tuning_schedule = [  # list of (stddev, max_iteration, population_size) tuples
            (3, 50, 16),
            (3, 50, 16),
            (3, 50, 16),
            (3, 50, 16),
            (2, 50, 16),
            (1.7, 60, 16),
            (1.4, 100, 16),
            (1.2, 200, 16),
            (0.9, 300, 16),
            (0.6, 500, 16),
    ]
def tune_all_hyperparameters():
    """
    Tune all of the hyperparameters which we want to run training with.
    """
    tuner = hyperparameter_tuning_parent2.HyperparameterTuningMethod(
            tuning_directory,
            task_name,
            optimizer_names,
            tuning_schedule=tuning_schedule,
            plot_loss_range=reasonable_loss_range,
            update_interval=60
    )
    best_hyperparameters = tuner.tune_all()
    log_file.write("\nBest hyperparameters are shown below:\n" + str(best_hyperparameters) + "\n")
    return best_hyperparameters

best_hyperparameters = tune_all_hyperparameters()

# Cut all the learning rates in half for the longer training run
if task_name == "Autoregression" or task_name == "ClassificationHead":
    best_hyperparameters = {key:[hyperparameters[0]/2] + hyperparameters[1:] for key, hyperparameters in best_hyperparameters.items()}

# Do the main training run
if task_name == "Rosenbrock":
    step_milestone = 200
    time_milestone = 3
    metric_evaluation_interval = 1
elif task_name == "Autoregression":
    step_milestone = 300000
    time_milestone = 50000
#    step_milestone = 1000000
#    time_milestone = 150000
    metric_evaluation_interval = 1000
elif task_name == "ClassificationHead":
    step_milestone = 5000
    time_milestone = 500
    metric_evaluation_interval = 250
elif task_name == "NoisyQuadraticBowl":
    step_milestone = 100000
    time_milestone = 300
    metric_evaluation_interval = 100
elif task_name == "ResnetCIFAR10Classification":
    step_milestone = 3000
    time_milestone = 24000
#    metric_evaluation_interval = 100
    metric_evaluation_interval = 20
def train_with_all_optimizers():
    """
    Train on the task using multiple copies of each of the optimizers. Results are saved in training_directory.
    """
    if task_name == "Rosenbrock" or task_name == "ResnetCIFAR10Classification":
        all_hyperparameters = best_hyperparameters
    elif task_name == "Autoregression" or task_name == "ClassificationHead":
        all_hyperparameters = dict(best_hyperparameters, **{
            "LODO_modified_beta" : best_hyperparameters["LODO"],
            "LODO_modified_depth" : best_hyperparameters["LODO"],
            "LODO_modified_block_size" : best_hyperparameters["LODO"]
        })
    elif task_name == "NoisyQuadraticBowl":
        all_hyperparameters = dict(best_hyperparameters, **{
            "LODO_modified_beta" : best_hyperparameters["LODO"],
            "LODO_modified_depth" : best_hyperparameters["LODO"],
            "LODO_modified_block_size" : best_hyperparameters["LODO"]
        })
    all_optimizer_names = all_hyperparameters.keys()
    trainer = training_routine_parent2.TrainingRoutine(
            training_directory,
            task_name,
            all_optimizer_names,
            all_hyperparameters,
            save_iterations=[step_milestone],
            save_times=[time_milestone],
            population_size=8,
            plot_loss_range=reasonable_loss_range,
            update_interval=60,
            metric_evaluation_interval=metric_evaluation_interval
    )
    trainer.run_all()

train_with_all_optimizers()  # Submits training/training_routine_child jobs which write to output files

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
    if task_name == "ResnetCIFAR10Classification":
        learning_curve_data_no_ablations = {name:li for name, li in learning_curve_data_no_ablations.items() if name in ("Momentum", "RMSprop", "Adam", "LARS", "Yogi", "O-LBFGS", "LODO")}
    for mode in ("step", "time"):
        learning_curve_tools2.draw_learning_curve_lists(
                learning_curve_data_no_ablations.values(),
                experiment_directory + "learning_curves_by_" + mode + ".pdf",
                loss_range=reasonable_loss_range,
                x_max = step_milestone if mode == "step" else time_milestone,
                x_axis=mode
        )
    if task_name != "NoisyQuadraticBowl":
        learning_curve_tools2.draw_metric_lists(
                learning_curve_data_no_ablations.values(),
                experiment_directory + "validation_learning_curves_by_step.pdf",
                metric_name="validation loss",
                loss_range=reasonable_loss_range,
                x_max=step_milestone,
        )
    else:
        learning_curve_tools2.draw_metric_lists(
                [learning_curve_data_no_ablations["LODO"]],
                experiment_directory + "hessian_approx_error_by_step.pdf",
                metric_name="hessian approximation error",
                loss_range=(0, np.inf),
                x_max=step_milestone,
        )
    if task_name == "ClassificationHead" or task_name == "ResnetCIFAR10Classification":
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
    if task_name == "ResnetCIFAR10Classification":
        for mode in ("step", "time"):
            x_max = step_milestone if mode == "step" else time_milestone
            log_file.write("\nTest accuracy performance during final 90% of training up to " + mode + " " + str(x_max) + ":\n")
            x_axis_index = 0 if mode == "step" else 1
            for name in learning_curve_data.keys():
                means = []
                n_diverged = 0
                for learning_curve in learning_curve_data[name]:
                    if mode == "step":
                        x_max_ = x_max
                    elif mode == "time":
                        x_max_ = x_max*np.max(learning_curve.learning_curve[:,0])/np.max(learning_curve.learning_curve[:,1])
                    samples = learning_curve.tracked_quantities[np.logical_and(learning_curve.tracked_quantities[:,0] < x_max_, learning_curve.tracked_quantities[:,0] > 0.9*x_max_)][:,5]
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
    log_file.write("\nStep/second training speed during training time for all training runs:\n")
    for name in learning_curve_data.keys():
        ratios = []
        for learning_curve in learning_curve_data[name]:
            ratios.append(np.max(learning_curve.learning_curve[:,0])/np.max(learning_curve.learning_curve[:,1]))
        log_file.write("Average for " + name + ": " + str(float(np.mean(ratios))) + " +/- " + str(float(np.std(ratios))) + "\n")
    log_file.write("\nStep/second training speed during training+validation time for all training runs:\n")
    for name in learning_curve_data.keys():
        ratios = []
        for learning_curve in learning_curve_data[name]:
            ratios.append(np.max(learning_curve.learning_curve[:,0])/learning_curve.time_since_init)
        log_file.write("Average for " + name + ": " + str(float(np.mean(ratios))) + " +/- " + str(float(np.std(ratios))) + "\n")

print_training_rates()

# Generate a graph of parameterized LODO ablations of vs the final loss
def generate_parameterized_ablations_graph():
    """
    For LODO with modified beta, depth, and block_size, make a plot of how the beta, depth, or block size affects the final loss.
    """
    learning_curves_directory = training_directory + "learning_curve_data/"
    filenames = [f for f in os.listdir(learning_curves_directory) if os.path.isfile(os.path.join(learning_curves_directory, f))]  # get all the learning curves

    # Figure out the LODO Stddev to plot the error bars
    means = []
    for learning_curve in learning_curve_data["LODO"]:
        mean = np.mean(learning_curve.learning_curve[np.logical_and(learning_curve.learning_curve[:,0] < step_milestone, learning_curve.learning_curve[:,0] > 0.9*step_milestone)][:,2])
        if np.isfinite(mean):
            means.append(mean)
    LODO_stddev = float(np.std(means))

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
        ax.errorbar(ablation_parameters, np.clip(means, reasonable_loss_range[0], reasonable_loss_range[1]), yerr=LODO_stddev, capsize=5, color="k")
        ax.fill_between(ablation_parameters, np.clip(means-stddevs, reasonable_loss_range[0], reasonable_loss_range[1]), np.clip(means+stddevs, reasonable_loss_range[0], reasonable_loss_range[1]), facecolor=(0.5, 0.5, 0.5))
        ax.set_xlabel(ablation_mode)
        ax.set_ylabel("loss")
        ax.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.tick_params(axis='y', which='minor', bottom=False)
        plt.subplots_adjust(left=(0.20 if ablation_mode!="block_size" else 0.25), bottom=0.20, right=1.0, top=1.0)
        fig.set_size_inches((6, 4))
        plt.grid(b=True, which='major', color='0.6')
        plt.grid(b=True, which='minor', color='0.8')
        plt.savefig(experiment_directory + ablation_mode + "_variation.pdf")
        plt.close()

if task_name == "Autoregression":
    generate_parameterized_ablations_graph()

if task_name == "Rosenbrock":
    # Generate visualization of trajectory on Rosenbrock optimization task
    def visualize_all():
        task = rosenbrock.RosenbrockTask()
        training_iteration_handlers = []
        for name in optimizer_names:
            training_iteration_handler = learning_curve_tools2.TrainingIterationHandler(task, max_iteration=100, time_limit=10, break_condition="both", record_trajectory=True, name=name)
            minimizer = hyperparameter_setup.minimizer_setup_fns[name](*best_hyperparameters[name])
            minimum = minimizer(training_iteration_handler, task.get_initialization())
            training_iteration_handlers.append(training_iteration_handler)
        task.visualize([handler.trajectory for handler in training_iteration_handlers],
                       optimizer_names,
                       experiment_directory + "visualizations2/trajectories.pdf",
                       step_limit=step_milestone)
elif task_name == "Autoregression":
    # Generate some MNIST images
    def visualize_all():
        filenames = [f for f in os.listdir(models_directory) if os.path.isfile(os.path.join(models_directory, f))]
        for name in optimizer_names:
            for milestone_value, milestone_unit in [(step_milestone, "steps"), (time_milestone, "seconds")]:
                fname = name + "_1_parameters_at_" + str(int(milestone_value)) + "_" + milestone_unit
                with open(os.path.join(models_directory, fname), "rb") as f:
                    weights = pickle.load(f)
                task.visualize(weights, experiment_directory + "visualizations2/" + name + "_at_" + str(int(milestone_value)) + "_" + milestone_unit + ".pdf")

    # Generate example of preprocessed input images
    def generate_sample_input():
        task.sample_new_batch()
        images = tf.gather_nd(autoregression_mnist.training_set_mnist_images, task.batch_indices[:,np.newaxis])
        queried_points = np.random.randint(autoregression_mnist.training_set_mnist_images.shape[1], size=(task.batch_size, 2))
        CNN_input = task.preprocess(images, queried_points)
        CNN_input = tf.reshape(tf.transpose(CNN_input[:5], [0, 3, 1, 2]), [25, 28, 28])
        CNN_input = tf.cast(127.5*(CNN_input+1), tf.int32)
        task.save_image_grid(CNN_input, experiment_directory + "visualizations2/sample_input.pdf")
elif task_name == "ClassificationHead":
    def visualize_all():
        pass
elif task_name == "NoisyQuadraticBowl":
    def visualize_all():
        pass
elif task_name == "ResnetCIFAR10Classification":
    def visualize_all():
        pass

log_file.close()

#visualize_all()
if task_name == "Autoregression":
    generate_sample_input()
