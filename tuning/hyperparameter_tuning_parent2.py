# This file contains code for automatically tuning the hyperparameters of an optimizer.
# It writes new .sh scripts with job submission code and submits them to the job scheduler.
# It waits for the results to come back before gathering all the results and figuring out the
# parameters for the next job to submit. It can tune multiple optimizers' hyperparameters in
# parallel.

import argparse
import numpy as np
import regex as re
import subprocess
import time
import os

import context
from training import hyperparameter_setup
from tuning import hyperparameter_normalization
from training import learning_curve_tools2

LLSUB_tasks_per_node = 1
assert 48 % LLSUB_tasks_per_node == 0

# tuning_schedule is a list of (stddev, max_iterations, population_size) pairs
default_tuning_schedule = [
        (10, 10, 32),
        (5, 10, 32),
        (3, 100, 32),
        (2, 100, 32),
        (1, 1000, 32), 
        (0.5, 4000, 32)
]

class HyperparameterTuningMethod():
    """
    This class contains methods relevant to automatic tuning of hyperparameters in parallel
    by automatic .sh script generation and submission. The main function to run is self.tune_all().
    Objects contain state which track the progress of the tuning procedure. Tuning is done
    by sampling generations of optimizer hyperparameter sets, running them, and taking the
    mean of the next generation as the mean of the better half performances of the previous
    generation. The variance in the generation is decreased as the running time is increased,
    as defined by the tuning_schedule.
    """

    def __init__(self, base_directory, task_name, optimizer_names, tuning_schedule=default_tuning_schedule, plot_loss_range=(-np.inf, np.inf), update_interval=10):
        """
        Create a hyperparameter tuning procedure. This will generate a folder system which is
        base_directory.
         - base_directory: folder to put results and files in
         - task_name: either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
         - optimizer_names: list of optimizer names as strings to optimize with
         - tuning_schedule: list of (stddev, max_iteration, population_size) tuples for parameters of successive tuning generations
         - plot_loss_range: (low, high), determines the loss range bounds for learning curve plots
         - update_interval: how often to check if the training job arrays finish, in seconds
        """
        self.update_interval = update_interval
        self.base_directory = base_directory
        self.task_name = task_name  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
        self.optimizer_names = optimizer_names
        self.plot_loss_range = plot_loss_range
        self.tuning_schedule = tuning_schedule
        # List of states are: "waiting", "ready", "running", "finished", and also "crashed". They go from left to right, but can also go from running to crashed.
        self.tuning_state_by_generation = [{key:"waiting" for key in self.optimizer_names} for evolution_parameters in self.tuning_schedule]
        # Every tuple here contains the LLsub number and the slurm number, respectively
        self.job_array_numbers = [{key:("","") for key in self.optimizer_names} for evolution_parameters in self.tuning_schedule]
        # Under this definition below, when best_run_means[-1] is accessed, we get the initial default hyperparameters.
        if self.task_name == "Rosenbrock":
            default_hyperparameters = {key:hyperparameter_setup.rosenbrock_defaults[key] for key in self.optimizer_names}
        elif self.task_name == "Autoregression":
            default_hyperparameters = {key:hyperparameter_setup.autoregression_defaults[key] for key in self.optimizer_names}
        elif self.task_name == "ClassificationHead":
            default_hyperparameters = {key:hyperparameter_setup.classification_head_defaults[key] for key in self.optimizer_names}
        elif self.task_name == "NoisyQuadraticBowl":
            default_hyperparameters = {key:hyperparameter_setup.noisy_quadratic_bowl_defaults[key] for key in self.optimizer_names}
        elif self.task_name == "ResnetCIFAR10Classification":
            default_hyperparameters = {key:hyperparameter_setup.resnet_defaults[key] for key in self.optimizer_names}
        self.best_run_means = [{key:None for key in self.optimizer_names} for evolution_parameters in self.tuning_schedule] + [default_hyperparameters]

        # build up the folder system
        for generation in range(len(self.tuning_schedule)):
            for name in self.optimizer_names:
                os.makedirs(self.base_directory + "generation_" + str(generation) + "/" + name + "/learning_curve_data/", exist_ok=True)

    def manage_tuning_states(self):
        """
        This function is run periodically to check the state of jobs and run new
        ones if needed. It calls on other functions to perform this functionality.
        """

        for name in self.optimizer_names:
            for generation, evolution_parameters in enumerate(self.tuning_schedule):
                prev_state = "finished" if generation == 0 else state
                state = self.tuning_state_by_generation[generation][name]
                if prev_state == "finished" and state == "waiting":  # if previous results are sufficient for the next job
                    n_output_files = self.count_output_files(generation, name)
                    if n_output_files >= evolution_parameters[2]:  # it's finished if enough current results are already there
                        state = "finished"
                    elif n_output_files == 0:  # ready to start next job if no current results
                        state = "ready"
                    else:  # something strange happened if only some results are already there
                        state = "crashed"
                if state == "ready":  # if ready, gather all necessary information to submit the next job
                    if generation > 0:
                        self.gather_output_from_previous_job(generation-1, name)
                    self.submit_next_job(generation, name)
                    state = "running"
                if state == "running":  # if job should be running, check if it has finished by producing all necessary output
                    n_output_files = self.count_output_files(generation, name)
                    if n_output_files >= evolution_parameters[2]:
                        state = "finished"
                if state == "running" and not self.test_if_currently_running(generation, name):  # if job should be running, check via commands that it is indeed running
                    state = "crashed"
                self.tuning_state_by_generation[generation][name] = state

    def gather_output_from_previous_job(self, generation, name):
        """
        Read output files from previous jobs to determine the parameters for the next job.
        Also generate plots of the previous job.
        """
        directory = self.get_output_directory(generation, name)

        # load data
        filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        match_pattern = "^curve_" + "_".join(["[+-]?[0-9]+\.[0-9]+(e[+-]?[0-9]+)?"]*hyperparameter_normalization.n_defaults[name]) + "$"
        filenames = list(filter(lambda fname: re.match(match_pattern, fname), filenames))
        learning_curve_data_list = []
        normalized_hyperparameters = []
        for fname in filenames:
            normalized_hyperparameters.append([normalizing_fn(float(hyperparameter)) for normalizing_fn, hyperparameter in zip(hyperparameter_normalization.normalizing_fns[name], fname[6:].split("_"))])
            learning_curve_data_list.append(learning_curve_tools2.LearningCurveFromFile(directory + fname))
        normalized_hyperparameters = np.array(normalized_hyperparameters)

        # Plot the data
        self.plot_learning_curve(generation, name, learning_curve_data_list)

        # Choose the half which performed the best over the last 10 percent of iterations for the next generation
        stddev, max_iteration, population_size = self.tuning_schedule[generation]
        learning_curve_final_losses = [np.mean(learning_curve_data.learning_curve[int(0.9*max_iteration):,2]) for learning_curve_data in learning_curve_data_list]
        learning_curve_final_losses = list(map(lambda x: np.inf if np.isnan(x) else x, learning_curve_final_losses))
        n_non_failures = sum([1 if np.isfinite(x) else 0 for x in learning_curve_final_losses])
        best_runs = np.argsort(learning_curve_final_losses)[:min(n_non_failures, int(0.5*population_size))]
        normalized_means = np.mean(normalized_hyperparameters[best_runs], axis=0)

        # Average the hyperparameters of the best performing runs from the generation
        self.best_run_means[generation][name] = [unnormalizing_fn(normalized_mean) for unnormalizing_fn, normalized_mean in zip(hyperparameter_normalization.unnormalizing_fns[name], normalized_means.tolist())]

    def submit_next_job(self, generation, name):
        """
        Write a .sh file which runs hyperparameter_tuning_child.py with information about how to produce the next generation.
        Submit this .sh file to the job scheduler and record down the job numbers.
        """
        directory = self.get_output_directory(generation, name)

        # write the script
        script_fname = directory[:directory.find("learning_curve_data/")] + "generation_" + str(generation) + "_" + name + ".sh"
        script_file_contents = "\n".join([
            "#!/bin/bash",
            "source /etc/profile",
            "module load anaconda/2021a",
            " ".join([
                "python tuning/hyperparameter_tuning_child2.py",
                "-n " + name,
                "-x " + " ".join(list(map(str, self.best_run_means[generation-1][name]))),
                "-s " + str(self.tuning_schedule[generation][0]),
                "-t " + str(self.tuning_schedule[generation][1]),
                "-g " + str(generation),
                "-b \"" + str(self.base_directory) + "\"",
                "-T " + self.task_name,
            ])
        ])
        with open(script_fname, mode="w") as f:
            f.write(script_file_contents)

        # run the script, record the job numbers
        output = str(subprocess.Popen(["chmod", "u+x", script_fname], stdout = subprocess.PIPE).communicate())
        output = str(subprocess.Popen(["LLsub", "./" + script_fname, "[" + str(int(np.ceil(self.tuning_schedule[generation][2]/LLSUB_tasks_per_node))) + "," + str(LLSUB_tasks_per_node) + ",1]"], stdout = subprocess.PIPE).communicate())
        LLsub_number = output[output.find("LLSUB.")+6:output.find("LLSUB.")+output[output.find("LLSUB."):].find("/")]
        slurm_number = output[output.find("Submitted batch job ")+20:output.find("Submitted batch job ")+28]
        self.job_array_numbers[generation][name] = (LLsub_number, slurm_number)

    def count_output_files(self, generation, name):
        """
        Count how many output files a job has produced.
        """
        directory = self.get_output_directory(generation, name)
        filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        match_pattern = "^curve_" + "_".join(["[+-]?[0-9]+\.[0-9]+(e[+-]?[0-9]+)?"]*hyperparameter_normalization.n_defaults[name]) + "$"
        filenames = list(filter(lambda fname: re.match(match_pattern, fname), filenames))
        return len(filenames)

    def test_if_currently_running(self, generation, name):
        """
        Test via commands whether a job is running.
        """
        slurm_number = self.job_array_numbers[generation][name][1]
        output = str(subprocess.Popen("LLstat", stdout = subprocess.PIPE).communicate())
        return slurm_number in output

    def get_output_directory(self, generation, name):
        return self.base_directory + "generation_" + str(generation) + "/" + name + "/learning_curve_data/"

    def plot_learning_curve(self, generation, name, learning_curve_data_list):
        """
        Plot the learning curves of an entire generation of a certain optimizer on the same plot.
        """
        directory = self.get_output_directory(generation, name)
        directory = directory[:directory.find("learning_curve_data/")]
        learning_curve_tools2.draw_learning_curves(
                learning_curve_data_list,
                directory + "learning_curve_by_step.pdf",
                x_axis="step",
                loss_range=self.plot_loss_range)
        learning_curve_tools2.draw_learning_curves(
                learning_curve_data_list,
                directory + "learning_curve_by_time.pdf",
                x_axis="time (s)",
                loss_range=self.plot_loss_range)

    def to_string(self, state):
        """
        For debugging purposes, print out a representation for the state of each job and overall
        tuning process. Indicate job numbers for any crashes which happen so the output files
        generated can be viewed.
        """
        message = "\nEach column is a generation which depends on the generation to the left. w=waiting r=running f=finished c=crashed. States are shown below:\n"
        for name in self.optimizer_names:
            current_progress_message = "finished"
            for generation in range(len(self.tuning_schedule)):
                message = message + state[generation][name][0]
                if state[generation][name] == "running":
                    current_progress_message = "running job: LLSUB " + str(self.job_array_numbers[generation][name][0]) + " slurm " + str(self.job_array_numbers[generation][name][1])
                elif state[generation][name] == "ready":
                    current_progress_message = "ready to run"
                elif state[generation][name] == "crashed":
                    current_progress_message = "crashed job: LLSUB " + str(self.job_array_numbers[generation][name][0]) + " slurm " + str(self.job_array_numbers[generation][name][1])
            message = message + " " + name + " " + current_progress_message + "\n"
        return message
    def copy_state(self, state):
        return [{name:state_ for name, state_ in generation_dict.items()} for generation_dict in state]
    def states_equal(self, state1, state2):
        for generation in range(len(self.tuning_schedule)):
            for name in self.optimizer_names:
                if state1[generation][name] != state2[generation][name]:
                    return False
        return True
    def is_state_frozen(self, state):
        """
        Determine from state whether there are any jobs still running.
        """
        for generation in range(len(self.tuning_schedule)):
            for name in self.optimizer_names:
                if state[generation][name] == "running":
                    return False
        return True

    def tune_all(self):
        """
        Tune all the hyperparameters, and return the tuned hyperparameters.
        """
        while True:
            old_state = self.copy_state(self.tuning_state_by_generation)
            self.manage_tuning_states()
            if not self.states_equal(old_state, self.tuning_state_by_generation):
                print(self.to_string(self.tuning_state_by_generation))
            if self.is_state_frozen(self.tuning_state_by_generation):
                break
            time.sleep(self.update_interval)
        for name in self.optimizer_names:
            self.gather_output_from_previous_job(len(self.tuning_schedule)-1, name)
        return self.best_run_means[len(self.tuning_schedule)-1]
