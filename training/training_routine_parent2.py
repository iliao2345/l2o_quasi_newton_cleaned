# This file contains code for automatically running full training using multiple optimizers.
# It writes new .sh scripts with job array submission code and submits them to the job scheduler.
# It waits for the results to come back before gathering all the results.

import argparse
import numpy as np
import regex as re
import subprocess
import time
import os

import context
from training import learning_curve_tools2

submit_mode = "sbatch"  # either "sbatch" or "LLsub"
#submit_mode = "LLsub"  # either "sbatch" or "LLsub"
#LLSUB_tasks_per_node = 8
#assert 48 % LLSUB_tasks_per_node == 0

class TrainingRoutine():
    """
    This class contains methods relevant to automatic training with multiple optimizers in parallel
    by automatic .sh script generation and submission. The main function to run is self.run_all().
    Objects contain state which track the progress of the training procedure. The results are collected
    once finished.
    """

    def __init__(self, base_directory, task_name, optimizer_names, hyperparameters, save_iterations=[100000], save_times=[100000], population_size=5, plot_loss_range=(-np.inf, np.inf), update_interval=10, metric_evaluation_interval=1000):
        """
        Create a parallel training procedure. This will generate a folder system which is
        base_directory. task_name is either "Rosenbrock" or "Autoregression" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification".
         - base_directory: folder to put results and files in
         - task_name: either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
         - optimizer_names: list of optimizer names as strings to optimize with
         - hyperparameters: list of hyperparameter tuples, one to each optimizer
         - save_iterations: list of training iteration numbers at which to save a model in a file
         - save_times: list of training times at which to save a model in a file
         - population_size: number of copies of experiment to run, to measure empirical uncertainty of results
         - plot_loss_range: (low, high), determines the loss range bounds for learning curve plots
         - update_interval: how often to check if the training job arrays finish, in seconds
        """
        self.base_directory = base_directory
        self.task_name = task_name  # either "Rosenbrock" or "Autoregression" or "ClassificationHead" or "NoisyQuadraticBowl" or "ResnetCIFAR10Classification"
        self.optimizer_names = optimizer_names  # list of optimizer names as strings
        self.hyperparameters = hyperparameters
        self.save_iterations = save_iterations
        self.save_times = save_times
        self.population_size = population_size
        self.plot_loss_range = plot_loss_range
        self.update_interval = update_interval
        self.metric_evaluation_interval = metric_evaluation_interval
        # List of states are: "running", "finished", and also "crashed". States are initialized to None and are set to one of the three states once training starts.
        self.training_states = {key:None for key in self.optimizer_names}
        # Every tuple here contains the LLsub number and the slurm number, respectively
        if submit_mode == "sbatch":
            self.job_array_numbers = {key:"" for key in self.optimizer_names}
        elif submit_mode == "LLsub":
            self.job_array_numbers = {key:("","") for key in self.optimizer_names}

        # build up the folder system
        for name in self.optimizer_names:
            os.makedirs(self.base_directory + "learning_curve_data/", exist_ok=True)
            os.makedirs(self.base_directory + "models/", exist_ok=True)
            os.makedirs(self.base_directory + "scripts/", exist_ok=True)

    def submit_all_jobs(self):
        """
        This function submits any training job arrays which don't already all their results.
        """

        for name in self.optimizer_names:
            n_output_files = self.count_output_files(name)
            if n_output_files >= self.population_size:  # it's finished if enough current results are already there
                self.training_states[name] = "finished"
            elif n_output_files == 0:  # ready to start next job array if no current results
                self.submit_job(name)
                self.training_states[name] = "running"
            else:  # something strange happened if only some results are already there
                self.training_states[name] = "crashed"

    def submit_job(self, name):
        """
        Write a .sh file which runs training_routine_child.py with information pertinnent to the training run.
        Submit this .sh file to the job scheduler and record the job array numbers so that we can check if it is finished later.
        """
        directory = self.base_directory + "scripts/"

        # write the script
        script_fname = directory + name + ".sh"
        if submit_mode == "sbatch":
            script_file_contents = "\n".join([
                "#!/bin/bash",
                "#SBATCH -a 1-" + str(self.population_size),
                "#SBATCH --exclusive",
                "#SBATCH --gres=gpu:volta:1",
                "#SBATCH -c 40",
                "source /etc/profile",
                "module load anaconda/2021a",
                "export TF_CUDNN_USE_AUTOTUNE=1",
                " ".join([
                    "python training/training_routine_child2.py",
                    "-n " + name,
                    "-x " + " ".join(list(map(str, self.hyperparameters[name]))),
                    "-s " + " ".join(list(map(str, self.save_iterations))),
                    "-t " + " ".join(list(map(str, self.save_times))),
                    "-r $SLURM_ARRAY_TASK_ID",
                    "-R " + str(self.population_size),
                    "-d " + self.base_directory,
                    "-T " + self.task_name,
                    "-v " + str(self.metric_evaluation_interval)
                ])
            ])
        elif submit_mode == "LLsub":
            script_file_contents = "\n".join([
                "#!/bin/bash",
                "source /etc/profile",
                "module load anaconda/2021a",
                " ".join([
                    "python training/training_routine_child2.py",
                    "-n " + name,
                    "-x " + " ".join(list(map(str, self.hyperparameters[name]))),
                    "-s " + " ".join(list(map(str, self.save_iterations))),
                    "-t " + " ".join(list(map(str, self.save_times))),
                    "-r $LLSUB_RANK",
                    "-R " + str(self.population_size),
                    "-d " + self.base_directory,
                    "-T " + self.task_name,
                    "-v " + str(self.metric_evaluation_interval)
                ])
            ])
        with open(script_fname, mode="w") as f:
            f.write(script_file_contents)

        # run the script, record the job array numbers
        if submit_mode == "sbatch":
            output = str(subprocess.Popen(["sbatch", "./" + script_fname, "[" + str(self.population_size) + ",1,1] -g volta:2"], stdout = subprocess.PIPE).communicate())
            slurm_number = output[output.find("Submitted batch job ")+20:output.find("Submitted batch job ")+28]
            self.job_array_numbers[name] = slurm_number
        elif submit_mode == "LLsub":
            output = str(subprocess.Popen(["chmod", "u+x", script_fname], stdout = subprocess.PIPE).communicate())
            output = str(subprocess.Popen(["LLsub", "./" + script_fname, "[" + str(int(np.ceil(self.population_size/LLSUB_tasks_per_node))) + "," + str(LLSUB_tasks_per_node) + ",1]"], stdout = subprocess.PIPE).communicate())
            LLsub_number = output[output.find("LLSUB.")+6:output.find("LLSUB.")+output[output.find("LLSUB."):].find("/")]
            slurm_number = output[output.find("Submitted batch job ")+20:output.find("Submitted batch job ")+28]
            self.job_array_numbers[name] = (LLsub_number, slurm_number)

    def manage_training_states(self):
        """
        This function is run periodically to check the presence of output from job arrays.
        """

        for name in self.optimizer_names:
            state = self.training_states[name]
            if self.training_states[name] == "running":  # if job array should be running, check if it has finished by producing all necessary output
                n_output_files = self.count_output_files(name)
                if n_output_files >= self.population_size:
                    state = "finished"
            if state == "running" and not self.test_if_currently_running(name):  # if job array should be running, check via commands that it is indeed running
                state = "crashed"
            self.training_states[name] = state

    def count_output_files(self, name):
        """
        Count how many output files a job array has produced.
        """

        directory = self.base_directory + "learning_curve_data/"
        filenames = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        match_pattern = "^" + name + "[0-9_\.]+$"
        filenames = list(filter(lambda fname: re.match(match_pattern, fname), filenames))
        return len(filenames)

    def test_if_currently_running(self, name):
        """
        Test via commands whether a job array is running.
        """
        if submit_mode == "sbatch":
            slurm_number = self.job_array_numbers[name]
        elif submit_mode == "LLsub":
            slurm_number = self.job_array_numbers[name][1]
        output = str(subprocess.Popen("LLstat", stdout = subprocess.PIPE).communicate())
        return slurm_number in output

    def to_string(self, state):
        """
        For debugging purposes, print out a representation for the state of each job array and overall
        training process. Indicate job numbers for any crashes which happen so the output files
        generated can be viewed.
        """
        message = "\n"
        for name in self.optimizer_names:
            current_progress_message = "finished"
            if state[name] == "running":
                if submit_mode == "sbatch":
                    current_progress_message = "running job: slurm " + str(self.job_array_numbers[name])
                elif submit_mode == "LLsub":
                    current_progress_message = "running job: LLSUB " + str(self.job_array_numbers[name][0]) + " slurm " + str(self.job_array_numbers[name][1])
            elif state[name] == "crashed":
                if submit_mode == "sbatch":
                    current_progress_message = "crashed job: slurm " + str(self.job_array_numbers[name])
                elif submit_mode == "LLsub":
                    current_progress_message = "crashed job: LLSUB " + str(self.job_array_numbers[name][0]) + " slurm " + str(self.job_array_numbers[name][1])
            message = message + " " + name + " " + current_progress_message + "\n"
        return message

    def copy_state(self, state):
        return {name:state_ for name, state_ in state.items()}

    def states_equal(self, state1, state2):
        for name in self.optimizer_names:
            if state1[name] != state2[name]:
                return False
        return True

    def is_state_frozen(self, state):
        """
        Determine from state whether there are any jobs still running.
        """
        for name in self.optimizer_names:
            if state[name] == "running":
                return False
        return True

    def run_all(self):
        """
        Run all the training tasks in parallel, and block until they are all finished.
        """
        self.submit_all_jobs()
        print(self.to_string(self.training_states))
        while True:
            old_state = self.copy_state(self.training_states)
            self.manage_training_states()
            if not self.states_equal(old_state, self.training_states):
                print(self.to_string(self.training_states))
            if self.is_state_frozen(self.training_states):
                break
            time.sleep(self.update_interval)
