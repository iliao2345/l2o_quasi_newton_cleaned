# This file contains common methods relating to saving, loading, writing to during optimization, and plotting training curves.
# This file also deals with stopping logic for the training loop.

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pickle
import os

plt.rcParams['font.size'] = '20'

class LearningCurveData():
    """
    This class represents any object capable of holding a memory past losses and saving it.
    """

    def __init__(self, name="", tracked_quantity_names = []):
        """
        Create an empty learning curve.
        """
        self.name = name
        self.learning_curve_arr = np.array([0, 3], dtype=float)
        self.tracked_quantity_names = tracked_quantity_names
        self.tracked_quantities_arr = np.array([0, 2+len(tracked_quantity_names)], dtype=float)

    @property
    def learning_curve(self):
        """
        We use a getter to extract the learning curve data so that subclasses can preprocess
        learning curve lists into arrays whenever they are needed.
        """
        return self.learning_curve_arr

    @property
    def tracked_quantities(self):
        """
        We use a getter to extract the tracked quantities so that subclasses can preprocess
        their lists into arrays whenever they are needed.
        """
        return self.tracked_quantities_arr

    @property
    def time_since_init(self):
        """
        Get the amount of total time (training and metric evaluation) which elapsed since the beginning of training.
        """
        return 0

    def save_learning_curve(self, fname):
        """
        Save the learning curve to a file.
        """
        tup = (self.learning_curve, self.tracked_quantity_names, self.tracked_quantities, self.time_since_init)
        with open(fname, "wb") as f:
            pickle.dump(tup, f)


class LearningCurveFromFile(LearningCurveData):
    """
    This class loads a learning curve from a file upon initialization.
    """

    def __init__(self, fname, name=""):
        self.name = name
        if not os.path.isfile(fname):
            raise FileNotFoundError
        with open(fname, "rb") as f:
            tup = pickle.load(f)
        self.learning_curve_arr, self.tracked_quantity_names, self.tracked_quantities_arr, self.time_since_init_ = tup

    @property
    def time_since_init(self):
        """
        Get the amount of total time (training and metric evaluation) which elapsed since the beginning of training.
        """
        return self.time_since_init_

class TrainingIterationHandler(LearningCurveData):
    """
    This class makes an empty learning curve which records losses to during training.
    It also has a function to compute whether or not to exit the training loop.
    """

    def __init__(self, task, name="", time_limit=np.inf, max_iteration=1<<30, break_condition="either", record_trajectory=False, tracked_quantity_evaluation_interval=1000, tracked_quantity_names=[], tracked_quantity_fns=[]):
        """
        Make an iteration handler. loss_and_grad_fn is a function which takes x and returns loss and grad as a tuple.
        name is stored such that it can be used to plot. The training loop breaks once either time_limit seconds has
        passed or max_iteration training iterations have passed. If record_trajectory, then the TrainingIterationHandler
        stores x for every step as well. break_condition can be "either" or "both", indicating whether either or both of
        the time limit and max iteration must be reached to break. Every tracked_quantity_evaluation_interval steps,
        evaluate all the tracked_auantity_fns with tracked_quantity_names.
        """
        self.record_trajectory = record_trajectory
        self.task = task
        self.learning_curve_list = []
        self.tracked_quantity_list = []
        self.tracked_quantity_fns = tracked_quantity_fns
        self.tracked_quantity_names = tracked_quantity_names
        self.trajectory_list = []
        self.samples_taken = 0
        self.max_iteration = max_iteration
        self.time_limit = time_limit
        self.name = name
        self.break_condition = break_condition
        self.tracked_quantity_evaluation_interval = tracked_quantity_evaluation_interval  # measured in steps, not time

        self.init_time = time.time()
        self.start_time = 0
        self.training_time = 0
        self.metric_evaluation_time = 0
        self.evaluation_mode = "uninitialized"  # "train" or "metrics", indicates whether the last evaluation done was training or to evaluate metrics

    def sample_training_loss(self, x, new_batch=True):
        """
        Evaluate the loss function, but store step, timing, and loss information on the way.
        """
        self.switch_evaluation_mode("train")
        self.samples_taken += 1
        loss = self.task(x, new_batch=new_batch)

        self.learning_curve_list.append([self.samples_taken, self.training_time+time.time()-self.start_time, float(loss)])
        if self.record_trajectory:
            self.trajectory_list.append(x.numpy())

        if self.samples_taken >= (len(self.tracked_quantity_list)+1) * self.tracked_quantity_evaluation_interval:
            x_without_gradienttape_recording = tf.convert_to_tensor(x.numpy())
            self.switch_evaluation_mode("metrics")
            self.tracked_quantity_list.append([self.samples_taken, self.metric_evaluation_time+time.time()-self.start_time] + [float(metric(x_without_gradienttape_recording)) for metric in self.tracked_quantity_fns])
            self.switch_evaluation_mode("train")
        
        return loss

    def sample_training_loss_and_grad(self, x, new_batch=True):
        """
        Evaluate the loss_and_grad_fn, but store step, timing, and loss information on the way.
        """
        self.switch_evaluation_mode("train")
        self.samples_taken += 1
        loss, grad = self.task.loss_and_grad_fn(x, new_batch=new_batch)

        self.learning_curve_list.append([self.samples_taken, self.training_time+time.time()-self.start_time, float(loss)])
        if self.record_trajectory:
            self.trajectory_list.append(x.numpy())
        
        if self.samples_taken >= (len(self.tracked_quantity_list)+1) * self.tracked_quantity_evaluation_interval:
            x_without_gradienttape_recording = tf.convert_to_tensor(x.numpy())
            self.switch_evaluation_mode("metrics")
            self.tracked_quantity_list.append([self.samples_taken, self.metric_evaluation_time+time.time()-self.start_time] + [float(metric(x_without_gradienttape_recording)) for metric in self.tracked_quantity_fns])
            self.switch_evaluation_mode("train")
        
        return loss, grad

    def switch_evaluation_mode(self, evaluation_mode):
        if evaluation_mode == "train":
            if self.evaluation_mode == "uninitialized":
                self.start_time = time.time()
                self.evaluation_mode = "train"
            elif self.evaluation_mode == "metrics":
                self.metric_evaluation_time += time.time() - self.start_time
                self.start_time = time.time()
                self.evaluation_mode = "train"
        elif evaluation_mode == "metrics":
            if self.evaluation_mode == "uninitialized":
                self.start_time = time.time()
                self.evaluation_mode = "metrics"
            elif self.evaluation_mode == "train":
                self.training_time += time.time() - self.start_time
                self.start_time = time.time()
                self.evaluation_mode = "metrics"

    @property
    def stopping_condition(self):
        """
        This is a getter which returns as stopping_condition a function whose output is True iff
        the time_limit or max_iteration are reached.
        """
        return lambda *args: self.samples_taken > 0 and self.training_completed_proportion > 1

    @property
    def training_completed_proportion(self):
        time_completed = (self.training_time + time.time() - self.start_time) / self.time_limit
        samples_completed = self.samples_taken / self.max_iteration
        return (max if self.break_condition=="either" else min)(time_completed, samples_completed)

    @property
    def learning_curve(self):
        """
        Preprocess and get the learning curve.
        """
        if len(self.learning_curve_list) == 0:
            return np.array([0, 3], dtype=float)
        return np.stack(self.learning_curve_list, axis=0)

    @property
    def tracked_quantities(self):
        """
        Preprocess and get any tracked quantities such as the validation loss.
        """
        if len(self.tracked_quantity_list) == 0:
            return np.array([0, 2+len(self.tracked_quantity_names)], dtype=float)
        return np.stack(self.tracked_quantity_list, axis=0)

    @property
    def trajectory(self):
        """
        Preprocess and get the learning trajectory.
        """
        return np.array(self.trajectory_list)

    @property
    def time_since_init(self):
        """
        Get the amount of time which has passed since this instance was made.
        """
        return time.time() - self.init_time

class TrainingIterationHandlerWithParameterSaving(TrainingIterationHandler):
    """
    This class makes an empty learning curve which records losses to during training.
    It also has a function to compute whether or not to exit the training loop.
    It also saves the parameters at specific times and steps.
    """

    def __init__(self, task, name="", time_limit=np.inf, max_iteration=1<<30, break_condition="either", save_at_steps=[], save_at_times=[], save_directory="", tracked_quantity_evaluation_interval=1000, tracked_quantity_names=[], tracked_quantity_fns=[]):
        """
        Make an iteration handler. loss_and_grad_fn is a function which takes x and returns loss and grad as a tuple.
        name is stored such that it can be used to plot. The training loop breaks once either time_limit seconds has
        passed or max_iteration training iterations have passed. If record_trajectory, then the TrainingIterationHandler
        stores x for every step as well. break_condition can be "either" or "both", indicating whether either or both of
        the time limit and max iteration must be reached to break.
        """

        super().__init__(task, name=name, time_limit=time_limit, max_iteration=max_iteration, break_condition=break_condition, tracked_quantity_evaluation_interval=tracked_quantity_evaluation_interval, tracked_quantity_names=tracked_quantity_names, tracked_quantity_fns=tracked_quantity_fns)
        self.save_directory = save_directory
        self.save_at_steps = save_at_steps
        self.save_at_times = save_at_times
        self.steps_saved = [False for step_num in save_at_steps]  # Records whether or not a save has been done for each step listed in save_at_steps
        self.times_saved = [False for time_ in save_at_times]  # Records whether or not a save has been done for each time listed in save_at_times

    def record_if_passed_milestone(self, x):
        """
        Save the parameters x upon reaching steps and times given by self.save_at_steps and self.save_at_times.
        """

        for i in range(len(self.save_at_steps)):
            if not self.steps_saved[i] and self.samples_taken >= self.save_at_steps[i]:
                self.steps_saved[i] = True
                with open(self.save_directory + "models/" + self.name + "_parameters_at_" + str(self.save_at_steps[i]) + "_steps", "wb") as f:
                    pickle.dump(x, f)
        for i in range(len(self.save_at_times)):
            if not self.times_saved[i] and self.samples_taken > 0 and self.evaluation_mode == "train" and self.training_time + time.time() - self.start_time >= self.save_at_times[i]:
                self.times_saved[i] = True
                with open(self.save_directory + "models/" + self.name + "_parameters_at_" + str(int(self.save_at_times[i])) + "_seconds", "wb") as f:
                    pickle.dump(x, f)

    def sample_training_loss(self, x, new_batch=True):
        """
        Evaluate the loss function, but store step, timing, and loss information on the way.
        """

        self.record_if_passed_milestone(x)
        return super().sample_training_loss(x, new_batch=new_batch)

    def sample_training_loss_and_grad(self, x, new_batch=True):
        """
        Evaluate the loss_and_grad_fn, but store step, timing, and loss information on the way.
        """

        self.record_if_passed_milestone(x)
        return super().sample_training_loss_and_grad(x, new_batch=new_batch)


# Choose a curve color based on the name of the learning_curve_data.
def default_color_fn(name):
    if "adam" in name.lower():
        return (1.0, 0.0, 0.0)
    elif "rmsprop" in name.lower():
        return (0.0, 1.0, 0.0)
    elif "momentum" in name.lower():
        return (0.0, 0.0, 1.0)
    elif "lars" in name.lower():
        return (0.0, 0.5, 0.5)
    elif "yogi" in name.lower():
        return (1.0, 0.75, 0.0)
#    elif "l-bfgs-no-line-search" in name.lower():
#        return (0.0, 0.4, 0.0)
    elif "l-bfgs" in name.lower():
        return (0.0, 0.4, 0.0)
    elif "o-lbfgs" in name.lower():
        return (0.8, 0.8, 0.0)
    elif "adahessian" in name.lower():
        return (0.5, 0.0, 0.5)
    elif "bfgs" in name.lower():
        return (34/255, 139/255, 34/255)
    elif "levenberg-marquardt" in name.lower():
        return (139/255, 69/255, 19/255)
    elif "lodo-diagonal" in name.lower():
        return (1.0, 0.0, 1.0)
    elif "lodo-global" in name.lower():
        return (0.0, 1.0, 1.0)
    elif "lodo-residuals" in name.lower():
        return (0.5, 0.0, 1.0)
    elif "lodo-sgd" in name.lower():
        return (139/255, 69/255, 19/255)
    elif "lodo-no-momentum" in name.lower():
        return (34/255, 139/255, 34/255)
    elif "pretrained lodo" in name.lower():
        return (1.0, 0.0, 0.0)
    elif "lodo" in name.lower():
        return (0.0, 0.0, 0.0)
    else:
        return (0.0, 0.0, 0.0)

def style_plot(fig, ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.00, 0.5))
    ax.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='y', which='minor', bottom=False)
    plt.grid(b=True, which='major', color='0.6')
    plt.grid(b=True, which='minor', color='0.8')

def draw_learning_curves(learning_curve_datas,
                         fname,
                         name_to_color_fn=default_color_fn,
                         name_to_label_fn=lambda x:x,
                         loss_range=(-np.inf, np.inf),
                         x_max=np.inf,
                         x_axis="step",  # One of two options: "step" or "time (s)"
                         ):
    """
    Plot a list of learning curves for comparison.
    learning_curve_datas: list of LearningCurveData objects.
    fname: filename to save the plot.
    name_to_color_fn: function which maps LearningCurveData names to learning curve colors.
    name_to_label_fn: function which maps LearningCurveData names to legend labels.
    loss_range: min and max losses to cap the learning curves at.
    x_max: the plot cuts the learning curves to a maximum step or time given by x_max.
    x_axis: "step" if we would like to plot the loss against the step, and
            "time (s)" if we would like to plot the loss against the training time.
    """

    # Figure out whether the learning curves need to be truncated in the x direction
    largest_x = 0
    x_axis_index = 0 if x_axis == "step" else 1
    for learning_curve_data in learning_curve_datas:
        largest_x = max(largest_x, learning_curve_data.learning_curve[-1,x_axis_index])
    x_max = min(largest_x+1e-5, x_max)
    
    # Plot each learning curve
    fig, ax = plt.subplots()
    for learning_curve_data in learning_curve_datas:
        # Read data from the learning_curve_data
        label = name_to_label_fn(learning_curve_data.name)
        color = name_to_color_fn(learning_curve_data.name)
        curve = learning_curve_data.learning_curve
        x_values = np.clip(curve[:,x_axis_index], 0, x_max)
        y_values = curve[:,2]
        
        # Determine how to bin the data
        n_bins = 500
        x_bin_size = x_max / n_bins
        x_plot_points = np.linspace(0, x_max, n_bins+1)
        x_bin_edge_indices = np.searchsorted(x_values, x_plot_points)
        n_bins_with_data = np.argmax(x_bin_edge_indices == x_values.shape[0])
        if n_bins_with_data == 0:
            n_bins_with_data = n_bins
        x_bin_edge_indices = x_bin_edge_indices[:n_bins_with_data+1]
        x_bin_start_indices = x_bin_edge_indices[:-1]
        x_bin_end_indices = np.maximum(x_bin_edge_indices[1:], x_bin_edge_indices[:-1]+1)

        # Average the data over each bin
        x_bin_centers = (x_values[x_bin_start_indices] + x_values[x_bin_end_indices-1]) / 2
        cumulative_y = np.concatenate([np.array([0.0]), np.cumsum(y_values)], axis=0)
        y_bin_totals = cumulative_y[x_bin_end_indices] - cumulative_y[x_bin_start_indices]
        y_bin_averages = y_bin_totals / (x_bin_end_indices - x_bin_start_indices)
        y_bin_averages = np.clip(y_bin_averages, loss_range[0], loss_range[1])

        # Plot the binned training curve
        if x_axis == "time":
            x_bin_centers = x_bin_centers / 3600
        ax.plot(x_bin_centers, y_bin_averages, label=label, color=color)
        
    if "noisy_quadratic_bowl" in fname.lower():
        theoretical_min = 7.411847
        ax.plot(np.array([0, x_max]), np.array([theoretical_min, theoretical_min]), '--', label="Newton method (theoretical)", color=(0, 0, 0))

    # Format the plot
    if "rosenbrock" in fname.lower():
        ax.set_yscale("log")
    if x_axis == "step":
        ax.set_xlabel("step")
    if x_axis == "time":
        ax.set_xlabel("time (h)")
    ax.set_ylabel("training loss")
    style_plot(fig, ax)
    ax.get_legend().remove()                         ############################################################################################################################# Remove this
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()

def draw_learning_curve_lists(learning_curve_datas_list,
                              fname,
                              name_to_color_fn=default_color_fn,
                              name_to_label_fn=lambda x:x,
                              loss_range=(-np.inf, np.inf),
                              x_max=np.inf,
                              x_axis="step",  # One of two options: "step" or "time (s)"
                              ):
    """
    Plot a list of learning curves for comparison. Draw error margins based on the variance
    of learning curve y values.

    learning_curve_datas: list of LearningCurveData objects.
    fname: filename to save the plot.
    name_to_color_fn: function which maps LearningCurveData names to learning curve colors.
    name_to_label_fn: function which maps LearningCurveData names to legend labels.
    loss_range: min and max losses to cap the learning curves at.
    x_max: the plot cuts the learning curves to a maximum step or time given by x_max.
    x_axis: "step" if we would like to plot the loss against the step, and
            "time (s)" if we would like to plot the loss against the training time.
    """

    # Figure out whether the learning curves need to be truncated in the x direction
    largest_x = 0
    x_axis_index = 0 if x_axis == "step" else 1
    for learning_curve_datas in learning_curve_datas_list:
        for learning_curve_data in learning_curve_datas:
            largest_x = max(largest_x, learning_curve_data.learning_curve[-1,x_axis_index])
    x_max = min(largest_x+1e-5, x_max)
    
    # Plot each learning curve
    fig, ax = plt.subplots()
    for learning_curve_datas in learning_curve_datas_list:
        # Figure out the common prefix between the names of the learning curves
        longest_common_prefix = learning_curve_datas[0].name
        def longest_common_prefix(strings):
            for i in range(min(list(map(len, strings)))):
                if not all([string[i]==strings[0][i] for string in strings]):
                    return strings[0][:i]
            return strings[0][:i+1]
        name = longest_common_prefix([learning_curve_data.name for learning_curve_data in learning_curve_datas])
        label = name_to_label_fn(name)
        color = name_to_color_fn(name)

        # Determine how to bin the data
        n_bins = min(500, int(np.min([np.sum((learning_curve_data.learning_curve[:,x_axis_index]<x_max).astype(np.float64)) for learning_curve_data in learning_curve_datas])))
        x_bin_size = x_max / n_bins
        x_plot_points = np.linspace(0, x_max, n_bins+1)

        # Gather the means and standard deviations over all learning curves with the same name
        curve_zeroeth_moments = 0
        curve_first_moments = 0
        curve_second_moments = 0
        for learning_curve_data in learning_curve_datas:
            # Read x values from the learning_curve_data to figure out which bins have data
            curve = learning_curve_data.learning_curve
            curve = curve[curve[:,x_axis_index]<x_max]
            x_values = curve[:,x_axis_index]
            y_values = curve[:,2]
            x_bin_edge_indices = np.searchsorted(x_values, x_plot_points)

            # Read data from the learning_curve_data
            x_bin_start_indices = x_bin_edge_indices[:-1]
            x_bin_end_indices = x_bin_edge_indices[1:]

            # Accumulate the data over each bin, calculate the learning curve
            x_bin_centers = x_plot_points[:-1]+x_max/n_bins/2
            cumulative_y = np.concatenate([np.array([0.0]), np.cumsum(y_values)], axis=0)
            cumulative_y_squared = np.concatenate([np.array([0.0]), np.cumsum(y_values**2)], axis=0)
            valid_bins = np.isfinite(cumulative_y[x_bin_end_indices] - cumulative_y[x_bin_start_indices])
            bin_samples = np.where(valid_bins, (x_bin_end_indices - x_bin_start_indices), 0)
            y_bin_totals = np.where(valid_bins, cumulative_y[x_bin_end_indices] - cumulative_y[x_bin_start_indices], 0)
            y_bin_means = y_bin_totals / bin_samples

            # Summarize the learning curve over multiple runs
            curve_zeroeth_moments = curve_zeroeth_moments + valid_bins.astype(np.float64)
            curve_first_moments = curve_first_moments + np.where(valid_bins, y_bin_means, 0)
            curve_second_moments = curve_second_moments + np.where(valid_bins, y_bin_means**2, 0)

        curve_means = curve_first_moments / curve_zeroeth_moments
        curve_stddevs = np.sqrt(np.maximum(0, curve_second_moments*curve_zeroeth_moments - curve_first_moments**2)) / curve_zeroeth_moments

        x_bin_centers = x_bin_centers[np.isfinite(curve_stddevs)]
        curve_means = curve_means[np.isfinite(curve_stddevs)]
        curve_stddevs = curve_stddevs[np.isfinite(curve_stddevs)]

        # Plot the binned training curve
        if x_axis == "time":
            x_bin_centers = x_bin_centers / 3600
        ax.plot(x_bin_centers, np.clip(curve_means, loss_range[0], loss_range[1]), label=label, color=color)
        if name != "LODO-Diagonal":
            ax.fill_between(x_bin_centers, np.clip(curve_means-curve_stddevs, loss_range[0], loss_range[1]), np.clip(curve_means+curve_stddevs, loss_range[0], loss_range[1]), facecolor=tuple([0.5+channel/2 for channel in color]), alpha=0.5)
        
    if "noisy_quadratic_bowl" in fname.lower():
        theoretical_min = 7.411847
        ax.plot(np.array([0, x_max]), np.array([theoretical_min, theoretical_min]), '--', label="Newton method (theoretical)", color=(0, 0, 0))

    # Format the plot
    if "rosenbrock" in fname.lower():
        ax.set_yscale("log")
    if x_axis == "step":
        ax.set_xlabel("step")
    if x_axis == "time":
        ax.set_xlabel("time (h)")
    ax.set_ylabel("training loss")
    bot, top = ax.get_ylim()
    if top/bot > 3:
        bot = 0
    ax.set_ylim((bot, top))
    style_plot(fig, ax)
    print(fname)
    ax.get_legend().remove()                         ############################################################################################################################# Remove this
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()

def draw_metric_lists(learning_curve_datas_list,
                      fname,
                      name_to_color_fn=default_color_fn,
                      name_to_label_fn=lambda x:x,
                      metric_name="validation loss",
                      loss_range=(-np.inf, np.inf),
                      x_max=np.inf,
                      x_axis="step",  # Only "step" is supported
                      ):
    """
    Plot a list of validation learning curves for comparison. Draw error margins based on the variance
    of learning curve y values.

    learning_curve_datas: list of LearningCurveData objects.
    fname: filename to save the plot.
    name_to_color_fn: function which maps LearningCurveData names to learning curve colors.
    name_to_label_fn: function which maps LearningCurveData names to legend labels.
    loss_range: min and max losses to cap the learning curves at.
    x_max: the plot cuts the learning curves to a maximum step or time given by x_max.
    """

    # Figure out whether the learning curves need to be truncated in the x direction
    largest_x = 0
    for learning_curve_datas in learning_curve_datas_list:
        for learning_curve_data in learning_curve_datas:
            largest_x = max(largest_x, learning_curve_data.tracked_quantities[-1,0])
#            largest_x = min(largest_x, learning_curve_data.tracked_quantities[-1,0])
    x_max = min(largest_x+1e-5, x_max)
    print(x_max)
    
    # Plot each learning curve
    fig, ax = plt.subplots()
    for learning_curve_datas in learning_curve_datas_list:
        # Figure out the common prefix between the names of the learning curves
        longest_common_prefix = learning_curve_datas[0].name
        def longest_common_prefix(strings):
            for i in range(min(list(map(len, strings)))):
                if not all([string[i]==strings[0][i] for string in strings]):
                    return strings[0][:i]
            return strings[0][:i+1]
        name = longest_common_prefix([learning_curve_data.name for learning_curve_data in learning_curve_datas])
        label = name_to_label_fn(name)
        color = name_to_color_fn(name)

        # Gather the means and standard deviations over all learning curves with the same name
        x_values = np.zeros([0], dtype=np.float64)
        curve_zeroeth_moments = 0
        curve_first_moments = 0
        curve_second_moments = 0
        for learning_curve_data in learning_curve_datas:
            # Read values from the learning_curve_data
            step_and_time_values = learning_curve_data.tracked_quantities[:,0:2]
            y_values = learning_curve_data.tracked_quantities[:,2+learning_curve_data.tracked_quantity_names.index(metric_name)]
            curve = tf.concat([step_and_time_values, y_values[:,tf.newaxis]], axis=1)
            curve = curve[curve[:,0]<x_max]
            if x_values.shape[0] < curve.shape[0]:
                x_values = curve[:,0]
            y_values = curve[:,2]

            # Accumulate the data over each validation set loss evaluation, calculate the learning curve
            valid_bins = np.isfinite(y_values)
            bin_samples = np.where(valid_bins, 1, 0)
            y_bin_totals = np.where(valid_bins, y_values, 0)
            y_bin_means = y_bin_totals / bin_samples

            # Summarize the learning curve over multiple runs
            curve_zeroeth_moments = curve_zeroeth_moments + valid_bins.astype(np.float64)
            curve_first_moments = curve_first_moments + np.where(valid_bins, y_bin_means, 0)
            curve_second_moments = curve_second_moments + np.where(valid_bins, y_bin_means**2, 0)

        curve_means = curve_first_moments / curve_zeroeth_moments
        curve_stddevs = np.sqrt(np.maximum(0, curve_second_moments*curve_zeroeth_moments - curve_first_moments**2)) / curve_zeroeth_moments

        # Plot the binned training curve
        ax.plot(x_values, np.clip(curve_means, loss_range[0], loss_range[1]), label=label, color=color)
        if name != "LODO-Diagonal":
            ax.fill_between(x_values, np.clip(curve_means-curve_stddevs, loss_range[0], loss_range[1]), np.clip(curve_means+curve_stddevs, loss_range[0], loss_range[1]), facecolor=tuple([0.5+channel/2 for channel in color]), alpha=0.5)
        
    # Format the plot
    if "rosenbrock" in fname.lower() and "loss" in metric_name:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(metric_name)
    if "accuracy" in metric_name:
        ax.set_ylim((0, 1))
    style_plot(fig, ax)
#    ax.get_legend().remove()                         ############################################################################################################################# Remove this
    plt.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close()
