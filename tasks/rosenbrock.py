# This is a basic task aimed at visualizing the trajectory that various optimizers take through a loss landscape.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import task
from training.learning_curve_tools import default_color_fn

class RosenbrockTask(task.Task):
    """
    This class represents the task of minimizing the Rosenbrock test function.
    """

    def __call__(self, x, new_batch=True):
        """
        Return the value of the Rosenbrock function.
        """
        return 0.01*(1-x[0])**2 + (x[1]-x[0]**2)**2

    def get_initialization(self):
        """
        Get an initialization for the Rosenbrock function.
        """
        return tf.cast(tf.convert_to_tensor([-0.5, 2]), dtype=tf.float64)

    def evaluate_validation_loss(self, x):
        """
        Return the value of the Rosenbrock function.
        """
        return self(x)

    def evaluate_test_loss(self, x):
        """
        Return the value of the Rosenbrock function.
        """
        return self(x)

    def visualize(self, step_histories, labels, fname, step_limit=1000000):
        """
        Save an image of the trajectory which x takes while training.
        """

        plt.rcParams['font.size'] = '12'

        # Draw a heatmap of the Rosenbrock function
        left, right, bottom, top = -1.2, 1.5, -1, 2.5
        x, y = np.meshgrid(np.linspace(left, right, num=500), np.linspace(bottom, top, num=500), indexing="ij")
        coordinates = np.stack([x, y], axis=0)
        heatmap = self(coordinates)
        fig, ax = plt.subplots()
        ax.imshow(np.log(heatmap.T[::-1,:]+1e-3), cmap="RdBu", extent=(left, right, bottom, top))

        # Draw the training trajectories on top
        for i, history, label in zip(range(len(step_histories)), step_histories, labels):
            ax.plot(history[:step_limit,0], history[:step_limit,1], label=label, color=default_color_fn(label))

        # Mark the minimum
        ax.scatter(1, 1, s=200, color=(0, 1, 0), marker="*", label="Minimum")

        # Format the plot
        ax.set_xlim((left, right))
        ax.set_ylim((bottom, top))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.close()
