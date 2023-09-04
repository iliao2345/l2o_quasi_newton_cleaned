# This is a test task indented to explore how LODO learns the Hessian.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import task
from training.learning_curve_tools import default_color_fn

class NoisyQuadraticBowlTask(task.Task):
    """
    This class represents the task of tracking the moving minimum of a quadratic bowl.
    """
    def __init__(self):
        self.dimension = 100
        eigenvalues = np.e**(np.linspace(np.log(0.001), np.log(1), self.dimension))
        eigenvectors = np.linalg.qr(np.random.normal(0, 1, [self.dimension, self.dimension]))[0]
        self.hessian = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
        self.center = np.zeros([self.dimension], dtype=np.float64)

    def __call__(self, x, new_batch=True):
        """
        Return the loss.
        """
        if new_batch:
            self.center += np.random.normal(0, 1, [self.dimension])
        return 1/2*tf.einsum("i,ij,j->", x-self.center, self.hessian, x-self.center)

    def get_initialization(self):
        """
        Get an initialization for the Rosenbrock function.
        """
        return tf.zeros([self.dimension], dtype=tf.float64)
