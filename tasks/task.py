# This class is a base class which tasks can inherit from.

import numpy as np
import tensorflow as tf

class Task():
    """
    This class encapsulates the machine learning model being trained by our method.
    """

    def __call__(self, x, new_batch=True):
        """
        Evaluate the loss at point x.
        """
        raise NotImplementedError("Override this method")

    def loss_and_grad_fn(self, x, new_batch=True):
        """
        Evaluate the loss and gradient at point x.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self(x, new_batch=new_batch)
        grad = tape.gradient(loss, [x])[0]
        return loss, grad

    def get_initialization(self):
        """
        Produce a random value of x to use as initialization.
        """
        raise NotImplementedError("Override this method")

    def evaluate_validation_loss(self, CNN_weights):
        """
        Evaluate the full-batch loss on the validataion set.
        """
        raise NotImplementedError("Override this method")

    def evaluate_test_loss(self, CNN_weights):
        """
        Evaluate the full-batch loss on the test set.
        """
        raise NotImplementedError("Override this method")

    def visualize(self, data, labels, fname):
        """
        Visualize data with labels pertaining to this task, and save with fname.
        The data may be the x, or could be a history of past xs, for example.
        """
        raise NotImplementedError("Override this method")

    def save(self, x, fname):
        """
        Save x to a file with fname.
        """
        raise NotImplementedError("Override this method")
