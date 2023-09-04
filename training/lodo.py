# This file contains the training loop logic for using LODO to minimize a function defined by a training_iteration_handler.
# It also contains ablated versions of LODO.

import numpy as np
import tensorflow as tf
import time
import sys

import context
from training import minimization_procedure

class LODOMinimizationProcedure(minimization_procedure.MinimizationProcedureClass):
    """
    This class allows us to minimize using the LODO algorithm.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9, n_layers=16, block_size=4):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter
        self.n_layers = n_layers  # The number of layers in the neural network inside LODO (not including the transposed portion)
        self.block_size = block_size  # The block size of the block diagonal matrices contained in the neural network

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.hidden_dimension = 2*self.task_dimension
        self.hidden_dimension = self.hidden_dimension + self.block_size - (self.hidden_dimension % self.block_size)
        self.weights, self.permutations = self.initialize_layers()

    def initialize_layers(self):
        """
        Initialize the weights and permutation matrices for the neural network which LODO uses to choose the step.
        Weights are initialized to random orthogonal such that the neural network in predict_step is initialized
        to perform the -self.initial_lr*identity_matrix operation.
        """

        weights = []
        permutations = []
        for i in range(self.n_layers):
#            print(time.time(), "init 1")
#            sys.stdout.flush()
            permutations.append(np.random.permutation(self.hidden_dimension)[:,np.newaxis])
            weights.append(tf.Variable(np.linalg.svd(np.random.normal(0, 1, size=[self.hidden_dimension//self.block_size, self.block_size, self.block_size]))[2]))
        for i in range(self.n_layers):
#            print(time.time(), "init 2")
#            sys.stdout.flush()
            permutations.append(np.argsort(permutations[self.n_layers-1-i][:,0])[:,np.newaxis])
        return weights, permutations

    def predict_step(self, weights, gradient):
        """
        Use a neural network to choose the step given the gradient. This neural network has no bias nodes nor activations,
        and its transpose is applied afterwards, such that we guarantee that the whole system is represented by a negative-
        semidefinite symmetric matrix.
        """

        # First apply the regular portion
        x = tf.concat([gradient, tf.zeros([self.hidden_dimension-self.task_dimension], dtype=tf.float64)], axis=0)
        for permutation, weight in zip(self.permutations[:len(self.permutations)//2], weights):
            x = tf.gather_nd(x, permutation)
            x = tf.reshape(tf.einsum('ni,nio->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
        # Then apply its transpose
        for permutation, weight in zip(self.permutations[len(self.permutations)//2:], reversed(weights)):
            x = tf.reshape(tf.einsum('ni,noi->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
            x = tf.gather_nd(x, permutation)
        # And finally multiply by some constants
        return -self.initial_lr*x[:self.task_dimension]

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using LODO.
        """

        meta_optimizer = tf.keras.optimizers.Adam(lr=self.meta_lr)
#        print(time.time(), "prep")
#        sys.stdout.flush()
        self.get_ready_to_minimize(x)

        m = tf.zeros([self.task_dimension], dtype=tf.float64)

#        t = 0
        while True:
#            t += 1
#            print(time.time(), "step", t)
#            sys.stdout.flush()

            with tf.GradientTape() as tape:

                tape.watch(self.weights)
                step = self.predict_step(self.weights, m)
                x = x + tf.cast(step, x.dtype)

                tape.watch(x)
                loss = training_iteration_handler.sample_training_loss(x)

            all_gradients = tape.gradient(loss, self.weights + [x])
            weights_gradient, g = all_gradients[:-1], all_gradients[-1]

            meta_optimizer.apply_gradients(zip(weights_gradient, self.weights))

            m = self.beta*m + (1-self.beta)*tf.cast(g, tf.float64)

            if training_iteration_handler.stopping_condition():
                break

class LODOWithResidualsMinimizationProcedure(LODOMinimizationProcedure):
    """
    This class allows us to minimize using a different architecture with residuals as described in the ablations section of the paper.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9, n_layers=64):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter
        self.n_layers = n_layers  # The number of layers in the neural network inside LODO (not including the transposed portion)

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.hidden_dimension = 2*self.task_dimension
        self.weights, self.permutations = self.initialize_layers()
    
    def initialize_layers(self):
        """
        Initialize the weights and permutation matrices for the neural network which LODO uses to choose the step.
        Weights are initialized to small random normal such that the neural network in predict_step is initialized
        close to the -self.initial_lr*identity_matrix operation.
        """

        weights = []
        permutations = []
        for i in range(self.n_layers):
            permutations.append(np.random.permutation(self.hidden_dimension)[:,np.newaxis])
            weights.append(tf.Variable(np.random.normal(0, 1/np.sqrt(self.n_layers), size=[self.hidden_dimension])))
        for i in range(self.n_layers):
            permutations.append(np.argsort(permutations[self.n_layers-1-i][:,0])[:,np.newaxis])
        weights.append(tf.Variable(tf.ones([self.hidden_dimension], dtype=tf.float64)))
        return weights, permutations

    def predict_step(self, weights, gradient):
        """
        Use a neural network to choose the step given the gradient. This neural network has no bias nodes nor activations,
        and its transpose is applied afterwards, such that we guarantee that the whole system is represented by a negative-
        semidefinite symmetric matrix.
        """
        # First apply the regular portion
        x = tf.concat([gradient, tf.zeros([self.hidden_dimension-self.task_dimension], dtype=tf.float64)], axis=0)
        for permutation, weight in zip(self.permutations[:len(self.permutations)//2], weights[:-1]):
            x = x + weight*tf.gather_nd(x, permutation)
        # Apply diagonal rescaling
        x = x*weights[-1]
        # Then apply the transpose
        for permutation, weight in zip(self.permutations[len(self.permutations)//2:], reversed(weights[:-1])):
            x = x + tf.gather_nd(weight*x, permutation)
        # And finally multiply by some constants
        return -self.initial_lr*x[:self.task_dimension]

class LODODiagonalMinimizationProcedure(LODOMinimizationProcedure):
    """
    This class allows us to minimize using the LODO algorithm but with architecture consisting only of a diagonal.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.weights = [tf.Variable(np.ones([self.task_dimension], dtype=np.float64))]

    def predict_step(self, weights, gradient):
        """
        Rescale the gradients by a constant matrix to use as the step.
        """

        return -self.initial_lr*weights[0]*gradient

class LODOGlobalMinimizationProcedure(LODODiagonalMinimizationProcedure):
    """
    This class allows us to minimize using the LODO algorithm but with architecture consisting only of a diagonal.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.weights = [tf.Variable(np.ones([], dtype=np.float64))]

class LODOSGDMinimizationProcedure(minimization_procedure.MinimizationProcedureClass):
    """
    This class allows us to minimize using the LODO algorithm.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9, n_layers=16, block_size=4):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter
        self.n_layers = n_layers  # The number of layers in the neural network inside LODO (not including the transposed portion)
        self.block_size = block_size  # The block size of the block diagonal matrices contained in the neural network

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.hidden_dimension = 2*self.task_dimension
        self.hidden_dimension = self.hidden_dimension + self.block_size - (self.hidden_dimension % self.block_size)
        self.weights, self.permutations = self.initialize_layers()

    def initialize_layers(self):
        """
        Initialize the weights and permutation matrices for the neural network which LODO uses to choose the step.
        Weights are initialized to random orthogonal such that the neural network in predict_step is initialized
        to perform the -self.initial_lr*identity_matrix operation.
        """

        weights = []
        permutations = []
        for i in range(self.n_layers):
            permutations.append(np.random.permutation(self.hidden_dimension)[:,np.newaxis])
            weights.append(tf.Variable(np.linalg.svd(np.random.normal(0, 1, size=[self.hidden_dimension//self.block_size, self.block_size, self.block_size]))[2]))
        for i in range(self.n_layers):
            permutations.append(np.argsort(permutations[self.n_layers-1-i][:,0])[:,np.newaxis])
        return weights, permutations

    def predict_step(self, weights, gradient):
        """
        Use a neural network to choose the step given the gradient. This neural network has no bias nodes nor activations,
        and its transpose is applied afterwards, such that we guarantee that the whole system is represented by a negative-
        semidefinite symmetric matrix.
        """

        # First apply the regular portion
        x = tf.concat([gradient, tf.zeros([self.hidden_dimension-self.task_dimension], dtype=tf.float64)], axis=0)
        for permutation, weight in zip(self.permutations[:len(self.permutations)//2], weights):
            x = tf.gather_nd(x, permutation)
            x = tf.reshape(tf.einsum('ni,nio->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
        # Then apply its transpose
        for permutation, weight in zip(self.permutations[len(self.permutations)//2:], reversed(weights)):
            x = tf.reshape(tf.einsum('ni,noi->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
            x = tf.gather_nd(x, permutation)
        # And finally multiply by some constants
        return -self.initial_lr*x[:self.task_dimension]

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using LODO.
        """

        meta_optimizer = tf.keras.optimizers.SGD(lr=self.meta_lr)
        self.get_ready_to_minimize(x)

        m = tf.zeros([self.task_dimension], dtype=tf.float64)

        while True:

            with tf.GradientTape() as tape:

                tape.watch(self.weights)
                step = self.predict_step(self.weights, m)
                x = x + step

                tape.watch(x)
                loss = training_iteration_handler.sample_training_loss(x)

            all_gradients = tape.gradient(loss, self.weights + [x])
            weights_gradient, g = all_gradients[:-1], all_gradients[-1]

            meta_optimizer.apply_gradients(zip(weights_gradient, self.weights))

            m = self.beta*m + (1-self.beta)*g

            if training_iteration_handler.stopping_condition():
                break


class LNDOMinimizationProcedure(minimization_procedure.MinimizationProcedureClass):
    """
    This class allows us to minimize using the LNDO algorithm.
    """

    def __init__(self, initial_lr=1.0, meta_lr=0.001, beta=0.9, n_layers=16, block_size=4):
        self.initial_lr = initial_lr  # The initial global learning rate used by LODO
        self.meta_lr = meta_lr  # The learning rate with which the neural network inside LODO is trained
        self.beta = beta  # A momentum parameter
        self.n_layers = n_layers  # The number of layers in the neural network inside LODO (not including the transposed portion)
        self.block_size = block_size  # The block size of the block diagonal matrices contained in the neural network
        self.grad_scale_beta = 0.999
        self.grad_scale_ema = 0
        self.grad_scale_normalization = 0

    def get_ready_to_minimize(self, x):
        """
        Initialize the neural network to choose steps in a parameter space of the same dimension as x.
        """

        self.task_dimension = x.shape[0]
        self.hidden_dimension = 2*self.task_dimension
        self.hidden_dimension = self.hidden_dimension + self.block_size - (self.hidden_dimension % self.block_size)
        self.weights, self.permutations = self.initialize_layers()

    def initialize_layers(self):
        """
        Initialize the weights and permutation matrices for the neural network which LODO uses to choose the step.
        Weights are initialized to random orthogonal such that the neural network in predict_step is initialized
        to perform the -self.initial_lr*identity_matrix operation.
        """

        weights = []
        permutations = []
        for i in range(self.n_layers):
            permutations.append(np.random.permutation(self.hidden_dimension)[:,np.newaxis])
            weights.append(tf.Variable(np.linalg.svd(np.random.normal(0, 1, size=[self.hidden_dimension//self.block_size, self.block_size, self.block_size]))[2]))
        for i in range(self.n_layers):
            permutations.append(np.argsort(permutations[self.n_layers-1-i][:,0])[:,np.newaxis])
        return weights, permutations

    def predict_step(self, weights, gradient):
        """
        Use a neural network to choose the step given the gradient. This neural network has no bias nodes nor activations,
        and its transpose is applied afterwards, such that we guarantee that the whole system is represented by a negative-
        semidefinite symmetric matrix.
        """

#        grad_scale = tf.math.sqrt(tf.mean(gradient**2))
#        self.grad_scale_ema = self.grad_scale_beta*self.grad_scale_ema + grad_scale
#        self.grad_scale_normalization = self.grad_scale_beta*self.grad_scale_normalization + 1
#        grad_scale_estimator = self.grad_scale_ema / self.grad_scale_normalization

        self.grad_scale_ema = self.grad_scale_beta*self.grad_scale_ema + gradient**2
        self.grad_scale_normalization = self.grad_scale_beta*self.grad_scale_normalization + 1
        grad_scale_estimator = np.sqrt(self.grad_scale_ema / self.grad_scale_normalization + 1e-8)
        

        # First apply the regular portion
        x = tf.concat([gradient/grad_scale_estimator, tf.random.normal([self.hidden_dimension-self.task_dimension], dtype=tf.float64)], axis=0)
        for permutation, weight in zip(self.permutations[:len(self.permutations)//2], weights):
            x = tf.gather_nd(x, permutation)
            x = tf.reshape(tf.einsum('ni,nio->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
        total_log_det = tf.reduce_sum([tf.reduce_sum(tf.math.log(tf.math.abs(tf.linalg.det(weight)))) for weight in weights[:len(self.permutations)//2]], axis=0)  # scalar
        loss = -total_log_det + 1/2*tf.reduce_sum(x**2)
        # Then apply its transpose
        for permutation, weight in zip(self.permutations[len(self.permutations)//2:], reversed(weights)):
            x = tf.reshape(tf.einsum('ni,noi->no', tf.reshape(x, [-1, self.block_size]), weight), [-1])
            x = tf.gather_nd(x, permutation)
        # And finally multiply by some constants
        return -self.initial_lr*x[:self.task_dimension], loss

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using LODO.
        """

        meta_optimizer = tf.keras.optimizers.Adam(lr=self.meta_lr)
        self.get_ready_to_minimize(x)

        m = tf.zeros([self.task_dimension], dtype=tf.float64)

        while True:

            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = training_iteration_handler.sample_training_loss(x)
            g = tape.gradient(loss, [x])[-1]

            with tf.GradientTape() as tape:
                tape.watch(self.weights)
                step, loss = self.predict_step(self.weights, g)
            weight_gradients = tape.gradient(loss, self.weights)

            m = self.beta*m + (1-self.beta)*step
            x = x + m
            meta_optimizer.apply_gradients(zip(weights_gradient, self.weights))

            if training_iteration_handler.stopping_condition():
                break


