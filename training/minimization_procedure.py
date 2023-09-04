# This file contains the training loop logic for minimizing a function defined by a training_iteration_handler.
# It allows for the use of a variety of different optimizers.

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class MinimizationProcedureClass():
    """
    This class represents an optimizer.
    """

    def __call__(self):
        raise NotImplementedError("Override this method")


class KerasMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using a keras optimizer.
    """

    def __init__(self, keras_optimizer):
        self.keras_optimizer = keras_optimizer

    def __call__(self, training_iteration_handler, x):
        opt = self.keras_optimizer()
        x = tf.Variable(x)
        while True:
            loss, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            opt.apply_gradients(zip([grad], [x]))
            if training_iteration_handler.stopping_condition():
                return x

class LBFGSMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using L-BFGS from the tensorflow_probability library.
    """

    def __call__(self, training_iteration_handler, x):
        return tfp.optimizer.lbfgs_minimize(training_iteration_handler.sample_training_loss_and_grad, 
                x,
                max_iterations=1<<30,
                stopping_condition=training_iteration_handler.stopping_condition).position

class LBFGSNoLineSearchMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using L-BFGS as defined in the paper linked below:
    http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf
    The method used is Algorithm 1 with modifications listed in the first paragraph of
    Section 3.3, as well as all stochastic modifications as described in Section 3.2.
    """

    def __init__(self, eta_0=0.1, tau=1e4, buffer_size=5, epsilon=1e-10):
        self.eta_0 = eta_0  # lr decay parameter
        self.tau = tau  # lr decay parameter
        self.buffer_size = buffer_size
        self.epsilon = epsilon  # inverse Hessian initialization size

    def __call__(self, training_iteration_handler, x):
        """
        This algorithm is taken from the O-LBFGS paper linked above.
        """

        t = 0
        buffer_ = [None for i in range(self.buffer_size)]  # stored as a list of (s, y) pairs
        while True:
            _, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            p = -grad
            alphas = []
            for i in range(1, min(t, self.buffer_size)+1):
                s_t_minus_i, y_t_minus_i = buffer_[(t-i)%self.buffer_size]
                alphas.append(tf.reduce_sum(s_t_minus_i*p)/tf.reduce_sum(s_t_minus_i*y_t_minus_i))
                p = p - alphas[-1]*y_t_minus_i
            if t > 0:
                s_t_minus_1, y_t_minus_1 = buffer_[(t-1)%self.buffer_size]
                p = tf.reduce_sum(s_t_minus_i*y_t_minus_1)/tf.reduce_sum(y_t_minus_1*y_t_minus_1)*p
            else:
                p = self.epsilon*p
            for i in reversed(range(1, min(t, self.buffer_size)+1)):
                s_t_minus_i, y_t_minus_i = buffer_[(t-i)%self.buffer_size]
                beta = tf.reduce_sum(y_t_minus_i*p)/tf.reduce_sum(y_t_minus_i*s_t_minus_i)
                p = p + (alphas[i-1] - beta)*s_t_minus_i
            s = self.eta_0*self.tau/(self.tau+t)*p
            x_new = x + s
            _, grad_prime = training_iteration_handler.sample_training_loss_and_grad(x_new, new_batch=False)
            y = grad_prime - grad

            x = x_new
            buffer_[t%self.buffer_size] = (s, y)
            t = t+1

            if training_iteration_handler.stopping_condition():
                return x

class OLBFGSMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using O-LBFGS as defined in the paper linked below:
    http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf
    The method used is Algorithm 1 with modifications listed in Section 3.3, as well as
    all stochastic modifications as described in Section 3.2.
    """

    def __init__(self, eta_0=0.1, tau=1e4, buffer_size=5, epsilon=1e-0):
        self.eta_0 = eta_0  # lr decay parameter
        self.tau = tau  # lr decay parameter
        self.buffer_size = buffer_size
        self.lam = 0  # regularization parameter
        self.epsilon = epsilon  # inverse Hessian initialization size

    def __call__(self, training_iteration_handler, x):
        """
        This algorithm is taken from the O-LBFGS paper linked above.
        """

        t = 0
        buffer_ = [None for i in range(self.buffer_size)]  # stored as a list of (s, y) pairs
        while True:
            _, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            p = -grad
            alphas = []
            for i in range(1, min(t, self.buffer_size)+1):
                s_t_minus_i, y_t_minus_i = buffer_[(t-i)%self.buffer_size]
                alphas.append(tf.reduce_sum(s_t_minus_i*p)/tf.reduce_sum(s_t_minus_i*y_t_minus_i))
                p = p - alphas[-1]*y_t_minus_i
            if t > 0:
                coefficient = 0
                for i in range(1, min(t, self.buffer_size)+1):
                    s_t_minus_i, y_t_minus_i = buffer_[(t-i)%self.buffer_size]
                    coefficient = coefficient + tf.reduce_sum(s_t_minus_i*y_t_minus_i)/tf.reduce_sum(y_t_minus_i*y_t_minus_i)
                p = p*(coefficient/min(t, self.buffer_size))
            else:
                p = self.epsilon*p
            for i in reversed(range(1, min(t, self.buffer_size)+1)):
                s_t_minus_i, y_t_minus_i = buffer_[(t-i)%self.buffer_size]
                beta = tf.reduce_sum(y_t_minus_i*p)/tf.reduce_sum(y_t_minus_i*s_t_minus_i)
                p = p + (alphas[i-1] - beta)*s_t_minus_i
            s = self.eta_0*self.tau/(self.tau+t)*p
            x_new = x + s
            _, grad_prime = training_iteration_handler.sample_training_loss_and_grad(x_new, new_batch=False)
            y = grad_prime - grad + self.lam*s

            x = x_new
            buffer_[t%self.buffer_size] = (s, y)
            t = t+1

            if training_iteration_handler.stopping_condition():
                return x

class LARSMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using LARS.
    """

    def __init__(self, lr=0.001, weight_decay=0.0005, momentum=0.9):
        self.lr = lr  # starting global learning rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def __call__(self, training_iteration_handler, x, layer_split_starts):
        """
        Minimize using LARS. layer_split_starts is a list of the index of x at which each layer's weights begins.
        """
        layer_split_ends = layer_split_starts[1:] + [x.shape[0]]

        t = 0
        momentum_splits = [np.zeros([end-start], dtype=np.float64) for start, end in zip(layer_split_starts, layer_split_ends)]
        while True:
            _, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            global_lr = self.lr*(1-training_iteration_handler.training_completed_proportion)**2
            new_x = []
            for i, start, end in zip(range(len(layer_split_starts)), layer_split_starts, layer_split_starts[1:] + [x.shape[0]]):
                x_split, g_split = x[start:end], grad[start:end]
                local_lr = tf.math.sqrt(tf.reduce_sum(x_split*x_split)) / (tf.math.sqrt(tf.reduce_sum(g_split*g_split)) + self.weight_decay*tf.math.sqrt(tf.reduce_sum(x_split*x_split)))
                momentum_splits[i] = self.momentum*momentum_splits[i] + global_lr*local_lr * (g_split + self.weight_decay*x_split)
                new_x.append(x_split - momentum_splits[i])
            x = tf.concat(new_x, axis=0)
            t = t + 1
            if training_iteration_handler.stopping_condition():
                return x

class AdaHessianMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using AdaHessian from the link below:
    https://arxiv.org/pdf/2006.00719.pdf
    but with no spatial averaging.
    It requires a second order gradient with a computational cost equal to that of backpropagation through the task,
    so requires the task to be twice differentiable. This may raise issues with tasks involving only ReLU activation.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1  # momentum decay
        self.beta2 = beta2  # Hessian decay

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using AdaHessian.
        """

        t = 0
        m = 0
        v = 0
        while True:
            t = t + 1
            rademacher_vector = np.sign(np.random.uniform(-1, 1, size=x.shape))
            with tf.GradientTape() as tape:
                tape.watch(x)
                _, grad = training_iteration_handler.sample_training_loss_and_grad(x)
                dot_product = tf.reduce_sum(grad*rademacher_vector)
            D_t = rademacher_vector*tape.gradient(dot_product, [x])[0]
            training_iteration_handler.samples_taken += 1

            m = self.beta1*m + (1-self.beta1)*grad
            v = self.beta2*v + (1-self.beta2)*D_t*D_t

            m_unbiased = m / (1-self.beta1**t)
            v_unbiased = v / (1-self.beta2**t)

            x = x - self.lr*m_unbiased/tf.sqrt(v_unbiased)
            if training_iteration_handler.stopping_condition():
                return x

class YogiMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using the Yogi algorithm.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=0.001):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using Yogi.
        """

        t = 0
        m = 0
        v = 0
        while True:
            t = t + 1
            _, grad = training_iteration_handler.sample_training_loss_and_grad(x)

            m = self.beta1*m + (1-self.beta1)*grad
            v = v - (1-self.beta2)*tf.math.sign(v - grad*grad)*grad*grad

            x = x - self.lr*m/(tf.sqrt(v)+self.epsilon)
            if training_iteration_handler.stopping_condition():
                return x

class LevenbergMarquardtMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using the Levenberg-Marquardt algorithm.
    """

    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using Levenberg-Marquardt.
        """

        t = 0
        while True:
            t = t + 1

            with tf.GradientTape() as tape:
                tape.watch(x)
                _, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            hessian = tape.jacobian(grad, x)

            hessian_squared = tf.matmul(hessian, hessian)
            x = x - tf.einsum('ab,bc,c->a', tf.linalg.inv(hessian_squared + self.lam*tf.linalg.diag(tf.linalg.diag_part(hessian_squared))), hessian, grad)

            if training_iteration_handler.stopping_condition():
                return x

class BFGSMinimizationProcedure(MinimizationProcedureClass):
    """
    This class allows us to minimize using the BFGS algorithm.
    """

    def __init__(self, learning_rate=0.01, H0_scale=1.0):
        self.learning_rate = learning_rate
        self.H0_scale = H0_scale

    def __call__(self, training_iteration_handler, x):
        """
        Minimize using BFGS.
        """

        t = 0
        H = tf.eye(x.shape[0])*self.H0_scale
        g = tf.cast(training_iteration_handler.sample_training_loss_and_grad(x)[1], tf.float32)
        while True:
            t = t + 1

            p = -tf.einsum('ab,b->a', H, g)
            s = self.learning_rate*p
            x = x + tf.cast(s, x.dtype)
            loss, grad = training_iteration_handler.sample_training_loss_and_grad(x)
            y = tf.cast(grad, g.dtype) - g
            Hys = tf.einsum('ab,b,c->ac', H, y, s)
            H = H + (tf.einsum('a,a->', s, y) + tf.einsum('a,ab,b->', y, H, y)) / tf.einsum('a,a->', s, y)**2 * tf.einsum('a,b->ab', s, s) \
                  - (Hys + tf.transpose(Hys, [1, 0])) / tf.einsum('a,a->', s, y)
            g = tf.cast(grad, g.dtype)

            if training_iteration_handler.stopping_condition():
                return x
