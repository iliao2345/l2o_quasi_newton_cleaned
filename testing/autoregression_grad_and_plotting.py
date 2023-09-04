# This file tests tasks/autoregression_mnist.py.
# It simply trains using Adam, generates some images, and makes a plot of the result.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import context
from tasks import autoregression_mnist

# Task setup
task = autoregression_mnist.AutoregressionTask()

opt = tf.keras.optimizers.Adam(lr=0.001)

x = task.get_initialization()
x = tf.Variable(x)

# Trajectory recording and training loop breaking logic
time_limit_seconds = 10
start_time = time.time()
def break_condition():
    return time.time() - start_time > time_limit_seconds
loss_curve = []

# Training loop
while True:
    loss, grad = task.loss_and_grad_fn(x)
    opt.apply_gradients(zip([grad], [x]))

    loss_curve.append(float(loss))

    if break_condition():
        break

# Trajectory postprocessing
loss_curve = np.array(loss_curve)

# Test visualization functionality
task.visualize(x, "testing/mnist_image_generation_test.png")

# Validation loss evaluation testing
print("Validation loss: ", float(task.evaluate_validation_loss(x)))
