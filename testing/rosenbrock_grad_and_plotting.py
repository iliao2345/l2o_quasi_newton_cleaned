# This file tests tasks/rosenbrock.py.
# It simply trains using Adam and makes a plot of the result.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import context
from tasks import rosenbrock

# Task setup
task = rosenbrock.RosenbrockTask()

opt = tf.keras.optimizers.Adam(lr=0.1)

x = task.get_initialization()
x = tf.Variable(x)

# Trajectory recording and training loop breaking logic
time_limit_seconds = 10
start_time = time.time()
def break_condition():
    return time.time() - start_time > time_limit_seconds
trajectory = []
loss_curve = []

# Training loop
while True:
    loss, grad = task.loss_and_grad_fn(x)
    opt.apply_gradients(zip([grad], [x]))

    trajectory.append(x.numpy())
    loss_curve.append(float(loss))

    if break_condition():
        break

# Trajectory postprocessing
trajectory = np.stack(trajectory, axis=0)
loss_curve = np.array(loss_curve)

# Test visualization functionality
task.visualize([trajectory], ["Adam"], "testing/rosenbrock_adam.png")
