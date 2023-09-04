# This file contains information about how to build each optimizer from its hyperparameters.

import numpy as np
import tensorflow as tf
from functools import reduce

import context
from training import minimization_procedure
from training import lodo
from tasks import autoregression_mnist

# Figure out where the divides are between 
def product(li):
    return reduce(lambda x,y: x*y, li, 1)
initialization = autoregression_mnist.CNN.get_initialization()
layer_splits = [0]
def take_shapes(layer_weights_and_biases):
    layer_splits.append(layer_splits[-1] + sum(list(map(lambda x: product(tuple(x.shape)), layer_weights_and_biases))))
take_shapes(initialization[0])
for i in range(5):
    for j in range(autoregression_mnist.n_residual_layers_per_pool):
        take_shapes(initialization[i+1][j][0:2])
        take_shapes(initialization[i+1][j][2:4])
        take_shapes(initialization[i+1][j][4:6])
take_shapes(initialization[-1])
layer_splits = layer_splits[:-1]

# List some optimizers
minimizer_setup_fns = {
        "Adam" : lambda lr=0.001, beta_1=0.9, beta_2=0.999: minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2)),
        "RMSprop" : lambda lr=0.001, rho=0.9, momentum=0.0: minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.RMSprop(lr=lr, rho=rho, momentum=momentum)),
        "Momentum" : lambda lr=0.01, momentum=0.0: minimization_procedure.KerasMinimizationProcedure(lambda: tf.keras.optimizers.SGD(lr=lr, momentum=momentum)),
        "LARS" : lambda lr=0.001, momentum=0.9, weight_decay=0.0005: lambda handler, initialization: minimization_procedure.LARSMinimizationProcedure(lr=lr, weight_decay=weight_decay, momentum=momentum)(handler, initialization, layer_splits),
        "Yogi" : lambda lr=0.001, beta1=0.9, beta2=0.999: minimization_procedure.YogiMinimizationProcedure(lr=lr, beta1=beta1, beta2=beta2),
#        "L-BFGS" : lambda: minimization_procedure.LBFGSMinimizationProcedure(),
        "L-BFGS-no-line-search" : lambda eta_0=0.001, tau=1e4, epsilon=1e-10: minimization_procedure.LBFGSNoLineSearchMinimizationProcedure(eta_0=eta_0, tau=tau, epsilon=epsilon, buffer_size=48),
        "O-LBFGS" : lambda eta_0=0.001, tau=1e4, epsilon=1e-10: minimization_procedure.OLBFGSMinimizationProcedure(eta_0=eta_0, tau=tau, epsilon=epsilon, buffer_size=48),
#        "AdaHessian" : lambda lr=0.001, beta1=0.9, beta2=0.999: minimization_procedure.AdaHessianMinimizationProcedure(lr=lr, beta1=beta1, beta2=beta2),
        "LODO" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODOMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta, n_layers=16, block_size=4),
        "LODO-Diagonal" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODODiagonalMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta),
        "LODO-Global" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODODiagonalMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta),
        "LODO-Residuals" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODOWithResidualsMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta, n_layers=64),
        "LODO-No-Momentum" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODOWithResidualsMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=0.0, n_layers=64),
        "LODO-SGD" : lambda initial_lr=1.0, meta_lr=0.001, beta=0.9: lodo.LODOSGDMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta, n_layers=16, block_size=4),
        "LNDO" : lambda initial_lr=0.001, meta_lr=0.001, beta=0.9: lodo.LODOMinimizationProcedure(initial_lr=initial_lr, meta_lr=meta_lr, beta=beta, n_layers=16, block_size=4),
        "Levenberg-Marquardt" : lambda lam=1.0: minimization_procedure.LevenbergMarquardtMinimizationProcedure(lam=lam),
        "BFGS" : lambda lr=0.01, H0_scale=1: minimization_procedure.BFGSMinimizationProcedure(learning_rate=lr, H0_scale=H0_scale),
}

# Define default hyperparameters for each of the tasks
# Make a dict of name : <list of default hyperparamers> pairs
rosenbrock_defaults = {key:(fn.__defaults__ if fn.__defaults__ is not None else ()) for key, fn in minimizer_setup_fns.items()}
autoregression_defaults = {key:(fn.__defaults__ if fn.__defaults__ is not None else ()) for key, fn in minimizer_setup_fns.items()}
classification_head_defaults = {key:(fn.__defaults__ if fn.__defaults__ is not None else ()) for key, fn in minimizer_setup_fns.items()}
noisy_quadratic_bowl_defaults = {key:(fn.__defaults__ if fn.__defaults__ is not None else ()) for key, fn in minimizer_setup_fns.items()}
resnet_defaults = {key:(fn.__defaults__ if fn.__defaults__ is not None else ()) for key, fn in minimizer_setup_fns.items()}

# Get all of the optimizer names
names = list(minimizer_setup_fns.keys())
