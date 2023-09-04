import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import levenberg_marquardt as lm


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

    def __call__(self, _, x):
        """
        Return the loss.
        """
        self.center += np.random.normal(0, 1, [self.dimension])
        return 1/2*tf.einsum("i,ij,j->", x-self.center, self.hessian, x-self.center)

    def get_initialization(self):
        """
        Get an initialization for the Rosenbrock function.
        """
        return tf.zeros([self.dimension], dtype=tf.float64)

input_size = 20000

x_train = np.zeros([1, 0], dtype=tf.float32)
y_train = np.zeros([1, 100], dtype=tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='linear')])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanSquaredError())

model_wrapper = lm.ModelWrapper(
    tf.keras.models.clone_model(model))

model_wrapper.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss=lm.MeanSquaredError())

print("Train using Adam")
t1_start = time.perf_counter()
model.fit(train_dataset, epochs=1000)
t1_stop = time.perf_counter()
print("Elapsed time: ", t1_stop - t1_start)

print("\n_________________________________________________________________")
print("Train using Levenberg-Marquardt")
t2_start = time.perf_counter()
model_wrapper.fit(train_dataset, epochs=100)
t2_stop = time.perf_counter()
print("Elapsed time: ", t2_stop - t2_start)
