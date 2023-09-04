# An autoregressive model on MNIST for use as a subtask.

import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

from . import task

validation_set_size = 64
(training_set_mnist_images, _), (test_set_mnist_images, _) = tf.keras.datasets.mnist.load_data('mnist.npz')
training_set_mnist_images, validation_set_mnist_images = training_set_mnist_images[:60000-validation_set_size,:,:], training_set_mnist_images[60000-validation_set_size:,:,:]

# CNN architecture constants
n_channels = 5
n_classes = 256
n_residual = 20
n_hidden = 40
depthwise_conv_multiplier = 3
n_residual_layers_per_pool = 4

# Function to take the product of a list of numbers
def product(li):
    return reduce(lambda x,y: x*y, li, 1)


class CNN():
    """
    This class defines the feedforward computation of the CNN and the initialization.
    """

    """
    Define the forward pass of the CNN bottom up
    """
    @staticmethod
    def depthwiseconvblock(x, weights):
        y = tf.nn.conv2d(x, weights[0], 1, "SAME") + weights[1]
        y = tf.math.atan(y)
        y = tf.nn.depthwise_conv2d(y, weights[2], [1,1,1,1], "SAME") + weights[3]
        y = tf.math.atan(y)
        y = tf.nn.conv2d(y, weights[4], 1, "SAME") + weights[5]
        return x + y
    @staticmethod
    def poolblock(x, weights_list):
        x = reduce(lambda images, weights: CNN.depthwiseconvblock(images, weights), weights_list, x)
        return tf.nn.avg_pool2d(x, (2, 2), (2, 2), padding="SAME")
    @staticmethod
    def evaluate(x, weights_list_list):
        x = tf.nn.conv2d(x, weights_list_list[0][0], 1, "SAME") + weights_list_list[0][1]
        x = reduce(lambda images, weights_list: CNN.poolblock(images, weights_list), weights_list_list[1:-1], x)
        return tf.einsum('io,ni->no', weights_list_list[-1][0], x[:,0,0,:]) + weights_list_list[-1][1]


    @staticmethod
    def get_initialization():
        """
        Get an initialization for this CNN.
        """
        params = []

        params.append([
            np.random.normal(0, 1/np.sqrt(n_channels+1), size=(1, 1, n_channels, n_residual)),
            np.random.normal(0, 1/np.sqrt(n_channels+1), size=(n_residual,))
        ])
        for i in range(5):
            params.append([])
            for j in range(n_residual_layers_per_pool):
                params[i+1].append([
                    np.random.normal(0, 1/np.sqrt(n_residual+1), size=(1, 1, n_residual, n_hidden)),
                    np.random.normal(0, 1/np.sqrt(n_residual+1), size=(n_hidden,)),
                    np.random.normal(0, 1/np.sqrt(3*3+1), size=(3, 3, n_hidden, depthwise_conv_multiplier)),
                    np.random.normal(0, 1/np.sqrt(3*3+1), size=(n_hidden*depthwise_conv_multiplier,)),
                    np.random.normal(0, 1/np.sqrt(n_hidden*depthwise_conv_multiplier+1), size=(1, 1, n_hidden*depthwise_conv_multiplier, n_residual)),
                    np.random.normal(0, 1/np.sqrt(n_hidden*depthwise_conv_multiplier+1), size=(n_residual,))
                ])
        params.append([
            np.random.normal(0, 1/np.sqrt(n_residual+1), size=(n_residual, n_classes)),
            np.random.normal(0, 1/np.sqrt(n_residual+1), size=(n_classes,))
        ])

        return params


class AutoregressionTask(task.Task):
    """
    This class represents the task of autoregression on MNIST.
    """

    def __init__(self):
        self.batch_size = 256
        self.sample_new_batch()

    def sample_new_batch(self):
        self.batch_indices = np.random.randint(training_set_mnist_images.shape[0], size=(self.batch_size,))

    def get_initialization(self):
        """
        Get an initialization for the CNN in flattened weight vector form.
        """

        folded_weights = CNN.get_initialization()
        flatten = lambda x: tf.reshape(x, [-1])
        flattened_weights = [flatten(folded_weights[0][0]), flatten(folded_weights[0][1])]
        for i in range(5):
            for j in range(n_residual_layers_per_pool):
                flattened_weights = flattened_weights + list(map(flatten, folded_weights[i+1][j]))
        flattened_weights = flattened_weights + [flatten(folded_weights[-1][0]), flatten(folded_weights[-1][1])]
        return tf.concat(flattened_weights, axis=0)
 

    def fold_weights(self, CNN_weights):
        """
        Reverse the weight flattening process.
        """

        params_used = 0
        def take_shape(shape):
            nonlocal params_used
            taken_weights = tf.reshape(CNN_weights[params_used:params_used+product(shape)], shape)
            params_used += product(shape)
            return taken_weights

        folded_weights = []
        folded_weights.append([take_shape((1, 1, n_channels, n_residual)), take_shape((n_residual,))])
        for i in range(5):
            folded_weights.append([])
            for j in range(n_residual_layers_per_pool):
                folded_weights[-1].append([])
                folded_weights[-1][-1] = folded_weights[-1][-1] + [take_shape((1, 1, n_residual, n_hidden)), take_shape((n_hidden,))]
                folded_weights[-1][-1] = folded_weights[-1][-1] + [take_shape((3, 3, n_hidden, depthwise_conv_multiplier)), take_shape((n_hidden*depthwise_conv_multiplier,))]
                folded_weights[-1][-1] = folded_weights[-1][-1] + [take_shape((1, 1, n_hidden*depthwise_conv_multiplier, n_residual)), take_shape((n_residual,))]
        folded_weights.append([take_shape((n_residual, n_classes)), take_shape((n_classes,))])
        
        return folded_weights

    def preprocess(self, images, queried_points):
        """
        Generate normalized CNN input given MNIST images and the indices of pixels to generate.
        """
        batch_size = queried_points.shape[0]
        pixel_x = np.tile(np.arange(training_set_mnist_images.shape[1])[np.newaxis,:,np.newaxis], (batch_size, 1, training_set_mnist_images.shape[2]))
        pixel_y = np.tile(np.arange(training_set_mnist_images.shape[2])[np.newaxis,np.newaxis,:], (batch_size, training_set_mnist_images.shape[1], 1))
        visible = np.logical_or(pixel_x < queried_points[:,0,np.newaxis,np.newaxis],
                                np.logical_and(pixel_x == queried_points[:,0,np.newaxis,np.newaxis],
                                               pixel_y < queried_points[:,1,np.newaxis,np.newaxis])).astype(np.float64)
        y_positioner = (pixel_y <= queried_points[:,1,np.newaxis,np.newaxis]).astype(np.float64)
        CNN_input = tf.stack([
                              tf.cast(images, dtype=tf.float64)*visible/256,
                              y_positioner*2-1,
                              visible*2-1,
                              pixel_x/training_set_mnist_images.shape[1]*2-1,
                              pixel_y/training_set_mnist_images.shape[2]*2-1,
                             ], axis=3)
#        CNN_input = tf.stack([tf.roll(CNN_input[i,:,:,:], np.array([images.shape[1], images.shape[2]])//2-queried_points[i,:], [0, 1]) for i in range(CNN_input.shape[0])], axis=0)
        return CNN_input

    def __call__(self, CNN_weights, new_batch=True):
        """
        Attempt to predict the color of a pixel.
        """
        folded_weights = self.fold_weights(CNN_weights)
        if new_batch:
            self.sample_new_batch()
        images = tf.gather_nd(training_set_mnist_images, self.batch_indices[:,np.newaxis])
        queried_points = np.random.randint(training_set_mnist_images.shape[1], size=(self.batch_size, 2))

        CNN_input = self.preprocess(images, queried_points)

        true_classes = tf.cast(tf.gather_nd(images, tf.concat([np.arange(self.batch_size)[:,np.newaxis], queried_points], axis=1)), tf.int32)
        logits = CNN.evaluate(CNN_input, folded_weights)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(true_classes, logits)

        return tf.math.reduce_mean(loss)

    def evaluate_loss_on_dataset(self, CNN_weights, dataset_images):
        """
        Evaluate the full-batch loss on a dataset.
        """
        folded_weights = self.fold_weights(CNN_weights)
        batch_size = dataset_images.shape[0]
        images = dataset_images
        losses = []
        for i in range(28):
            for j in range(28):
                queried_points = (np.array([[i, j]])*np.ones([batch_size, 2])).astype(np.int32)

                CNN_input = self.preprocess(dataset_images, queried_points)

                true_classes = tf.cast(tf.gather_nd(dataset_images, tf.concat([np.arange(batch_size)[:,np.newaxis], queried_points], axis=1)), tf.int32)
                logits = CNN.evaluate(CNN_input, folded_weights)
                losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(true_classes, logits))

        return tf.reduce_mean(losses)

    def evaluate_validation_loss(self, CNN_weights):
        """
        Evaluate the full-batch loss on the validataion set.
        """
        return self.evaluate_loss_on_dataset(CNN_weights, validation_set_mnist_images)

    def evaluate_test_loss(self, CNN_weights):
        """
        Evaluate the full-batch loss on the test set.
        """
        total_loss = 0
        for start in range(0, test_set_mnist_images.shape[0], 256):
            batch = test_set_mnist_images[start:start+256]
            total_loss = total_loss + self.evaluate_loss_on_dataset(CNN_weights, batch)*batch.shape[0]
        return total_loss / test_set_mnist_images.shape[0]
   
    def generate(self, CNN_weights, batch_size=16):
        """
        Generate new MNIST images with the learned model.
        """
        folded_weights = self.fold_weights(CNN_weights)
        flattened_partial_images = tf.zeros([batch_size, 0], dtype=tf.int32)
        for i in range(28*28):  # Sample all the pixels one by one
            images = tf.reshape(tf.concat([flattened_partial_images, tf.zeros([batch_size, 28*28-flattened_partial_images.shape[1]], dtype=tf.int32)], axis=1), [batch_size, 28, 28])
            queried_points = np.tile(np.array([[i//28, i%28]], dtype=np.int32), [batch_size, 1])
            CNN_input = self.preprocess(images, queried_points)
            probabilities = tf.nn.softmax(CNN.evaluate(CNN_input, folded_weights), axis=1)
            samples = tf.math.reduce_sum(tf.cast(np.random.uniform(0, 1, size=[batch_size, 1]) > tf.math.cumsum(probabilities, axis=1), tf.int32), axis=1)
            flattened_partial_images = tf.concat([flattened_partial_images, samples[:,tf.newaxis]], axis=1)
        return tf.reshape(flattened_partial_images, [batch_size, 28, 28])
    
    def visualize(self, CNN_weights, fname, batch_size=16):
        """
        Generate and save MNIST images from the learned model.
        """
        print(fname, end=" ")
        images = self.generate(CNN_weights, batch_size=batch_size)
        self.save_image_grid(images, fname)

    def save_image_grid(self, images, fname):
        """
        Draw and save an array of images in a grid format for visualization.
        """
        batch_size = images.shape[0]
        grid_width = int(np.floor(np.sqrt(batch_size)))
        grid_height = int(np.ceil(batch_size/grid_width))
        images = tf.concat([images, 255*np.ones([grid_width*grid_height-batch_size, 28, 28], dtype=np.int32)], axis=0)
        images = images.numpy().astype(np.uint8)
        images = images.reshape([grid_width, grid_height, 28, 28]).transpose([0, 2, 1, 3]).reshape([grid_width*28, grid_height*28])
        fig, ax = plt.subplots()
        ax.imshow(images, vmin=0, vmax=255, cmap="gray")
        plt.axis("off")
        plt.savefig(fname, format="pdf", bbox_inches="tight")
        plt.close()
