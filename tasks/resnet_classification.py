# An autoregressive model on MNIST for use as a subtask.

import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

from . import task
import tasks.resnet_helper as resnet_helper

padding = 4
image_size = 32
target_size = image_size + padding*2

# Code from https://github.com/lionelmessi6410/tensorflow2-cifar/tree/main. See LICENCE.
def get_dataset():
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images/255.0, test_images/255.0

    # One-hot labels
#    train_labels = _one_hot(train_labels, 10)
#    test_labels = _one_hot(test_labels, 10)
    return train_images, train_labels[:,0].astype(np.int64), test_images, test_labels[:,0].astype(np.int64)

def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std

def dataset_generator(images, labels, batch_size, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if repeat:
        ds = ds.as_numpy_iterator()
    return ds

#def _one_hot(train_labels, num_classes, dtype=np.float32):
#    """Create a one-hot encoding of labels of size num_classes."""
#    return np.array(train_labels == np.arange(num_classes), dtype)

def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels

#batch_size = 128
batch_size = 2048
#batch_size = 32

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#x_train, x_test = x_train/255.0, x_test/255.0
##(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
##(training_set_mnist_images, _), (test_set_mnist_images, _) = tf.keras.datasets.mnist.load_data('mnist.npz')
#n_validation = x_train.shape[0]//10
#n_train = x_train.shape[0]-n_validation
#n_test = x_test.shape[0]
#x_validation, y_validation = x_train[-n_validation:], y_train[-n_validation:]
#x_train, y_train = x_train[:n_validation], y_train[:n_validation]

train_images, train_labels, test_images, test_labels = get_dataset()
mean, std = get_mean_and_std(train_images)
train_images = normalize(train_images, mean, std)
test_images = normalize(test_images, mean, std)

train_dataset = dataset_generator(train_images, train_labels, batch_size, repeat=True)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).\
        batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#datasets = []
#for x, y, split in [(x_train, y_train, "train"), (x_validation, y_validation, "validation"), (x_test, y_test, "test")]:
#    x_dataset = tf.data.Dataset.from_tensor_slices(x)
#    y_dataset = tf.data.Dataset.from_tensor_slices(y)
#    x_dataset = x_dataset.batch(batch_size)
#    y_dataset = y_dataset.batch(batch_size)
#    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
#    dataset = dataset.shuffle(x.shape[0]//batch_size+1)
#    if split == "train":
#        dataset = dataset.repeat()
#    dataset = dataset.prefetch(2)
#    if split == "train":
#        dataset = dataset.as_numpy_iterator()
#    datasets.append(dataset)
#
#train_dataset, validation_dataset, test_dataset = datasets

product = lambda li: reduce(lambda x,y: x*y, li, 1)

class ResnetClassificationTask(task.Task):
    """
    This class represents the task of CIFAR10 classification.
    """

    def __init__(self, regularization=1e-4):
        self.regularization = regularization
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def get_initialization(self):
        """
        Get an initialization for the CNN in flattened weight vector form.
        """
        input_tensor = tf.keras.Input(shape=(32, 32, 3))
#        x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1, trainable=False)(input_tensor)
#        x = tf.keras.applications.efficientnet.EfficientNetB0(
#            include_top=True,
#            weights=None,
#            input_tensor=x,
#            classes=100,
#            classifier_activation=None
#        )(x)
#        self.keras_model = tf.keras.Model(input_tensor, x)

#        self.keras_model = resnet_helper.resnet_18(100)

#        self.keras_model = tf.keras.Sequential([
#            tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1, trainable=False),
#            resnet_helper.resnet_18(10),
#        ])

        self.keras_model = resnet_helper.ResNet('resnet18', 10)

        self.keras_model(input_tensor)

        weight_vector, self.shape = self.flatten(self.keras_model.trainable_weights)

        return weight_vector

    def flatten(self, weights):
        shape = []
        vect = tf.zeros([0], dtype=tf.float32)
        for weight in weights:
            shape.append(weight.shape)
            vect = tf.concat([vect, tf.reshape(weight, [-1])], axis=0)
        return vect, shape

    def fold_weights(self, weights):
        """
        Reverse the weight flattening process.
        """

        folded_weights = []
        cursor = 0
        for shape in self.shape:
            folded_weights.append(tf.reshape(weights[cursor:cursor+product(shape)], shape))
            cursor += product(shape)

        return folded_weights

    def get_trainable_layers(self, model):
#        if isinstance(model, resnet_helper.ResNetTypeI) or isinstance(model, tf.keras.Sequential) or isinstance(model, resnet_helper.BasicBlock):
        if isinstance(model, resnet_helper.BuildResNet) or isinstance(model, tf.keras.Sequential):
            layers = []
            for layer in model.layers:
                layers = layers + self.get_trainable_layers(layer)
            return layers
        elif model.trainable:
            return [model]
        else:
            return []

    def set_trainable_layers(self, updated_weights):
        weight_num = 0
        for layer in self.get_trainable_layers(self.keras_model):
            updated_weights_sublist = []
            for weight in layer.get_weights():
                if weight.shape == updated_weights[weight_num].shape:
                    updated_weights_sublist.append(updated_weights[weight_num])
                    weight_num += 1
                else:
                    updated_weights_sublist.append(weight)
            layer.set_weights(updated_weights_sublist)
        assert weight_num == len(updated_weights)

#    def set_trainable_layers(self, updated_weights):
#        weight_num = 0
#        print(len(self.keras_model.layers))
#        for layer in self.keras_model.layers[2].layers:
#            if layer.trainable:
#                updated_weights_sublist = []
#                for weight in layer.get_weights():
#                    if weight.shape == updated_weights[weight_num].shape:
#                        updated_weights_sublist.append(updated_weights[weight_num])
#                        weight_num += 1
#                    else:
#                        updated_weights_sublist.append(weight)
#                layer.set_weights(updated_weights_sublist)
#        assert weight_num == len(updated_weights)

    def __call__(self, weights, new_batch=True):
        """
        Attempt to predict the color of a pixel.
        NOTE: new_batch=False is unsupported.
        """
        folded_weights = self.fold_weights(weights)
        self.set_trainable_layers(folded_weights)

        images, true_classes = next(train_dataset)

        with tf.GradientTape() as tape:
            logits = self.keras_model(images)
            loss = self.loss_fn(true_classes, logits)
        grad = tape.gradient(loss, self.keras_model.trainable_weights)
        grad_vector, _ = self.flatten(grad)

        # first order approximation
        dot_product = tf.tensordot(weights, tf.stop_gradient(grad_vector), axes=1)
        regularization = self.regularization*tf.nn.l2_loss(weights)
        return tf.stop_gradient(loss - dot_product) + dot_product + regularization

    def evaluate_loss_on_dataset(self, weights, dataset):
        """
        Evaluate the full-batch loss on a dataset.
        """
        folded_weights = self.fold_weights(weights)
        self.set_trainable_layers(folded_weights)

        total_loss = 0
        total_n = 0
        for images, true_classes in dataset:
            output = self.keras_model(images)
            total_loss = total_loss + self.loss_fn(true_classes, output)*images.shape[0]
            total_n += images.shape[0]
        return total_loss / total_n

    def evaluate_validation_loss(self, weights):
        """
        Evaluate the full-batch loss on the validataion set.
        """
#        return self.evaluate_loss_on_dataset(weights, validation_dataset)
        return self.evaluate_loss_on_dataset(weights, test_dataset)

    def evaluate_test_loss(self, weights):
        """
        Evaluate the full-batch loss on the test set.
        """
        return self.evaluate_loss_on_dataset(weights, test_dataset)

    def evaluate_accuracy_on_dataset(self, weights, dataset):
        """
        Evaluate the full-batch loss on a dataset.
        """
        folded_weights = self.fold_weights(weights)
        self.set_trainable_layers(folded_weights)

        total_accuracy = 0
        total_n = 0
        for images, true_classes in dataset:
            logits = self.keras_model(images)
            total_accuracy = total_accuracy + tf.math.reduce_sum(tf.cast(tf.math.argmax(logits, axis=1)==true_classes, tf.float64))
            total_n += images.shape[0]
        return total_accuracy / total_n

    def evaluate_validation_accuracy(self, weights):
        """
        Evaluate the full-batch loss on the validatation set.
        """
#        return self.evaluate_accuracy_on_dataset(weights, validation_dataset)
        return self.evaluate_accuracy_on_dataset(weights, test_dataset)

    def evaluate_test_accuracy(self, weights):
        """
        Evaluate the full-batch loss on the validatation set.
        """
        return self.evaluate_accuracy_on_dataset(weights, test_dataset)
