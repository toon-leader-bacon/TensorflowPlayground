# Based off the tutorial from here:
# https://analyticsindiamag.com/getting-started-image-generation-tensorflow-keras/
# https://www.tensorflow.org/guide/keras/custom_layers_and_models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plot


def showImg(img):
    plot.imshow(img)
    plot.colorbar()
    plot.show()


def blab():
    fashion_data = keras.datasets.fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = fashion_data
    print(x_train.shape)  # (60000, 28, 28)
    print(x_test.shape)  # (10000, 28, 28)

    # showImg(x_train[10])  # Pixel value ranges from [0, 255]

    # We're making a "self-supervised" model so no need to split
    # data between train and test
    # Merge the data sets along the 0 axis.
    # Documentation and example: https://www.tensorflow.org/api_docs/python/tf/concat#for_example
    data = tf.concat([x_train, x_test], axis=0)
    # Convert images from 2D to 3D
    # Negative insertion index => append tensor to end
    data = tf.expand_dims(data, -1)
    print(data.shape)  # (70000, 28, 28, 1) <=> (count, height, width, channels)
    # Scale pixel values to range [0, 1]
    data = tf.cast(data, tf.float32)
    data = data / 255.0

    showImg(data[10])  # Pixel value ranges from [0.0, 1.0]


def buildEncoder():
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(
        32,  # dimensionality of the output space
        (3, 3),  # kernel_size of the convolution window
        activation="relu",
        strides=(2, 2),  # strides of the convolution window
        padding="same"  # zeros evenly pad the left/right/up/down of input
    )(encoder_inputs)
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        strides=2,
        padding="same"
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(
        16,
        activation="relu"
    )(x)

    mean = layers.Dense(2, name="z_mean")(x)
    log_sigma = layers.Dense(2, name="z_log_var")(x)
    z = Sampling()([mean, log_sigma])

    encoder = keras.Model(encoder_inputs, [mean, log_sigma, z], name="encoder")
    print(encoder.summary())

# Build the VAE Architecture
# Subclassing the Layer class: https://keras.io/guides/making_new_layers_and_models_via_subclassing/


class Sampling(layers.Layer):
    # Takes an image as input and outputs sampling representation

    # Call(...) is the layer's forward pass
    def call(self, inputs):
        # inputs is an image tensor
        # Math explanation here:
        # https://towardsdatascience.com/vae-variational-autoencoders-how-to-employ-neural-networks-to-generate-new-images-bdeb216ed2c0
        # Section: Building a Variational Autoencoder model
        mean, log_sigma = inputs
        batch = tf.shape(mean)[0]
        dimension = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(
            shape=(batch, dimension), mean=0., stddev=1.)
        return mean + tf.exp(0.5 * log_sigma) * eps


if __name__ == '__main__':
    buildEncoder()
