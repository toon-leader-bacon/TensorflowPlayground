# Tutorial from here: https://www.tensorflow.org/tutorials/generative/dcgan

from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

BUFFER_SIZE: int = 60_000
BATCH_SIZE: int = 256


def entrypoint():
    generator = build_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    discriminator = build_discriminator_model()
    decision = discriminator(generated_image)
    print(decision)  # Example output: tf.Tensor([[-0.00014535]], shape=(1, 1), dtype=float32)
    # effectively a random guess (real or fake) on a random noise img


def show_random_noise():
    generator = build_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()


def load_data():
    # Load and prepare dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.minst.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    # Set each pixel value to be in range [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
        .shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE)

    # Create the Generator and Discriminator models


def build_generator_model():
    # Uses Conv2DTranspose (upsampling) layers to generate image from random noise seed
    model = tf.keras.Sequential()

    # Inpujt layer takes in a random seed
    input_layer_nodes = 7 * 7 * 256
    model.add(layers.Dense(input_layer_nodes,
              use_bias=False,
              input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # NOTE: None is the batch size
    assert model.output_shape == (None, 7, 7, 256)

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    model.add(layers.Conv2DTranspose(
        filters=128,  # dimensionality of the output space
        kernel_size=(5, 5),  # (height, width) of 2D convolution window
        strides=(1, 1),  # (height, width) of window strides
        padding='same',  # pad with zeros so output has the same height/ width dimension as input
        use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=64,  # dimensionality of the output space
        kernel_size=(5, 5),  # (height, width) of 2D convolution window
        strides=(2, 2),  # (height, width) of window strides
        padding='same',  # pad with zeros so output has the same height/ width dimension as input
        use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=1,  # dimensionality of the output space
        kernel_size=(5, 5),  # (height, width) of 2D convolution window
        strides=(2, 2),  # (height, width) of window strides
        padding='same',  # pad with zeros so output has the same height/ width dimension as input
        use_bias=False,
        activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def build_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2DTranspose(
        filters=64,  # dimensionality of the output space
        kernel_size=(5, 5),  # (height, width) of 2D convolution window
        strides=(2, 2),  # (height, width) of window strides
        padding='same',  # pad with zeros so output has the same height/ width dimension as input
        input_shape=[28, 28, 1]))  # Input shape matches generator output shape
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(
        filters=164,  # dimensionality of the output space
        kernel_size=(5, 5),  # (height, width) of 2D convolution window
        strides=(2, 2),  # (height, width) of window strides
        padding='same',  # pad with zeros so output has the same height/ width dimension as input
    ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    # Output shape = 1
    # Positive => Real
    # Negative => Fake
    model.add(layers.Dense(1))

    return model


if __name__ == '__main__':
    entrypoint()
