# Tutorial from here: https://www.tensorflow.org/tutorials/generative/dcgan

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

from IPython import display
from tensorflow.keras import layers

BUFFER_SIZE: int = 60_000
BATCH_SIZE: int = 256

# Training
EPOCHS: int = 50
NOISE_DIM: int = 100
NUM_EXAMPLES_TO_GEN: int = 16


def entrypoint():
    train(load_data(), EPOCHS)


# @tf.function
# def train_step(images):
#     # Create a BATCH_SIZE number of noise input images
#     # Each with a dimension of [NOISE_DIM, NOISE_DIM]
#     noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

#     with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
#         # Use noise to generate fake images
#         generated_images = generator_model(noise, training=True)

#         # Discriminate on real and fake images
#         real_output = discriminator_model(images, training=True)
#         fake_output = discriminator_model(generated_images, training=True)

#         # Use the discriminator's predictions on the fake data as input
#         # into the generator's loss calculation error
#         gen_loss = generator_loss(fake_output)
#         # Use the discriminator's predictions on real and fake data
#         # as input into the discriminator's loss calculation error
#         disc_loss = discriminator_loss(real_output, fake_output)

#     # Run the gradient descent on the cross entropy loss calculated in the
#     # earlier code for the generator and discriminator
#     gradients_of_generator = generator_tape.gradient(
#         gen_loss, generator_model.trainable_variables)
#     gradients_of_discriminator = discriminator_tape.gradient(
#         disc_loss, discriminator_model.trainable_variables)

#     # Apply the gradient descent calculated earlier onto the models
#     generator_optimizer.apply_gradients(
#         zip(gradients_of_generator, generator_model.trainable_variables))
#     discriminator_optimizer.apply_gradients(
#         zip(gradients_of_discriminator, discriminator_model.trainable_variables))

#     # Training and back propagation complete!

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)

        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(
            generator_model,
            epoch + 1,
            seed
        )

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')
    # End training epoch loop

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(
        generator_model,
        epochs,
        seed
    )


def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch}.png')
    plt.show()


def no_training():
    generator = build_generator_model()
    noise = tf.random.normal([1, NOISE_DIM])
    generated_image = generator(noise, training=False)

    discriminator = build_discriminator_model()
    decision = discriminator(generated_image)
    # Example output: tf.Tensor([[-0.00014535]], shape=(1, 1), dtype=float32)
    print(decision)
    # effectively a random guess (real or fake) on a random noise img


def show_random_noise():
    generator = build_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.show()


def load_data():
    # Load and prepare dataset
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    # Set each pixel value to be in range [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
        .shuffle(BUFFER_SIZE) \
        .batch(BATCH_SIZE)
    return train_dataset
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


def generator_loss(fake_output):
    # Quantify how well this was able to trick the discriminator
    # If the generator is preforming well, the discriminator will classify the
    # fake images as real (or 1)

    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    cross_entropy = tf.keras.losses.BinaryCrossentropy(
        from_logits=True  # True => y_pred in range [-inf, +inf]
    )
    # Compare the discriminators decisions on the generated images to an
    # array of 1s
    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


def discriminator_loss(real_output, fake_output):
    # Quantify how well the discriminator is able to distinguish real images from fakes.

    # https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
    cross_entropy = tf.keras.losses.BinaryCrossentropy(
        from_logits=True  # True => y_pred in range [-inf, +inf]
    )
    # discriminator's predictions on real images to an array of 1s,
    real_loss = cross_entropy(
        # A tensor the same shape as real_output but only with value=1
        y_true=tf.ones_like(real_output),
        y_pred=real_output  # The model's predictions
    )
    # discriminator's predictions on fake (generated) images to an array of 0s.
    fake_loss = cross_entropy(
        # A tensor the same shape as fake_output but only with value=1
        tf.zeros_like(fake_output),
        fake_output  # THe model's predictions
    )
    total_loss = real_loss + fake_loss
    return total_loss


# Create tensorflow components
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_model = build_generator_model()
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_model = build_discriminator_model()

# Reuse seed during training to visualize progress in animated gif
seed = tf.random.normal([NUM_EXAMPLES_TO_GEN, NOISE_DIM])

# Checkpoints for long running training
checkpoint_dir = './TrainingCheckpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'NocabGAN_checkpoints.txt')
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    generator=generator_model,
    discriminator_optimizer=discriminator_optimizer,
    discriminator=discriminator_model
)


if __name__ == '__main__':
    entrypoint()
