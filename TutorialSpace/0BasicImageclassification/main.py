# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import typing

print("Tensorflow Version: ")
print(tf.__version__)

# Get the MNIST Fashion data
# The images are 28x28 NumPy arrays,
# with pixel values ranging from 0 to 255.
# The labels are an array of integers, ranging from 0 to 9.
# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

# Class names ranging from 0 to 9
class_names: list[str] = ['T-shirt/top',  # 0
                          'Trouser',      # 1
                          'Pullover',     # 2
                          'Dress',        # 3
                          'Coat',         # 4
                          'Sandal',       # 5
                          'Shirt',        # 6
                          'Sneaker',      # 7
                          'Bag',          # 8
                          'Ankle boot']   # 9


def explore_data(train_images: np.ndarray, array_name: str) -> None:
    print(f"Looking at array: \"{array_name}\"")
    print(train_images.shape)  # (60'000, 28, 28)
    # 60k images, each 28x28 pixels


print("========================")
explore_data(train_images, "train_images")
explore_data(train_labels, "train_labels")
print("========================")


def plot_image(img: np.ndarray, title: str = "") -> None:
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(title)
    plt.show()


plot_image(train_images[0], "train image 0")  # type = np.ndarray

# Each pixel is in the range of [0, 255]
# Scale each to the range [0, 1]
train_images: np.ndarray = train_images / 255
test_images: np.ndarray = test_images / 255

# Print several images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


def build_model() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        # convert 28x28 into 784x1
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # Dense layer for neural network magic
        tf.keras.layers.Dense(128, activation='relu'),

        # Output layer. Each node represents the % confidence of label
        tf.keras.layers.Dense(10)
    ])


def compile_model(model: tf.keras.Sequential) -> tf.keras.Sequential:
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


model: tf.keras.Sequential = compile_model(build_model())
# Train the model
model.fit(train_images, train_labels, epochs=10)

# Test the model
test_loss: float
test_acc: float
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"Accuracy from the model: %{test_acc * 100}")
# Accuracy is about 88%

# Convert model to output probability 
probability_model: tf.keras.Sequential = tf.keras.Sequential([model,
                                                             tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

