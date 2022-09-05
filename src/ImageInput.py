from encodings import normalize_encoding
import glob
import keras
import os
import tensorflow as tf
import matplotlib.pyplot as plot
import numpy

CACHE_DIR: str = "./images/"
DATA_DIR: str = f"{CACHE_DIR}datasets/flower_photos/"


def setup_directory():
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_pics():
    setup_directory()
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    keras.utils.get_file(origin=dataset_url,
                         fname='flower_photos',
                         untar=True,
                         cache_dir=CACHE_DIR)
    image_count = len(glob.glob(f'{DATA_DIR}**/*.jpg'))
    print(f"Downloaded {image_count} images")
    print("By default the download location is ~/.keras")


def LoadData() -> tuple():
    batch_size: int = 32
    img_height: int = 180
    img_width: int = 180
    percent_for_validation: float = 0.2

    # Requires the directory to be in the following format
    # main_directory/
    # ...class_a/
    # ......a_image_1.jpg
    # ......a_image_2.jpg
    # ...class_b/
    # ......b_image_1.jpg
    # ......b_image_2.jpg
    train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=percent_for_validation,
        subset="training",
        seed=123,
        # Use image_size or the keras.layers.Resizing(...) layer
        image_size=(img_height, img_width),
        batch_size=batch_size)
    validation_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=percent_for_validation,
        subset="validation",
        seed=123,
        # Use image_size or the keras.layers.Resizing(...) layer
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print("Found image data with these classes: " + str(train_ds.class_names))
    return (train_ds, validation_ds)


def VisualizeData(train_ds, validation_ds):
    plot.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plot.subplot(3, 3, i + 1)
            plot.imshow(images[i].numpy().astype("uint8"))
            plot.title(train_ds.class_names[labels[i]])
            plot.axis("off")
    plot.show()


def dataProcessing(train_ds, validation_ds):

    # Input images have shape: (32, 180, 180, 3)
    #   In english: 32 images, each 180x180 pixels, 3 color channels
    # Each pixel has the range [0, 255]
    # First, we should normalize this to the range [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1.0/255.0)
    # for range [-1, 1] use Rescaling(1./127.5, offset=-1)

    # Two options:
    #  - Apply layer to data set right now to pre-process data
    #  - Include the layer inside the model.

    # # Option 1 shown here:
    # normalized_data = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_data))
    # # Pixel values are now in range [0, 1]
    # first_image = image_batch[0]
    # print(numpy.min(first_image), numpy.max(first_image))

    # Option 2 shown here:
    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Define model
    num_classes = 5
    model = keras.Sequential([
        normalization_layer,  # Input Layer

        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)  # Output Layer
    ])

    # TODO: Study the compile function. What are details (pros, cons) of different optimizers/loss functions?
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    # Train
    # TODO: Write a training loop: https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
    model.fit(train_ds, validation_data=validation_ds, epochs=3)


if __name__ == '__main__':
    (train_ds, validation_ds) = LoadData()
    dataProcessing(train_ds, validation_ds)
    # VisualizeData(train_ds, validation_ds)
