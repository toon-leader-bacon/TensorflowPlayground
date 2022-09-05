import glob
import keras
import os
import tensorflow as tf

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


def blab():
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
        image_size=(img_height, img_width),
        batch_size=batch_size)
    validation_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=percent_for_validation,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    print("Found image data with these classes: " + str(train_ds.class_names))

if __name__ == '__main__':
    blab()
