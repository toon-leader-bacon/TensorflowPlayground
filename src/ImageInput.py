import keras
import pathlib
import os
import glob

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


if __name__ == '__main__':
    download_pics()
