import keras
import pathlib
import os

CACHE_DIR: str = "./images/"


def setup_directory():
    os.makedirs(CACHE_DIR, exist_ok=True)


def download_pics():
    setup_directory()
    dataset_url: str = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = keras.utils.get_file(origin=dataset_url,
                                    fname='flower_photos',
                                    untar=True,
                                    cache_dir=CACHE_DIR)
    data_dir = pathlib.Path(CACHE_DIR)
    image_count = len(list(data_dir.glob('*/*.jpeg')))
    print(f"Downloaded {image_count} images")
    print("By default the download location is ~/.keras")


if __name__ == '__main__':
    download_pics()
