import numpy
import matplotlib.pyplot as plot
# from PIL import Image, ImageOps
import cv2


def load_image(img_path: str) -> None:
    bgr_img = cv2.imread(img_path)
    b, g, r = cv2.split(bgr_img)
    img = cv2.merge([r, g, b])
    return img


def display_img(img, title: str = "image") -> None:
    print("# Dimensions: " + str(img.ndim))
    print("Image shape: " + str(img.shape))
    print("Data Type: " + str(img.dtype))
    print("Pixel RGB value at [0, 0]: " + str(img[0, 0]))

    figure = plot.figure(figsize=(5.0, 5.0))
    figure.add_subplot(1, 1, 1)
    plot.imshow(img)
    plot.colorbar()
    plot.title(title)
    plot.show()


def merge_all_images(dir_with_layers: str) -> numpy.ndarray:
    # TODO: This currently adds pixel values together.
    # I want it to replace pixel values (except for values with alpha)
    bg: numpy.ndarray = cv2.imread(dir_with_layers + "Background.png")
    grass: numpy.ndarray = cv2.imread(dir_with_layers + "Grass.png")
    surface: numpy.ndarray = cv2.imread(dir_with_layers + "Surface.png")
    trainer: numpy.ndarray = cv2.imread(dir_with_layers + "Trainer.png")

    # Merge all the layers together into a single result image
    result: numpy.ndarray = cv2.addWeighted(bg, 0, grass, 1, 0.0)
    result: numpy.ndarray = cv2.addWeighted(result, 0, surface, 1, 0.0)
    result: numpy.ndarray = cv2.addWeighted(result, 1, trainer, 1, 0.0)

    # Convert to rbg encoding at the end
    b, g, r = cv2.split(result)
    result = cv2.merge([r, g, b])
    return result


if __name__ == '__main__':
    # print_img("./images/PokemonMaps/Gen1/Route1/Background.png")
    display_img(merge_all_images("./images/PokemonMaps/Gen1/Route1/"))
