import numpy
import matplotlib.pyplot as plot
from PIL import Image


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
    bg_img: Image = Image.open(dir_with_layers + "Background.png")
    grass_img: Image = Image.open(dir_with_layers + "Grass.png")
    surface_img: Image = Image.open(dir_with_layers + "Surface.png")
    trainer_img: Image = Image.open(dir_with_layers + "Trainer.png")

    result = Image.alpha_composite(bg_img, grass_img)
    result = Image.alpha_composite(result, surface_img)
    result = Image.alpha_composite(result, trainer_img)

    return numpy.asarray(result)


if __name__ == '__main__':
    display_img(merge_all_images("./images/PokemonMaps/Gen1/Route1/"))
