from PIL import Image
from src.ImageProcessing import ColorMap as colors
from src.NocabUtility import utility as nocabUtil
from typing import Dict
import matplotlib.pyplot as plot
import numpy


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


# Convert an image to an array of integer values.
# Pixel color corresponds to number value
def blab(target_file_name: str) -> numpy.ndarray:
    grass_img: Image = Image.open("./images/PokemonMaps/Gen1/Route3/" + "Grass.png")
    result: numpy.ndarray = numpy.zeros((grass_img.width, grass_img.height), dtype=numpy.int8)

    # Validate the data. Does the image have the correct RGBA channels?
    band_names: tuple = grass_img.getbands()
    if band_names != ("R", "G", "B", "A"):
        print("Invalid Image! Provided image does NOT have the required bands (R, G, B, A).")
        return

    # Convert every pixel to it's associated integer
    for x in range(grass_img.width):
        for y in range(grass_img.height):
            result[x, y] = color_to_data(grass_img.getpixel((x, y)))

    return result


def layer_to_data(layer: Image, color_map: Dict[tuple, numpy.int8]) -> numpy.ndarray:
    result: numpy.ndarray = numpy.zeros((layer.width, layer.height), dtype=numpy.int8)
    # Validate the data. Does the image have the correct RGBA channels?
    band_names: tuple = layer.getbands()
    if band_names != ("R", "G", "B", "A"):
        print("Invalid Image Layer! Provided layer does NOT have the required bands (R, G, B, A).")
        return result
    # Convert every pixel to it's associated integer
    for x in range(layer.width):
        for y in range(layer.height):
            result[x, y] = color_to_data(layer.getpixel((x, y)), color_map)
    return result


def color_to_data(pixel: tuple, color_map: Dict[tuple, numpy.int8]) -> numpy.int8:
    if len(pixel) != 4:
        print("Invalid pixel! Provided pixel does NOT have the expected 4 bands RGBA: " + pixel)

    if pixel[3] == 0:
        # If alpha is totally transparent
        return 0
    if pixel in color_map:
        return color_map[pixel]
    # Else unknown color
    print("Unexpected pixel color! Assuming transparent\n" + pixel)
    return 0


def color_to_data_grass(pixel: tuple) -> numpy.int8:
    return color_to_data(pixel, colors.GRASS_COLOR_TO_DATA)


def color_to_data_bg(pixel: tuple) -> numpy.int8:
    return color_to_data(pixel, colors.BG_COLOR_TO_DATA)


def color_to_data_surface(pixel: tuple) -> numpy.int8:
    return color_to_data(pixel, colors.SURFACE_COLOR_TO_DATA)


def color_to_data_TRAINER_COLOR_TO_DATA(pixel: tuple) -> numpy.int8:
    return color_to_data(pixel, colors.TRAINER_COLOR_TO_DATA)


if __name__ == '__main__':
    grass_img: Image = Image.open("./images/PokemonMaps/Gen1/Route3/" + "Grass.png")
    arr: numpy.ndarray = layer_to_data(grass_img, colors.GRASS_COLOR_TO_DATA)

    import sys
    numpy.set_printoptions(threshold=sys.maxsize)
    print(arr)
    nocabUtil.showImg(grass_img)
