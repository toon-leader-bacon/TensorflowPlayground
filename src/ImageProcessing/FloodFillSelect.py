from typing import Tuple, Set
from PIL import Image

from src.ImageProcessing import ColorDist


# Get the 4 points (N/S/E/W) of the provided pix_xy
# If the neighbor point is outside of the range [0, max_width] or [0, max_height]
# it will be excluded from the returned set
def make_cardinal_directions(pix_xy: Tuple[int, int], max_width: int, max_height: int) -> Set[Tuple[int, int]]:
    (x0, y0) = pix_xy
    
    xLeft: int = x0 - 1
    xRight: int = x0 + 1
    yUp: int = y0 - 1
    yDown: int = y0 + 1
    
    neighbors: Set[Tuple[int, int]] = set()
    if (0 <= xLeft and xLeft < max_width):
        neighbors.add((xLeft, y0))
    if (0 <= xRight and xRight < max_width):
        neighbors.add((xRight, y0))
    if (0 <= yUp and yUp < max_height):
        neighbors.add((x0, yUp))
    if (0 <= yDown and yDown < max_height):
        neighbors.add((x0, yDown))
    return neighbors

def flood_fill_select(pix_xy: Tuple[int, int], image: Image.Image, distance_epsilon: float = 10):
    width: int = image.width
    height: int = image.height

    initial_pixel_rbg = image.getpixel(pix_xy)

    to_explore: Set[Tuple[int, int]] = {pix_xy}
    explored: Set[Tuple[int, int]] = set()
    selected_pixels: Set[Tuple[int, int]] = {pix_xy}

    while len(to_explore) > 0:
        current_pix_xy: Tuple[int, int] = to_explore.pop()
        if (current_pix_xy in explored):
            continue

        # Else this is an unexplored pixel
        explored.add(current_pix_xy)
        current_pix_rgb = image.getpixel(current_pix_xy)
        if (not ColorDist.color_dist_nonlinear(initial_pixel_rbg, current_pix_rgb) <= distance_epsilon):
            # if this pixel is NOT similar to the initial pixel
            # Then don't add it's neighbors
            continue
        # Else this is a similar pixel and should be selected
        selected_pixels.add(current_pix_xy)
        # continue the search on its neighbors
        to_explore = to_explore.union(make_cardinal_directions(current_pix_xy, width, height))
    return selected_pixels


def main() -> None:
    bg_image: Image.Image = Image.open("./images/PokemonMaps/Gen1/Route3/" + "Background.png")
    bottom_left_xy: Tuple[int, int] = (0, bg_image.height - 1)
    mountain_edge_pixs_xy: set[Tuple[int, int]] = flood_fill_select(bottom_left_xy, bg_image)
    print(mountain_edge_pixs_xy)


if __name__ == "__main__":
    main()