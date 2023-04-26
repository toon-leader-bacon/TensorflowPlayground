from typing import Tuple, Set


def color_dist_eculid(c1, c2) -> float:
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2

    # Return simple eculidian distance between the 3d points
    return ((r1 - r2) ** 2) + ((g1 - g2) ** 2) + ((b1 - b2) ** 2)


def color_dist_nonlinear(c1, c2) -> float:
    # https://en.wikipedia.org/wiki/Color_difference
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2

    deltaRSqr: float = (r1 - r2) ** 2
    deltaGSqr: float = (g1 - g2) ** 2
    deltaBSqr: float = (b1 - b2) ** 2

    rAve: float = 0.5 * (r1 + r2)
    if rAve < 128:
        return (2 * deltaRSqr) + (4 * deltaGSqr) + (3 * deltaBSqr)
    else:
        return (3 * deltaRSqr) + (4 * deltaGSqr) + (2 * deltaBSqr)


def color_dist_redmean(c1, c2) -> float:
    # https://en.wikipedia.org/wiki/Color_difference
    (r1, g1, b1) = c1
    (r2, g2, b2) = c2

    deltaRSqr: float = (r1 - r2) ** 2
    deltaGSqr: float = (g1 - g2) ** 2
    deltaBSqr: float = (b1 - b2) ** 2

    rAve: float = 0.5 * (r1 + r2)

    return ((2 + (rAve / 256)) * deltaRSqr) + \
           (4 * deltaGSqr) + \
           ((2 + ((255-rAve) / 256)) * deltaBSqr)


def find_nearest_color(pix: Tuple[float], colors: Set[Tuple[float]], dist_func):
    nearest_dist: float = 999999
    return_color: Tuple[float] = pix
    for color in colors:
        dist: float = dist_func(pix, color)
        if (dist < nearest_dist):
            # If the pixel is closer to this color than the previous nearest
            nearest_dist = dist
            return_color = color
    return return_color
