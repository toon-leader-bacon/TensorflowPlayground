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
