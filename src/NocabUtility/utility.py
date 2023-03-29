import matplotlib.pyplot as plot
from PIL import Image


def showImg(img: Image):
    plot.imshow(img)
    plot.colorbar()
    plot.show()
