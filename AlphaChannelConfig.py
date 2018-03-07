###Function for setting Alpha Levels of an image mask
###Makes background transparent

import sys
from PIL import Image

def defAlpha(image, alphaLevel, backgroundLevel):
    image = image.convert('RGBA')
    pixeldata = list(image.getdata())
    for i,pixel in enumerate(pixeldata):
        if pixel[:3] == (255,255,255):
            pixeldata[i] = (255,255,255,backgroundLevel)

        else:
            pixeldata[i] = (pixel[0], pixel[1], pixel[2], alphaLevel)

    image.putdata(pixeldata)
    del pixeldata
    return(image)
