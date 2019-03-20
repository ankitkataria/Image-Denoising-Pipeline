import numpy as np
import sys


def enlarge(x, frontPadSize, endRowPadSize, endColPadSize):
    nlin, ncol = x.shape
    tmp = np.concatenate((np.arange((frontPadSize-1), -1, -1), np.arange(0, ncol), np.arange(ncol-1, ncol-endColPadSize-1, -1)))
    y = x[:, tmp]
    y = y[np.concatenate((np.arange(frontPadSize-1, -1, -1), np.arange(0, nlin), np.arange(nlin-1, nlin-endRowPadSize-1, -1))),]

    return y

def image_enlarge_tl(image, BMparam):
    aa0, bb0 = image.shape
    dim = BMparam['dim']

    frontPadSize = dim - 1
    endRowPadSize = frontPadSize
    endColPadSize = frontPadSize

    enlargedImage = enlarge(image, frontPadSize, endRowPadSize, endColPadSize)
    BMparam['aa0'] = aa0
    BMparam['bb0'] = bb0
    BMparam['aa'] = aa0 + frontPadSize + endRowPadSize
    BMparam['bb'] = bb0 + frontPadSize + endColPadSize
    BMparam['frontPadSize'] = frontPadSize

    return enlargedImage, BMparam
