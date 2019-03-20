import numpy as np


def image_patch(im, psize):
    f = psize
    N = im.shape[0] - f + 1
    M = im.shape[1] - f + 1
    L = M*N
    X = np.zeros((f*f, L), np.float)
    k = -1

    row, col = im.shape
    for i in range(f):
        for j in range(f):
            k = k + 1
            blk = im[i:row - f + i + 1, j:col - f + j + 1]
            X[k,] = blk.flatten(1).conj().transpose()

    return X
