import numpy as np


def aggregate(patches, weights, param):
    aa = param['aa']
    bb = param['bb']
    aa0 = param['aa0']
    bb0 = param['bb0']

    frontPadSize = param['frontPadSize']
    dim = param['dim']
    Mimage = aa - dim + 1
    Nimage = bb - dim  + 1

    r = np.arange(0, Mimage)
    c = np.arange(0, Nimage)

    im_out = np.zeros((aa, bb))
    im_wei = np.zeros((aa, bb))

    k = -1

    for i in range(dim):
        for j in range(dim):
            k = k + 1
            t = np.multiply(
                    patches[k,].conj().transpose().reshape((Mimage, Nimage), order='F'),
                    weights.conj().transpose().reshape((Mimage, Nimage), order='F')
                )

            x1 = 0
            for x in i + r:
                y1 = 0
                for y in c + j:
                    im_out[x][y] = np.add(im_out[x][y], t[x1][y1])
                    y1 += 1
                x1 += 1

            t1 =  weights.conj().transpose().reshape((Mimage, Nimage), order='F')
            x1 = 0
            for x in i + r:
                y1 = 0
                for y in c + j:
                    im_wei[x][y] = np.add(im_wei[x][y], t1[x1][y1])
                    y1 += 1
                x1 += 1

    im_out = np.divide(im_out, (im_wei + 10**-5))
    Xr = im_out[frontPadSize:frontPadSize + aa0, frontPadSize: frontPadSize + bb0]

    return Xr
