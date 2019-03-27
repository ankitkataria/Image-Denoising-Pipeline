import numpy as np
import scipy.io


def bm_fix(extractPatch, BMparam):
    aa =   BMparam['aa']
    bb =   BMparam['bb']
    dim =   BMparam['dim']
    searchWindowSize =   BMparam['searchWindowSize']
    stride =   BMparam['BMstride']
    tensorSize =   BMparam['tensorSize']

    # row / col index size in noisy image
    Nimage = aa-dim+1
    Mimage = bb-dim+1

    # reference patch row / col indexing
    # normal one indexed
    r = np.arange(0, aa-searchWindowSize+1, stride)
    c = np.arange(0, bb-searchWindowSize+1, stride)

    # patchs in search window
    swidth = searchWindowSize - dim + 1
    swidth2 = swidth ** 2

    # noisy all possible patch indexing
    L =   Nimage*Mimage
    I =   np.arange(0, L)
    I =   I.reshape(Nimage, Mimage)

    # reference patch indexing
    N1 = len(r)
    M1 = len(c)

    # BM result indexing table
    pos_arr         =   np.zeros((tensorSize, N1*M1))
    error_arr       =   np.zeros((tensorSize, N1*M1))
    numPatch_arr    =   np.ones((1, N1*M1)) * tensorSize

    # middle index (refernce patch index), wihtin the search window
    mid             =   np.mod(swidth, 2) * ((swidth2+1)/2) + np.mod(swidth+1, 2) * (swidth2+swidth)/2

    # BMError
    for i in range(0, N1):
        for j in range(0, M1):
            row =  r[i]
            col =  c[j]

            idx = I[row: row+swidth, col:col+swidth]

            idx = np.ravel(idx)

            B = extractPatch.take(idx, axis=1)
            mid = int(mid)

            v = extractPatch[:, idx[mid-1]]
            v1 = np.tile(v,(swidth2, 1)).transpose()

            dis =   np.subtract(B, v1)
            metric =  (dis ** 2).mean(axis=0)
            BMerror = np.msort(metric)
            ind = np.argsort(metric)

            # ye high level be dekhna padega
            pos_arr[:, (j)*N1 + i]    =  idx[ind[0:tensorSize]]
            error_arr[:, (j)*N1 + i]  =  BMerror[ind[0:tensorSize]]

    return pos_arr, error_arr, numPatch_arr
