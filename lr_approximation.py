import numpy as np
import math
import scipy.io


def lr_approximation(patchNoisy, blk_arr, blk_pSize, param):
    numTensorPatch      =   param['numTensorPatch']
    n                   =   param['n']
    thr                 =   param['thr']

    denoisedPatch       =   np.zeros((n, numTensorPatch))
    Weights             =   np.zeros((1, numTensorPatch))
    rankWeightTable    =   np.zeros((1, numTensorPatch))
    numRef              =   blk_arr.shape[1]
    blk_arr = np.asarray(blk_arr, np.int)
    blk_pSize = np.asarray(blk_pSize, np.int)

    print(numRef)
    for k in range(numRef):
        print(k)
        curTensorSize = blk_pSize[0][k]
        curTensorInd = blk_arr[0:curTensorSize, k]
        curTensorInd = curTensorInd - 1
        Scenter = patchNoisy[:, curTensorInd]
        mB = Scenter.mean(axis = 1)

        v = np.tile(mB,(Scenter.shape[1], 1)).transpose()
        Scenter = np.double(np.subtract(Scenter, v))
        eigenVal, bas = np.linalg.eig(np.matmul(Scenter, Scenter.conj().transpose()))

        eigenVal = eigenVal[::-1]

        diat            =   eigenVal / curTensorSize

        thr2            =   thr ** 2
        diatthr         =   (diat>thr2)

        diatthr[-1]    =   True
        rankWeightTable[:, curTensorInd] = rankWeightTable[:, curTensorInd] + (1 / sum(diatthr))

        ys              =   np.matmul(np.matmul(np.matmul(bas, np.diag(diatthr)) ,bas.conj().transpose()), Scenter)
        ys              =  np.add(ys, np.tile(mB,(ys.shape[1], 1)).transpose())

        denoisedPatch[:, curTensorInd] =   denoisedPatch[:, curTensorInd] +   ys
        Weights[:, curTensorInd] =   Weights[:, curTensorInd] + 1

    return denoisedPatch, Weights, rankWeightTable
