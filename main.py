#!/usr/bin/python3
import sys

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import cv2


from image_enlarge_tl import image_enlarge_tl
from image_patch import image_patch
from tl_approximation import tl_approximation
from bm_fix import bm_fix
from lr_approximation import lr_approximation
from f1_reconstruction import f1_reconstruction
from aggregate import aggregate
from PSNR import PSNR


def strollr2d_imagedenoising(data, param):
    """
    This is the entrypoint function for the imagedenoising. The input parameters
    are defined as

    data: A python object containing the image data, the structure contains two fields
    - noisy: a*b size gray-scale image matrix for denoising.
    - oracle (optional): a*b size gray-scale matrix as the ground truth for calculation
    of PSNR.

    param: Structure containing parameters for the algorithm.
    """
    try:
        noisy = data['noisy']
        oracle = data.get('oracle', None)

        sig = param['sig']
        dim = param['dim']
        # Kronecker product
        # dct(np.eye(8), axis=0) is the cosine transform of order 8
        W = np.kron(dct(np.eye(dim), axis=0, norm='ortho'), dct(np.eye(dim), axis=0, norm='ortho'))
        threshold = param['TLthr0'] * sig
        param['threshold'] = threshold

        thr = param['thr0'] * sig
        param['thr'] = thr

        print('[+] Parameters loaded')
        noisy, param = image_enlarge_tl(noisy, param)
        print('[+] Image Enlarged for TL ONLY')

        patchNoisy = image_patch(noisy, dim)
        print('[+] Image patch done')
        patches = patchNoisy

        # patchNoisy is a 2D numpy array
        numTensorPatch = patchNoisy.shape[1]
        param['numTensorPatch'] = numTensorPatch

        W, sparseCode, nonZeroTable = tl_approximation(patches, W, param)
        print('[+] Module TL approx done')

        nonZeroTable[nonZeroTable == 0] = param['zeroWeight']
        TLsparsityWeight = np.divide(1, nonZeroTable)
        blk_arr, _, blk_pSize = bm_fix(patches, param)
        print('[+] Module BM fix done')
        blk_arr = np.asarray(blk_arr)
        blk_pSize = np.asarray(blk_pSize)

        LRpatch, LRweights, LRrankWeight = lr_approximation(patches, blk_arr, blk_pSize, param)
        print('[+] Module LRapprox done')
        nonZerosLR = LRweights > 0
        LRrankWeight[nonZerosLR] = np.divide(LRrankWeight[nonZerosLR], LRweights[nonZerosLR])

        patchRecon = f1_reconstruction(sparseCode, W, LRpatch, LRweights, patches, param, TLsparsityWeight, LRrankWeight)
        print('[+] Module F1 Reconstruction done')
        Xr = aggregate(patchRecon, TLsparsityWeight, param)

        plt.imshow(Xr, cmap='gray', vmin=0, vmax=255)
        plt.show()

        psnrXr = PSNR(Xr - oracle)
        print('[+] PSNR value is : {}'.format(psnrXr))
        return Xr, psnrXr

    except KeyError as e:
        print('The parameter provided to strollr2d are not valid: {}'.format(e))
        sys.exit(1)


if __name__ == "__main__":
    print('Starting program')
    param = {}

    param['dim'] = 8
    param['n'] = param['dim'] * param['dim']
    param['stride'] = 1
    param['BMstride'] = 1
    param['TLthr0'] = 2.5
    param['isMeanRemoved'] = True
    param['zeroWeight'] = 0.2
    param['learningIter'] = 40

    param['lambda0'] = 0.031

    param['searchWindowSize'] = 35
    param['csim'] = 5
    param['tensorSize'] = param['csim'] * param['n']
    param['thr0'] = 1.5
    param['maxTensorPatch'] = 25

    param['numIter'] = 1
    param['gamma_f'] = 10 ** -6
    param['gamma_l'] = 0.01
    param['iterx'] = 5
    param['sig'] = 20

    print('[+] Parameters: {}'.format(param))

    data = {}
    data['noisy'] = cv2.imread("demo_data/test.png", 0)
    data['oracle'] = cv2.imread("demo_data/barbara.png", 0)

    plt.imshow(data['noisy'], cmap='gray', vmin=0, vmax=255)
    plt.show()

    Xr, psnrXr = strollr2d_imagedenoising(data, param)
    plt.imshow(Xr)
    plt.show()


# Calculated PSNR values
# PSNR - 1 = PSNR value is : 31.87406554528377
