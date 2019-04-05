import numpy as np


def f1_reconstruction(sparseCode, W, LRpatch, LRweights, patchNoisy, param, TLsparsityWeight, LRrankWeight):
    sig = param['sig']
    gamma_f = param['gamma_f'] / sig
    gamma_l = param['gamma_l'] * sig
    LRvsRLcoef = np.divide(LRrankWeight, TLsparsityWeight.conj().transpose())

    W = np.asarray(W)

    TLrecon = np.matmul(W.conj().transpose(), sparseCode)
    patchRecon = TLrecon + gamma_f * patchNoisy + np.dot(LRpatch, gamma_l) * LRvsRLcoef
    patchRecon = np.divide(patchRecon, (1 + gamma_f + gamma_l * np.multiply(LRweights, LRvsRLcoef)))

    return patchRecon
