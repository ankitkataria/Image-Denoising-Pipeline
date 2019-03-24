import numpy as np

from tl_ortho_penalty import tl_ortho_penalty
from sparse_l0 import sparse_l0

import sys

def tl_approximation(extractPatchAll, W, TLparam):
    print('[+] Inside module TLapprox')
    learningIter = TLparam['learningIter']
    threshold = TLparam['threshold']

    W = tl_ortho_penalty(W, extractPatchAll, learningIter, threshold)

    sparseCode = np.matmul(W, extractPatchAll)

    sparseCode, nonZeroTable = sparse_l0(sparseCode, threshold)

    return W, sparseCode, nonZeroTable
