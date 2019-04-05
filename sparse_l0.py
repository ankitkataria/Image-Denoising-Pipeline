import numpy as np

def sparse_l0(X, threshold):
    maxInd = np.argmax(np.absolute(X), axis=0)
    n, N = X.shape
    nonZeroMap = np.greater_equal(np.absolute(X), threshold)
    base = np.arange(0, n*N, n)
    maxInd = np.add(maxInd, base)

    row, cols = nonZeroMap.shape

    tmp = nonZeroMap.flatten(1)
    for i in range(len(maxInd)):
        val = maxInd[i]
        nonZeroMap[val % row][val // row] = True

    X = X * nonZeroMap

    # Fix the complex conjugate thing here.
    nonZeroMap = nonZeroMap.sum(axis=0).conj().transpose()

    return X, nonZeroMap
