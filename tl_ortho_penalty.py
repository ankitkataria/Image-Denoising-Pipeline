import numpy as np


def tl_ortho_penalty(D, Y, numiter, tau):
    n = Y.shape[0]

    for i in range(1, numiter + 1):
        X = np.matmul(D, Y)
        X1 = np.absolute(X)

        maxVal, maxInd = X1.max(0), X1.argmax(0)

        X[np.absolute(X) < tau] = 0
        row, cols = X.shape
        for j in range(cols):
            X[maxInd[j]][j] = maxVal[j]

        U, _, VH = np.linalg.svd(np.matmul(Y, X.conj().transpose()))
        V = VH.T.conj()
        D = np.matmul(V[:, 0:n], U.conj().transpose())

    return D
