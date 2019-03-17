import numpy as np
import math

def PSNR(X):
    aa = len(X)
    bb = len(X[0])
    psnr = 20*math.log10((math.sqrt(aa*bb))*255/(np.linalg.norm(X,'fro')));
    return psnr

