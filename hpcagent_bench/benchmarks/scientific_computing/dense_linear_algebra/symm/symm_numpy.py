import numpy as np


def kernel(alpha, beta, C, A, B):
    A_sym = np.tril(A) + np.tril(A, -1).T
    C[:] = beta * C + alpha * (A_sym @ B)
