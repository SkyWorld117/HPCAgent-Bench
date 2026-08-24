import numpy as np


def kernel(alpha, A, B):
    strictly_upper = np.tril(A, -1).T
    B[:] = alpha * (B + strictly_upper @ B)
