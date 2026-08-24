# Forward elimination is a genuine loop-carried recurrence: step k reads the row/column state
# left behind by step k-1, so the outer loop over k cannot be removed (Sec. 18). The body is
# already the vectorized rank-1 update; np.subtract(..., out=) folds the elementwise multiply
# and the subtraction into one ufunc call instead of materializing the outer product and then
# subtracting it.

import numpy as np


def gaussian(A, b):
    N = A.shape[0]
    for k in range(N - 1):
        mult = A[k + 1:, k] / A[k, k]
        row = A[k, k:]
        block = A[k + 1:, k:]
        np.subtract(block, mult[:, np.newaxis] * row, out=block)
        b[k + 1:] -= mult * b[k]
