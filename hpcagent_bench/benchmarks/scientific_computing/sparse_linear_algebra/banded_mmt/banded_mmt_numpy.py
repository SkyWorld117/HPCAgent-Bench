import numpy as np
import scipy.sparse as sp


def unpack_banded(packed, lbound, ubound, N):
    # Packed layout (see generate_banded): row i holds its band, columns
    # start(i):stop(i) of the dense row, in the FIRST stop(i)-start(i) packed slots --
    # not a centered/shifted slice, so a plain diagonal offset would miss the boundary
    # rows. One tap per packed column (band width many, << N) instead of a Python loop
    # over every cell.
    row = np.arange(N)
    start = np.maximum(row - lbound, 0)
    stop = np.minimum(N, row + ubound + 1)
    dense = np.zeros((N, N), dtype=packed.dtype)
    width = lbound + ubound + 1
    for j in range(width):
        col = start + j
        valid = col < stop
        dense[row[valid], col[valid]] = packed[valid, j]
    return dense


def banded_mmt(A, a_lbound: int, a_ubound: int, B, b_lbound: int, b_ubound: int, ret_out):
    if sp.issparse(A) and sp.issparse(B):
        ret_out[:] = (A @ B @ A.T).toarray()
        return
    N = ret_out.shape[0]
    A_dense = unpack_banded(A, a_lbound, a_ubound, N)
    B_dense = unpack_banded(B, b_lbound, b_ubound, N)
    ret_out[:] = A_dense @ B_dense @ A_dense.T
