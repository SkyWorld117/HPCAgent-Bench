import numpy as np


def go_fast(a, out):
    trace = float(np.tanh(np.diagonal(a), dtype=np.float64).sum())
    out[:] = a + trace
