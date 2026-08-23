import numpy as np


def average_pooling_1d(x, kernel_size, stride, padding, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, out):
    k = int(avg_pool_kernel_size)
    s = k if avg_pool_stride is None else int(avg_pool_stride)
    p = int(avg_pool_padding)
    padded = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * p), dtype=x.dtype)
    padded[:, :, p:p + x.shape[2]] = x
    span = out.shape[2] * s
    acc = np.zeros(out.shape, dtype=x.dtype)
    # kh taps, not H*W points: the loop is 3 iterations and each body is one whole-array slice add.
    for kx in range(k):
        acc += padded[:, :, kx:kx + span:s]
    out[:] = acc / k
