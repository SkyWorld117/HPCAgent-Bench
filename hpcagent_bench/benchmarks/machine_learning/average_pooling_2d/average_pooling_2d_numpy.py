import numpy as np


def average_pooling_2d(x, kernel_size, stride, padding, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding,
                        out):
    k = int(avg_pool_kernel_size)
    s = k if avg_pool_stride is None else int(avg_pool_stride)
    p = int(avg_pool_padding)
    padded = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * p, x.shape[3] + 2 * p), dtype=x.dtype)
    padded[:, :, p:p + x.shape[2], p:p + x.shape[3]] = x
    span_h = out.shape[2] * s
    span_w = out.shape[3] * s
    acc = np.zeros(out.shape, dtype=x.dtype)
    # kh*kw taps, each body a whole-array slice, not a materialized kh*kw window pair of axes.
    for ky in range(k):
        for kx in range(k):
            acc += padded[:, :, ky:ky + span_h:s, kx:kx + span_w:s]
    out[:] = acc / (k * k)
