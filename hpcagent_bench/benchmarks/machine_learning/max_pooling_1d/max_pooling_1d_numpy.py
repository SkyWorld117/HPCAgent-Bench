import numpy as np


def max_pooling_1d(x, kernel_size, stride, padding, dilation, return_indices, maxpool_kernel_size, maxpool_stride,
                    maxpool_padding, out):
    k = int(maxpool_kernel_size)
    s = k if maxpool_stride is None else int(maxpool_stride)
    p = int(maxpool_padding)
    fill = np.finfo(x.dtype).min if np.issubdtype(x.dtype, np.floating) else np.iinfo(x.dtype).min
    padded = np.full((x.shape[0], x.shape[1], x.shape[2] + 2 * p), fill, dtype=x.dtype)
    padded[:, :, p:p + x.shape[2]] = x
    span = out.shape[2] * s
    acc = np.full(out.shape, fill, dtype=x.dtype)
    # k taps, each body a whole-array slice, not a materialized k-wide window axis.
    for kx in range(k):
        acc = np.maximum(acc, padded[:, :, kx:kx + span:s])
    out[:] = acc
