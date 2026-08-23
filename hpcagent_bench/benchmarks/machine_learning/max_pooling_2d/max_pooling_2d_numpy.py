import numpy as np


def max_pooling_2d(x, kernel_size, stride, padding, dilation, maxpool_kernel_size, maxpool_stride, maxpool_padding,
                    out):
    k = int(maxpool_kernel_size)
    s = k if maxpool_stride is None else int(maxpool_stride)
    p = int(maxpool_padding)
    fill = np.finfo(x.dtype).min if np.issubdtype(x.dtype, np.floating) else np.iinfo(x.dtype).min
    padded = np.full((x.shape[0], x.shape[1], x.shape[2] + 2 * p, x.shape[3] + 2 * p), fill, dtype=x.dtype)
    padded[:, :, p:p + x.shape[2], p:p + x.shape[3]] = x
    span_h = out.shape[2] * s
    span_w = out.shape[3] * s
    acc = np.full(out.shape, fill, dtype=x.dtype)
    # kh*kw taps, each body a whole-array slice, not a materialized kh*kw window pair of axes.
    for ky in range(k):
        for kx in range(k):
            acc = np.maximum(acc, padded[:, :, ky:ky + span_h:s, kx:kx + span_w:s])
    out[:] = acc
