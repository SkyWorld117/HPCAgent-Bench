import numpy as np


def _maxpool1d(x, kernel_size, stride, padding):
    n, c, length = x.shape
    out_len = (length + 2 * padding - kernel_size) // stride + 1
    padded = np.full((n, c, length + 2 * padding), -np.inf, dtype=x.dtype)
    padded[:, :, padding:padding + length] = x
    span = stride * (out_len - 1) + 1
    out = np.full((n, c, out_len), -np.inf, dtype=x.dtype)
    for k in range(kernel_size):
        out = np.maximum(out, padded[:, :, k:k + span:stride])
    return out


def max_pooling_1d(x, maxpool_kernel_size, maxpool_stride, maxpool_padding, out):
    out[:] = _maxpool1d(x, maxpool_kernel_size, maxpool_stride, maxpool_padding)
