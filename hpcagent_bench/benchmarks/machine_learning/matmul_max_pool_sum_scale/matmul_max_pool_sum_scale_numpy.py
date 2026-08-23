import numpy as np


def matmul_max_pool_sum_scale(x, kernel_size, scale_factor, matmul_weight, matmul_bias, out):
    y = x @ matmul_weight.T + matmul_bias
    k = int(kernel_size)
    out_len = y.shape[1] // k
    span = out_len * k
    fill = np.finfo(y.dtype).min if np.issubdtype(y.dtype, np.floating) else np.iinfo(y.dtype).min
    acc = np.full((y.shape[0], out_len), fill, dtype=y.dtype)
    # padding is 0 and stride defaults to kernel_size (non-overlapping), so k taps of
    # whole-array slices over the out_features axis cover this pool.
    for kx in range(k):
        acc = np.maximum(acc, y[:, kx:kx + span:k])
    out[:] = acc.sum(axis=1) * scale_factor
