import numpy as np


def _conv3d(x, weight, bias):
    # stride=1, padding=0, dilation=1, groups=1 -- the only case this kernel ever calls.
    n, c_in, d, h, w = x.shape
    c_out, _, kd, kh, kw = weight.shape
    od, oh, ow = d - kd + 1, h - kh + 1, w - kw + 1
    out = np.zeros((n, c_out, od, oh, ow), dtype=x.dtype)
    for kz in range(kd):
        for ky in range(kh):
            for kx in range(kw):
                patch = x[:, :, kz:kz + od, ky:ky + oh, kx:kx + ow]
                out += np.einsum('ncdhw,oc->nodhw', patch, weight[:, :, kz, ky, kx])
    return out + bias[None, :, None, None, None]


def _maxpool3d(x, k):
    n, c, d, h, w = x.shape
    od, oh, ow = (d - k) // k + 1, (h - k) // k + 1, (w - k) // k + 1
    span_d, span_h, span_w = od * k, oh * k, ow * k
    out = np.full((n, c, od, oh, ow), -np.inf, dtype=x.dtype)
    for kz in range(k):
        for ky in range(k):
            for kx in range(k):
                out = np.maximum(out, x[:, :, kz:kz + span_d:k, ky:ky + span_h:k, kx:kx + span_w:k])
    return out


def conv3d_divide_max_global_avg_pool_bias_add_sum(x, divisor, pool_size, conv_weight, conv_bias, bias, out, *, sum_dim=1):
    x = _conv3d(x, conv_weight, conv_bias)
    x = x / divisor
    x = _maxpool3d(x, pool_size)
    x = x.mean(axis=(2, 3, 4), keepdims=True)
    x = x + bias
    x = np.sum(x, axis=sum_dim, keepdims=False)
    out[:] = x
