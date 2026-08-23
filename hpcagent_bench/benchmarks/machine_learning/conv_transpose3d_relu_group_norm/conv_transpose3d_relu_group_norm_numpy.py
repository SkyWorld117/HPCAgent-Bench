import numpy as np


def _as_tuple(value, dims):
    if isinstance(value, tuple):
        return value
    return tuple(value for _ in range(dims))


def _conv_transpose3d(x, weight, bias, stride, padding, output_padding, dilation, groups):
    stride = _as_tuple(stride, 3)
    padding = _as_tuple(padding, 3)
    output_padding = _as_tuple(output_padding, 3)
    dilation = _as_tuple(dilation, 3)
    n, c_in, d, h, w = x.shape
    _, c_out_per_group, kd, kh, kw = weight.shape
    c_out = c_out_per_group * groups
    od = (d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kd - 1) + output_padding[0] + 1
    oh = (h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kh - 1) + output_padding[1] + 1
    ow = (w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kw - 1) + output_padding[2] + 1
    in_per_group = c_in // groups

    # Uncropped scatter target: o_full = i*stride + k*dilation, o = o_full - padding.
    # Accumulate at full (uncropped) size, crop the padding border once at the end.
    full_d = (d - 1) * stride[0] + dilation[0] * (kd - 1) + 1
    full_h = (h - 1) * stride[1] + dilation[1] * (kh - 1) + 1
    full_w = (w - 1) * stride[2] + dilation[2] * (kw - 1) + 1
    ndhwc = np.transpose(x, (0, 2, 3, 4, 1))
    acc = np.zeros((n, full_d, full_h, full_w, c_out), x.dtype)
    for kz in range(kd):
        oz0 = kz * dilation[0]
        oz_stop = oz0 + (d - 1) * stride[0] + 1
        for ky in range(kh):
            oy0 = ky * dilation[1]
            oy_stop = oy0 + (h - 1) * stride[1] + 1
            for kx in range(kw):
                ox0 = kx * dilation[2]
                ox_stop = ox0 + (w - 1) * stride[2] + 1
                for g in range(groups):
                    ic = slice(g * in_per_group, (g + 1) * in_per_group)
                    oc = slice(g * c_out_per_group, (g + 1) * c_out_per_group)
                    acc[:, oz0:oz_stop:stride[0], oy0:oy_stop:stride[1], ox0:ox_stop:stride[2],
                        oc] += ndhwc[..., ic] @ weight[ic, :, kz, ky, kx]

    out_ndhwc = np.zeros((n, od, oh, ow, c_out), x.dtype)
    core_d = min(full_d - padding[0], od)
    core_h = min(full_h - padding[1], oh)
    core_w = min(full_w - padding[2], ow)
    out_ndhwc[:, :core_d, :core_h, :core_w, :] = acc[:, padding[0]:padding[0] + core_d, padding[1]:padding[1] + core_h,
                                                       padding[2]:padding[2] + core_w, :]
    out = np.transpose(out_ndhwc, (0, 4, 1, 2, 3))
    out += bias.reshape(1, -1, 1, 1, 1)
    return out


def _group_norm(x, num_groups, weight, bias, eps):
    n, c = x.shape[0], x.shape[1]
    y = x.reshape((n, num_groups, c // num_groups) + x.shape[2:])
    mean = np.mean(y, axis=tuple(range(2, y.ndim)), keepdims=True)
    var = np.var(y, axis=tuple(range(2, y.ndim)), keepdims=True)
    y = ((y - mean) / np.sqrt(var + eps)).reshape(x.shape)
    shape = (1, c) + (1,) * (x.ndim - 2)
    return y * weight.reshape(shape) + bias.reshape(shape)


def conv_transpose3d_relu_group_norm(x, conv_transpose_weight, conv_transpose_bias, group_norm_num_groups, group_norm_weight, group_norm_bias, group_norm_eps, out):
    x = _conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, 1, 0, 0, 1, 1)
    x = np.maximum(x, 0)
    x = _group_norm(x, group_norm_num_groups, group_norm_weight, group_norm_bias, group_norm_eps)
    out[:] = x
