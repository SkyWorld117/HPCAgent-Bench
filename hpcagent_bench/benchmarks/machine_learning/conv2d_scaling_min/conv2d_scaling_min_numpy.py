import numpy as np


def _conv2d(x, weight, bias, stride, padding, dilation, groups):
    if isinstance(stride, (int, np.integer)): stride = (stride, stride)
    if isinstance(padding, (int, np.integer)): padding = (padding, padding)
    if isinstance(dilation, (int, np.integer)): dilation = (dilation, dilation)
    n, c_in, h, w = x.shape
    c_out, c_per_group, kh, kw = weight.shape
    oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    padded = np.zeros((n, c_in, h + 2 * padding[0], w + 2 * padding[1]), dtype=x.dtype)
    padded[:, :, padding[0]:padding[0] + h, padding[1]:padding[1] + w] = x
    # NHWC once, then one 2-D matmul per kernel tap contracts the in-channel axis per group.
    nhwc = np.transpose(padded, (0, 2, 3, 1))
    span_h, span_w = (oh - 1) * stride[0] + 1, (ow - 1) * stride[1] + 1
    out_per_group, in_per_group = c_out // groups, c_in // groups
    acc = np.zeros((n, oh, ow, c_out), dtype=x.dtype)
    for g in range(groups):
        ic0, ic1 = g * in_per_group, (g + 1) * in_per_group
        oc0, oc1 = g * out_per_group, (g + 1) * out_per_group
        group_in = nhwc[..., ic0:ic1]
        group_acc = acc[..., oc0:oc1]
        for ky in range(kh):
            for kx in range(kw):
                y0, x0 = ky * dilation[0], kx * dilation[1]
                patch = group_in[:, y0:y0 + span_h:stride[0], x0:x0 + span_w:stride[1], :]
                tap = np.transpose(weight[oc0:oc1, :, ky, kx])
                group_acc += (patch.reshape(n * oh * ow, in_per_group) @ tap).reshape(n, oh, ow, out_per_group)
    out = np.transpose(acc, (0, 3, 1, 2)) + bias.reshape(1, c_out, 1, 1)
    return out


def conv2d_scaling_min(x, in_channels, out_channels, kernel_size, scale_factor, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, out):
    x = _conv2d(x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups)
    x = x * scale_factor
    x = np.min(x, axis=1, keepdims=True)
    out[:] = x
