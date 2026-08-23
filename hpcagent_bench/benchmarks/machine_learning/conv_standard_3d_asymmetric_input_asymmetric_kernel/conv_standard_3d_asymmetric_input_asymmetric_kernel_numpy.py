import numpy as np


def _conv3d(x, weight, bias, stride, padding, dilation, groups):
    if isinstance(stride, (int, np.integer)): stride = (stride, stride, stride)
    if isinstance(padding, (int, np.integer)): padding = (padding, padding, padding)
    if isinstance(dilation, (int, np.integer)): dilation = (dilation, dilation, dilation)
    n, c_in, d, h, w = x.shape
    c_out, c_per_group, kd, kh, kw = weight.shape
    od = (d + 2 * padding[0] - dilation[0] * (kd - 1) - 1) // stride[0] + 1
    oh = (h + 2 * padding[1] - dilation[1] * (kh - 1) - 1) // stride[1] + 1
    ow = (w + 2 * padding[2] - dilation[2] * (kw - 1) - 1) // stride[2] + 1
    padded = np.zeros((n, c_in, d + 2 * padding[0], h + 2 * padding[1], w + 2 * padding[2]), dtype=x.dtype)
    padded[:, :, padding[0]:padding[0] + d, padding[1]:padding[1] + h, padding[2]:padding[2] + w] = x
    # NDHWC once, then one matmul per kernel tap contracts the channel axis; far cheaper than the 9-deep loop nest.
    ndhwc = np.transpose(padded, (0, 2, 3, 4, 1))
    out_per_group = c_out // groups
    in_per_group = c_in // groups
    acc = np.zeros((n * od * oh * ow, c_out), dtype=x.dtype)
    span_d = (od - 1) * stride[0] + 1
    span_h = (oh - 1) * stride[1] + 1
    span_w = (ow - 1) * stride[2] + 1
    for g in range(groups):
        in_g = ndhwc[..., g * in_per_group:(g + 1) * in_per_group]
        oc_lo, oc_hi = g * out_per_group, (g + 1) * out_per_group
        for kz in range(kd):
            zz = kz * dilation[0]
            for ky in range(kh):
                yy = ky * dilation[1]
                for kx in range(kw):
                    xx = kx * dilation[2]
                    patch = in_g[:, zz:zz + span_d:stride[0], yy:yy + span_h:stride[1], xx:xx + span_w:stride[2], :]
                    tap_w = np.transpose(weight[oc_lo:oc_hi, :, kz, ky, kx])
                    acc[:, oc_lo:oc_hi] += np.reshape(patch, (n * od * oh * ow, in_per_group)) @ tap_w
    out = np.transpose(np.reshape(acc, (n, od, oh, ow, c_out)), (0, 4, 1, 2, 3))
    return out + bias.reshape((1, c_out, 1, 1, 1))


def conv_standard_3d_asymmetric_input_asymmetric_kernel(x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, conv3d_weight, conv3d_bias, conv3d_stride, conv3d_padding, conv3d_dilation, conv3d_groups, out):
    out[:] = _conv3d(x, conv3d_weight, conv3d_bias, conv3d_stride, conv3d_padding, conv3d_dilation, conv3d_groups)
