import numpy as np


def _conv2d_depthwise(x, weight, bias, stride, padding, dilation):
    # groups == in_channels == out_channels (depthwise): each output channel only
    # sees its own input channel, so the channel contraction collapses to a
    # per-channel broadcast multiply -- no matmul/tensordot needed, just taps.
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    padded = np.zeros((n, c_in, h + 2 * padding, w + 2 * padding), dtype=x.dtype)
    padded[:, :, padding:padding + h, padding:padding + w] = x
    span_h, span_w = stride * (oh - 1) + 1, stride * (ow - 1) + 1
    out = np.zeros((n, c_out, oh, ow), dtype=x.dtype)
    w_tap = weight[:, 0]
    for ky in range(kh):
        iy0 = ky * dilation
        for kx in range(kw):
            ix0 = kx * dilation
            out += w_tap[None, :, ky, kx, None, None] * padded[:, :, iy0:iy0 + span_h:stride, ix0:ix0 + span_w:stride]
    out += bias[None, :, None, None]
    return out


def conv_depthwise_2d_square_input_square_kernel(x, conv2d_weight, conv2d_bias, conv2d_stride, conv2d_padding,
                                                   conv2d_dilation, conv2d_groups, out):
    out[:] = _conv2d_depthwise(x, conv2d_weight, conv2d_bias, conv2d_stride, conv2d_padding, conv2d_dilation)
