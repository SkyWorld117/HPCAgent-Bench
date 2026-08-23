import numpy as np


def conv2d(input, weights, output):
    K = weights.shape[0]  # Assuming square kernel
    N, H, W, C_in = input.shape
    H_out = H - K + 1
    W_out = W - K + 1
    C_out = weights.shape[3]
    acc = np.zeros((N, H_out, W_out, C_out), dtype=input.dtype)
    # Loop over the K*K kernel taps, not the H_out*W_out output pixels: one matmul per tap
    # contracts C_in, so each tap is a single wide, compiled pass over the whole array.
    for ky in range(K):
        for kx in range(K):
            patch = input[:, ky:ky + H_out, kx:kx + W_out, :]
            tap = weights[ky, kx, :, :]
            acc += (patch.reshape(N * H_out * W_out, C_in) @ tap).reshape(N, H_out, W_out, C_out)
    output[:] = acc


def conv2d_bias(input, weights, bias, out):
    conv2d(input, weights, out)
    out += bias
