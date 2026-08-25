"""Box convolution: 49 accumulation passes over the output collapse into one contraction.

The reference walks the (2R+1)^2 taps and does ``out_grid += w * slice`` for each,
so the output is read and written 49 times and 49 full-size temporaries are born.
A sliding-window view turns the padded input into an (N, N, K, K) strided view at
zero copy cost, and ``einsum`` contracts the tap axes against ``w_box`` in a single
pass that writes the output exactly once.

``optimize=False`` is deliberate: the contraction is already a single term, and the
optimizing path builds an intermediate that is slower here (measured).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_2d(in_grid, out_grid, w_box, N, R):
    padded = np.pad(in_grid, pad_width=R, mode="edge")
    windows = sliding_window_view(padded, (2 * R + 1, 2 * R + 1))
    np.einsum("ijkl,kl->ij", windows, w_box, out=out_grid, optimize=False)
