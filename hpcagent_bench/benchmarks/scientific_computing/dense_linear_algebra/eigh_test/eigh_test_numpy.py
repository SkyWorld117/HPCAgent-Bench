from scipy.linalg import eigh as _sci_eigh
import numpy as np


def eigh_test(a, b, wout, vout, lower=False):
    w, v = _sci_eigh(a, b, lower=lower)
    wout[:] = w
    # An eigenvector is fixed only up to a unit phase, so two LAPACK builds disagree by e^(i theta)
    # per column: pin the gauge on the largest-magnitude entry or the native legs mismatch by O(1).
    lead = v[np.argmax(np.abs(v), axis=0), np.arange(v.shape[1])]
    vout[:] = v * (np.abs(lead) / lead)
