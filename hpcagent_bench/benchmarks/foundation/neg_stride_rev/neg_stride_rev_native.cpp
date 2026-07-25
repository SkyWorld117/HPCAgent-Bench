/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// neg_stride_rev_d (s112): for i = len_1d-1 .. 0: a[i] = b[i] + 1
void neg_stride_rev_d(double *__restrict__ a, const double *__restrict__ b, const int len_1d) {
  for (int i = len_1d - 1; i >= 0; --i) {
    a[i] = b[i] + 1.0;
  }
}

} // extern "C"
