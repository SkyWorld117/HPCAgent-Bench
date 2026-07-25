/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// -------------------------------------------------------------------------
// Masked stores
// -------------------------------------------------------------------------

// masked_store_const_d: predicated store keyed on int mask
void masked_store_const_d(double *__restrict__ a, const double *__restrict__ b,
                                  const std::int64_t *__restrict__ mask, const int len_1d) {
  for (int i = 0; i < len_1d; ++i) {
    if (mask[i] > 0) {
      a[i] = b[i];
    }
  }
}

} // extern "C"
