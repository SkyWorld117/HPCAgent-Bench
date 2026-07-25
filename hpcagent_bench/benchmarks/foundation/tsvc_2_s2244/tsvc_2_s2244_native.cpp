/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ============================================================================
// s2244_d_single
// ============================================================================
void s2244_d_single(double *__restrict__ a, const double *__restrict__ b,
                     const double *__restrict__ c, const double *__restrict__ e,
                     const int iterations, const int len_1d) {

  {
    
      for (int i = 0; i < len_1d - 1; ++i) {
        a[i + 1] = b[i] + e[i];
        a[i] = b[i] + c[i];
      }
    
  }

}

} // extern "C"
