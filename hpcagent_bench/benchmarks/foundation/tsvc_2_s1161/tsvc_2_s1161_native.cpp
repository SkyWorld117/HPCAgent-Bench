/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ------------------------------------------------------------
// s1161_d_single
// ------------------------------------------------------------
void s1161_d_single(double *__restrict__ a, double *__restrict__ b,
                     double *__restrict__ c, const double *__restrict__ d,
                     const double *__restrict__ e, const int iterations,
                     const int len_1d) {

  {
    
      for (int i = 0; i < len_1d; ++i) {
        if (c[i] < 0.0) {
          b[i] = a[i] + d[i] * d[i];
        } else {
          a[i] = c[i] + d[i] * e[i];
        }
      }
    
  }
}

} // extern "C"
