/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ------------------------------------------------------------
// s2251_d_single
// ------------------------------------------------------------
void s2251_d_single(double *__restrict__ a, double *__restrict__ b,
                     const double *__restrict__ c, const double *__restrict__ d,
                     const double *__restrict__ e, int iterations, int len_1d) {

  {
    
      double s = 0.0;
      for (int i = 0; i < len_1d; i++) {
        a[i] = s * e[i];
        s = b[i] + c[i];
        b[i] = a[i] + d[i];
      }
    
  }

}

} // extern "C"
