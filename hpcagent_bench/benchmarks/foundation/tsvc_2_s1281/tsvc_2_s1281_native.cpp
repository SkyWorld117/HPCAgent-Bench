/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ------------------------------------------------------------
// s1281_d_single
// ------------------------------------------------------------
void s1281_d_single(double *__restrict__ a, double *__restrict__ b,
                     const double *__restrict__ c, const double *__restrict__ d,
                     const double *__restrict__ e, int iterations, int len_1d) {

  {
    
      for (int i = 0; i < len_1d; i++) {
        double x = b[i] * c[i] + a[i] * d[i] + e[i];
        a[i] = x - 1.0;
        b[i] = x;
      }
    
  }

}

} // extern "C"
