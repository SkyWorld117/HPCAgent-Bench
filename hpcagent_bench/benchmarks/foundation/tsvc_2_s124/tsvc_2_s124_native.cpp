/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// s124_d_single: induction variable under both sides of if (same value)
void s124_d_single(double *__restrict__ a, const double *__restrict__ b,
                    const double *__restrict__ c, const double *__restrict__ d,
                    const double *__restrict__ e, const int iterations,
                    const int len_1d) {
  {
    int j;
    
      j = -1;
      for (int i = 0; i < len_1d; i++) {
        if (b[i] > 0.0) {
          j++;
          a[j] = b[i] + d[i] * e[i];
        } else {
          j++;
          a[j] = c[i] + d[i] * e[i];
        }
      }
    
  }
}

} // extern "C"
