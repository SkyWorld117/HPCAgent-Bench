/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// s1115_d_single: triangular saxpy loop variant
void s1115_d_single(double *__restrict__ aa, const double *__restrict__ bb,
                     const double *__restrict__ cc, const int iterations,
                     const int len_2d) {
  {
    
      for (int i = 0; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
          aa[i * len_2d + j] =
              aa[i * len_2d + j] * cc[j * len_2d + i] + bb[i * len_2d + j];
        }
      }
    
  }
}

} // extern "C"
