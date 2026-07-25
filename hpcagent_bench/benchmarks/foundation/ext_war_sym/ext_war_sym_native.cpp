/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ext_war_sym_d: a[i] = a[i + k] + b[i] (symbolic-offset WAR)
void ext_war_sym_d(double *__restrict__ a, const double *__restrict__ b, const int len_1d, const int k) {
  for (int i = 0; i < len_1d - k; ++i) {
    a[i] = a[i + k] + b[i];
  }
}

} // extern "C"
