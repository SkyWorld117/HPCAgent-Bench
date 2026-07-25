/*
 * HPCAgent-Bench C++ native timing baseline for a foundation microkernel. The tsvc_2 /
 * tsvc_2_5 kernels derive from TSVC_2 (github.com/UoB-HPC/TSVC_2, NCSA/MIT, UIUC); the
 * extended microkernels are HPCAgent-Bench's own tsvc-style additions.
 */

#include <cstdint>
#include <cmath>

extern "C" {

// ext_strided_load_2_d: dst[i] = src[i * 2] * scale (constant-stride sibling)
void ext_strided_load_2_d(double *__restrict__ dst, const double *__restrict__ src, const double scale,
                                  const int len_1d) {
  for (int i = 0; i < len_1d; ++i) {
    dst[i] = src[i * 2] * scale;
  }
}

} // extern "C"
