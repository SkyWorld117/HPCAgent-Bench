// Frozen upstream source for the bout_arakawa benchmark.
//
// Transcribed from BOUT++ (github.com/boutproject/BOUT-dev, LGPL-3.0-or-later),
// revision ebdcb73c9, src/mesh/difops.cxx, function
//   Field3D bracket(const Field3D& f, const Field3D& g, BRACKET_METHOD method,
//                   CELL_LOC outloc, Solver* solver)
// case BRACKET_ARAKAWA (lines 1319-1424), the #if not(BOUT_USE_METRIC_3D) path.
//
// What changed and why:
//   * BOUT++ Field3D/Field2D containers become raw pointers with the same
//     (x, y, z) row-major layout Field3D uses internally, so f(x, y) - a pointer
//     to the start of a z row - becomes &f[idx3(x, y, 0)].
//   * BOUT_FOR over result.getRegion2D("RGN_NOBNDRY") - the region that excludes
//     the MXG x-guard cells and the y-boundary rows - becomes an explicit loop
//     over the stencil-defined interior 1 <= jx <= NX-2, 0 <= jy <= NY-1. The
//     MPI/domain-decomposition halo width is replaced by the one-cell halo the
//     stencil actually reads (docs/kernel_extraction.md step 9).
//   * The OpenMP pragma BOUT_FOR expands to is dropped; the reference is serial.
// Nothing else is altered: the three-block z split, the operation order inside
// each Jacobian, and the reciprocal spacingFactor multiply are upstream's.

extern "C" void bout_arakawa_reference(const double *__restrict__ dx, const double *__restrict__ dz,
                                       const double *__restrict__ f, const double *__restrict__ g, int NX, int NY,
                                       int NZ, double *__restrict__ result) {
  const int ncz = NZ;

  for (int jx = 1; jx < NX - 1; ++jx) {
    for (int jy = 0; jy < NY; ++jy) {
      const int j2D = jx * NY + jy;
      const double spacingFactor = 1.0 / (12 * dz[j2D] * dx[j2D]);
      const int xm = jx - 1;
      const int xp = jx + 1;

      const double *Fxm = &f[(xm * NY + jy) * NZ];
      const double *Fx = &f[(jx * NY + jy) * NZ];
      const double *Fxp = &f[(xp * NY + jy) * NZ];
      const double *Gxm = &g[(xm * NY + jy) * NZ];
      const double *Gx = &g[(jx * NY + jy) * NZ];
      const double *Gxp = &g[(xp * NY + jy) * NZ];
      double *out = &result[(jx * NY + jy) * NZ];

      {
        const int jz = 0;
        const int jzp = 1;
        const int jzm = ncz - 1;

        // J++ = DDZ(f)*DDX(g) - DDX(f)*DDZ(g)
        const double Jpp =
            (((Fx[jzp] - Fx[jzm]) * (Gxp[jz] - Gxm[jz])) - ((Fxp[jz] - Fxm[jz]) * (Gx[jzp] - Gx[jzm])));

        // J+x
        const double Jpx = ((Gxp[jz] * (Fxp[jzp] - Fxp[jzm])) - (Gxm[jz] * (Fxm[jzp] - Fxm[jzm])) -
                            (Gx[jzp] * (Fxp[jzp] - Fxm[jzp])) + (Gx[jzm] * (Fxp[jzm] - Fxm[jzm])));

        // Jx+
        const double Jxp = ((Gxp[jzp] * (Fx[jzp] - Fxp[jz])) - (Gxm[jzm] * (Fxm[jz] - Fx[jzm])) -
                            (Gxm[jzp] * (Fx[jzp] - Fxm[jz])) + (Gxp[jzm] * (Fxp[jz] - Fx[jzm])));

        out[jz] = (Jpp + Jpx + Jxp) * spacingFactor;
      }

      for (int jz = 1; jz < ncz - 1; jz++) {
        const int jzp = jz + 1;
        const int jzm = jz - 1;

        const double Jpp =
            (((Fx[jzp] - Fx[jzm]) * (Gxp[jz] - Gxm[jz])) - ((Fxp[jz] - Fxm[jz]) * (Gx[jzp] - Gx[jzm])));

        const double Jpx = ((Gxp[jz] * (Fxp[jzp] - Fxp[jzm])) - (Gxm[jz] * (Fxm[jzp] - Fxm[jzm])) -
                            (Gx[jzp] * (Fxp[jzp] - Fxm[jzp])) + (Gx[jzm] * (Fxp[jzm] - Fxm[jzm])));

        const double Jxp = ((Gxp[jzp] * (Fx[jzp] - Fxp[jz])) - (Gxm[jzm] * (Fxm[jz] - Fx[jzm])) -
                            (Gxm[jzp] * (Fx[jzp] - Fxm[jz])) + (Gxp[jzm] * (Fxp[jz] - Fx[jzm])));

        out[jz] = (Jpp + Jpx + Jxp) * spacingFactor;
      }

      {
        const int jz = ncz - 1;
        const int jzp = 0;
        const int jzm = ncz - 2;

        const double Jpp =
            (((Fx[jzp] - Fx[jzm]) * (Gxp[jz] - Gxm[jz])) - ((Fxp[jz] - Fxm[jz]) * (Gx[jzp] - Gx[jzm])));

        const double Jpx = ((Gxp[jz] * (Fxp[jzp] - Fxp[jzm])) - (Gxm[jz] * (Fxm[jzp] - Fxm[jzm])) -
                            (Gx[jzp] * (Fxp[jzp] - Fxm[jzp])) + (Gx[jzm] * (Fxp[jzm] - Fxm[jzm])));

        const double Jxp = ((Gxp[jzp] * (Fx[jzp] - Fxp[jz])) - (Gxm[jzm] * (Fxm[jz] - Fx[jzm])) -
                            (Gxm[jzp] * (Fx[jzp] - Fxm[jz])) + (Gxp[jzm] * (Fxp[jz] - Fx[jzm])));

        out[jz] = (Jpp + Jpx + Jxp) * spacingFactor;
      }
    }
  }
}
