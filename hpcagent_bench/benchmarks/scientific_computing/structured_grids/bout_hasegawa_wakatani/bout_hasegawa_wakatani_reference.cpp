// Frozen upstream source for the bout_hasegawa_wakatani benchmark.
//
// Transcribed from BOUT++ (github.com/boutproject/BOUT-dev, LGPL-3.0-or-later),
// revision ebdcb73c9:
//   examples/hasegawa-wakatani-3d/hw.cxx, HW::rhs()  (the BOUT_FOR_RAJA body)
//   include/bout/single_index_ops.hxx, bracket(), DDZ(), Delp2(),
//                                      Div_par_Grad_par(), i_zp/i_zm/i_xp/i_xm
//
// What changed and why:
//   * FieldAccessor/Field3D become raw pointers with Field3D's own row-major
//     (x, y, z) layout; CoordinatesAccessor's per-cell metric lookups become
//     Field2D-shaped (x, y) arrays, which is what they are for BOUT_USE_METRIC_3D=OFF.
//   * i_zp/i_zm wrap in z exactly as upstream (z is periodic, no z guard cells);
//     i_xp/i_xm/i_yp/i_ym are the same +-1 index offsets spelled out.
//   * f.yup / f.ydown alias f: the default parallel transform is "identity"
//     (src/mesh/coordinates.cxx:865), which is what this example runs with.
//   * BOUT_FOR_RAJA over n.getRegion("RGN_NOBNDRY") - the region excluding the
//     MXG/MYG guard cells - becomes an explicit loop over the stencil-defined
//     interior 1 <= jx <= NX-2, 1 <= jy <= NY-2, 0 <= jz <= NZ-1. The
//     MPI/domain-decomposition halo width is replaced by the one-cell halo the
//     stencil actually reads (docs/kernel_extraction.md step 9).
//   * The RAJA/OpenMP dispatch is dropped; the reference is serial.
//   * The elliptic solve phi = phiSolver->solve(vort, phi), the halo exchange
//     mesh->communicate(...) and the model's option parsing are OUTSIDE the
//     extraction boundary: phi arrives as an input with its halos already filled.
// The arithmetic - operand order, association, the shared div_current, the
// division by (12*dx*dz) rather than a reciprocal multiply - is upstream's.

extern "C" void bout_hasegawa_wakatani_reference(
    const double *__restrict__ n, const double *__restrict__ vort, const double *__restrict__ phi,
    const double *__restrict__ dx, const double *__restrict__ dy, const double *__restrict__ dz,
    const double *__restrict__ J, const double *__restrict__ g_22, const double *__restrict__ g11,
    const double *__restrict__ g33, const double *__restrict__ g13, const double *__restrict__ G1,
    const double *__restrict__ G3, const double *__restrict__ d1_dx, double alpha, double kappa, double Dn,
    double Dvort, int NX, int NY, int NZ, double *__restrict__ pmn, double *__restrict__ ddt_n,
    double *__restrict__ ddt_vort) {

  // Field3D phi_minus_n = phi - n;
  for (int i = 0; i < NX * NY * NZ; ++i) {
    pmn[i] = phi[i] - n[i];
  }

  for (int jx = 1; jx < NX - 1; ++jx) {
    for (int jy = 1; jy < NY - 1; ++jy) {
      for (int jz = 0; jz < NZ; ++jz) {
        const int i = (jx * NY + jy) * NZ + jz;
        const int izp = (jz < NZ - 1) ? (i + 1) : (i - (NZ - 1));
        const int izm = (jz > 0) ? (i - 1) : (i + (NZ - 1));
        const int ixp = i + NY * NZ;
        const int ixm = i - NY * NZ;
        const int iyp = i + NZ;
        const int iym = i - NZ;
        const int izpxp = izp + NY * NZ;
        const int izpxm = izp - NY * NZ;
        const int izmxp = izm + NY * NZ;
        const int izmxm = izm - NY * NZ;

        const int m = jx * NY + jy;   // Field2D index of this cell
        const int myp = m + 1;        // ... of the y+1 cell
        const int mym = m - 1;        // ... of the y-1 cell

        // Div_par_Grad_par(phi_minus_n_acc, i)
        const double gradient_upper = 2. * (pmn[iyp] - pmn[i]) / (dy[m] + dy[myp]);
        const double flux_upper = gradient_upper * (J[m] + J[myp]) / (g_22[m] + g_22[myp]);
        const double gradient_lower = 2. * (pmn[i] - pmn[iym]) / (dy[m] + dy[mym]);
        const double flux_lower = gradient_lower * (J[m] + J[mym]) / (g_22[m] + g_22[mym]);
        const double div_par_grad_par = (flux_upper - flux_lower) / (dy[m] * J[m]);

        const double div_current = alpha * div_par_grad_par;

        // bracket(phi_acc, n_acc, i)
        const double Jpp_n =
            ((phi[izp] - phi[izm]) * (n[ixp] - n[ixm]) - (phi[ixp] - phi[ixm]) * (n[izp] - n[izm]));
        const double Jpx_n = (n[ixp] * (phi[izpxp] - phi[izmxp]) - n[ixm] * (phi[izpxm] - phi[izmxm]) -
                              n[izp] * (phi[izpxp] - phi[izpxm]) + n[izm] * (phi[izmxp] - phi[izmxm]));
        const double Jxp_n = (n[izpxp] * (phi[izp] - phi[ixp]) - n[izmxm] * (phi[ixm] - phi[izm]) -
                              n[izpxm] * (phi[izp] - phi[ixm]) + n[izmxp] * (phi[ixp] - phi[izm]));
        const double bracket_n = (Jpp_n + Jpx_n + Jxp_n) / (12 * dx[m] * dz[m]);

        // bracket(phi_acc, vort_acc, i)
        const double Jpp_w =
            ((phi[izp] - phi[izm]) * (vort[ixp] - vort[ixm]) - (phi[ixp] - phi[ixm]) * (vort[izp] - vort[izm]));
        const double Jpx_w = (vort[ixp] * (phi[izpxp] - phi[izmxp]) - vort[ixm] * (phi[izpxm] - phi[izmxm]) -
                              vort[izp] * (phi[izpxp] - phi[izpxm]) + vort[izm] * (phi[izmxp] - phi[izmxm]));
        const double Jxp_w = (vort[izpxp] * (phi[izp] - phi[ixp]) - vort[izmxm] * (phi[ixm] - phi[izm]) -
                              vort[izpxm] * (phi[izp] - phi[ixm]) + vort[izmxp] * (phi[ixp] - phi[izm]));
        const double bracket_w = (Jpp_w + Jpx_w + Jxp_w) / (12 * dx[m] * dz[m]);

        // DDZ(phi_acc, i)
        const double ddz_phi = 0.5 * (phi[izp] - phi[izm]) / dz[m];

        // Delp2(n_acc, i)
        const double delp2_n = (G1[m] + d1_dx[m] * g11[m]) * (n[ixp] - n[ixm]) / (2.0 * dx[m]) +
                               G3[m] * (n[izp] - n[izm]) / (2.0 * dz[m]) +
                               g11[m] * (n[ixp] - 2.0 * n[i] + n[ixm]) / (dx[m] * dx[m]) +
                               g33[m] * (n[izp] - 2.0 * n[i] + n[izm]) / (dz[m] * dz[m]) +
                               2 * g13[m] * ((n[izpxp] - n[izpxm]) - (n[izmxp] - n[izmxm])) / (4. * dz[m] * dx[m]);

        // Delp2(vort_acc, i)
        const double delp2_w =
            (G1[m] + d1_dx[m] * g11[m]) * (vort[ixp] - vort[ixm]) / (2.0 * dx[m]) +
            G3[m] * (vort[izp] - vort[izm]) / (2.0 * dz[m]) +
            g11[m] * (vort[ixp] - 2.0 * vort[i] + vort[ixm]) / (dx[m] * dx[m]) +
            g33[m] * (vort[izp] - 2.0 * vort[i] + vort[izm]) / (dz[m] * dz[m]) +
            2 * g13[m] * ((vort[izpxp] - vort[izpxm]) - (vort[izmxp] - vort[izmxm])) / (4. * dz[m] * dx[m]);

        ddt_n[i] = -bracket_n - div_current - kappa * ddz_phi + Dn * delp2_n;
        ddt_vort[i] = -bracket_w - div_current + Dvort * delp2_w;
      }
    }
  }
}
