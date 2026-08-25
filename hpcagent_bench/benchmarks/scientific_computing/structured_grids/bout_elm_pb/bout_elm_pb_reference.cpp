// Frozen upstream reference -- BOUT++ high-beta reduced-MHD (peeling-ballooning) RHS.
//
// Provenance: boutproject/BOUT-dev @ ebdcb73c9 (LGPL-3.0-or-later),
//   examples/elm-pb-outerloop/elm_pb_outerloop.cxx, the fused BOUT_FOR_RAJA loop in
//   ELMpb::rhs (elm_pb_outerloop.cxx:1563-1638), and the operator templates it calls in
//   include/bout/single_index_ops.hxx (Grad_par, DDX/DDY/DDZ, Delp2, bracket,
//   b0xGrad_dot_Grad).
//
// Adaptation, and only this: BOUT++'s FieldAccessor/CoordinatesAccessor containers are
// replaced by raw restrict-qualified pointers, and the RGN_NOBNDRY region iterator by
// explicit loop bounds. Every arithmetic expression keeps upstream's operand order and
// association, so this reproduces the application bit for bit -- verified against a live
// BOUT++ Mesh in check_elmpb.cxx.
//
// Layout. Every buffer is a separate flat contiguous array (SoA): the 3-D fields are
// (NX, NY, NZ) with z fastest, index i = (jx*NY + jy)*NZ + jz; the equilibrium profiles
// and the metric are (NX, NY), index i2d = i / NZ. BOUT++ interleaves the 24 metric
// quantities into one strided array (CoordinatesAccessor::stripe_size); that AoS packing
// is unpacked here into one array per quantity.
//
// Note: the fused loop only ever reads B0phi through Grad_par, so the centre buffer of
// that product is not a parameter -- its two parallel slices are.
//
// Parallel slices. The example runs with the shifted-metric parallel transform, so a
// field's y-neighbours live in separate yup/ydown buffers produced by an FFT phase shift
// during mesh->communicate(). They are inputs here, exactly as the fused loop sees them.
// Field2D has no z dependence, so upstream's Field2D::yup() returns the field itself and
// the 2-D profiles need no slice buffers.
//
// Configuration. The compile-time switches at the top of elm_pb_outerloop.cxx select the
// terms; this is the shipped default set, which is what examples/elm-pb-outerloop/data
// runs and what was profiled:
//   EVOLVE_JPAR false, RELAX_J_VAC false, EHALL false, DIAMAG_PHI0 true,
//   DIAMAG_GRAD_T false, HYPERRESIST true, EHYPERVISCOS false, INCLUDE_RMP false,
//   GRADPARJ true, VISCOS_PERP false, EVOLVE_PRESSURE true, NONLINEAR false.
// Upstream's EVAL_IF(false, expr) expands to the literal 0.0, so the disabled terms are
// dropped here rather than added as zero; the surviving terms keep their original order.
// With NONLINEAR false the GRAD_PARP macro reduces to Grad_par.

#include <cmath>

// Ind3D neighbour arithmetic, single_index_ops.hxx:8-35. z is periodic and wraps.
static inline int elm_i_zp(int id, int nz) {
  const int jz = id % nz;
  const int jzmax = nz - 1;
  return (jz < jzmax) ? (id + 1) : (id - jzmax);
}

static inline int elm_i_zm(int id, int nz) {
  const int jz = id % nz;
  const int jzmax = nz - 1;
  return (jz > 0) ? (id - 1) : (id + jzmax);
}

// DDX/DDY/DDZ on a Field3D, single_index_ops.hxx:140-201.
static inline double elm_ddx3(const double *__restrict__ f, int i, int ny, int nz,
                              const double *__restrict__ dx) {
  return 0.5 * (f[i + ny * nz] - f[i - ny * nz]) / dx[i / nz];
}

static inline double elm_ddy3(const double *__restrict__ f_yup, const double *__restrict__ f_ydown, int i, int nz,
                              const double *__restrict__ dy) {
  return 0.5 * (f_yup[i + nz] - f_ydown[i - nz]) / dy[i / nz];
}

static inline double elm_ddz3(const double *__restrict__ f, int i, int nz, const double *__restrict__ dz) {
  return 0.5 * (f[elm_i_zp(i, nz)] - f[elm_i_zm(i, nz)]) / dz[i / nz];
}

// Grad_par(f) = DDY(f) / sqrt(g_22), single_index_ops.hxx:375-379.
static inline double elm_grad_par(const double *__restrict__ f_yup, const double *__restrict__ f_ydown, int i, int nz,
                                  const double *__restrict__ dy, const double *__restrict__ g_22) {
  return elm_ddy3(f_yup, f_ydown, i, nz, dy) / std::sqrt(g_22[i / nz]);
}

// Delp2: perpendicular (x-z) Laplacian, single_index_ops.hxx:210-236. The non-uniform mesh
// correction (d1_dx) is always included.
static inline double elm_delp2(const double *__restrict__ f, int i, int ny, int nz,
                               const double *__restrict__ dx_a, const double *__restrict__ dz_a,
                               const double *__restrict__ G1, const double *__restrict__ G3,
                               const double *__restrict__ d1_dx, const double *__restrict__ g11,
                               const double *__restrict__ g13, const double *__restrict__ g33) {
  const int ixp = i + ny * nz;
  const int ixm = i - ny * nz;
  const int izp = elm_i_zp(i, nz);
  const int izm = elm_i_zm(i, nz);

  const int izpxp = izp + ny * nz;
  const int izpxm = izp - ny * nz;
  const int izmxp = izm + ny * nz;
  const int izmxm = izm - ny * nz;

  const int c = i / nz;
  const double dx = dx_a[c];
  const double dz = dz_a[c];

  return (G1[c] + d1_dx[c] * g11[c]) * (f[ixp] - f[ixm]) / (2.0 * dx)
         + G3[c] * (f[izp] - f[izm]) / (2.0 * dz)
         + g11[c] * (f[ixp] - 2.0 * f[i] + f[ixm]) / (dx * dx)
         + g33[c] * (f[izp] - 2.0 * f[i] + f[izm]) / (dz * dz)
         + 2 * g13[c] * ((f[izpxp] - f[izpxm]) - (f[izmxp] - f[izmxm])) / (4. * dz * dx);
}

// Arakawa bracket [f, g] with f a Field2D, single_index_ops.hxx:42-79. The J++ and J+x
// terms collapse because a Field2D has no z derivative.
static inline double elm_bracket_2d3d(const double *__restrict__ f, const double *__restrict__ g, int i, int ny,
                                      int nz, const double *__restrict__ dx, const double *__restrict__ dz) {
  const int i2d = i / nz;
  const int i2dxp = i2d + ny;
  const int i2dxm = i2d - ny;

  const int izp = elm_i_zp(i, nz);
  const int izm = elm_i_zm(i, nz);

  const int izpxp = izp + ny * nz;
  const int izpxm = izp - ny * nz;
  const int izmxp = izm + ny * nz;
  const int izmxm = izm - ny * nz;

  const double Jpp = -(f[i2dxp] - f[i2dxm]) * (g[izp] - g[izm]);

  const double Jpx = (-g[izp] * (f[i2dxp] - f[i2dxm]) + g[izm] * (f[i2dxp] - f[i2dxm]));

  const double Jxp = (g[izpxp] * (f[i2d] - f[i2dxp]) - g[izmxm] * (f[i2dxm] - f[i2d])
                      - g[izpxm] * (f[i2d] - f[i2dxm]) + g[izmxp] * (f[i2dxp] - f[i2d]));

  return (Jpp + Jpx + Jxp) / (12 * dx[i2d] * dz[i2d]);
}

// b0xGrad_dot_Grad(phi, f) with phi a Field3D and f a Field2D, single_index_ops.hxx:278-306.
// b0 x Grad(phi) is the E x B velocity; it is dotted into Grad(f).
static inline double elm_b0xgrad_3d2d(const double *__restrict__ phi, const double *__restrict__ phi_yup,
                                      const double *__restrict__ phi_ydown, const double *__restrict__ f, int i,
                                      int ny, int nz, const double *__restrict__ dx, const double *__restrict__ dy,
                                      const double *__restrict__ dz, const double *__restrict__ J,
                                      const double *__restrict__ g_12, const double *__restrict__ g_22,
                                      const double *__restrict__ g_23) {
  const int c = i / nz;

  const double dpdx = elm_ddx3(phi, i, ny, nz, dx);
  const double dpdy = elm_ddy3(phi_yup, phi_ydown, i, nz, dy);
  const double dpdz = elm_ddz3(phi, i, nz, dz);

  const double vx = g_22[c] * dpdz - g_23[c] * dpdy;
  const double vy = g_23[c] * dpdx - g_12[c] * dpdz;

  const int i2d = c;
  const int ixp = i2d + ny;
  const int ixm = i2d - ny;
  const int iyp = i2d + 1;
  const int iym = i2d - 1;

  const double vddx = vx * (f[ixp] - f[ixm]) / (2. * dx[c]);
  const double vddy = vy * (f[iyp] - f[iym]) / (2. * dy[c]);

  return (vddx + vddy) / (J[c] * std::sqrt(g_22[c]));
}

// b0xGrad_dot_Grad(phi, f) with phi a Field2D and f a Field3D, single_index_ops.hxx:308-336.
// The equilibrium potential has no z derivative, so the velocity picks up a z component
// instead.
static inline double elm_b0xgrad_2d3d(const double *__restrict__ phi, const double *__restrict__ f,
                                      const double *__restrict__ f_yup, const double *__restrict__ f_ydown, int i,
                                      int ny, int nz, const double *__restrict__ dx, const double *__restrict__ dy,
                                      const double *__restrict__ dz, const double *__restrict__ J,
                                      const double *__restrict__ g_12, const double *__restrict__ g_22,
                                      const double *__restrict__ g_23) {
  const int c = i / nz;

  const double dpdx = 0.5 * (phi[c + ny] - phi[c - ny]) / dx[c];
  const double dpdy = 0.5 * (phi[c + 1] - phi[c - 1]) / dy[c];

  const double vx = -g_23[c] * dpdy;
  const double vy = g_23[c] * dpdx;
  const double vz = g_12[c] * dpdy - g_22[c] * dpdx;

  const int ixp = i + ny * nz;
  const int ixm = i - ny * nz;
  const int iyp = i + nz;
  const int iym = i - nz;
  const int izp = elm_i_zp(i, nz);
  const int izm = elm_i_zm(i, nz);

  const double vddx = vx * (f[ixp] - f[ixm]) / (2. * dx[c]);
  const double vddy = vy * (f_yup[iyp] - f_ydown[iym]) / (2. * dy[c]);
  const double vddz = vz * (f[izp] - f[izm]) / (2. * dz[c]);

  return (vddx + vddy + vddz) / (J[c] * std::sqrt(g_22[c]));
}

extern "C" void bout_elm_pb_reference(
    const double *__restrict__ B0, const double *__restrict__ B0phi_ydown,
    const double *__restrict__ B0phi_yup,
    const double *__restrict__ G1, const double *__restrict__ G3, const double *__restrict__ J,
    const double *__restrict__ J0, const double *__restrict__ Jpar, const double *__restrict__ Jpar_ydown,
    const double *__restrict__ Jpar_yup, const double *__restrict__ P, const double *__restrict__ P0,
    const double *__restrict__ P_ydown, const double *__restrict__ P_yup, const double *__restrict__ Psi,
    const double *__restrict__ Psi_ydown, const double *__restrict__ Psi_yup, const double *__restrict__ U,
    const double *__restrict__ U_ydown, const double *__restrict__ U_yup, const double *__restrict__ d1_dx,
    double *__restrict__ ddt_P, double *__restrict__ ddt_Psi, double *__restrict__ ddt_U,
    const double *__restrict__ dx, const double *__restrict__ dy, const double *__restrict__ dz,
    const double *__restrict__ eta, const double *__restrict__ g11, const double *__restrict__ g13,
    const double *__restrict__ g33, const double *__restrict__ g_12, const double *__restrict__ g_22,
    const double *__restrict__ g_23, double hyperresist, const double *__restrict__ phi,
    const double *__restrict__ phi0, const double *__restrict__ phi_ydown, const double *__restrict__ phi_yup,
    int NX, int NY, int NZ) {
  // RGN_NOBNDRY: the two guard cells at each end of x and y are excluded, z is periodic.
  for (int jx = 2; jx < NX - 2; ++jx) {
    for (int jy = 2; jy < NY - 2; ++jy) {
      for (int jz = 0; jz < NZ; ++jz) {
        const int i = (jx * NY + jy) * NZ + jz;
        const int i2d = i / NZ;

        ////////////////////////////////////////////////////
        // Parallel electric field: evolve the vector potential.
        // Induction, resistive diffusion of the parallel current, advection of Psi by
        // the equilibrium E x B flow, and hyper-resistivity.

        ddt_Psi[i] = -elm_grad_par(B0phi_yup, B0phi_ydown, i, NZ, dy, g_22) / B0[i2d] + eta[i] * Jpar[i]

                     - elm_bracket_2d3d(phi0, Psi, i, NY, NZ, dx, dz) * B0[i2d]

                     - eta[i] * hyperresist
                           * elm_delp2(Jpar, i, NY, NZ, dx, dz, G1, G3, d1_dx, g11, g13, g33);

        ////////////////////////////////////////////////////
        // Vorticity equation: field-line bending against the equilibrium current,
        // the parallel current term, and advection by the equilibrium flow.

        ddt_U[i] = (B0[i2d] * B0[i2d])
                       * elm_b0xgrad_3d2d(Psi, Psi_yup, Psi_ydown, J0, i, NY, NZ, dx, dy, dz, J, g_12, g_22, g_23)

                   - (B0[i2d] * B0[i2d]) * elm_grad_par(Jpar_yup, Jpar_ydown, i, NZ, dy, g_22)

                   - elm_b0xgrad_2d3d(phi0, U, U_yup, U_ydown, i, NY, NZ, dx, dy, dz, J, g_12, g_22, g_23);

        ////////////////////////////////////////////////////
        // Pressure equation: the perturbed flow across the equilibrium pressure
        // gradient, plus advection of the perturbation by the equilibrium flow.

        ddt_P[i] = -elm_b0xgrad_3d2d(phi, phi_yup, phi_ydown, P0, i, NY, NZ, dx, dy, dz, J, g_12, g_22, g_23)

                   - elm_b0xgrad_2d3d(phi0, P, P_yup, P_ydown, i, NY, NZ, dx, dy, dz, J, g_12, g_22, g_23);
      }
    }
  }
}
