/*
 * Frozen upstream source -- PETSc, not HPCAgent-Bench code.  Kept verbatim under
 * PETSc's own licence (below); do NOT relicense or reformat.
 *
 * Provenance
 *   project : PETSc (Portable, Extensible Toolkit for Scientific Computation)
 *   version : v3.25.4  (release branch)
 *   commit  : ab79ce791859c84e6d524d329ee8a429d6dc37ad
 *   upstream: https://gitlab.com/petsc/petsc.git
 *   files   : src/mat/impls/aij/seq/aijfact.c   lines 2466-2519  (MatSolve_SeqAIJ)
 *             src/mat/impls/aij/seq/aij.h       lines 541-547    (PetscSparseDenseMinusDot)
 *
 * MatSolve_SeqAIJ is the sparse triangular solve applied once per Krylov iteration by
 * PCApply for -pc_type ilu.  The PetscSparseDenseMinusDot macro is included because the
 * routine's inner loop is written entirely in terms of it; the variant reproduced here is
 * the one this build selected (neither PETSC_KERNEL_USE_UNROLL_4 nor _UNROLL_2 is defined
 * in arch-gh200-opt/include/petscconf.h, so the #else branch is what compiled).
 *
 * ---------------------------------------------------------------------------------------
 * PETSc licence (SPDX-License-Identifier: BSD-2-Clause), reproduced verbatim from
 * the LICENSE file at the commit above:
 *
 * Copyright (c) 1991-2025, UChicago Argonne, LLC and the PETSc Developers and Contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ---------------------------------------------------------------------------------------
 */

/* ---- src/mat/impls/aij/seq/aij.h, lines 541-547 ---- */
#else
  #define PetscSparseDenseMinusDot(sum, r, xv, xi, nnz) \
    do { \
      PetscInt __i; \
      for (__i = 0; __i < nnz; __i++) sum -= xv[__i] * r[xi[__i]]; \
    } while (0)
#endif

/* ---- src/mat/impls/aij/seq/aijfact.c, lines 2466-2519 ---- */
PetscErrorCode MatSolve_SeqAIJ(Mat A, Vec bb, Vec xx)
{
  Mat_SeqAIJ        *a     = (Mat_SeqAIJ *)A->data;
  IS                 iscol = a->col, isrow = a->row;
  PetscInt           i, n = A->rmap->n, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag, nz;
  const PetscInt    *rout, *cout, *r, *c;
  PetscScalar       *x, *tmp, sum;
  const PetscScalar *b;
  const MatScalar   *aa, *v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatSeqAIJGetArrayRead(A, &aa));
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArrayWrite(xx, &x));
  tmp = a->solve_work;

  PetscCall(ISGetIndices(isrow, &rout));
  r = rout;
  PetscCall(ISGetIndices(iscol, &cout));
  c = cout;

  /* forward solve the lower triangular */
  tmp[0] = b[r[0]];
  v      = aa;
  vi     = aj;
  for (i = 1; i < n; i++) {
    nz  = ai[i + 1] - ai[i];
    sum = b[r[i]];
    PetscSparseDenseMinusDot(sum, tmp, v, vi, nz);
    tmp[i] = sum;
    v += nz;
    vi += nz;
  }

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + adiag[i + 1] + 1;
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    sum = tmp[i];
    PetscSparseDenseMinusDot(sum, tmp, v, vi, nz);
    x[c[i]] = tmp[i] = sum * v[nz]; /* v[nz] = aa[adiag[i]] */
  }

  PetscCall(ISRestoreIndices(isrow, &rout));
  PetscCall(ISRestoreIndices(iscol, &cout));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &aa));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArrayWrite(xx, &x));
  PetscCall(PetscLogFlops(2.0 * a->nz - A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
