# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""PETSc's sparse triangular solve over a packed ILU(0) factor (``MatSolve_SeqAIJ``).

Ported from PETSc v3.25.4, ``src/mat/impls/aij/seq/aijfact.c:2466`` -- the frozen upstream
source sits beside this file as ``petsc_ilu_trisolve_reference.c``.  This is the routine
``PCApply`` calls once per Krylov iteration for ``-pc_type ilu``; it was the top event by
``%T`` in a ``-log_view`` profile of ``src/ksp/ksp/tutorials/ex45`` on a 321^3 grid.

Storage conventions (PETSc's, not a textbook CSR -- each one silently produces a plausible
wrong answer if mis-read; all four were confirmed against arrays dumped from a live PETSc
factor rather than inferred from the source):

* **One array holds both factors.**  ``fact_values``/``fact_cols`` pack L and U together.
  L row ``i`` is ``l_rowptr[i] .. l_rowptr[i+1]-1`` and its **unit diagonal is not stored**
  (so L row 0 is empty and ``l_rowptr[0] == l_rowptr[1] == 0``).
* **U is stored backwards.**  ``u_diagptr`` is strictly DECREASING.  U row ``i`` occupies
  ``u_diagptr[i+1]+1 .. u_diagptr[i]``, so row ``n-1`` sits at the low end of the array and
  row ``0`` at the high end.  Within a row the strictly-upper entries come first and the
  **diagonal is last**, at ``u_diagptr[i]``.
* **The stored diagonal is its own reciprocal.**  ``MatLUFactorNumeric_SeqAIJ`` writes
  ``1.0/rtmp[i]`` there, so the back substitution MULTIPLIES by ``fact_values[u_diagptr[i]]``.
  A port that divides is wrong by construction and still converges on a tame matrix.
* **The factor carries both permutations.**  The factorization is of ``B = P_r A P_c``, so
  the solve reads the right-hand side through ``perm_row`` (``b[perm_row[i]]``) and scatters
  the result through ``perm_col`` (``x[perm_col[i]]``).  Both are kept here rather than
  pinning the driver to a natural ordering -- the profiled run used
  ``-pc_factor_mat_ordering_type rcm``, where both are non-trivial.

``solve_work`` is PETSc's ``a->solve_work`` scratch vector, passed in rather than allocated
so the buffer is explicit; it is fully written before it is read, so its incoming contents
do not matter.

Parallelism: **none, deliberately.**  ``aijfact.c`` contains no OpenMP pragma at all -- the
forward sweep's ``solve_work[i]`` depends on ``solve_work[j]`` for ``j < i`` and the backward
sweep on ``j > i``, so both loops are sequential dependence chains, and that dependence is
the point of the kernel.  (PETSc's ``parallel for`` pragmas live in ``aij.c``/``inode.c`` on
the SpMV row loops, not here.)

Simplifications from upstream, all of them error handling or bookkeeping rather than
arithmetic: ``PetscFunctionBegin``/``PetscCall``/``PetscLogFlops`` and the ``Vec``/``IS``
get-and-restore wrappers are dropped, and the ``n == 0`` early return is dropped because the
manifest never generates an empty grid.  The arithmetic, its order, and the index
expressions are unchanged; the inner loops are ``PetscSparseDenseMinusDot`` spelled out (the
non-unrolled ``#else`` variant, which is what this build compiled).  Indices are ``int64``
here where PETSc's build used 32-bit ``PetscInt``; this is the benchmark's ABI convention and
does not change any value.
"""


def petsc_ilu_trisolve(b, fact_cols, fact_values, l_rowptr, perm_col, perm_row, solve_work, u_diagptr, x):
    n = l_rowptr.shape[0] - 1

    # forward solve the lower triangular -- L has a unit diagonal that is not stored, so
    # row 0 is just the permuted right-hand side and the loop starts at 1.
    solve_work[0] = b[perm_row[0]]
    for i in range(1, n):
        sum_ = b[perm_row[i]]
        for k in range(l_rowptr[i], l_rowptr[i + 1]):
            sum_ = sum_ - fact_values[k] * solve_work[fact_cols[k]]
        solve_work[i] = sum_

    # backward solve the upper triangular -- the strictly-upper entries of row i run from
    # u_diagptr[i+1]+1 up to (not including) u_diagptr[i], which holds 1/U[i,i].
    for i in range(n - 1, -1, -1):
        sum_ = solve_work[i]
        for k in range(u_diagptr[i + 1] + 1, u_diagptr[i]):
            sum_ = sum_ - fact_values[k] * solve_work[fact_cols[k]]
        solve_work[i] = sum_ * fact_values[u_diagptr[i]]
        x[perm_col[i]] = solve_work[i]
