# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic inputs for ``petsc_ilu_trisolve``: a real PETSc-layout ILU(0) factor.

Reproduces what the profiled run actually factored -- the 7-point star Laplacian that
``src/ksp/ksp/tutorials/ex45`` assembles on a ``DMDA_STENCIL_STAR`` grid, reordered with RCM
(the profiled option string used ``-pc_factor_mat_ordering_type rcm``), then ILU(0)-factored
into PETSc's packed ``Mat_SeqAIJ`` factor layout.

The factorization here is a transcription of ``MatLUFactorNumeric_SeqAIJ``
(``src/mat/impls/aij/seq/aijfact.c:218``) restricted to zero fill, so the buffers the kernel
receives are the buffers PETSc would hand it.  Verified against a live PETSc factor: on a
4x3x2 grid this produces bit-identical ``l_rowptr``/``u_diagptr``/``fact_cols``/``fact_values``
for both natural and RCM orderings (see ``test_petsc_ilu_trisolve_reference.py``).

Zero fill is what makes this affordable: ILU(0) keeps exactly the sparsity of the permuted
matrix, so the pattern is known up front and no symbolic phase is needed.  The elimination
runs over Python lists rather than NumPy scalars -- the per-row work is a handful of
operations and NumPy's scalar-indexing overhead would dominate it by an order of magnitude.
"""
from typing import Optional

import numpy as np


def _laplacian_csr(NX, NY, NZ, datatype):
    """7-point star Laplacian in CSR with sorted column indices, diagonal 6, off-diagonals -1.

    Same stencil shape and coefficients (up to the uniform mesh scaling ex45 applies) as the
    operator ``ComputeMatrix`` assembles for ``DMDA_STENCIL_STAR`` with one degree of freedom.
    """
    import scipy.sparse as sp

    def tri(m):
        return sp.diags([-np.ones(m - 1), np.zeros(m), -np.ones(m - 1)], [-1, 0, 1], format="csr")

    Ix, Iy, Iz = (sp.identity(m, format="csr") for m in (NX, NY, NZ))
    A = (sp.kron(sp.kron(Iz, Iy), tri(NX)) + sp.kron(sp.kron(Iz, tri(NY)), Ix) +
         sp.kron(sp.kron(tri(NZ), Iy), Ix))
    A = (A + 6.0 * sp.identity(NX * NY * NZ)).tocsr()
    A.sort_indices()
    return A.astype(datatype)


def _ilu0_packed(indptr, indices, values, perm_row, inv_perm_col, n, datatype):
    """ILU(0) of ``B = P_r A P_c`` into PETSc's packed factor layout.

    Returns ``(l_rowptr, u_diagptr, fact_cols, fact_values)``.  Follows
    ``MatLUFactorNumeric_SeqAIJ``: a dense sparse-accumulator row (``rtmp``), IKJ elimination
    against already-finished rows, then a scatter back into the packed array with the
    diagonal stored as its reciprocal.
    """
    indptr = indptr.tolist()
    indices = indices.tolist()
    values = values.tolist()
    perm_row = perm_row.tolist()
    inv_perm_col = inv_perm_col.tolist()

    # --- pattern: ILU(0) keeps the sparsity of the permuted matrix, row by row ---
    rows_cols = []
    lcnt = [0] * n
    ucnt = [0] * n
    for i in range(n):
        orig = perm_row[i]
        cols = sorted(inv_perm_col[j] for j in indices[indptr[orig]:indptr[orig + 1]])
        if i not in cols:
            raise ValueError(f"row {i} has no diagonal entry; ILU(0) needs one")
        rows_cols.append(cols)
        lcnt[i] = sum(1 for cc in cols if cc < i)
        ucnt[i] = sum(1 for cc in cols if cc > i)

    l_rowptr = [0] * (n + 1)
    for i in range(n):
        l_rowptr[i + 1] = l_rowptr[i] + lcnt[i]
    nnz_l = l_rowptr[n]
    total = nnz_l + sum(ucnt) + n

    # u_diagptr DECREASES: U rows are stored back to front, diagonal last within each row.
    u_diagptr = [0] * (n + 1)
    u_diagptr[n] = nnz_l - 1
    for i in range(n - 1, -1, -1):
        u_diagptr[i] = u_diagptr[i + 1] + ucnt[i] + 1

    fact_cols = [0] * total
    fact_values = [0.0] * total
    for i in range(n):
        cols = rows_cols[i]
        pos = l_rowptr[i]
        for cc in cols:
            if cc < i:
                fact_cols[pos] = cc
                pos += 1
        pos = u_diagptr[i + 1] + 1
        for cc in cols:
            if cc > i:
                fact_cols[pos] = cc
                pos += 1
        fact_cols[u_diagptr[i]] = i

    # --- numeric phase ---
    rtmp = [0.0] * n
    for i in range(n):
        for k in range(l_rowptr[i], l_rowptr[i + 1]):
            rtmp[fact_cols[k]] = 0.0
        for k in range(u_diagptr[i + 1] + 1, u_diagptr[i] + 1):
            rtmp[fact_cols[k]] = 0.0
        orig = perm_row[i]
        for k in range(indptr[orig], indptr[orig + 1]):
            rtmp[inv_perm_col[indices[k]]] = values[k]
        for k in range(l_rowptr[i], l_rowptr[i + 1]):
            row = fact_cols[k]
            pc = rtmp[row]
            if pc != 0.0:
                multiplier = pc * fact_values[u_diagptr[row]]
                rtmp[row] = multiplier
                for kk in range(u_diagptr[row + 1] + 1, u_diagptr[row]):
                    rtmp[fact_cols[kk]] -= multiplier * fact_values[kk]
        for k in range(l_rowptr[i], l_rowptr[i + 1]):
            fact_values[k] = rtmp[fact_cols[k]]
        for k in range(u_diagptr[i + 1] + 1, u_diagptr[i]):
            fact_values[k] = rtmp[fact_cols[k]]
        if rtmp[i] == 0.0:
            raise ValueError(f"zero pivot at row {i}")
        fact_values[u_diagptr[i]] = 1.0 / rtmp[i]

    return (np.array(l_rowptr, dtype=np.int64), np.array(u_diagptr, dtype=np.int64),
            np.array(fact_cols, dtype=np.int64), np.array(fact_values, dtype=datatype))


def initialize(NX, NY, NZ, datatype=np.float64, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random.default_rng(42)
    if NX < 2 or NY < 2 or NZ < 2:
        raise ValueError("each grid dimension must be at least 2")

    n = NX * NY * NZ
    A = _laplacian_csr(NX, NY, NZ, datatype)

    from scipy.sparse.csgraph import reverse_cuthill_mckee
    # RCM is a symmetric reordering, so the row and column permutations coincide -- which is
    # also what PETSc's MatGetOrdering returns for MATORDERINGRCM.
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    perm_row = perm
    perm_col = perm
    inv_perm_col = np.zeros(n, dtype=np.int64)
    inv_perm_col[perm_col] = np.arange(n, dtype=np.int64)

    l_rowptr, u_diagptr, fact_cols, fact_values = _ilu0_packed(
        A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data, perm_row, inv_perm_col, n,
        datatype)

    # A smooth, strictly non-degenerate right-hand side: no zeros, O(1), and independent of
    # the grid shape so the graded output does not collapse at any preset.
    b = np.ascontiguousarray(1.0 + np.sin(np.arange(1, n + 1, dtype=datatype)), dtype=datatype)
    solve_work = np.zeros(n, dtype=datatype)
    x = np.zeros(n, dtype=datatype)

    return (b, fact_cols, fact_values, l_rowptr, perm_col, perm_row, solve_work, u_diagptr, x)
