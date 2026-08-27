# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Correctness gate for petsc_ilu_trisolve against PETSc itself.

The frozen upstream source (``petsc_ilu_trisolve_reference.c``) is verbatim PETSc C that only
compiles against PETSc's headers and private ``Mat_SeqAIJ`` struct, so it cannot be executed
from pytest the way a self-contained C or Python reference could.  This test therefore checks
the port against something strictly stronger than a re-run of that file: the **actual arrays
PETSc produced at run time**.  ``petsc_ilu_trisolve_petsc_dump.txt`` holds ``l_rowptr``,
``u_diagptr``, ``fact_cols``, ``fact_values``, both permutations and ``MatSolve_SeqAIJ``'s own
output vector, read straight out of a live factored ``Mat`` on the machine the port was
measured on (PETSc v3.25.4, commit ab79ce791859c84e6d524d329ee8a429d6dc37ad).

Both orderings PETSc can dispatch are covered.  ``natural`` makes both permutations the
identity -- the case that would hide a permutation bug -- and ``rcm`` makes both non-trivial,
which is what the profiled ``ex45`` run used.  Agreement is asserted EXACT (atol=rtol=0): the
port performs the same operations in the same order on the same inputs, so anything short of
bit-identical is a real difference, not a rounding artifact.
"""
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_DUMP = _HERE / "petsc_ilu_trisolve_petsc_dump.txt"

#: The grid the dump was taken on -- deliberately non-cubic so a transposed axis cannot hide.
_NX, _NY, _NZ = 4, 3, 2


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _parse_dump():
    """Split the dump into ``{ordering: {field: ndarray}}``."""
    blocks, cur = {}, None
    lines = [ln for ln in _DUMP.read_text().splitlines() if ln and not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if parts[0] == "ordering":
            cur = blocks.setdefault(parts[1], {})
            i += 1
        elif parts[0] in ("n", "inode_size_csr"):
            cur[parts[0]] = int(parts[1])
            i += 1
        else:
            name, count = parts[0], int(parts[1])
            vals = lines[i + 1].split() if count else []
            assert len(vals) == count, f"{name}: expected {count} values, got {len(vals)}"
            cur[name] = np.array([float(v) for v in vals])
            i += 2
    return blocks


_DUMPS = _parse_dump()


@pytest.mark.parametrize("ordering", ["natural", "rcm"])
def test_initialize_reproduces_petsc_factor(ordering: str) -> None:
    """``initialize``'s ILU(0) reproduces PETSc's packed factor bit-for-bit.

    Runs the benchmark's own factorization on PETSc's dumped permutation (rather than the RCM
    scipy computes, which need not be the same RCM) so the comparison isolates the
    factorization and the packed layout from the choice of ordering.
    """
    d = _DUMPS[ordering]
    mod = _load("petsc_ilu_trisolve")
    n = d["n"]
    assert n == _NX * _NY * _NZ

    A = mod._laplacian_csr(_NX, _NY, _NZ, np.float64)
    # the generated operator IS the one PETSc factored
    np.testing.assert_array_equal(A.indptr.astype(np.int64), d["A_i"].astype(np.int64))
    np.testing.assert_array_equal(A.indices.astype(np.int64), d["A_j"].astype(np.int64))
    np.testing.assert_array_equal(A.data, d["A_a"])

    perm_row = d["F_r"].astype(np.int64)
    perm_col = d["F_c"].astype(np.int64)
    inv_perm_col = np.zeros(n, dtype=np.int64)
    inv_perm_col[perm_col] = np.arange(n, dtype=np.int64)

    l_rowptr, u_diagptr, fact_cols, fact_values = mod._ilu0_packed(
        A.indptr.astype(np.int64), A.indices.astype(np.int64), A.data, perm_row, inv_perm_col, n,
        np.float64)

    np.testing.assert_array_equal(l_rowptr, d["F_i"].astype(np.int64))
    np.testing.assert_array_equal(u_diagptr, d["F_diag"].astype(np.int64))
    np.testing.assert_array_equal(fact_cols, d["F_j"].astype(np.int64))
    np.testing.assert_array_equal(fact_values, d["F_a"])

    # the layout conventions the docstring claims, asserted rather than assumed
    assert l_rowptr[0] == 0 and l_rowptr[1] == 0, "L row 0 must be empty (unit diagonal unstored)"
    assert np.all(np.diff(u_diagptr) < 0), "u_diagptr must strictly decrease (U stored backwards)"
    assert u_diagptr[n] == l_rowptr[n] - 1, "the U region must start right after the L region"
    for i in range(n):
        assert fact_cols[u_diagptr[i]] == i, "each U row must store its diagonal last"


@pytest.mark.parametrize("ordering", ["natural", "rcm"])
def test_kernel_reproduces_petsc_matsolve(ordering: str) -> None:
    """The port reproduces ``MatSolve_SeqAIJ``'s own output vector, bit-for-bit.

    Driven from PETSc's dumped factor arrays, so this checks the kernel alone -- a fault in
    the benchmark's factorization cannot mask a fault in the solve or vice versa.
    """
    d = _DUMPS[ordering]
    kernel = _load("petsc_ilu_trisolve_numpy").petsc_ilu_trisolve
    n = d["n"]

    b = d["rhs"]
    x = np.zeros(n)
    solve_work = np.zeros(n)
    kernel(b, d["F_j"].astype(np.int64), d["F_a"], d["F_i"].astype(np.int64),
           d["F_c"].astype(np.int64), d["F_r"].astype(np.int64), solve_work,
           d["F_diag"].astype(np.int64), x)

    np.testing.assert_array_equal(x, d["sol"])


def test_petsc_did_not_use_inode_routines() -> None:
    """The dump records the dispatch this port assumes.

    ``MatLUFactorNumeric_SeqAIJ`` picks ``MatSolve_SeqAIJ_Inode`` whenever the factor has
    inodes, which is a DIFFERENT algorithm.  A scalar (one-dof) star stencil gives inodes of
    size 1, so PETSc declines them -- if that ever stopped being true, this port would be
    reproducing the wrong routine.
    """
    for ordering, d in _DUMPS.items():
        assert d["inode_size_csr"] == 0, f"{ordering}: PETSc used inode routines"


def test_output_clears_the_oracle_floor() -> None:
    """The graded buffer is O(1), not O(1e-12).

    The numerical oracle compares with an absolute tolerance around 1e-9; if this kernel
    produced values near that floor, a backend that wrote zeros would pass.
    """
    mod = _load("petsc_ilu_trisolve")
    kernel = _load("petsc_ilu_trisolve_numpy").petsc_ilu_trisolve
    args = mod.initialize(7, 5, 6)
    kernel(*args)
    x = args[-1]
    assert np.all(np.isfinite(x))
    assert np.abs(x).min() > 1e-3, f"smallest |x| = {np.abs(x).min():.3e} is too close to the floor"
