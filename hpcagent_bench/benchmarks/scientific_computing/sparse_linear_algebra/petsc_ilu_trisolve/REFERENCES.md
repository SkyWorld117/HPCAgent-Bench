<!--
Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
SPDX-License-Identifier: GPL-3.0-or-later
-->
# `petsc_ilu_trisolve` -- provenance, measurement, and fidelity

The kernel is **PETSc's `MatSolve_SeqAIJ`**: the sparse triangular solve applied once per
Krylov iteration by `PCApply` when the preconditioner is `-pc_type ilu`. It walks a factor in
which L and U share one array, U's rows are stored back to front, the diagonal is stored as
its reciprocal, and both a row and a column permutation are folded into the traversal.

## Provenance & licensing

- **Upstream**: PETSc (Portable, Extensible Toolkit for Scientific Computation),
  `https://gitlab.com/petsc/petsc.git`, docs at `https://petsc.org/release/`.
- **Version**: **v3.25.4**, `release` branch, commit
  **`ab79ce791859c84e6d524d329ee8a429d6dc37ad`** (`git describe`: `v3.25.4-22-gab79ce79185`).
- **Ported file**: `src/mat/impls/aij/seq/aijfact.c`, lines 2466-2519 (`MatSolve_SeqAIJ`),
  together with the `PetscSparseDenseMinusDot` macro from `src/mat/impls/aij/seq/aij.h`
  lines 541-547, in terms of which the routine's inner loops are written.
- **Licence**: PETSc is **BSD-2-Clause**. `petsc_ilu_trisolve_reference.c` is the frozen
  upstream excerpt and keeps PETSc's own notice verbatim -- it is provenance, not the oracle,
  and must not be relicensed or reformatted. Everything else here is original HPCAgent-Bench
  work under **GPL-3.0-or-later**.

## How the kernel was chosen

Built optimized on Alps/Daint (GH200, aarch64) inside `uenv prgenv-gnu/25.6:v2`:

```
./configure PETSC_ARCH=arch-gh200-opt \
    --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
    --with-debugging=0 --with-cuda=0 \
    --with-blaslapack-lib=<uenv>/openblas-0.3.29-.../lib/libopenblas.so \
    COPTFLAGS='-O3 -g -ffp-contract=off' \
    CXXOPTFLAGS='-O3 -g -ffp-contract=off' \
    FOPTFLAGS='-O3 -g -ffp-contract=off'
```

`petscconf.h` for that arch: real scalars, double precision, **32-bit `PetscInt`**, no
`PETSC_USE_DEBUG`, no `PETSC_HAVE_OPENMP`, and neither `PETSC_KERNEL_USE_UNROLL_4` nor
`_UNROLL_2` -- so `PetscSparseDenseMinusDot` compiled to its plain `#else` gather loop, which
is the form reproduced in the port.

Driver and full option string (`src/ksp/ksp/tutorials/ex45`, 3-D Poisson on a `DMDA`
`DMDA_STENCIL_STAR` grid with one degree of freedom):

```
./ex45 -da_grid_x 321 -da_grid_y 321 -da_grid_z 321 \
       -pc_type ilu -pc_factor_mat_ordering_type rcm \
       -ksp_rtol 1e-12 -ksp_converged_reason -log_view
```

`-pc_factor_mat_ordering_type rcm` is load-bearing, not decoration. `MatLUFactorNumeric_SeqAIJ`
ends by choosing the solve implementation three ways:

```c
if (b->inode.size_csr)              C->ops->solve = MatSolve_SeqAIJ_Inode;
else if (row_identity && col_identity) C->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
else                                C->ops->solve = MatSolve_SeqAIJ;
```

Under PETSc's default ILU ordering (`natural`) both permutations are the identity and the
dispatch lands on `MatSolve_SeqAIJ_NaturalOrdering`, a reduced variant with no permutation
and no scratch vector. RCM makes both permutations non-trivial, which is what selects the
general routine -- the one that is worth porting, and the one whose permutation handling is
the part most easily got wrong. The inode branch is never taken here: a one-dof star stencil
gives inodes of size 1, and PETSc says so itself
(`MatSeqAIJCheckInode(): Found 4913 nodes out of 4913 rows. Not using Inode routines`).

## What the measurement said

33,076,161 unknowns; **1694 iterations, `CONVERGED_RTOL`**, residual 2.92e-12 (a real
convergence, not the `-ksp_max_it` cap). Wall clock **1497.0 s untraced**, 1406.6 s under
`-log_view` -- the profiler is not what makes this run long.

| `-log_view` event | calls | time (s) | %T | Mflop/s |
|---|---:|---:|---:|---:|
| `KSPSolve` | 1 | 1391.0 | 99 | 3777 |
| `KSPGMRESOrthog` (composite) | 1694 | 563.9 | 40 | 6134 |
| **`MatSolve`** | **1751** | **538.3** | **38** | **1395** |
| `PCApply` | 1751 | 538.3 | 38 | 1395 |
| `VecMAXPY` | 1751 | 365.1 | 26 | 5045 |
| `MatMult` | 1751 | 259.1 | 18 | 2898 |
| `VecMDot` | 1694 | 220.5 | 16 | 7844 |
| `PCSetUp` | 1 | 7.3 | 1 | 98 |
| `MatILUFactorSym` | 1 | 2.9 | 0 | 0 |
| `MatLUFactorNum` | 1 | 1.7 | 0 | 425 |

`MatSolve` is the largest single **leaf** event, and `PCApply` is the same 38% because
`PCApply_ILU` is nothing but this solve. `KSPGMRESOrthog` reads higher at 40%, but it is a
composite of `VecMDot` + `VecMAXPY` (16 + 26), and `VecMDot` resolves to BLAS `dgemv` under
the default `-vec_mdot_use_gemv`. This is also why the table must be read by `%T` and not by
`Mflop/s`: `MatSolve` has the *worst* flop rate in the table (1395) and the largest share of
the time.

`perf` (781K samples of `cycles:u`, `--call-graph=dwarf`, BLAS threads pinned to 1) agrees on
the leaf ranking and names the C function:

| self | symbol | object |
|---:|---|---|
| **33.43%** | **`MatSolve_SeqAIJ`** | `libpetsc.so.3.25.4` |
| 23.53% | `dgemv_t_NEOVERSEV1` | `libopenblasp-r0.3.29.so` |
| 22.82% | `VecMAXPY_Seq` | `libpetsc.so.3.25.4` |
| 16.21% | `MatMult_SeqAIJ` | `libpetsc.so.3.25.4` |
| 0.10% | `MatLUFactorNumeric_SeqAIJ` | `libpetsc.so.3.25.4` |

with the call chain `MatSolve_SeqAIJ <- MatSolve <- PCApply_ILU <- PCApply <- PCApplyBAorAB
<- KSPGMRESCycle <- KSPSolve_GMRES <- KSPSolve <- main`. So the top three are: PETSc's own
irregular solve, a **vendor BLAS** call, and Krylov vector arithmetic whose context `gmres`
already covers in the corpus. The first is the only one worth porting.

The hand-off expected `MatLUFactorNumeric_SeqAIJ` to be a top event as well. **It is not**:
one call against `MatSolve`'s 1751, 1.68 s, 0% of wall. Under a driver that re-factors (a
nonlinear or time-stepping run, or `-ksp_reuse_preconditioner false` with changing
coefficients) it would matter; under a single `KSPSolve` it does not, so this branch ports
the solve only.

### CPU vs GPU

Not decided by measurement here: the assignment fixed a **CPU-only build** (`--with-cuda=0`),
so there is no GPU busy fraction to report, and the run above uses no device at all. That is
the one item of the skill's checklist left unmeasured, and the reason is a decision taken
upstream of the work rather than a gap.

It does not change the answer. On the `aijcusparse` path `MatLUFactorNumeric` sets
`fact->ops->solve = MatSolve_SeqAIJCUSPARSE_LU`
(`src/mat/impls/aij/seq/seqcusparse/aijcusparse.cu`), which is a wrapper around
`cusparseSpSV_solve` / `cusparseXcsrsv_*` -- i.e. **cuSPARSE**, which the boundary rule puts
out of scope. A GPU-bound measurement of this event would therefore have sent the port back
to PETSc's CPU implementation of the same operation, which is what was ported.

## Preset timings

Measured with `scripts/run_benchmark.py -b petsc_ilu_trisolve -f numpy -p <preset>` on a
compute node (median of the timed repetitions):

| preset | NX x NY x NZ | rows | nonzeros | median |
|---|---|---:|---:|---:|
| S | 82 x 85 x 89 | 620,330 | 4,298,644 | **1.310 s** |
| M | 95 x 98 x 101 | 940,310 | 6,524,564 | 2.031 s |
| L | 106 x 109 x 111 | 1,282,494 | 8,906,620 | 2.702 s |
| XL | 117 x 119 x 123 | 1,712,529 | 11,901,801 | **3.618 s** |

Sized by the clock at both ends, as the skill requires, not by the 4 GB memory floor -- XL's
buffers are about 273 MB. A size chosen to reach 4 GB would run for roughly a minute under
the NumPy reference and make a corpus-wide sweep unusable.

## Fidelity: checked against PETSc itself, not against a transcription

`petsc_ilu_trisolve_petsc_dump.txt` holds the index arrays, factor values, both permutations
and `MatSolve_SeqAIJ`'s own output vector, read out of a live factored `Mat` (see that file's
header for the exact call sequence). On that data, for **both** the `natural` and `rcm`
orderings:

- `initialize`'s ILU(0) reproduces PETSc's `l_rowptr`, `u_diagptr`, `fact_cols` and
  `fact_values` **exactly** (`assert_array_equal`, difference 0.0);
- the kernel reproduces PETSc's solution vector **exactly**.

Independently, `tests/test_ported_references.py::test_petsc_ilu_trisolve_matches_reference`
unpacks the factor into dense `L`/`U` -- inverting the reciprocal diagonal on the way -- and
solves with `scipy.linalg.solve_triangular`, sharing no index arithmetic with the kernel.

## Parallelism

**Serial, deliberately.** `src/mat/impls/aij/seq/aijfact.c` contains **no OpenMP pragma at
all**; PETSc's `PetscPragmaUseOMPKernels(parallel for)` appears on the SpMV row loops in
`aij.c` (`MatMult_SeqAIJ`, `MatMultAdd_SeqAIJ`) and `inode.c` (`MatMult_SeqAIJ_Inode`), never
on the factorization or the solve. Both sweeps here are dependence chains -- forward reads
`solve_work[j]` for `j < i`, backward for `j > i` -- and that dependence is the kernel.

Confirmed three ways rather than assumed: `parallelism.py` reports
`loop_is_parallel_safe=False, loop_reduction=None` for all four loops; the emitted OpenMP C
(`emit_c_omp`) contains **no `#pragma omp` line**; and numba reports
`parallel=True was specified but no transformation for parallel execution was possible`.

## Sizing

The manifest's size symbols are the grid extents `NX/NY/NZ`, deliberately non-cubic and
mutually distinct so a transposed axis cannot hide. Every buffer shape is an exact closed
form in them -- an `NX x NY x NZ` star stencil has exactly
`7*NX*NY*NZ - 2*(NY*NZ + NX*NZ + NX*NY)` entries and ILU(0) preserves that count -- so the
oracle's proportional down-scaling stays self-consistent and no `NO_SCALE` entry is needed.
