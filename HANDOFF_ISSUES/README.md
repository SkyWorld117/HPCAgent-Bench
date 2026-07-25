# Fortran-emit handoff issues (numpy translators)

**BACKLOG SWEEP 2026-07-25: this whole directory re-verified against current `main`.** Tally:
01, 02, 03, 05, and all 6 kernels in `DACE_SDFG_FAILURES.md` = **FIXED, re-confirmed by actually
running each reproducer** (not just re-reading the old status). 04 (advisory) = still accurate,
nothing regressed, nothing newly fixed except what was already marked. 06 (GPU translator
targets) = still fully open, zero implementation found. `HANDOVER_translator_followups_20260712.md`
= 3 of 7 sections done (pluto gate, docs, + partial argmax/argmin), 3 open (JAX H1/H2,
`skip:too-long` enumeration, dedup decisions), 1 not independently checkable. `bloat-review-backlog.md`
= closed, unchanged. See each file's own top-of-section STATUS line for the evidence. One new,
out-of-scope-for-this-backlog finding surfaced during re-verification: an `autoopt`-pipeline
failure on `correlation`/`durbin` (see `DACE_SDFG_FAILURES.md`) -- not a numpy->dace emitter bug,
reported there for visibility, not fixed.

Context for the concurrent session that owns
`hpcagent_bench/numpy_translators/src/numpyto_common/{lib_nodes.py,numpy_desugar.py}`.

A parallel session hardened the **Fortran scalar-local dtype** path in
`numpyto_fortran/emit.py` and `numpyto_common/lowering.py` (both clean, sitting
uncommitted in the working tree). Those fixes are **entangled** with your live
`lib_nodes.py` / `numpy_desugar.py` WIP and cannot be committed independently --
they depend on the `lib_nodes.py` `_CallHoister` int-inherit (see Issue 01).
This directory lists every verified error with a runnable reproducer so you can
carry the invariants forward when you land your WIP.

## How to run a reproducer

Each reproducer is a pytest that uses the existing oracle harness
`hpcagent_bench/numpy_translators/tests/_op_oracle.py::run_op` (compiles the emitted
C / C++ / Fortran, runs it forked, compares bit-exact vs numpy). Drop the
function into a file under `hpcagent_bench/numpy_translators/tests/` (the `conftest.py`
there puts `_op_oracle` on the path) and run:

```
cd hpcagent_bench
rm -rf .dacecache
python -m pytest hpcagent_bench/numpy_translators/tests/<file>.py -q -p no:cacheprovider -n2
```

Use `-n2` for native-compile batches -- `-n4` OOMs a 12 GB box (gfortran/g++/pythran
fork per kernel). `run_op` returns `{backend: "ok" | "skip:<why>" | "FAIL:<why>"}`;
`fortran` legitimately `skip`s when `gfortran` is absent (that skip is the subject
of Issue 02).

## Issues

| # | Severity | Title | Repro | Status (2026-07-25) |
|---|----------|-------|-------|--------|
| [01](01-fortran-int-array-reduction-miscompile.md) | High | Int reduction gets a `real` accumulator -> gfortran `merge` kind mismatch | yes (green with fix, `FAIL:compile` without) | **FIXED**, committed on `main`, re-run green |
| [02](02-fortran-skip-test-coverage-gap.md) | Medium | `_ok` accepts a fortran skip -> int-reduction guard unverified on gfortran-less CI | yes (behavioural) | **FIXED**, commit `d16a125d`, re-run green |
| [03](03-recorded-int-must-widen-int64.md) | Low-Med | Integer local bit-mixed with int64 must be declared int64, not its recorded int32 kind | yes (green with fix, `FAIL:compile` without) | **FIXED**, committed on `main`, re-run green |
| [04](04-low-and-cleanup.md) | Low | nest-forge scatter-emitter nits + reuse/weight cleanups | partial | advisory, unchanged (A still open in nest-forge, B/C re-confirmed correct-as-is) |
| [05](05-numpyto-dace-corpus-lowering-failures.md) | High | numpy->**dace** (`numpyto_c/dace_emit.py`) corpus lowering: 4 original bugs (nested-ternary, `np_float` dtype, reduction symbol/descriptor clash, `for x in array` iterator) **FIXED + unit-tested**; 3 deeper dace-broadcast/SSA residuals remain (mandelbrot1, nbody, contour_integral) | yes (residuals, one script) | **ALL FIXED** (4 original + 3 residuals), re-run green, 288 dace-emit tests pass |
| [06](06-gpu-translator-targets.md) | -- | numpy->X GPU translator targets (design/requirements doc, no bug) | n/a (design doc) | **STILL OPEN**, zero implementation found |

Also swept `DACE_SDFG_FAILURES.md` (all 6 kernels now fixed, seissol_batched_gemm's the last one
to close), `HANDOVER_translator_followups_20260712.md` (pluto gate + docs done; JAX H1/H2, the
`skip:too-long` enumeration, and the dedup decisions still open), and `bloat-review-backlog.md`
(closed, unchanged) -- see each file's own top for the dated re-verification note.

All three fortran reproducers (01, 03 in full; 01's `merge` error and 03's were captured
directly by toggling the fix off) were verified against the current tree. Issue 05 covers the
numpy->dace target (`numpyto_c/dace_emit.py::emit_dace`), verified via `Kernel.to_sdfg` -- distinct
from the fortran issues above. Issue 05, 2026-07-14: all four original bugs are fixed in
`dace_emit.py` (committed, guarded in `test_dace_emit.py`); `nussinov` is fully green, and three
deeper broadcast/SSA-frontend residuals (Bug E/F/G) remain -- see the doc's status table.
