# 05 -- numpy->dace translator: corpus lowering failures

**Translator:** `numpyto_c/dace_emit.py::emit_dace` (frontend `numpyto_common/frontend.py::parse_kernel`),
driven by `hpcagent_bench/autogen.py::_emit_dace`.
**Owner:** hpcagent_bench (numpy->X). nest-forge only imports the generated `*_dace.py` and calls
`Kernel.to_sdfg`; the failures are in the hpcagent_bench-generated source, before nest-forge touches anything.

## Status (2026-07-25): RE-VERIFIED, still all clear

Re-ran the exact repro script below against current HEAD (`emit_targets` + fresh `to_sdfg(simplify=True)`
for all three): `OK mandelbrot1`, `OK nbody`, `OK contour_integral`. Also re-ran
`numpy_translators/tests/test_dace_emit.py` in full: 288 passed (up from 259 at the 07-22 count --
more guards added since). No regression. Settled; nothing here needs re-investigation.

## Status (2026-07-22): ALL RESIDUALS CLEARED

Re-ran the reproducer below against the current tree: **mandelbrot1, nbody, contour_integral and
nussinov all lower** (`to_sdfg(simplify=True)`, dace 2.0.0a4). Bugs E and F needed no emitter
change -- the DaCe frontend now infers the broadcasts it used to reject, and the emitted source
still carries the original `X + Y[:, None]` / `x.T - x` forms. Bug G is gone at the emitter: the
functional->in-place conversion emits `X[:] = -X`, not a rebinding `X = -X`, so single-assignment
is never violated.

Nothing here is guarded by a test at the corpus level; `tests/test_dace_port_lowering.py` covers
the microapp ports, not these four. If they regress it will surface in nest-forge again.

## Status (2026-07-14, historical)

The **four originally-reported bugs are FIXED** in `numpyto_c/dace_emit.py` (committed; unit-tested in
`hpcagent_bench/numpy_translators/tests/test_dace_emit.py`). Each fix let `to_sdfg` parse further into its kernel;
three of the four kernels then hit a **new, deeper** dace-frontend limitation (mandelbrot1, nbody,
contour_integral). `nussinov` is now **fully green**.

| kernel | original bug (FIXED) | fix | residual (still red) |
|--------|----------------------|-----|----------------------|
| nussinov | `Add(Scalar, IfExp)` -- nested ternary as a value | `_DesugarTernary` now hoists nested ternaries to a guarded scalar temp | -- **fully green** |
| mandelbrot1 | `Use of undefined variable np_float` | `_RewriteFrameworkDtype` maps `np_float`/`np_complex` -> `dc_float`/`dc_complex_float` | Bug E (meshgrid broadcast) |
| nbody | `Cannot create symbol __rd0_d1 ... used by a data descriptor` | `_inline_transient_shape_scalars` inlines `__rd*_d* = <transient>.shape[k]` into its uses | Bug F (transpose-difference broadcast) |
| contour_integral | `Iterator of ast.For must be a function or a subscript` (`for z in int_pts`) | `_DesugarArrayIteration` rewrites to `for i in range(...): z = int_pts[i]` | Bug G (SSA reassignment) |

The four fixes are guarded by `test_nussinov_nested_ternary_hoisted_no_ifexp`,
`test_mandelbrot_no_leaked_framework_dtype_token`,
`test_nbody_reduction_shape_scalar_inlined_no_descriptor_symbol_clash`, and
`test_contour_integral_array_iteration_rewritten_to_indexed_range`.

## Why the self-check misses the residuals
`_emit_dace` runs only `ast.parse(src)` -- a syntactic self-check. Every residual below is syntactically
valid Python but semantically invalid dace, failing only at `to_sdfg()` (the in-memory SDFG build, which
is compile-free and cheap -- DaCe owns codegen). A stronger self-check would `to_sdfg()` a size-stamped copy
before writing.

## Repro (all three residuals)
```bash
cd /home/primrose/Work/hpcagent_bench
OMPI_MCA_pml=ob1 MPI4PY_RC_INITIALIZE=0 python3 - <<'PY'
import importlib.util, os
from hpcagent_bench.spec import BenchSpec
from hpcagent_bench.autogen import emit_targets
import hpcagent_bench.infrastructure.dace_framework as fw
from dace import float64, complex128
fw.dc_float, fw.dc_complex_float = float64, complex128
for name, rel, attr in [
    ("mandelbrot1", "hpc/map_reduce/mandelbrot1/mandelbrot1_dace.py", "mandelbrot"),
    ("nbody", "hpc/n_body_methods/nbody/nbody_dace.py", "nbody"),
    ("contour_integral", "hpc/dense_linear_algebra/contour_integral/contour_integral_dace.py", "contour_integral"),
]:
    emit_targets(BenchSpec.load(name), ["dace"])
    p = os.path.join("hpcagent_bench/benchmarks", rel)
    spec = importlib.util.spec_from_file_location("m", p); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    try:
        getattr(m, attr).to_sdfg(simplify=True); print("OK  ", name)
    except Exception as e:
        print("FAIL", name, "->", type(e).__name__, str(e).splitlines()[0][:90])
PY
```

---

## Bug E -- meshgrid add: dace can't broadcast a bare 1-D `X` against a column `Y[:, None]`
- **File:** `hpcagent_bench/benchmarks/hpc/map_reduce/mandelbrot1/mandelbrot1_dace.py:13-15`
- **Emitted:** `X = np.linspace(...).astype(dc_float)  # (xn,)` ... `C = X + Y[:, None] * 1j`
- **dace error:** `IndexError: operands could not be broadcast together with shapes (xn, 1), (yn, 1)`
- **Root cause:** numpy broadcasts `(xn,)` + `(yn,1)` to `(yn, xn)` by prepending a row axis to `X`; dace's
  stricter broadcaster promotes the bare `X` to a **column** `(xn, 1)` instead, which cannot broadcast
  against `(yn, 1)`. The numpy oracle is effectively `C = X[None, :] + Y[:, None] * 1j`.
- **Fix direction:** emit the row orientation explicitly -- `X[None, :]` -- when a 1-D array is added to a
  column-broadcast (`[:, None]`) array, rather than relying on numpy's implicit row-axis prepend.

## Bug F -- transpose-difference: dace can't infer the `(N, N)` shape of `X.T - X`
- **File:** `hpcagent_bench/benchmarks/hpc/n_body_methods/nbody/nbody_dace.py:23-30`
- **Emitted:** `__inl1_x = pos[:, 0:1]  # (N,1)` ... `__inl1_dx = __inl1_x.T - __inl1_x` ... then
  `for __mi0_1 in range(__inl1_inv_r3.shape[1]):`
- **dace error:** `IndexError: list index out of range` (`.shape[1]` on a transient dace resolved as < 2-D)
- **Root cause:** the pairwise outer difference `pos[:, k:k+1].T - pos[:, k:k+1]` broadcasts `(1, N)` -
  `(N, 1)` to `(N, N)` in numpy, but dace's frontend does not infer the 2-D result, so the transient is
  rank-1 and `.shape[1]` is out of range. Same broadcast-inference gap family as Bug E.
- **Fix direction:** make the outer-difference broadcast explicit / 2-D during translation, or teach the
  shape inference that `a.T - a` on a column vector yields an `(N, N)` transient.

## Bug G -- result variable reassigned (`X = solve(...)` then `X = -X`); dace is single-assignment
- **File:** `hpcagent_bench/benchmarks/hpc/dense_linear_algebra/contour_integral/contour_integral_dace.py:20-24`
- **Emitted:** `X = np.linalg.solve(Tz, Y)` ... `X = -X`
- **dace error:** `DaceSyntaxError: Cannot reassign value to variable "X"`
- **Root cause:** dace's frontend is single-assignment for a name bound to an array / library-call result;
  rebinding `X` (solve -> negate) is rejected.
- **Fix direction:** version the reassigned target during translation (`X_0 = solve(...)`, `X_1 = -X_0`,
  threading the latest version forward) whenever a name holding an array/library-call result is assigned
  more than once. See the related SSA-versioning design note.

---

## Not in this doc (nest-forge-side, fixed separately)
The functional->in-place conversions (`atax`->`out`, `azimint_hist`->`out`, `azimint_naive`->`res`,
`gramschmidt`->`Q`/`R`) and `tsvc_full` lane-timeout trims are nest-forge test-side updates, not hpcagent_bench
emitter bugs.
