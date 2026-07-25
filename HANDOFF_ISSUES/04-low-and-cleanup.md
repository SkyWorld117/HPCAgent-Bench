# 04 -- Low-severity nits + reuse / weight cleanups

Advisory. No shipping bug; grouped so nothing is dropped.

## A. nest-forge scatter-conflict emitter (repo: `../nest-forge`, already pushed)

File `nestforge/emit_libnode.py`, `emit_scatter_conflict_check`.

1. **Negative index silently wraps.** `owner[idx[i]]` in the emitted numpy reference
   uses numpy negative indexing for a negative index value (wrap-around -> wrong
   duplicate count), while the translated C reads out of bounds. Scatter indices are
   non-negative by construction, so it is latent, but the two backends diverge on an
   invalid input with no guard/assert.
2. **`next(...)` without default.** `tag = next(e for e in state.out_edges(node) if
   e.src_conn == "_count_out")` raises a bare `StopIteration` (not
   `UnsupportedLibraryNode`) if a malformed node lacks the count-out connector -- the
   error escapes `state_body`'s `UnsupportedLibraryNode -> UnsupportedNest` mapping.

No unit-test reproducer (both need a deliberately malformed node); fix is a guard +
`next(..., None)` with an explicit raise.

**RE-CHECKED 2026-07-25: STILL PRESENT, NOT FIXED.** `nest-forge/nestforge/emit_libnode.py`
still emits `{owner}[{idx}[{i}]] = {i}` (plain numpy negative-indexing subscript, line ~417) and
`tag = next(e for e in state.out_edges(node) if e.src_conn == "_count_out")` with no default
(line 416). Both nits as originally described. Out of this repo's scope to fix (nest-forge-owned,
`../nest-forge`) -- still advisory.

## B. hpcagent_bench reuse -- shared `is_integer(dtype)` helper

**STATUS: DONE, PARTIALLY -- and the "two spellings of the same predicate" premise is WRONG.**

`dtypes.is_integer(dtype)` now exists and backs the `lowering.py` sites (2026-07-22, commit
d16a125d). The two `_fortran_type(dt).startswith("integer")` sites in `numpyto_fortran/emit.py`
were **deliberately left alone**: the predicates were compared over all 26 registry keys, aliases
and scalar-kind spellings and they disagree on the five fp8 tokens. `float8_e4m3` / `float8_e5m2`
(plus the `fp8_*` and `float8_e4m3fn` aliases) are FLOAT dtypes stored as `integer(c_int8_t)`, so
`is_integer` is False while `_fortran_type(...).startswith("integer")` is True.

Those two sites ask what Fortran **emits**, not what the dtype **is** -- an fp8 array is a Fortran
integer array and needs exactly the same `/= 0` condition-site treatment as a genuinely integer one.
Converting them would silently drop every fp8 array out of `_int_array_names`. Do not "finish" this
item; the reason is recorded inline at both sites.

**RE-CHECKED 2026-07-25:** both sites still present, unconverted, exactly as described
(`emit.py:2152` and `:2526`). `dtypes.is_integer` exists and backs the `lowering.py` call sites.
No regression, no action needed -- correctly left alone.

## C. hpcagent_bench weight -- `_collect_implicit_locals`

`numpyto_fortran/emit.py::_collect_implicit_locals` is ~270 lines mixing the
`local_dtypes` classification, the int64 bitwise fixed-point propagation, the
real-assignment detection, and the classify tower. The parallel session already
**folded the four separate `local_dtypes` walks into one pass** (build
`recorded_ftype` + `complex_names` + `recorded_int64_local` + `recorded_real_local`
once; the later seeds became set-unions). Further extraction of the dtype-tag
classification into a `numpyto_common` helper would let the C and Fortran emitters
share one source of truth. Optional, no behaviour change.

**RE-CHECKED 2026-07-25: the single-pass fold is CONFIRMED DONE and still in place**
(`emit.py:2390-2413`: `recorded_ftype`, `complex_names`, `recorded_int64_local`,
`recorded_real_local` all built in one walk over `kir.local_dtypes`). The `_collect_implicit_locals`
function itself is still ~257 lines. The further `numpyto_common` extraction was, and remains,
not done -- optional, unchanged status.
