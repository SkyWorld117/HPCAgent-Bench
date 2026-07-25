# Deep bloat/readability review -- CLOSED

**RE-VERIFIED 2026-07-25:** commits `59c85911` and `21f404c3` both present on `main`. The
deliberately-not-done item below (the two `_fortran_type(dt).startswith("integer")` sites in
`numpyto_fortran/emit.py`) is still exactly as described (`emit.py:2152`, `:2526`), still
correctly left alone -- no regression, nothing to re-investigate.

All 80 items are applied (2026-07-22). The last 35 landed in two commits:

* `59c85911` -- 18 items (redundant imports, two single-caller abstractions inlined, the
  underscore-prefixed module-level names the repo convention bans, `LEVELS` backing both the
  validator and the `@lvl<n>` selector parser, two comment blocks restating their own docstring).
* `21f404c3` -- the remaining 17 (`combine_grades` / `binding_shapes` / `implausible_speedup` /
  `feedback_source` / `subst_map` / `LINK_LANG_ORDER`, the dead `SuiteScore.verified_count`, the
  double query parse in `do_GET`, harbor's single-use `_mpi_distribution_json`, `from __future__
  import annotations` in `api.py`, and the stale `spec.py` module docstring).

## Deliberately NOT done

**`numpyto_fortran/emit.py`'s two `_fortran_type(dt).startswith("integer")` sites.** The review
called them a second spelling of `dtypes.is_integer`. They are not -- see `04-low-and-cleanup.md`
section B. fp8 is a float dtype emitted as `integer(c_int8_t)`, so the two predicates disagree on
exactly the five fp8 tokens, and those sites ask what Fortran *emits*.
