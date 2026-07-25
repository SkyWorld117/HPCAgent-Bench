# 02 -- `_ok` accepts a fortran skip -> int-reduction guard unverified on gfortran-less CI

**STATUS: FIXED** (2026-07-22, commit d16a125d). `_ok` now accepts exactly one skip reason from a
first-party backend -- `skip:no-compiler`. c/cpp/fortran are OUR emitters, so `skip:unsupported:*`
from one of them is a translator gap, not an environment fact, and used to read as a pass.
`test_scatter_conflict_check_tagcount` additionally asserts `fortran == "ok"` wherever gfortran
exists, since that test's whole subject is the Fortran accumulator.

**RE-VERIFIED 2026-07-25:** commit `d16a125d` present on `main`; `test_libnode_emit_ops_gaps.py`
still contains the `shutil.which("gfortran")` -> `assert res.get("fortran") == "ok"` guard
(lines 250-251 of the current file) and it passed on a re-run with gfortran on PATH. Settled.

**Severity:** Medium (test-coverage hole -- a Fortran-emit regression ships green on
any machine without `gfortran`).

**Affected file:** `numpy_translators/tests/test_libnode_emit_ops_gaps.py`.

## Symptom

The suite's `_ok` helper accepts a result as passing when **every** backend is
`ok`-or-`skip` and **at least one** ran:

```python
def _ok(res):
    return (all(v == "ok" or v.startswith("skip") for v in res.values())
            and any(v == "ok" for v in res.values())), res
```

On a box without `gfortran`, `fortran` returns `skip:...`, `c`/`cpp` return `ok`,
so `_ok` passes -- even though `test_scatter_conflict_check_tagcount` exists *only*
to guard the Fortran int-reduction fix (Issue 01). The one backend the test is
about never compiled, and nothing flags it.

## Fix

For the int-reduction regression guard specifically, require Fortran to actually
run when `gfortran` is on `PATH`; skip *loudly* (recorded reason) only when it is
genuinely absent. This keeps portability (gfortran-less dev boxes still pass) while
making the guard mean something wherever gfortran exists -- including CI.

```python
import shutil
# inside the int-reduction guard, after run_op(...):
if shutil.which("gfortran"):
    assert res.get("fortran") == "ok", res      # must really compile+run where gfortran exists
ok, r = _ok(res)
assert ok, r
```

(Apply only to the int-reduction / dtype-sensitive guards; the general `_ok`
skip-tolerance is fine for ops that are not Fortran-specific.)

## Reproducer

Behavioural, not a green unit test: run the suite with `gfortran` removed from
`PATH` and confirm `test_scatter_conflict_check_tagcount` still passes while
`fortran` never compiled.

```
PATH="$(python - <<'PY'
import os
print(os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep)
                      if not os.path.exists(os.path.join(p, "gfortran"))))
PY
)" python -m pytest hpcagent_bench/numpy_translators/tests/test_libnode_emit_ops_gaps.py::test_scatter_conflict_check_tagcount -q -p no:cacheprovider
# -> passes today (fortran skipped); with the fix it xfails/skips loudly instead of silently passing
```
