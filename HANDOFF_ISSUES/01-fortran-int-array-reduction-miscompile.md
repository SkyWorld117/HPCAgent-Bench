# 01 -- Fortran int-array reduction gets a `real` accumulator -> gfortran `merge` kind mismatch

**STATUS: FIXED** (verified 2026-07-22 -- the reproducer below now compiles and runs green on
gfortran). Kept for the invariant, which the emitter must preserve: a value-preserving reduction
(max/min/sum/prod) over an int array declares an INTEGER accumulator. Guarded by
`test_libnode_emit_ops_gaps.py::test_scatter_conflict_check_tagcount`, which since issue 02 asserts
Fortran actually ran rather than accepting a skip.

**RE-VERIFIED 2026-07-25:** all three parts are committed on `main` (the `lib_nodes.py`
`_CallHoister` int-inherit landed as far back as commit `f376a43f`, 2026-07-16; the
`dtypes.is_integer` gate on it via `d16a125d`, 2026-07-22). Re-ran
`test_libnode_emit_ops_gaps.py::test_scatter_conflict_check_tagcount` against current HEAD with
`gfortran` present on `PATH` -- 6 passed, `fortran == "ok"` asserted and true. No re-investigation
needed.

**Severity:** High (Fortran backend fails to compile; C / C++ silently promote, so
the bug is invisible unless gfortran is exercised).

**Affected files**
- `numpyto_common/lowering.py` -- part 1 (in working tree, clean).
- `numpyto_common/lib_nodes.py` -- **part 2 (YOUR file -- must be preserved when you land WIP).**
- `numpyto_fortran/emit.py` -- part 3 (in working tree, clean).

## Symptom

A value-preserving reduction (`np.max` / `np.min` / `np.sum` / `np.prod`, including
via `int(np.max(int_array))`) over an **integer** array declares its running
accumulator as the emitter's default scalar type `real(c_double)`. The Fortran
running-max/min update then mixes an integer element with a real accumulator:

```
f.f90:13:39:
   13 |  x_cb1 = merge(idx((x_r0)+1), x_cb1, ((idx((x_r0)+1) > x_cb1) .OR. (idx((x_r0)+1) /= idx((x_r0)+1))))
      |                                    1
Error: 'fsource' argument of 'merge' intrinsic at (1) must be the same type and kind as 'tsource'
```

C and C++ promote the int to double silently, so only gfortran flags it -- a
correctness divergence, not just a build break.

## Root cause + fix (all three parts required)

1. **`lowering.py` (`_lp_libnode_expand`)** -- seed the int/uint *element* dtype of
   every kernel-parameter array into `ctx.local_dtypes`, so the hoister can see
   that e.g. `idx` in `int(np.max(idx))` is `int64`. Only int/uint are seeded
   (float would flip untagged-default paths; complex has its own pass); params are
   declared from the ABI signature, so tagging never re-declares them.

2. **`lib_nodes.py` (`_CallHoister`, scalar branch) -- YOUR CODE** -- when hoisting a
   value-preserving reduction `{max, min, sum, prod}` whose operand is an
   int-tagged `Name`, inherit the int dtype onto the **reduction temp** and record
   it in `local_dtypes`. This is the load-bearing link: without it the temp carries
   no dtype tag and part 3 has nothing to key on.

3. **`emit.py` (`_collect_implicit_locals`)** -- declare a scalar local integer when
   `local_dtypes` tags it integer (now via the unified `recorded_ftype` map;
   exact int32/int64 kind from the registry, never a literal).

## Reproducer

```python
import numpy as np
from _op_oracle import run_op


def test_fortran_int_reduction_accumulator():
    """int(np.max(int64_array)) must declare an INTEGER accumulator; a real one
    makes the Fortran running-max `merge(int, real)` a kind mismatch."""
    idx = np.array([0, 2, 2, 5, 5, 5, 1, 9, 9], dtype=np.int64)
    src = (
        "import numpy as np\n"
        "def f(idx, cnt):\n"
        "    m = int(np.max(idx))\n"
        "    owner = np.full(m + 1, -1, np.int64)\n"
        "    for i in range(idx.shape[0]):\n"
        "        owner[idx[i]] = i\n"
        "    c = 0\n"
        "    for i in range(idx.shape[0]):\n"
        "        if owner[idx[i]] != i:\n"
        "            c += 1\n"
        "    cnt[0] = c\n"
    )
    res = run_op(src, "f", {"idx": idx}, {"cnt": (1,)}, {"N": 9},
                 shapes={"idx": "(N,)", "cnt": "(1,)"},
                 dtypes={"idx": "int64", "cnt": "int64"},
                 backends=("c", "cpp", "fortran"))
    assert res.get("fortran") == "ok", res   # FAIL:compile (merge kind mismatch) before the fix
```

**Verified:** green with the 3-part fix in the tree; `fortran: FAIL:compile` (the
exact `merge` error above) when any part is removed. The scatter form above is also
the shipped `test_libnode_emit_ops_gaps.py::test_scatter_conflict_check_tagcount`.
