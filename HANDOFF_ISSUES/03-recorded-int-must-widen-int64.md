# 03 -- An integer local bit-mixed with int64 must be declared int64, not its recorded int32 kind

**STATUS: FIXED** (verified 2026-07-22 -- reproducer green). Kept for the invariant only.

**RE-VERIFIED 2026-07-25:** the widen (`if rec.startswith("integer") and name in int64_uses: return
int64_kind`) is present verbatim in `numpyto_fortran/emit.py::_collect_implicit_locals._classify`
(current line ~2586). Re-ran the reproducer below (dropped into the tests dir with `_op_oracle` on
path, then removed) against current HEAD -- 1 passed, `fortran == "ok"`. Settled.

**Severity:** Low-Medium (latent gfortran `-std=f2018` kind clash; invariant to
preserve when you change `local_dtypes` tagging or the hoister).

**Affected file:** `numpyto_fortran/emit.py` (`_collect_implicit_locals._classify`).
Fix is in the working tree; this documents the invariant so future `local_dtypes`
/ hoister edits keep it.

## Symptom

`_classify` returns the exact recorded integer kind for a tagged local. A temp
recorded `int32` that then participates in a bitwise / kind context with an
`int64` operand needs to be declared `int64`, or gfortran rejects the mismatch:

```
f.f90:11:53:
   11 |  x_cb1 = merge(INT(a((x_r0)+1), c_int64_t), x_cb1, ...)
      |                                             1
Error: 'fsource' argument of 'merge' intrinsic at (1) must be the same type and kind as 'tsource'
```

Here `x_cb1` is the int32 accumulator; the element is promoted to `c_int64_t`
because `a` flows into an int64 context, so `merge(int64, int32)` clashes.

## Fix (in tree)

In `_classify`, when the recorded dtype is integer, widen to the int64 kind if the
name is in the int64 fixed-point set -- mirroring what the non-recorded integer path
already does:

```python
rec = recorded_ftype.get(name)
if rec is not None:
    if rec.startswith("integer") and name in int64_uses:
        return int64_kind          # widen: bitwise/kind source demands int64
    return rec
```

**Invariant for your changes:** a recorded integer local is *at least* as wide as
whatever the `int64_uses` propagation says it meets. If you retag temps or change
the hoister so an int-tagged temp participates in int64 ops, keep this widen (or
tag the temp int64 directly).

## Reproducer

```python
import numpy as np
from _op_oracle import run_op


def test_fortran_int32_temp_widens_when_bitmixed_with_int64():
    """An int32-recorded reduction temp XORed with an int64 value must be declared
    int64, else gfortran rejects the merge/IEOR kind mismatch."""
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    src = (
        "import numpy as np\n"
        "def f(a, out):\n"
        "    t = int(np.max(a))\n"          # int32-tagged temp
        "    out[0] = t ^ np.int64(255)\n"  # bit-mixed with int64
    )
    res = run_op(src, "f", {"a": a}, {"out": (1,)}, {"N": 4},
                 shapes={"a": "(N,)", "out": "(1,)"},
                 dtypes={"a": "int32", "out": "int64"},
                 backends=("c", "cpp", "fortran"))
    assert res.get("fortran") == "ok", res   # FAIL:compile (merge kind mismatch) without the widen
```

**Verified:** green with the widen; `fortran: FAIL:compile` (exact `merge` error
above) when the widen is disabled -- so this is a genuine regression guard, not a
vacuous pass.
