"""A config knob the manifest PINNED to one value is emitted as a constant, not an argument.

``config: {max_iter: {value: 100}}`` says the knob has that value for every preset and every fuzz
draw. Carrying it across the ABI spells a compile-time constant as a runtime argument: the loop
bound, the stride and the padding are all knowable while the kernel compiles, and only a constant
lets the compiler unroll on them. So it is declared -- BY NAME, so the emitted code still reads
like the reference -- as a C ``constexpr`` / Fortran ``parameter``, and leaves ``param_order()``,
both binding JSONs and ``binding_from_spec``. A knob with a ``domain:`` is a real axis and stays a
parameter; see tests/test_spec_dimensions_config.py for that half.
"""
import json
import pathlib
import tempfile

from numpyto_c.emit import emit_c, emit_cpp
from numpyto_common.frontend import parse_kernel
from numpyto_common.lowering import lower
from numpyto_fortran.emit import emit_fortran

_SRC = ("import numpy as np\n"
        "def f(x, max_iter, tol, out):\n"
        " out[:] = x\n"
        " for _ in range(max_iter):\n"
        "  out[:] = out * 0.5\n"
        "  if np.max(np.abs(out)) < tol:\n"
        "   break\n")


def _kir(pinned=True):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(_SRC)
    bench = {
        "name": "k",
        "short_name": "k",
        "relative_path": "",
        "module_name": "k",
        "func_name": "f",
        "parameters": {
            "S": {
                "n": 8,
                "max_iter": 100,
                "tol": 1.0e-06
            }
        },
        "input_args": ["x", "max_iter", "tol", "out"],
        "array_args": ["x", "out"],
        "output_args": ["out"],
        "init": {
            "shapes": {
                "x": "(n,)",
                "out": "(n,)"
            }
        },
    }
    if pinned:
        bench["pinned_config"] = {"max_iter": 100, "tol": 1.0e-06}
    (d / "bi.json").write_text(json.dumps({"benchmark": bench}))
    return lower(parse_kernel(d / "k_numpy.py", d / "bi.json"))


def test_pinned_knobs_leave_the_abi_and_are_declared_as_constants():
    kir = _kir()
    assert kir.pinned_consts == {"max_iter": 100, "tol": 1.0e-06}
    # Sec. 4 order over what REMAINS: pointers by name, then the size symbol.
    assert kir.param_order() == ["out", "x", "n"]
    c = emit_c(kir, fn_name="f")
    assert "constexpr int64_t max_iter = 100;" in c
    assert "constexpr double tol = 1e-06;" in c
    assert "void f(double *restrict out, const double *restrict x, const int64_t n)" in c
    assert "max_iter" not in c.split("void f(", 1)[1].split("{", 1)[0], "the knob must not be a parameter"
    assert "constexpr int64_t max_iter = 100;" in emit_cpp(kir, fn_name="f")
    f90 = emit_fortran(kir, fn_name="f")
    assert "integer(c_int64_t), parameter :: max_iter = 100_8" in f90
    assert "real(c_double), parameter :: tol = 1e-06_8" in f90
    assert "subroutine f(out, x, n)" in f90


def test_without_the_pinned_declaration_the_same_knobs_stay_parameters():
    # The control: identical source and identical `parameters`, only the manifest's `config:` block
    # differs. Without it the knobs are ordinary by-value scalars, which is what a `domain:` knob
    # and every legacy manifest keep.
    kir = _kir(pinned=False)
    assert kir.pinned_consts == {}
    assert "max_iter" in kir.param_order() and "tol" in kir.param_order()
    assert "constexpr int64_t max_iter" not in emit_c(kir, fn_name="f")
