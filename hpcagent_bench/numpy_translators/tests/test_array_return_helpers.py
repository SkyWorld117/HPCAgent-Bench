"""Array-returning non-inlinable helpers emitted as native out-param functions.

The scalar-return sibling (``test_helper_functions.py``) emits a helper with an
early ``return`` as a native function that returns by value. A helper that
returns a whole ARRAY cannot come back by value in C/Fortran, so it is emitted
with a trailing out-param the body writes into (``return fac`` ->
``__hret[:] = fac``); the call site materialises any slice arguments into
contiguous temps and stores the filled result. Config-flag arguments that are
compile-time constants at the call site are folded into the helper body and the
now-dead branches pruned -- so a QE-``g2_convolution``-style helper (whose vcut /
gamma branches carry un-lowerable tuples) reduces to its live path.
"""
from typing import Dict, Optional

import numpy as np

from _op_oracle import run_op

_ALL = ("c", "cpp", "fortran", "numba", "pythran", "jax")


def _all_ok(res):
    return all(v == "ok" or v.startswith("skip") for v in res.values()), res


def test_array_return_slice_target():
    # Early-return array helper, result stored into a row slice of the output.
    src = ("import numpy as np\n"
           "def clamp_row(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = clamp_row(x[i, :], thr)\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "thr": 0.5
               }, {"out": (4, 5)}, {
                   "M": 4,
                   "n": 5
               },
               shapes={
                   "x": "(M,n)",
                   "out": "(M,n)"
               },
               backends=_ALL))
    assert ok, res


def test_array_return_bare_target():
    # Whole-array target: the out-param is filled in place (no temp copy).
    src = ("import numpy as np\n"
           "def screen(v, s):\n"
           " if s > 0.0:\n"
           "  return v * s\n"
           " return v + 1.0\n"
           "def f(x, s, out):\n"
           " out[:] = screen(x, s)\n")
    x = np.linspace(-3.0, 3.0, 12).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "s": 2.0
               }, {"out": (12, )}, {"n": 12},
               shapes={
                   "x": "(n,)",
                   "out": "(n,)"
               },
               backends=_ALL))
    assert ok, res


def test_array_return_helper_pointer_params_sort_against_source_order():
    # Three same-typed pointers (zz, aa and the synthesized out buffer) whose ABI order
    # (__hret_0, aa, zz) is a non-trivial permutation of the source order. Transposing two of them
    # compiles and links clean in C, so only numerics can catch a definition/call-site drift; the
    # body is asymmetric in zz and aa so a swap changes the answer.
    src = ("import numpy as np\n"
           "def mix(zz, aa, s):\n"
           " if s > 0.0:\n"
           "  return zz * 2.0 + aa\n"
           " return zz - aa\n"
           "def f(x, y, s, out):\n"
           " out[:] = mix(x, y, s)\n")
    x = np.linspace(-3.0, 3.0, 12).astype(np.float64)
    y = np.linspace(4.0, -1.0, 12).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "y": y,
                   "s": 2.0
               }, {"out": (12, )}, {"n": 12},
               shapes={
                   "x": "(n,)",
                   "y": "(n,)",
                   "out": "(n,)"
               },
               backends=_ALL))
    assert ok, res


def test_array_return_specialized_config_flag():
    # A ``g2_convolution``-shaped helper: a config flag (``use_alt``) is a
    # compile-time ``False`` at the call site, so its early-return branch folds
    # away; a strided column arg ``xk[:, k]`` is materialised; the live path runs
    # a reduction + ``np.where`` and stores into a column slice.
    src = ("import numpy as np\n"
           "def gconv(g, xk, scale, use_alt):\n"
           " q = xk[:, None] + g\n"
           " qq = np.sum(q ** 2, axis=0)\n"
           " if use_alt:\n"
           "  return q[0, :] * 0.0 + 7.0\n"
           " nz = qq > 1e-08\n"
           " qn = np.where(nz, qq, 1.0)\n"
           " return np.where(nz, scale / qn, -1.0)\n"
           "def f(g, xk, scale, out):\n"
           " K = out.shape[1]\n"
           " for k in range(K):\n"
           "  out[:, k] = gconv(g, xk[:, k], scale, False)\n")
    rng = np.random.default_rng(0)
    ngm, K = 8, 3
    g = rng.standard_normal((3, ngm))
    xk = rng.standard_normal((3, K))
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "g": g,
                   "xk": xk,
                   "scale": 2.0
               }, {"out": (ngm, K)}, {
                   "ngm": ngm,
                   "K": K,
                   "three": 3
               },
               shapes={
                   "g": "(three,ngm)",
                   "xk": "(three,K)",
                   "out": "(ngm,K)"
               },
               backends=_ALL))
    assert ok, res


def test_array_return_helper_native_desugar_bug3():
    # BUG-3: a NON-inlined array-returning helper used to keep native constructs
    # the kernel body had already shed -- the desugars only ran on the kernel, not
    # on ``_build_helper_kirs`` bodies. This helper is non-inlinable (an early
    # ``if s < 0: return`` inside the body) and carries a ``.ndim`` validation
    # guard plus an ``np.newaxis``; both must be desugared away on the HELPER for
    # the native backends to emit. Before DI-2 the ``.ndim`` / ``newaxis`` reached
    # the emitter and it failed.
    src = ("import numpy as np\n"
           "def scale_row(v, s):\n"
           " if s < 0.0:\n"
           "  return -v\n"
           " if v.ndim != 1:\n"
           "  raise ValueError('expected 1-D input')\n"
           " w = v[:, np.newaxis]\n"
           " return w[:, 0] * s\n"
           "def f(x, s, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = scale_row(x[i, :], s)\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "s": 1.5
               }, {"out": (4, 5)}, {
                   "M": 4,
                   "n": 5
               },
               shapes={
                   "x": "(M,n)",
                   "out": "(M,n)"
               },
               backends=_ALL))
    assert ok, res


def test_array_helper_emitted_as_outparam_c_function():
    # Structural: the helper is a ``void`` C function with a trailing out-param,
    # and the call site is a SINGLE opaque call (not a per-element loop calling
    # the whole-array helper once per element).
    from numpyto_c.emit import emit_c
    src = ("import numpy as np\n"
           "def clamp_row(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " for i in range(x.shape[0]):\n"
           "  out[i, :] = clamp_row(x[i, :], thr)\n")
    kir = _helper_kir(src, shape="(M,n)", params={"M": 4, "n": 5})
    assert len(kir.helpers) == 1 and kir.helpers[0].return_kind == "__hret_0"
    c = emit_c(kir, fn_name="f")
    # Helper ABI == kernel ABI (abi_contract.md Sec. 4): pointers by name, then scalars by name,
    # the out buffer sorting like any other pointer (``__hret_0`` < ``v``).
    assert ("static void clamp_row(double *restrict __hret_0, const double *restrict v, "
            "const double lo, const int64_t n)") in c
    # a single call statement, not ``__hret_tmp_0[..] = clamp_row(..)`` per element
    assert "clamp_row(__hret_tmp_0, __harg_0_0, thr, n);" in c


def _helper_kir(src: str, precision: str = "", shape: str = "(n,)", params: Optional[Dict[str, int]] = None):
    """Lower ``src``'s one-array-in/one-array-out kernel ``f`` at ``precision`` (``""`` = fp64).

    ``x`` and ``out`` share ``shape``, which every array-helper case here does -- the helper is what
    is under test, not the ABI's shape handling.
    """
    import json
    import pathlib
    import tempfile
    from numpyto_common.frontend import parse_kernel
    from numpyto_common.ir import apply_precision
    from numpyto_common.lowering import lower
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "k_numpy.py").write_text(src)
    bi = {
        "benchmark": {
            "name": "k",
            "short_name": "k",
            "relative_path": "",
            "module_name": "k",
            "func_name": "f",
            "parameters": {
                "S": params if params is not None else {
                    "n": 8
                }
            },
            "input_args": ["x", "thr", "out"],
            "array_args": ["x", "out"],
            "output_args": ["out"],
            "init": {
                "shapes": {
                    "x": shape,
                    "out": shape
                }
            },
            "scalars": {
                "thr": 0.5
            }
        }
    }
    (d / "bi.json").write_text(json.dumps(bi))
    kir = lower(parse_kernel(d / "k_numpy.py", d / "bi.json", precision=precision))
    return apply_precision(kir, precision) if precision else kir


#: A helper whose array argument is a kernel-local allocated with numpy's ``dtype=x.dtype``
#: idiom -- the shape conv2d_instance_norm_divide / conv3d_multiply_instance_norm_clamp_multiply_max
#: reach after their conv helper is inlined. The early ``return`` keeps ``scale_up`` from being
#: inlined, so it survives as its own native function with a trailing out-param.
_DTYPE_OF_SRC = ("import numpy as np\n"
                 "def scale_up(v, s):\n"
                 " if s < 0.0:\n"
                 "  return -v\n"
                 " return v * s\n"
                 "def f(x, thr, out):\n"
                 " t = np.zeros(8, dtype=x.dtype)\n"
                 " t[:] = x + 1.0\n"
                 " out[:] = scale_up(t, thr)\n")


def test_array_return_helper_buffers_follow_kernel_precision():
    # ``dtype=x.dtype`` is "whatever x is", so the helper's argument and its synthesized out-param
    # must narrow with the kernel. Read as the literal tag ``"dtype"`` they missed every emitter's
    # dtype table and fell back to double, and an fp32 caller then handed a ``float *`` to a
    # ``double *`` dummy -- rejected by all three native toolchains.
    from numpyto_c.emit import emit_c, emit_cpp
    from numpyto_fortran.emit import emit_fortran
    kir = _helper_kir(_DTYPE_OF_SRC, "float32")
    assert [(a.name, a.dtype) for a in kir.helpers[0].arrays] == [("v", "float32"), ("__hret_0", "float32")]
    assert ("static void scale_up(float *restrict __hret_0, const float *restrict v, "
            "const int64_t n, const float s)") \
        in emit_c(kir, fn_name="f")
    assert ("static void scale_up(float *__restrict__ __hret_0, const float *__restrict__ v, "
            "const int64_t n, const float s)") \
        in emit_cpp(kir, fn_name="f")
    f90 = emit_fortran(kir, fn_name="f")
    assert "real(c_float), intent(in) :: v(8)" in f90
    assert "real(c_float), intent(inout) :: x_hret_0(n)" in f90


def test_fp64_helper_buffers_are_unchanged():
    # The default (no ``--precision``) path must stay exactly where it was: fp64 everywhere.
    from numpyto_c.emit import emit_c, emit_cpp
    from numpyto_fortran.emit import emit_fortran
    kir = _helper_kir(_DTYPE_OF_SRC, "")
    assert [(a.name, a.dtype) for a in kir.helpers[0].arrays] == [("v", "float64"), ("__hret_0", "float64")]
    assert ("static void scale_up(double *restrict __hret_0, const double *restrict v, "
            "const int64_t n, const double s)") \
        in emit_c(kir, fn_name="f")
    assert ("static void scale_up(double *__restrict__ __hret_0, const double *__restrict__ v, "
            "const int64_t n, const double s)") \
        in emit_cpp(kir, fn_name="f")
    f90 = emit_fortran(kir, fn_name="f")
    assert "real(c_double), intent(in) :: v(8)" in f90
    assert "real(c_double), intent(inout) :: x_hret_0(n)" in f90


def test_an_unresolvable_buffer_dtype_refuses():
    # A refusal beats a silently wrong emit: a dtype expression nothing can resolve used to be
    # stored verbatim and rendered as double. No emitter can pick a width for it, so it stops here.
    import pytest
    src = _DTYPE_OF_SRC.replace("dtype=x.dtype", "dtype=SOME_DTYPE")
    with pytest.raises(NotImplementedError, match="does not resolve to a known dtype"):
        _helper_kir(src, "float32")


#: ``t = h(t, s)``: the target IS the first argument, so the helper reads and writes ONE buffer.
_INOUT_SRC = ("import numpy as np\n"
              "def scale_in_place(v, s):\n"
              " if s < 0.0:\n"
              "  return -v\n"
              " return v * s\n"
              "def f(x, thr, out):\n"
              " t = np.zeros(8, dtype=x.dtype)\n"
              " t[:] = x + 1.0\n"
              " t = scale_in_place(t, thr)\n"
              " out[:] = t\n")


def test_inout_target_takes_one_abi_slot():
    # An in-out buffer is ONE parameter. Appending a separate out-param put the same pointer in two
    # slots, and in C both carry ``restrict`` -- a promise the call itself breaks, so the compiler
    # is licensed to keep a stale copy of what the helper just wrote (vgg16's _maxpool2d(h, h, n)).
    from numpyto_c.emit import emit_c
    from numpyto_fortran.emit import emit_fortran
    kir = _helper_kir(_INOUT_SRC)
    helper = kir.helpers[0]
    assert [a.name for a in helper.arrays] == ["v"], "the in-out buffer must not gain a second descriptor"
    assert helper.return_kind == "v" and helper.arrays[0].is_output
    assert helper.param_order() == ["v", "s"]
    c = emit_c(kir, fn_name="f")
    assert "static void scale_in_place(double *restrict v, const double s)" in c
    assert "scale_in_place(t, thr);" in c
    assert "scale_in_place(t, t," not in c, "the same buffer must not be passed twice"
    f90 = emit_fortran(kir, fn_name="f")
    assert "subroutine scale_in_place(v, s)" in f90
    assert "call scale_in_place(t, thr)" in f90


def test_helper_specialised_on_a_rebound_shape_is_refused():
    # A local rebound to a DIFFERENT shape resolves to whichever binding came first, and the helper
    # built from it bakes those extents in as literals. vgg16's _maxpool2d was emitted for
    # (batch, 3, 224, 224) and called on (batch, 512, 14, 14): wrong numbers and reads past the end,
    # reported by no compiler. Refuse while the inference cannot tell the two apart.
    import pytest
    src = ("import numpy as np\n"
           "def half(v):\n"
           " if v[0] > 0.0:\n"
           "  return v[0:4]\n"
           " return -v[0:4]\n"
           "def f(x, thr, out):\n"
           " t = np.zeros(8, dtype=x.dtype)\n"
           " t[:] = x + 1.0\n"
           " t = half(t)\n"
           " t = half(t)\n"
           " out[0:4] = t\n")
    with pytest.raises(NotImplementedError, match="rebound to both"):
        _helper_kir(src)


def test_a_fresh_local_bound_by_a_helper_call_gets_its_buffer():
    """``y = pick(x, thr)`` binds a local nothing allocated.

    The bare-Name target is passed as the helper's out-param, so it is never a Store anywhere in
    the tree. ``_promote_free_names_to_params`` then rescued the name as a scalar int PARAMETER: it
    entered the emitted ABI the binding never passes, and the C call handed an integer to a pointer
    dummy (``passing argument 1 of 'build_up_b' makes pointer from integer without a cast``). mlp,
    vision_transformer, squeezenet_fire_module, mamba2, channel_flow and cp2k_density_matrix_trs4
    all sat on it.
    """
    src = ("import numpy as np\n"
           "def pick(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " y = pick(x, thr)\n"
           " out[:] = y * 3.0\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "thr": 0.5
               }, {"out": (4, 5)}, {
                   "M": 4,
                   "n": 5
               },
               shapes={
                   "x": "(M, n)",
                   "out": "(M, n)"
               },
               backends=_ALL))
    assert ok, res


def test_an_array_valued_expression_argument_is_passed_as_a_buffer():
    """mlp's shape: ``y = step(x * 2.0 + 1.0, thr)``, a helper called on an EXPRESSION.

    ``_infer_param_desc`` reached the resolver for a Name and a Subscript only, so every other node
    fell through to the by-value scalar default: the helper declared a ``double`` where the call
    passes a buffer (``Rank mismatch in argument 'v' (scalar and rank-2)``; in C a pointer added to
    a double). Numbers here, not a signature, because the by-value form also loses the argument.
    """
    src = ("import numpy as np\n"
           "def step(v, lo):\n"
           " if lo > 0.0:\n"
           "  return np.maximum(v, lo)\n"
           " return -v\n"
           "def f(x, thr, out):\n"
           " y = step(x * 2.0 + 1.0, thr)\n"
           " out[:] = y * 3.0\n")
    x = np.linspace(-2.0, 2.0, 20).reshape(4, 5).astype(np.float64)
    ok, res = _all_ok(
        run_op(src,
               "f", {
                   "x": x,
                   "thr": 0.5
               }, {"out": (4, 5)}, {
                   "M": 4,
                   "n": 5
               },
               shapes={
                   "x": "(M, n)",
                   "out": "(M, n)"
               },
               backends=_ALL))
    assert ok, res
