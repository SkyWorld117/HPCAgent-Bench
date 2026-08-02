# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The genuine Pluto path: the TIMED ``pluto`` library is compiled from ``polycc --pet`` output, never
from the untransformed C++ the ``llvm`` column builds. These tests pin the invocation, the transformed
-source selection, the Pluto argument order, the decline-don't-fake policy, the preflight gate, and --
end to end, gated on a real polycc -- numerical correctness. Mirrors tests/numerical_oracle.py::_run_pluto.
"""
import ctypes
import json
import shutil
import types
import pathlib

import numpy as np
import pytest

from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks.errors import NotSupportedByFramework
from hpcagent_bench.frameworks.pluto_framework import PlutoFramework
from hpcagent_bench.harness import preflight


def _write(p: pathlib.Path, text: str = "") -> pathlib.Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


# --------------------------------------------------------------------------------------------------
# Decline-don't-fake: polycc absent / no scop / non-affine / polycc failure -> NotSupportedByFramework,
# NEVER a silent fall back to the untransformed C++ artifact (requirements 1 & 2).
# --------------------------------------------------------------------------------------------------

def test_missing_polycc_declines(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: None)
    with pytest.raises(NotSupportedByFramework) as ei:
        cpp_runtime._ensure_built_pluto(tmp_path, "k")
    assert "polycc" in ei.value.reason and "untransformed" in ei.value.reason


def test_no_scop_declines(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: "/usr/bin/polycc")
    with pytest.raises(NotSupportedByFramework) as ei:
        cpp_runtime._ensure_built_pluto(tmp_path, "k")  # no *_pluto_input.c present
    assert "scop" in ei.value.reason.lower()


def test_nonaffine_declines(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: "/usr/bin/polycc")
    _write(tmp_path / "k_fp64_pluto_input.c", "#pragma scop\n#pragma endscop\n")
    import hpcagent_bench.pluto_affine as pa
    monkeypatch.setattr(pa, "scop_nonaffine_reason", lambda text: "data-dependent-bound")
    with pytest.raises(NotSupportedByFramework) as ei:
        cpp_runtime._ensure_built_pluto(tmp_path, "k")
    assert "affine" in ei.value.reason.lower()


def test_polycc_failure_declines(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: "/usr/bin/polycc")
    _write(tmp_path / "k_fp64_pluto_input.c", "scop")
    import hpcagent_bench.pluto_affine as pa
    monkeypatch.setattr(pa, "scop_nonaffine_reason", lambda text: None)

    def fake_run(cmd, **kw):  # polycc rejects the scop
        return types.SimpleNamespace(returncode=1, stdout="",
                                     stderr="pet: data dependent conditions not supported")

    monkeypatch.setattr(cpp_runtime.subprocess, "run", fake_run)
    with pytest.raises(NotSupportedByFramework) as ei:
        cpp_runtime._ensure_built_pluto(tmp_path, "k")
    assert "polycc" in ei.value.reason


def test_pluto_reject_reason_extracts_cause():
    assert cpp_runtime._pluto_reject_reason("pet: data dependent conditions not supported")
    assert cpp_runtime._pluto_reject_reason("nothing notable here") == ""


# --------------------------------------------------------------------------------------------------
# Invocation + transformed-source selection (requirements 3-6): polycc --pet runs on the input, and the
# TRANSFORMED <base>_fpNN_pluto.c (never the .cpp) is what gets compiled, as C, with clang + OpenMP.
# --------------------------------------------------------------------------------------------------

def test_polycc_invoked_and_transformed_c_compiled(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_runtime.shutil, "which", lambda name: "/usr/bin/polycc")
    _write(tmp_path / "k_fp64_pluto_input.c", "scop64")
    _write(tmp_path / "k_fp32_pluto_input.c", "scop32")
    import hpcagent_bench.pluto_affine as pa
    monkeypatch.setattr(pa, "scop_nonaffine_reason", lambda text: None)

    seen_cmds = []

    def fake_run(cmd, **kw):  # fake polycc: honour -o, record --pet
        seen_cmds.append(cmd)
        out = pathlib.Path(cmd[cmd.index("-o") + 1])
        out.write_text("/* transformed */\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cpp_runtime.subprocess, "run", fake_run)

    captured = {}

    def fake_build(sources, so, build_dir=None, compiler=None, extra_flags=""):
        captured.update(sources=sources, so=so, compiler=compiler, extra_flags=extra_flags)
        return [["/bin/true"]]

    import hpcagent_bench.languages as languages
    monkeypatch.setattr(languages, "build_kernel_lib_commands", fake_build)
    monkeypatch.setattr(cpp_runtime.subprocess, "check_call",
                        lambda cmd, **kw: (tmp_path / "build" / "libk_pluto.so").write_bytes(b""))

    so = cpp_runtime._ensure_built_pluto(tmp_path, "k")

    assert so.name == "libk_pluto.so"
    assert all("--pet" in cmd for cmd in seen_cmds)  # requirement: polycc --pet
    langs = [lang for lang, _ in captured["sources"]]
    names = [p.name for _, p in captured["sources"]]
    assert langs == ["c", "c"]                                      # compiled AS C (requirement 4)
    assert names == ["k_fp64_pluto.c", "k_fp32_pluto.c"]           # TRANSFORMED source (requirement 3)
    assert captured["compiler"] == "clang"                         # clang, not clangpp
    assert "-fopenmp" in captured["extra_flags"]                   # OpenMP preserved (requirement 5)


def test_generated_source_reports_transformed(tmp_path, monkeypatch):
    bd = tmp_path / "build"
    _write(bd / "k_fp64_pluto.c", "#pragma omp parallel for\nfor(...)\n")
    import hpcagent_bench.languages as languages
    monkeypatch.setattr(languages, "annotate_generated", lambda src, lang: src.read_text())
    text = cpp_runtime.pluto_generated_source_text(tmp_path, "k")
    assert text is not None
    assert "omp parallel for" in text
    assert "compiled + timed source" in text
    assert "does not run" not in text  # the old untransformed disclaimer is gone


# --------------------------------------------------------------------------------------------------
# Pluto argument order (requirement 3): call_args marshals in the binding's symbols-first order, which
# the transformed VLA signature needs -- NOT the default C-ABI order the cc/llvm columns use.
# --------------------------------------------------------------------------------------------------

def test_call_args_uses_pluto_binding_order(tmp_path, monkeypatch):
    _write(tmp_path / "k_fp64_pluto_binding.json",
           json.dumps({"args": [{"name": "N", "kind": "i64"}, {"name": "A", "kind": "ptr_f64"},
                                {"name": "B", "kind": "ptr_f64"}, {"name": "C", "kind": "ptr_f64"}]}))
    fw = PlutoFramework("pluto")
    arg = lambda name, kind: types.SimpleNamespace(name=name, kind=kind, shape=None, dtype="float64")
    # Default ABI order (arrays then scalar) -- deliberately DIFFERENT from the Pluto order.
    default_abi = [arg("A", "ptr"), arg("B", "ptr"), arg("C", "ptr"), arg("N", "scalar")]
    monkeypatch.setattr(fw, "_abi_args", lambda bench: default_abi)
    monkeypatch.setattr(fw, "_cpp_backend", lambda bench: tmp_path)
    monkeypatch.setattr(fw, "_native_base", lambda bench: "k")

    bench = types.SimpleNamespace(bname="k")
    resolved = {"A": "arrA", "B": "arrB", "C": "arrC"}
    bdata = {"N": 8, "A": "arrA", "B": "arrB", "C": "arrC"}
    args, kwargs = fw.call_args(bench, None, resolved, bdata)
    assert args == [8, "arrA", "arrB", "arrC"]  # N first (symbol), then arrays -- Pluto order
    assert kwargs == {}


# --------------------------------------------------------------------------------------------------
# Preflight gate (requirement 7): a pluto job with polycc absent is FATAL up front.
# --------------------------------------------------------------------------------------------------

def test_preflight_requires_polycc(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    code, report, _env = preflight.run(["pluto"])
    assert code == 1 and any("polycc" in line for line in report)

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    code_ok, report_ok, _ = preflight.run(["pluto"])
    assert code_ok == 0 and any("polycc present" in line for line in report_ok)


# --------------------------------------------------------------------------------------------------
# End-to-end NUMERICAL correctness through the genuine build path (requirement 10). Gated on a real
# polycc: transform a hand-written affine matmul scop, compile the polycc output as C with clang, and
# check the timed .so computes A @ B. This exercises _ensure_built_pluto with the real toolchain.
# --------------------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("polycc") is None or shutil.which("clang") is None,
                    reason="genuine Pluto needs polycc + clang on PATH (source slurm/hpcagent-env.sh)")
def test_pluto_transformed_so_is_numerically_correct(tmp_path):
    _write(tmp_path / "mm_fp64_pluto_input.c",
           "#include <stdint.h>\n"
           "void mm_fp64(const int64_t N, double (*restrict A)[N], double (*restrict B)[N],\n"
           "             double (*restrict C)[N]) {\n"
           "#pragma scop\n"
           "  for (int64_t i=0;i<N;i++) for (int64_t j=0;j<N;j++) for (int64_t k=0;k<N;k++)\n"
           "      C[i][j] += A[i][k] * B[k][j];\n"
           "#pragma endscop\n"
           "}\n")
    so_path = cpp_runtime._ensure_built_pluto(tmp_path, "mm")
    assert so_path.name == "libmm_pluto.so"
    # The transformed C the build actually compiled must carry polycc's marks (tiling / omp).
    transformed = (tmp_path / "build" / "mm_fp64_pluto.c").read_text()
    assert "omp parallel for" in transformed or "32" in transformed

    lib = ctypes.CDLL(str(so_path))
    fn = lib["mm_fp64"]
    fn.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_double),
                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    fn.restype = None
    n = 16
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(rng.random((n, n)))
    b = np.ascontiguousarray(rng.random((n, n)))
    c = np.zeros((n, n))
    ptr = lambda arr: arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fn(n, ptr(a), ptr(b), ptr(c))
    np.testing.assert_allclose(c, a @ b, rtol=1e-12, atol=1e-12)
