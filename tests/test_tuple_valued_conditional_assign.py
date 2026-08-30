# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tuple-returning helper has no ABI to be called across, so the frontend splices it into its
call site as ONE expression: a conditional selecting between tuple literals. The tuple unpack that
receives it survives lowering, and C and Fortran have no tuple to receive it with -- both emitters
have to project the conditional per element instead of refusing the expression.

Three conv_transpose kernels reached emit this way and every native backend refused them with
``expression Tuple``; the shape below is their ``_tap_span`` in miniature.
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hpcagent_bench" / "numpy_translators" / "src"))

from numpyto_common.frontend import parse_kernel  # noqa: E402
from numpyto_common.lowering import lower  # noqa: E402
from numpyto_c.emit import emit_c  # noqa: E402
from numpyto_fortran.emit import emit_fortran  # noqa: E402

#: Two guarded early returns and a fall-through one, each a 2-tuple, over locals the splice folds
#: in. Only the LAST arm computes ``min(...)``, which is what makes per-element projection visible:
#: ``hi`` must carry it and ``lo`` must not.
TUPLE_HELPER_KERNEL = """import numpy as np


def _span(n, k, s):
    off = k - s
    lo = 0 if off >= 0 else -off
    if off < 0 or lo >= n:
        return lo, lo
    hi = min(n, off + 1)
    return lo, hi


def k(a, out):
    for i in range(3):
        lo, hi = _span(N, i, 1)
        for j in range(lo, hi):
            out[j] = a[j] + 1.0
"""

BENCH_INFO = {
    "benchmark": {
        "name": "k",
        "short_name": "k",
        "relative_path": ".",
        "module_name": "k",
        "func_name": "k",
        "kind": "m",
        "domain": "d",
        "dwarf": "d",
        "parameters": {
            "S": {
                "N": 8
            }
        },
        "init": {
            "func_name": "",
            "input_args": [],
            "output_args": [],
            "arrays": {
                "a": "(N,)",
                "out": "(N,)"
            }
        },
        "input_args": ["a", "out"],
        "array_args": ["a", "out"],
        "output_args": ["out"],
    }
}


@pytest.fixture(name="kir")
def _kir(tmp_path):
    (tmp_path / "k_numpy.py").write_text(TUPLE_HELPER_KERNEL)
    (tmp_path / "k.json").write_text(json.dumps(BENCH_INFO))
    return lower(parse_kernel(tmp_path / "k_numpy.py", tmp_path / "k.json"))


def _assignments(src, name):
    """Every emitted statement that binds ``name``, declarations excluded."""
    hits = []
    for line in src.splitlines():
        stripped = line.strip()
        head = stripped.split("=", 1)[0].strip()
        if head == name and "==" not in stripped:
            hits.append(stripped)
    return hits


def test_the_c_emit_projects_the_conditional_element_by_element(kir):
    src = emit_c(kir)
    lo_lines = _assignments(src, "lo")
    hi_lines = _assignments(src, "hi")
    assert len(lo_lines) == 1, f"lo must be bound exactly once, got {lo_lines}"
    assert len(hi_lines) == 1, f"hi must be bound exactly once, got {hi_lines}"
    # Element 1 of the final arm is the only ``min(...)`` in the helper: it belongs to hi alone.
    # A projection that took the wrong element, or the whole tuple, shows up right here.
    assert "min(" in hi_lines[0], f"hi lost its own element: {hi_lines[0]}"
    assert "min(" not in lo_lines[0], f"lo picked up hi's element: {lo_lines[0]}"
    assert "int64_t lo;" in src and "int64_t hi;" in src


def test_the_fortran_emit_binds_both_targets_from_distinct_values(kir):
    src = emit_fortran(kir)
    lo_lines = _assignments(src, "lo")
    hi_lines = _assignments(src, "hi")
    assert len(lo_lines) == 1, f"lo must be bound exactly once, got {lo_lines}"
    assert len(hi_lines) == 1, f"hi must be bound exactly once, got {hi_lines}"
    # Fortran has no conditional expression, so each element lands in its own hoisted temp; the two
    # targets reading ONE temp would mean the tuple was projected once and copied.
    assert lo_lines[0].split("=", 1)[1] != hi_lines[0].split("=", 1)[1]
    assert "min(" in src, "hi's element (the only min in the helper) never reached the emit"
