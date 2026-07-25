# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ratchet: the -Wall -Wextra warning count on the native (C / C++ / Fortran) corpus
must never grow. See ``hpcagent_bench/flags.py:WARNINGS_BASIC`` and the ``warnings_ref``
wiring in ``hpcagent_bench/envs/compilers.yaml`` -- deliberately no ``-Werror``, so a
regression here is a number moving, not a build breaking.

Builds a small, fast foundation sample through the SAME path the real native corpus
build uses (``hpcagent_bench.benchmarks.cpp_runtime._ensure_built`` calls
:func:`hpcagent_bench.languages.build_kernel_lib_commands`), but into an isolated
``tmp_path`` build dir instead of the tracked ``cpp_backend/build/`` directories.
"""
import pathlib
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

import pytest

from hpcagent_bench import paths
from hpcagent_bench.flags import Mode
from hpcagent_bench.languages import build_kernel_lib_commands

#: (cpp_backend dir relative to paths.BENCHMARKS, short name) for foundation kernels that
#: already have generated fp64/fp32 sources on disk -- no codegen step, so the sample
#: builds fast. The same 10 kernels the ratchet count below was measured against.
#: Every kernel keeps its sources under its OWN ``<kernel>/cpp_backend/``. The shared
#: ``foundation/cpp_backend/`` still holds leftovers from the pre-flatten layout that no
#: emit refreshes, so pointing at it measures whatever was last written there -- which is
#: how the first count for this ratchet came out too high.
_KERNELS: Tuple[Tuple[str, str], ...] = (
    ("foundation/disjoint_halves_gather/cpp_backend", "disjoint_halves_gather"),
    ("foundation/halo_broadcast/cpp_backend", "halo_broadcast"),
    ("foundation/safety_column_stencil/cpp_backend", "safety_column_stencil"),
    ("foundation/wf_north_west/cpp_backend", "wf_north_west"),
    ("foundation/cond_reduce_sum/cpp_backend", "cond_reduce_sum"),
    ("foundation/ext_gather_load/cpp_backend", "ext_gather_load"),
    ("foundation/fuse_diamond/cpp_backend", "fuse_diamond"),
    ("foundation/iv_additive/cpp_backend", "iv_additive"),
    ("foundation/jacobi2d_tiled_sym/cpp_backend", "jacobi2d_tiled_sym"),
    ("foundation/wavefront_2d/cpp_backend", "wavefront_2d"),
)

#: (framework, lang, source ext, forced compilers.yaml block) mirroring
#: cpp_runtime.FRAMEWORK_LANG / FRAMEWORK_COMPILER for the 3 native flavors a real sweep
#: builds: cc -> gcc (first "c" block, unforced), llvm -> clangpp (forced, matches the
#: real "llvm" flavor), fortran -> gfortran (first "fortran" block, unforced).
_FLAVORS: Tuple[Tuple[str, str, str, Optional[str]], ...] = (
    ("cc", "c", "c", None),
    ("llvm", "cpp", "cpp", "clangpp"),
    ("fortran", "fortran", "f90", None),
)

#: KNOWN-BAD COUNT -- measured 2026-07-25 on this dev box (gcc 15.2.0, clang 21.1.8,
#: gfortran) with:
#:   OMP_NUM_THREADS=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp PMIX_MCA_gds=hash \
#:   UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl python3 -m pytest tests/test_warnings_ratchet.py
#: ZERO, and it starts at zero deliberately -- the first measurement was 40, every one of
#: them clang++'s -Wunused-const-variable on the C++ prelude's file-scope M_PI/M_E, which
#: numpyto_c/emit.py now marks [[maybe_unused]]. Starting a ratchet at a number that a
#: one-line emitter fix removes would have frozen that warning in as acceptable.
#: MAY ONLY BE LOWERED: a change needing a higher number is a regression to fix, never a
#: ratchet bump.
_KNOWN_BAD_COUNT = 0

#: Below this many successful (kernel, flavor) builds the sample itself is broken (missing
#: sources / no compiler) rather than clean, and the assertion above would pass vacuously.
_MIN_BUILDS = 20

_WARNING_RE = re.compile(r"warning:", re.IGNORECASE)

_REQUIRED_COMPILERS: Tuple[str, ...] = ("gcc", "g++", "clang", "clang++", "gfortran")


def _run_build(cmds: List[List[str]], cwd: pathlib.Path) -> Tuple[bool, int]:
    """Run a compile/link argv sequence, returning ``(ok, warning_count)`` summed over every
    step's stderr. ``ok`` is False on the first nonzero exit -- the new -Wall -Wextra flags
    being REJECTED is a build break, not a warning to count, and must fail loudly."""
    warnings = 0
    for argv in cmds:
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True)
        if proc.returncode != 0:
            return False, warnings
        warnings += len(_WARNING_RE.findall(proc.stderr))
    return True, warnings


@pytest.mark.integration
def test_warnings_ratchet(tmp_path: pathlib.Path) -> None:
    """-Wall -Wextra warning count on a representative foundation sample must not exceed
    the known-bad count above; lower it here whenever a fix reduces the real count."""
    missing = [c for c in _REQUIRED_COMPILERS if shutil.which(c) is None]
    if missing:
        pytest.skip(f"compiler(s) not installed: {missing}")

    total_warnings = 0
    total_builds = 0
    for backend_rel, short in _KERNELS:
        backend = paths.BENCHMARKS / backend_rel
        for framework, lang, ext, compiler in _FLAVORS:
            sources = [(lang, p) for p in (backend / f"{short}_fp64.{ext}", backend / f"{short}_fp32.{ext}")
                       if p.exists()]
            if not sources:
                continue
            build_dir = tmp_path / short / framework
            build_dir.mkdir(parents=True)
            out_so = build_dir / f"lib{short}_{framework}.so"
            cmds = build_kernel_lib_commands(sources,
                                             out_so,
                                             build_dir=build_dir,
                                             mode=Mode.SINGLE_CORE,
                                             compiler=compiler)
            ok, warnings = _run_build(cmds, build_dir)
            assert ok, f"{short} ({framework}): the new -Wall -Wextra flags broke the build"
            total_builds += 1
            total_warnings += warnings

    assert total_builds >= _MIN_BUILDS, (f"only {total_builds} (kernel, flavor) pairs built -- "
                                         f"the ratchet sample is broken, not clean")
    assert total_warnings <= _KNOWN_BAD_COUNT, (
        f"-Wall -Wextra warning count grew to {total_warnings} (ratchet: {_KNOWN_BAD_COUNT}); fix the "
        f"new warning(s), or lower the ratchet constant if the count instead went down")
