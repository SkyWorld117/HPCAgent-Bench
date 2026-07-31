# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Framework binding for the Pluto polyhedral native backend: kept separate from NativeFramework because
polycc is a distinct toolchain (a polyhedral source-to-source transform producing a different generated
source), not merely a compiler flag like ``polly``. Reuses the native wrapper/C-ABI machinery via subclass."""

import pathlib
import shlex
import shutil
import subprocess
import tempfile

from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks import Benchmark
from hpcagent_bench.frameworks.native_framework import NativeFramework
from hpcagent_bench.pluto_affine import scop_nonaffine_reason
from typing import Any, List, Optional

#: How the transformation report invokes ``polycc``, and why each flag is there.
#:
#: * ``--pet``      -- the emitted scop uses ``int64_t`` counters, which the default clan extractor
#:                     rejects; this is the same extractor ``tests/numerical_oracle`` runs Pluto with.
#: * ``--tile``     -- the repo's documented Pluto invocation (``numpy_translators/README.md``). Tiling
#:                     is off by default in polycc, and untiled output makes the report's whole
#:                     "After tiling" section vacuous.
#: * ``--parallel`` -- also off by default. Without it polycc marks no loop parallel and emits no
#:                     ``#pragma omp parallel for``, so the report could never answer "what did Pluto
#:                     parallelize" -- the question this column exists to ask.
#: * ``--debug``    -- promotes the band/parallel decisions to stdout. At default verbosity polycc
#:                     prints the transformation matrices but never says WHICH loop it marked
#:                     parallel or which bands it tiled (measured: ``[pluto_mark_parallel] parallel
#:                     loops`` and ``Bands for intra tile optimization`` appear only under --debug).
#:                     ``--moredebug`` triples the size with per-dependence solver traces that answer
#:                     no question a reader of this file has.
POLYCC_REPORT_ARGS = ("--pet", "--tile", "--parallel", "--debug")


class PlutoFramework(NativeFramework):
    """The Pluto polyhedral native backend (base ``pluto``); a thin NativeFramework subclass dispatching
    to the wrapper's ``kernel_pluto`` entry point. Its own base/class since polycc is a distinct toolchain."""

    def opt_report(self, program: Any, bench: Benchmark) -> Optional[str]:
        """Pluto's polyhedral transformation report, followed by the C++ compiler's vectorization report.

        Two reports because two tools shape this column, and they answer different questions: polycc
        says which bands it tiled, which loops it marked parallel and how it fused them; the compiler
        says what it then vectorized. Concatenated rather than split across kinds so the pair is read
        together -- the vectorizer's verdict on a tiled loop is only meaningful next to the tiling.

        polycc runs in a scratch directory and its output is discarded, so this cannot disturb the
        timed ``.so`` (which, today, polycc played no part in building -- see :meth:`polycc_report`).
        """
        parts = [p for p in (self.polycc_report(bench), super().opt_report(program, bench)) if p]
        return "\n\n".join(parts) if parts else None

    def polycc_report(self, bench: Benchmark) -> Optional[str]:
        """polycc's transformation report for this kernel's emitted scops, or ``None`` when there is none.

        ``None`` covers two normal answers: polycc is not installed, and the translator emitted no
        ``#pragma scop`` for this kernel. A scop outside Pluto's affine model is reported as a skip
        rather than run, using :func:`hpcagent_bench.pluto_affine.scop_nonaffine_reason` -- the same
        detector the numerical oracle gates on -- because polycc may silently MISCOMPILE a non-affine
        scop rather than reject it, and a report from a run that had no business happening is worse
        than no report.

        .. warning::
           This describes what polycc does to the emitted scop, NOT the binary this column timed.
           ``pluto`` currently builds ``<base>_fp{64,32}.cpp`` -- the same sources as ``llvm``, with the
           same ``clang++`` -- and never invokes polycc (see ``benchmarks/cpp_runtime.py``
           ``FRAMEWORK_LANG`` / ``_native_sources``), so the transformation below is absent from the
           timed artifact. The report says so in its own header rather than reading as a description
           of what ran.
        """
        exe = shutil.which("polycc")
        if exe is None:
            return None
        cpp_backend = self._cpp_backend(bench)
        base = self._native_base(bench)
        scops = sorted(cpp_backend.glob(f"{base}_fp*_pluto_input.c"))
        if not scops:
            return None
        chunks: List[str] = [
            "==== polycc transformation report ====\n"
            "NOTE: the `pluto` column compiles the untransformed C++ (same sources as `llvm`) and does\n"
            "      not invoke polycc, so the transformation below is NOT in the timed binary."
        ]
        with tempfile.TemporaryDirectory(prefix="pluto_opt_report_") as scratch:
            for scop in scops:
                nonaffine = scop_nonaffine_reason(scop.read_text())
                if nonaffine is not None:
                    chunks.append(f"---- {scop.name} ----\nskipped: outside Pluto's affine model ({nonaffine})")
                    continue
                out = pathlib.Path(scratch) / f"{scop.stem}_pluto.c"
                cmd = [exe, *POLYCC_REPORT_ARGS, str(scop), "-o", str(out)]
                proc = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True)
                if proc.returncode != 0:
                    chunks.append(f"---- {scop.name} ----\nskipped: polycc rejected the scop\n{proc.stderr}")
                    continue
                chunks.append(f"---- {scop.name} ----\n$ {shlex.join(cmd)}\n{proc.stdout}{proc.stderr}")
        return "\n\n".join(chunks)

    def generated_source(self, program: Any, bench: Benchmark) -> Optional[str]:
        """The sources this column compiled. Overridden only to record that they are the UNTRANSFORMED
        C++: the base class's docstring promises "the polyhedrally-transformed code" for a
        source-to-source backend, which this column does not currently produce (see
        :meth:`polycc_report`)."""
        text = cpp_runtime.generated_source_text(self._cpp_backend(bench), self._native_base(bench), self.fname)
        if text is None:
            return None
        return f"// NOTE: compiled as emitted -- polycc does not run in this column's build.\n{text}"
