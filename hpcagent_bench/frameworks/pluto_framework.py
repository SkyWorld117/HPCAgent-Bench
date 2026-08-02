# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Framework binding for the Pluto polyhedral native backend: kept separate from NativeFramework because
polycc is a distinct toolchain (a polyhedral source-to-source transform producing a different generated
source), not merely a compiler flag like ``polly``. Reuses the native wrapper/C-ABI machinery via subclass."""

import json
import pathlib
import shlex
import shutil
import subprocess
import tempfile

from hpcagent_bench.benchmarks import cpp_runtime
from hpcagent_bench.frameworks import Benchmark
from hpcagent_bench.frameworks.native_framework import NativeFramework
from hpcagent_bench.pluto_affine import scop_nonaffine_reason
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """The Pluto polyhedral native backend (base ``pluto``); a NativeFramework subclass dispatching to
    the wrapper's ``kernel_pluto`` entry point. Its timed ``.so`` is compiled from polycc's ``--pet``
    transformation of the emitted scop (``cpp_runtime._ensure_built_pluto``), NOT from the untransformed
    C++ the ``llvm`` column builds; the transformed function keeps Pluto's symbols-first VLA signature,
    so :meth:`call_args` marshals arguments in that order instead of the default C-ABI one."""

    def call_args(self, bench: Benchmark, impl: Any, resolved: Dict[str, Any],
                  bdata: Dict[str, Any]) -> Tuple[Sequence[Any], Dict[str, Any]]:
        """Marshal arguments in the Pluto binding's order -- symbols, then arrays, then scalars -- which
        is the order polycc's transformed ``<base>_fpNN`` VLA signature expects, NOT the default C-ABI
        order (sorted pointers then scalars) the cc/llvm columns use. The transformed function takes each
        shape SYMBOL before the array whose ``[N]`` VLA dimension it sizes, so a default-ordered call
        would hand a pointer where a length is expected -- a silent wrong answer, not a crash.

        The order comes from ``<base>_fp64_pluto_binding.json`` (emitted beside the scop; the order is
        precision-independent). Reuses the base class's resolved ABI descriptors and output allocation,
        only REORDERING them; if the binding is missing or its names disagree with the descriptors,
        falls back to the base ordering (which then fails validation rather than fabricating a result).
        """
        order = self._pluto_arg_order(bench)
        abi = self._abi_args(bench)
        if order is None or abi is None:
            return super().call_args(bench, impl, resolved, bdata)
        by_name = {a.name: a for a in abi}
        if set(order) != set(by_name):
            return super().call_args(bench, impl, resolved, bdata)
        out: List[Any] = []
        for name in order:
            a = by_name[name]
            if name in resolved:
                out.append(resolved[name])
            elif name in bdata:
                out.append(bdata[name])
            elif a.kind == "ptr":
                out.append(self._alloc_output(a, bdata))
            else:
                raise KeyError(f"{bench.bname}: Pluto ABI scalar {name!r} has no value in resolved/bdata")
        return out, {}

    def _pluto_arg_order(self, bench: Benchmark) -> Optional[List[str]]:
        """Argument NAMES in Pluto binding order from ``<base>_fp64_pluto_binding.json``, or ``None``
        when it is absent/unreadable. fp64 is always emitted and the order does not vary by precision."""
        cpp_backend = self._cpp_backend(bench)
        base = self._native_base(bench)
        pb = cpp_backend / f"{base}_fp64_pluto_binding.json"
        if not pb.exists():
            return None
        try:
            return [a["name"] for a in json.loads(pb.read_text())["args"]]
        except Exception:  # noqa: BLE001 -- a malformed binding must not crash the run; fall back
            return None

    def opt_report(self, program: Any, bench: Benchmark) -> Optional[str]:
        """Pluto's polyhedral transformation report, followed by clang's vectorization report on the
        TRANSFORMED C. Two tools shape this column and answer different questions: polycc says which
        bands it tiled and which loops it marked parallel; clang says what it then vectorized. Both
        describe the ``<base>_fpNN_pluto.c`` that was actually compiled and timed. Each runs compile-only
        into a scratch dir, so neither disturbs the timed ``.so``."""
        clang_report = cpp_runtime.pluto_opt_report_text(self._cpp_backend(bench), self._native_base(bench))
        parts = [p for p in (self.polycc_report(bench), clang_report) if p]
        return "\n\n".join(parts) if parts else None

    def polycc_report(self, bench: Benchmark) -> Optional[str]:
        """polycc's transformation decisions for this kernel's emitted scops, or ``None`` when there is
        none (polycc absent, or no ``#pragma scop`` was emitted). A scop outside Pluto's affine model is
        reported as skipped rather than run (:func:`hpcagent_bench.pluto_affine.scop_nonaffine_reason`,
        the same detector the build gates on), because polycc may silently MISCOMPILE a non-affine scop.

        This is a VERBOSE re-run (``--tile --parallel --debug``) that surfaces the band/parallel
        decisions polycc makes; the timed build applies ``polycc --pet`` to the SAME scop
        (``cpp_runtime._ensure_built_pluto``), so the transformation described here is the one compiled
        and timed. Run into a throwaway directory, so it cannot disturb the timed ``.so``."""
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
            "The `pluto` column compiles polycc's --pet output for these scops; the decisions below\n"
            "(a verbose --tile --parallel --debug re-run of the same scop) describe the transformation\n"
            "that IS compiled and timed."
        ]
        with tempfile.TemporaryDirectory(prefix="pluto_opt_report_") as scratch:
            # Same aarch64 pet-parse include shim the timed build uses, so the report can extract the
            # scop where the build did (see cpp_runtime.pet_parse_env).
            pet_env = cpp_runtime.pet_parse_env(pathlib.Path(scratch))
            for scop in scops:
                nonaffine = scop_nonaffine_reason(scop.read_text())
                if nonaffine is not None:
                    chunks.append(f"---- {scop.name} ----\nskipped: outside Pluto's affine model ({nonaffine})")
                    continue
                out = pathlib.Path(scratch) / f"{scop.stem}_pluto.c"
                cmd = [exe, *POLYCC_REPORT_ARGS, str(scop), "-o", str(out)]
                proc = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, env=pet_env)
                if proc.returncode != 0:
                    chunks.append(f"---- {scop.name} ----\nskipped: polycc rejected the scop\n{proc.stderr}")
                    continue
                chunks.append(f"---- {scop.name} ----\n$ {shlex.join(cmd)}\n{proc.stdout}{proc.stderr}")
        return "\n\n".join(chunks)

    def generated_source(self, program: Any, bench: Benchmark) -> Optional[str]:
        """The polycc-TRANSFORMED C that was compiled and timed (``<base>_fpNN_pluto.c``), so the
        recorded source matches the artifact. Falls back to ``None`` before the transform has run."""
        return cpp_runtime.pluto_generated_source_text(self._cpp_backend(bench), self._native_base(bench))
