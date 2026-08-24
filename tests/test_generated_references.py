# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every loop_level_reasoning kernel ships a committed reference in ALL THREE languages.

The agent reads that FILE. There is no route that serves it any more -- ``materialize_shared.sh``
copies it into the task folder and the prompt points at it -- so a missing or malformed file is
not a degraded experience, it is the agent starting from nothing in the one language where the
ABI is the thing that trips submissions.

Coverage and SHAPE only. Whether the file computes the right answer is the e2e sweep's job; what
is asserted here is what a numeric test cannot see: that the file exists at all, that the Fortran
one is the bare ``bind(C)`` SUBROUTINE the judge can actually load, and that no timer leaked into
a file the score DIVIDES by.
"""
import pytest

from hpcagent_bench import paths
from hpcagent_bench.spec import KERNELS

#: The three the agent may submit in, and the one extension each reference carries.
LANGUAGE_EXT = (("c", ".c"), ("cpp", ".cpp"), ("fortran", ".f90"))

#: A clock read in a baseline is measured AS kernel work, so the speedup every submission is
#: graded by would be inflated by the timer. Same list the collector refuses on.
TIMING_TOKENS = ("system_clock", "cpu_time", "date_and_time", "time_ns", "omp_get_wtime", "chrono", "clock_highres",
                 "clock_gettime")


def foundation_specs():
    """``(registry_key, spec)`` for the loop_level_reasoning track, by the same PREFIX test the
    collector uses -- kernels live at ``loop_level_reasoning/<stem>``, so an equality test on the
    track name silently matches nothing."""
    return [(key, spec) for key, spec in sorted(KERNELS.specs().items())
            if str(spec.relative_path).startswith("loop_level_reasoning")]


#: The line every emitted baseline's attribution header carries. It is what separates a file the
#: judge could LOAD from a vendored provenance copy: a ``*_reference.c`` may be the upstream TSVC
#: original and a ``*_reference.cpp`` may be a TSVC microkernel adaptation, and neither is written
#: against the judge's ABI -- they exist to show where the kernel came from, not to be submitted.
EMITTED_MARKER = "baseline reference for HPCAgent-Bench kernel"


def reference_path(spec, ext):
    return paths.BENCHMARKS / spec.relative_path / f"{spec.module_name}_reference{ext}"


def is_emitted(path) -> bool:
    """Whether ``path`` is a translator-emitted baseline rather than vendored provenance."""
    return path.exists() and EMITTED_MARKER in path.read_text(errors="replace")


@pytest.mark.parametrize("language,ext", LANGUAGE_EXT)
def test_every_foundation_kernel_ships_a_reference_in_this_language(language, ext):
    """Coverage, stated as the whole set rather than per kernel: a per-kernel parametrization
    reports the first gap and hides the other 241, and the number missing is what says whether a
    translator regressed or one manifest was renamed."""
    missing = [key for key, spec in foundation_specs() if not reference_path(spec, ext).exists()]
    assert not missing, (f"{len(missing)} loop_level_reasoning kernels have no {ext} reference "
                         f"(first few: {missing[:5]}); run scripts/collect_reference_sources.py")


def test_no_generated_reference_carries_a_timer():
    """The score DIVIDES by these files. A clock read inside one is counted as kernel work, which
    inflates the measured baseline and hands every submission graded against it a free speed-up."""
    offenders = []
    for key, spec in foundation_specs():
        for _language, ext in LANGUAGE_EXT:
            path = reference_path(spec, ext)
            if not path.exists():
                continue
            if not is_emitted(path):
                continue  # vendored provenance keeps whatever the upstream shipped
            body = path.read_text(errors="replace").lower()
            hits = [t for t in TIMING_TOKENS if t in body]
            if hits:
                offenders.append(f"{key}{ext}: {hits}")
    assert not offenders, "timing instrumentation in a baseline: " + "; ".join(offenders[:10])


def test_every_fortran_reference_is_the_loadable_abi_shape():
    """The Fortran ABI is where Fortran submissions fail, and this file is what the agent copies
    that shape from. It must be a BARE ``bind(C)`` subroutine: wrapped in a module, or written as
    a function, the build still succeeds and the LOAD fails -- so a file that is merely valid
    Fortran teaches the exact mistake the skill page spends a section warning about."""
    bad = []
    for key, spec in foundation_specs():
        path = reference_path(spec, ".f90")
        if not is_emitted(path):
            continue  # the four vendored ICON / LULESH ports are provenance, not baselines
        # Comments only ever start a line in an emitted file, so dropping them needs no parser.
        body = "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("!"))
        if "bind(c" not in body.lower():
            bad.append(f"{key}: no bind(C)")
        elif "\nmodule " in f"\n{body}" or body.lstrip().startswith("module "):
            bad.append(f"{key}: wrapped in a module (the load will fail)")
        elif not body.lstrip().lower().startswith("subroutine "):
            bad.append(f"{key}: not a bare subroutine")
    assert not bad, "Fortran references off the loadable ABI: " + "; ".join(bad[:10])
