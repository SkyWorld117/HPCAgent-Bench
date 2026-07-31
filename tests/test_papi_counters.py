# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The PAPI counting wrapper: what it resolves, what it refuses, and what a crash costs.

Most of this runs on ANY host, because the parts that can be wrong on a host without PAPI are
pure: the metric -> event ladder, the sign arithmetic of a derived expression, and the rule that
an unexpressible metric is reported missing rather than filled with a different quantity.

The handful of tests that genuinely need counters are gated on an EXPLICIT predicate --
``find_library("papi")`` on Linux -- rather than a swallowed import error, so a skip here always
means "this host has no PAPI" and never "something changed and the guard stopped noticing".
"""
import ctypes.util
import json
import os
import signal
import subprocess

import pytest

from hpcagent_bench import osinfo
from hpcagent_bench.harness import papi, profiling

#: The environment predicate the skips key on. A name, not an exception: PAPI is a system
#: library, so its absence is a property of the host that can be stated before anything is run.
PAPI_LIBRARY = ctypes.util.find_library("papi")

requires_papi = pytest.mark.skipif(
    not (osinfo.IS_LINUX and PAPI_LIBRARY),
    reason="no libpapi on this host (ctypes.util.find_library('papi') found nothing), so hardware "
    "counters cannot be read; install PAPI to exercise these")

#: The event set of the machine this was developed on -- an AMD Zen4 with 5 counters, where
#: PAPI_L1_DCM exists and PAPI_L1_ICM, PAPI_L3_DCM, PAPI_L1_DCH, PAPI_INT_INS do not. Frozen as
#: data so the fallback ladder is tested against a REAL availability map on every host.
ZEN4_AVAILABLE = ("PAPI_L1_DCM", "PAPI_L2_DCM", "PAPI_L2_ICM", "PAPI_L2_TCM", "PAPI_TLB_DM", "PAPI_FMA_INS",
                  "PAPI_TOT_INS", "PAPI_FP_INS", "PAPI_TOT_CYC", "PAPI_L2_DCH", "PAPI_L1_DCA", "PAPI_FP_OPS")

#: The seven quantities the wrapper exists to report.
WANTED = ("data_cache_misses", "instruction_cache_misses", "instructions", "cache_hits", "fp_ops",
          "integer_instructions", "fma_instructions")


def test_every_requested_quantity_has_a_metric() -> None:
    assert set(papi.METRICS) == set(WANTED)


def test_no_candidate_swaps_operations_for_instructions() -> None:
    """The one substitution that must never happen: PAPI_FP_INS counts INSTRUCTIONS, and one
    packed FMA is one instruction and many operations. A fallback from ops to instructions would
    report a number an order of magnitude wrong under a name that looks right."""
    for candidate in papi.METRICS["fp_ops"]:
        assert all(term.endswith("_OPS") for term in candidate), candidate


def test_resolve_takes_the_direct_event_when_the_cpu_has_it() -> None:
    assert papi.resolve("data_cache_misses", ZEN4_AVAILABLE) == ("PAPI_L1_DCM", )
    assert papi.resolve("instructions", ZEN4_AVAILABLE) == ("PAPI_TOT_INS", )


def test_resolve_derives_cache_hits_when_the_preferred_event_is_missing() -> None:
    """PAPI_L1_DCH exists on almost no CPU. Accesses minus misses is the SAME number, exactly,
    from two events that still fit the counter budget -- so the metric survives the gap."""
    assert "PAPI_L1_DCH" not in ZEN4_AVAILABLE
    assert papi.resolve("cache_hits", ZEN4_AVAILABLE) == ("PAPI_L1_DCA", "-PAPI_L1_DCM")


def test_resolve_falls_through_to_the_next_cache_level() -> None:
    """L1 instruction misses are unavailable here; L2 answers the same question one level down,
    and the expression that ships with the count says which level it was."""
    assert papi.resolve("instruction_cache_misses", ZEN4_AVAILABLE) == ("PAPI_L2_ICM", )


def test_resolve_reports_nothing_rather_than_a_different_quantity() -> None:
    """No integer-instruction preset on this CPU, and nothing else counts integer instructions.
    The answer is None -- the caller then reports the metric missing."""
    assert papi.resolve("integer_instructions", ZEN4_AVAILABLE) is None


def test_resolve_needs_every_event_of_a_derived_candidate() -> None:
    """A derivation is only usable when BOTH its events are countable; half of one is not a
    partial answer, it is a wrong one."""
    assert papi.resolve("cache_hits", ("PAPI_L1_DCA", )) is None
    assert papi.resolve("cache_hits", ("PAPI_L1_DCM", )) is None


def test_expression_and_combine_agree_about_the_signs() -> None:
    terms = ("PAPI_L1_DCA", "-PAPI_L1_DCM")
    assert papi.expression(terms) == "PAPI_L1_DCA - PAPI_L1_DCM"
    assert papi.combine(terms, [1000, 40]) == 960
    assert papi.expression(("PAPI_DP_OPS", "PAPI_SP_OPS")) == "PAPI_DP_OPS + PAPI_SP_OPS"
    assert papi.combine(("PAPI_DP_OPS", "PAPI_SP_OPS"), [7, 5]) == 12


def test_a_missing_metric_is_never_confusable_with_zero() -> None:
    row = papi.missing("fp_ops", "nope")
    assert row["count"] is None and row["missing"] == "nope"


def segfaulting_worker(*args, **kwargs):
    """Stand-in for :func:`papi.counting_worker` that dies the way an agent's kernel dies."""
    os.kill(os.getpid(), signal.SIGSEGV)


def raising_worker(*args, **kwargs):
    raise RuntimeError("PAPI_start failed: no such event")


def test_a_segfaulting_counted_run_costs_one_metric_not_the_process(monkeypatch) -> None:
    """The isolation contract: metric k's child dying is metric k's number lost, the parent alive,
    and the signal NAMED -- not an exception the caller has to guess the meaning of."""
    monkeypatch.setattr(papi, "counting_worker", segfaulting_worker)
    row = papi.count_metric("/nonexistent.so", None, {}, "c", "instructions", rep_timeout=5.0)
    assert row["count"] is None
    assert "SIGSEGV" in row["missing"]
    assert os.getpid()  # the parent is still here to run metric k+1


def test_a_papi_failure_inside_the_child_is_reported_as_that_metric_s_reason(monkeypatch) -> None:
    monkeypatch.setattr(papi, "counting_worker", raising_worker)
    row = papi.count_metric("/nonexistent.so", None, {}, "c", "fp_ops", rep_timeout=5.0)
    assert row["metric"] == "fp_ops" and row["count"] is None
    assert "PAPI_start failed" in row["missing"]


def counted_line(metric: str) -> str:
    """A counting child's one machine-readable stdout line, as the parent expects to parse it."""
    return profiling.RESULT_PREFIX + json.dumps({
        "metric": metric,
        "expression": "PAPI_X",
        "events": ["PAPI_X"],
        "derived": False,
        "count": 7,
        "elapsed_ns": 1,
        "reps_counted": 1,
        "hardware_counters": 5
    })


def test_one_wedged_metric_does_not_cost_the_others(monkeypatch, tmp_path) -> None:
    """The sweep must survive a metric that hangs. subprocess signals a deadline overrun by
    RAISING, so without a catch here the first wedge takes down every metric after it -- and the
    call graph the profile already paid for."""

    def fake_run(argv, **kwargs):
        metric = argv[argv.index("--metric") + 1]
        if metric == "fp_ops":
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout=counted_line(metric) + "\n", stderr="")

    monkeypatch.setattr(profiling.subprocess, "run", fake_run)
    counters = profiling.count_metrics(tmp_path, tmp_path / "request.json", timeout=1.0)
    rows = {row["metric"]: row for row in counters["metrics"]}
    assert set(rows) == set(papi.METRICS) and counters["runs"] == len(papi.METRICS)
    assert rows["fp_ops"]["count"] is None and "wedged" in rows["fp_ops"]["missing"]
    assert all(rows[m]["count"] == 7 for m in papi.METRICS if m != "fp_ops")


def test_a_dead_counting_process_is_decoded_into_that_metric_s_reason(monkeypatch, tmp_path) -> None:
    """No result line means the process never got that far; its exit status and stderr are the
    reason, not a silent zero."""

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, -11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr(profiling.subprocess, "run", fake_run)
    rows = profiling.count_metrics(tmp_path, tmp_path / "request.json", timeout=1.0)["metrics"]
    assert all(row["count"] is None and "Segmentation fault" in row["missing"] for row in rows)


def test_the_rendered_table_carries_the_expression_and_the_ratio() -> None:
    """A raw miss count is not a finding; misses per thousand instructions is. And a metric this
    CPU cannot express must render its reason, not a blank that reads as zero."""
    counters = {
        "threads":
        1,
        "runs":
        3,
        "metrics": [{
            "metric": "instructions",
            "expression": "PAPI_TOT_INS",
            "events": ["PAPI_TOT_INS"],
            "derived": False,
            "count": 1_000_000
        }, {
            "metric": "cache_hits",
            "expression": "PAPI_L1_DCA - PAPI_L1_DCM",
            "events": ["PAPI_L1_DCA", "PAPI_L1_DCM"],
            "derived": True,
            "count": 25_000
        },
                    papi.missing("integer_instructions", "no candidate is available on this CPU")]
    }
    text = "\n".join(profiling.render_counters(counters))
    assert "PAPI_L1_DCA - PAPI_L1_DCM" in text
    assert "25.00" in text, f"25000 of 1000000 is 25 per 1k instructions:\n{text}"
    assert "no candidate is available on this CPU" in text
    # The instruction row's own ratio is 1000 by construction and says nothing, so it is blanked.
    assert "1000.00" not in text


def test_check_names_a_cause_a_caller_can_branch_on() -> None:
    """Shaped like perf_reports.PerfUnavailable on purpose, so one handler covers both."""
    if osinfo.IS_LINUX and PAPI_LIBRARY:
        assert papi.check() is not None
        return
    with pytest.raises(papi.PapiUnavailable) as excinfo:
        papi.check()
    assert excinfo.value.cause in ("not_linux", "papi_missing")


@requires_papi
def test_the_version_probe_finds_the_installed_papi() -> None:
    """No PAPI version is hardcoded anywhere, so this is what proves the probe works at all."""
    assert papi.initialised() is not None


@requires_papi
def test_availability_comes_from_papi_and_is_a_strict_subset_of_the_presets() -> None:
    events = papi.available_events()
    assert events, "PAPI enumerated no countable preset on a host that has PAPI"
    assert all(name.startswith("PAPI_") for name in events)
    # Availability is per-CPU, so the only safe universal claim is that SOMETHING is unavailable:
    # no CPU implements the whole preset table, which is exactly why this is discovered.
    every_candidate = {papi.event_name(t) for cands in papi.METRICS.values() for c in cands for t in c}
    assert not every_candidate.issubset(set(events))


@requires_papi
def test_at_least_one_metric_resolves_on_a_real_cpu() -> None:
    resolved = {m: papi.resolve(m, papi.available_events()) for m in papi.METRICS}
    assert any(v is not None for v in resolved.values()), resolved


@requires_papi
def test_the_counter_budget_is_reported_so_multiplexing_stays_checkable() -> None:
    """Every candidate must fit the budget, or the counts this module returns are estimates."""
    budget = papi.hardware_counters()
    assert budget > 0
    assert max(len(c) for cands in papi.METRICS.values() for c in cands) <= budget


@requires_papi
def test_the_count_covers_the_timed_call_and_nothing_else() -> None:
    """The assertion this whole module exists for: gemm at preset S does 2*NI*NJ*NK multiply-adds,
    so a correctly bracketed fp-op count lands ON that number. A count that also swept up
    interpreter start-up, the seeded input generation or the per-rep buffer copies would not --
    which is how a wrong number wearing a right label gets caught.

    Skipped by name (not silently) on a CPU with no fp-op preset: there the metric is honestly
    unavailable, and this test has nothing to check.
    """
    from hpcagent_bench.harness.agent import reference_source
    from hpcagent_bench.harness.envelope import Submission
    from hpcagent_bench.harness.grading import _data_seeded
    from hpcagent_bench.harness.sandbox import Sandbox
    from hpcagent_bench.harness.task import Task
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import binding_from_spec

    if papi.resolve("fp_ops", papi.available_events()) is None:
        pytest.skip(f"no fp-op preset on this CPU; available: {papi.available_events()}")
    spec = BenchSpec.load("gemm")
    sizes = spec.parameters["S"]
    expected = 2 * sizes["NI"] * sizes["NJ"] * sizes["NK"]
    binding = binding_from_spec(spec)
    task = Task("gemm", "restricted", "c")
    with Sandbox(binding) as sandbox:
        built = sandbox.build(Submission(language="c", source=reference_source(task)), debug=True)
        assert built.ok, built.log[-2000:]
        row = papi.count_metric(built.lib,
                                binding,
                                _data_seeded("gemm", "S", "float64", 42),
                                "c",
                                "fp_ops",
                                reps=2,
                                rep_timeout=300.0)
    assert row["count"] is not None, row.get("missing")
    assert row["reps_counted"] == 2 and row["elapsed_ns"] > 0
    # 5% either side: the reference also scales C by beta and alpha, which is O(NI*NJ) more work.
    assert 0.95 * expected <= row["count"] <= 1.05 * expected, (
        f"{row['expression']} counted {row['count']}, expected ~{expected} (2*NI*NJ*NK): the "
        "counted region is not the timed call")
