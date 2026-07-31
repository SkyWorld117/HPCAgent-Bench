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
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess

import pytest

from hpcagent_bench import flags, osinfo
from hpcagent_bench.flags import Mode
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
        "hardware_counters": 5,
        "threads_counted": 3,
        "scope": "all_threads",
        "smt": True
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
    counters = profiling.count_metrics(tmp_path, tmp_path / "request.json", threads=2, timeout=1.0)
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
    rows = profiling.count_metrics(tmp_path, tmp_path / "request.json", threads=2, timeout=1.0)["metrics"]
    assert all(row["count"] is None and "Segmentation fault" in row["missing"] for row in rows)


def test_the_rendered_table_carries_the_expression_and_the_ratio() -> None:
    """A raw miss count is not a finding; misses per thousand instructions is. And a metric this
    CPU cannot express must render its reason, not a blank that reads as zero."""
    counters = {
        "threads":
        4,
        "threads_counted":
        5,
        "smt":
        True,
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
    # The threading scope is part of the reading, not a footnote: an SMT box that was not pinned
    # and a count taken on one thread of eight are different numbers with the same units.
    assert "4 thread(s), 5 counted" in text and "SMT on" in text


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


def test_feature_set_is_the_intersection_of_what_we_want_and_what_the_cpu_has(monkeypatch) -> None:
    """The query an agent, a test and the skill all start from -- answerable without a workload."""
    monkeypatch.setattr(papi, "available_events", lambda: ZEN4_AVAILABLE)
    monkeypatch.setattr(papi, "hardware_counters", lambda: 5)
    features = papi.feature_set()
    assert set(features["supported"]) | set(features["unsupported"]) == set(papi.METRICS)
    assert not set(features["supported"]) & set(features["unsupported"]), "a metric cannot be both"
    assert features["supported"]["cache_hits"]["expression"] == "PAPI_L1_DCA - PAPI_L1_DCM"
    assert features["supported"]["cache_hits"]["derived"] is True
    assert features["supported"]["instructions"]["derived"] is False
    assert "PAPI_INT_INS" in features["unsupported"]["integer_instructions"]
    assert features["hardware_counters"] == 5


def test_feature_set_answers_for_a_subset_when_asked(monkeypatch) -> None:
    monkeypatch.setattr(papi, "available_events", lambda: ZEN4_AVAILABLE)
    monkeypatch.setattr(papi, "hardware_counters", lambda: 5)
    features = papi.feature_set(("instructions", "integer_instructions"))
    assert set(features["supported"]) == {"instructions"}
    assert set(features["unsupported"]) == {"integer_instructions"}


def test_no_candidate_can_exceed_a_plausible_counter_budget() -> None:
    """A candidate wider than the CPU's counter registers would be MULTIPLEXED -- the estimate
    this module exists to refuse. Four is the narrowest budget in the wild."""
    assert max(len(c) for cands in papi.METRICS.values() for c in cands) <= 4


def test_thread_ids_put_the_calling_thread_first() -> None:
    """Order is load-bearing: the calling thread's set needs no attach, so a fallback drops
    everything after it and is left with exactly the single-threaded answer."""
    tids = papi.thread_ids()
    assert tids[0] == os.getpid()
    assert len(set(tids)) == len(tids)
    assert sorted(tids[1:]) == list(tids[1:])


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


#: A gemm submission that really does run on OpenMP worker threads, so a count that only saw the
#: master thread is off by the thread count and this test says so. Hand-written rather than the
#: generated reference precisely because the reference is serial -- a serial kernel cannot
#: distinguish "counted every thread" from "counted the only thread".
OPENMP_GEMM = """
#include <stdint.h>
void gemm_fp64(double *A, double *B, double *C, int64_t NI, int64_t NJ, int64_t NK,
               double alpha, double beta, void *workspace, int64_t workspace_size) {
    (void) workspace; (void) workspace_size;
    #pragma omp parallel for
    for (int64_t i = 0; i < NI; ++i) {
        for (int64_t j = 0; j < NJ; ++j) C[i * NJ + j] *= beta;
        for (int64_t k = 0; k < NK; ++k) {
            double a = alpha * A[i * NK + k];
            for (int64_t j = 0; j < NJ; ++j) C[i * NJ + j] += a * B[k * NJ + j];
        }
    }
}
"""


def refusing_open_counter(original):
    """A host that will not attach: the calling thread opens, every worker is refused.

    Closes over the REAL function -- reading it back off the module would find this stand-in and
    recurse forever, which is the shape of a patch that tests nothing.
    """

    def refuse(lib, tid: int, codes):
        if tid == os.getpid():
            return original(lib, tid, codes)
        return ctypes.c_int(papi.PAPI_NULL), f"cannot attach to thread {tid}: simulated refusal"

    return refuse


@requires_papi
def test_a_threaded_kernel_is_counted_on_every_thread_and_degrades_out_loud(monkeypatch) -> None:
    """The headline of the multithreaded path: an OpenMP kernel's fp-op count must equal
    2*NI*NJ*NK whatever the thread count, because the WORK does not change.

    Counting only the calling thread would return roughly 1/N of that -- which is what PAPI does
    by default, and exactly the wrong number wearing the right label. dace solves this by calling
    PAPI from inside each OpenMP thread, which a judge holding an opaque .so cannot do; the master
    thread attaches an event set per worker instead.

    The same build then proves the OTHER half of the contract: a host that refuses the attach
    still answers, with the master thread's share and a STATED scope. A silent fraction of the
    kernel's work is the one outcome that must be impossible.
    """
    from hpcagent_bench.harness.envelope import Submission
    from hpcagent_bench.harness.grading import _data_seeded
    from hpcagent_bench.harness.sandbox import Sandbox
    from hpcagent_bench.spec import BenchSpec
    from hpcagent_bench.support.bindings.contract import binding_from_spec

    if papi.resolve("fp_ops", papi.available_events()) is None:
        pytest.skip(f"no fp-op preset on this CPU; available: {papi.available_events()}")
    threads = min(4, flags.ncores())
    if threads < 2:
        pytest.skip(f"only {flags.ncores()} physical core(s) available; nothing to parallelise over")
    for key, value in {**flags.cpu_env(Mode.MULTI_CORE, threads=threads), **papi.PINNED_ENV}.items():
        monkeypatch.setenv(key, value)  # set BEFORE the .so loads, which is when OpenMP reads them

    spec = BenchSpec.load("gemm")
    sizes = spec.parameters["S"]
    expected = 2 * sizes["NI"] * sizes["NJ"] * sizes["NK"]
    binding = binding_from_spec(spec)
    data = _data_seeded("gemm", "S", "float64", 42)
    with Sandbox(binding) as sandbox:
        built = sandbox.build(Submission(language="c", source=OPENMP_GEMM), debug=True)
        assert built.ok, built.log[-2000:]
        counted = papi.count_metric(built.lib, binding, data, "c", "fp_ops", reps=1, warmup=1, rep_timeout=300.0)
        # The patch is inherited across the fork count_metric does, so it reaches the worker.
        monkeypatch.setattr(papi, "open_counter", refusing_open_counter(papi.open_counter))
        refused = papi.count_metric(built.lib, binding, data, "c", "fp_ops", reps=1, warmup=1, rep_timeout=300.0)

    assert counted["count"] is not None, counted.get("missing")
    assert counted["scope"] == "all_threads", counted.get("fallback")
    assert counted["threads_counted"] > 1, "the OpenMP pool was never seen, so this proves nothing"
    assert 0.95 * expected <= counted["count"] <= 1.05 * expected, (
        f"counted {counted['count']} on {counted['threads_counted']} thread(s), expected "
        f"~{expected}; {counted['count'] / expected:.2f}x suggests only some threads were counted")

    assert refused["scope"] == "calling_thread" and refused["threads_counted"] == 1
    assert "simulated refusal" in refused["fallback"]
    assert refused["count"] < 0.9 * expected, (
        "the master thread alone cannot have done all the work -- if it did, the kernel never "
        "parallelised and the all-threads assertion above proved nothing")


@requires_papi
def test_open_counter_names_a_thread_it_cannot_attach_to() -> None:
    """The refusal is a REASON, not a warning: one thread we cannot see makes the sum wrong."""
    lib = papi.initialised()
    code = ctypes.c_int(0)
    lib.PAPI_event_name_to_code(b"PAPI_TOT_CYC", ctypes.byref(code))
    _eventset, why = papi.open_counter(lib, 0x7FFFFFF0, [code])  # a tid that cannot exist
    assert why is not None and "0x7ffffff0" not in why  # decimal, as /proc/self/task reports it
    assert str(0x7FFFFFF0) in why
