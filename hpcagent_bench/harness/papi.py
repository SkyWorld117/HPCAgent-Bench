# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hardware counters through PAPI: ONE metric per run, each run in its own crashable child.

The ``perf`` half of "measure the machine, not the clock" lives in
:mod:`hpcagent_bench.perf_reports` (sampling -> call graph). This is the counting half, and it
answers a different question: perf says WHERE the cycles went, a counter says WHAT the hardware
did while they went there.

Five decisions, each one load-bearing:

**One metric per run.** A CPU has a small fixed number of counter registers (5 on the AMD Zen4
this was written on, 4-8 is typical). Asking for more events than that forces PAPI -- and perf --
to MULTIPLEX: time-slice the events and scale each partial count back up to a full-run estimate.
The number that comes out still looks like a count and is not one. So a metric never shares a run
with another metric, and the price is stated where a caller can see it: turning counters on costs
one extra measured run per metric.

**Availability is discovered, never assumed.** Preset events are a property of the CPU, and the
mapping is not guessable: on the machine this was developed on ``PAPI_L1_DCM`` is available while
``PAPI_L1_ICM``, ``PAPI_L3_DCM`` and ``PAPI_L1_TCM`` are not, and an Intel server splits them
differently. :func:`available_events` asks PAPI's own preset enumeration -- the same
``PAPI_enum_event`` + ``PAPI_query_event`` pair ``papi_avail`` prints its table from.

**ctypes, and only ctypes.** ``libpapi.so`` is already installed wherever PAPI is, so ctypes needs
no build step and no pip dependency. The alternatives were rejected rather than kept as fallbacks:
``papi_command_line`` counts its OWN synthetic loop, not a supplied program, so it cannot measure
anything here; scraping ``papi_avail``'s text would be a SECOND availability oracle free to
disagree with the first. There is one mechanism.

**A missing number is named, never substituted.** Each metric carries an ordered list of candidate
expressions and the first one whose every event this CPU has wins; a metric with no surviving
candidate comes back with ``missing`` saying why. Candidates may differ in cache LEVEL (an L2 miss
count answers the same question about the memory hierarchy, and the payload always states the
expression it used) but never in QUANTITY -- an instruction count is not an operation count, so
``PAPI_FP_INS`` is not a fallback for ``PAPI_FP_OPS``.

**Counters bracket exactly what the judge times.** A count that includes interpreter start-up and
input generation is a wrong number wearing a right label, so :func:`counting_worker` drives
:func:`~hpcagent_bench.harness.native_call._call_native_impl` with a ``timed_call`` of its own --
the seam that already promises to wrap ``fn(*c_args)`` and nothing else. That is also why this
module sits in the harness rather than beside the perf half, which deliberately imports nothing
from here.

Crash safety is :func:`~hpcagent_bench.frameworks.forked.run_forked`, the repo's one fork
isolation (it is what :func:`~hpcagent_bench.harness.native_call._call_isolated` is built from).
A segfault, an OOM kill or a PAPI bring-up failure during metric *k* costs metric *k*'s number and
nothing else.
"""
import ctypes
import ctypes.util
import functools
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hpcagent_bench import osinfo
from hpcagent_bench.frameworks.forked import forked_failure_reason, run_forked
from hpcagent_bench.harness.native_call import _call_native_impl, _current_vmsize_bytes
from hpcagent_bench.support.bindings.contract import Binding

#: PAPI's success code; everything else is an error whose text ``PAPI_strerror`` owns.
PAPI_OK = 0

#: The "no event set yet" handle ``PAPI_create_eventset`` expects to be handed.
PAPI_NULL = -1

#: High bit of a PRESET event code -- where the preset enumeration starts.
PRESET_MASK = 0x80000000

#: ``PAPI_enum_event`` modifiers: 1 seeds the walk at the first preset, 0 steps to the next.
ENUM_FIRST = 1
ENUM_NEXT = 0

#: ``PAPI_MAX_STR_LEN``: the buffer ``PAPI_event_code_to_name`` writes into.
NAME_LEN = 128

#: (major, minor) pairs tried against ``PAPI_library_init``, newest first. The version it demands
#: is ``PAPI_VER_CURRENT`` -- a header constant, ``major << 24 | minor << 16`` -- so a literal here
#: would pin one PAPI release, and there is nothing better to read it from: ``papi.h`` is not
#: guaranteed installed and the library exports no version symbol. The check runs BEFORE PAPI
#: touches any state and rejects a mismatch with ``PAPI_EINVAL``, so probing is safe and costs
#: microseconds.
VERSION_MAJORS = range(9, 2, -1)
VERSION_MINORS = range(15, -1, -1)

#: Metric -> candidate expressions, best first. A candidate is a tuple of preset event names; a
#: leading ``-`` subtracts. The first candidate whose every event this CPU reports wins.
#:
#: Names describe the QUANTITY the events actually deliver, not the question they are usually
#: asked. PAPI has no integer-*operation* or FMA-*operation* preset on any CPU -- both presets
#: count INSTRUCTIONS -- and one packed AVX-512 FMA is one instruction and thirty-two operations,
#: so calling either an "op" count would be the exact mislabeling this module refuses.
#:
#: ``cache_hits`` shows why derivation is worth having: almost no CPU exposes ``PAPI_L1_DCH``
#: directly, but accesses minus misses is the same number, exactly, from two events that fit in
#: the counter budget together.
METRICS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "data_cache_misses": (("PAPI_L1_DCM", ), ("PAPI_L2_DCM", ), ("PAPI_L3_DCM", )),
    "instruction_cache_misses": (("PAPI_L1_ICM", ), ("PAPI_L2_ICM", ), ("PAPI_L3_ICM", )),
    "instructions": (("PAPI_TOT_INS", ), ),
    "cache_hits":
    (("PAPI_L1_DCH", ), ("PAPI_L1_DCA", "-PAPI_L1_DCM"), ("PAPI_L2_DCH", ), ("PAPI_L2_DCA", "-PAPI_L2_DCM")),
    "fp_ops": (("PAPI_FP_OPS", ), ("PAPI_DP_OPS", "PAPI_SP_OPS")),
    "integer_instructions": (("PAPI_INT_INS", ), ),
    "fma_instructions": (("PAPI_FMA_INS", ), ),
}


class PapiUnavailable(RuntimeError):
    """PAPI cannot count here. ``cause`` is the machine-readable reason (``not_linux`` /
    ``papi_missing`` / ``papi_init_failed`` / ``not_native``); the message names the fix.

    Deliberately shaped like :class:`hpcagent_bench.perf_reports.PerfUnavailable`, down to the
    ``cause`` attribute: a caller that already handles "this host cannot sample" handles "this
    host cannot count" with the same branch instead of a second one."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


@functools.lru_cache(maxsize=None, typed=True)
def check() -> ctypes.CDLL:
    """The loaded ``libpapi`` handle, or :class:`PapiUnavailable` naming the cause and the fix.

    LOADS but does not initialize, which splits the two questions that have different answers at
    different times: "can this host count at all" is a property of the machine and must be
    answerable in the PARENT before anything is compiled, while bring-up belongs in the process
    that counts (:func:`initialised`). PAPI's own guidance is to initialize per process; a forked
    child inheriting an initialized parent measured fine here but is not contractual, and the
    split costs nothing.
    """
    if not osinfo.IS_LINUX:
        raise PapiUnavailable(
            "not_linux", "PAPI counting is wired for Linux only; on macOS the hardware counters "
            "are behind Instruments' 'CPU Counters' template, which cannot be driven from here")
    name = ctypes.util.find_library("papi") or "libpapi.so"
    try:
        return ctypes.CDLL(name)
    except OSError as exc:
        raise PapiUnavailable(
            "papi_missing", f"libpapi could not be loaded ({exc}); install PAPI (Debian/Ubuntu: "
            "'apt install libpapi-dev', or build it from https://github.com/icl-utk-edu/papi) and "
            "make sure the library is on the loader path") from exc


@functools.lru_cache(maxsize=None, typed=True)
def initialised() -> ctypes.CDLL:
    """``check()`` plus ``PAPI_library_init``, version discovered by asking (:data:`VERSION_MAJORS`).

    Belongs in the process that will do the counting; see :func:`check` for why the host gate is
    a separate call.
    """
    lib = check()
    for major in VERSION_MAJORS:
        for minor in VERSION_MINORS:
            wanted = (major << 24) | (minor << 16)
            if lib.PAPI_library_init(wanted) == wanted:  # returns the version on success, <0 otherwise
                return lib
    raise PapiUnavailable(
        "papi_init_failed", "PAPI_library_init rejected every version from "
        f"{VERSION_MAJORS.start}.x down to {VERSION_MAJORS.stop + 1}.x: the loaded libpapi is "
        "either newer than this range or broken ('papi_avail' will print the same failure)")


def strerror(lib: ctypes.CDLL, code: int) -> str:
    """PAPI's own text for an error code -- its table, so it stays right across PAPI versions."""
    lib.PAPI_strerror.restype = ctypes.c_char_p
    text = lib.PAPI_strerror(code)
    return text.decode() if text else f"PAPI error {code}"


def demand(lib: ctypes.CDLL, code: int, what: str) -> None:
    """Raise unless ``code`` is :data:`PAPI_OK`. Raising is right here: every caller is already
    inside a forked child, so the failure becomes ONE metric's ``missing`` reason."""
    if code != PAPI_OK:
        raise PapiUnavailable("papi_init_failed", f"{what} failed: {strerror(lib, code)}")


@functools.lru_cache(maxsize=None, typed=True)
def available_events() -> Tuple[str, ...]:
    """Every PAPI PRESET event THIS CPU can actually count, in PAPI's enumeration order.

    The oracle is PAPI's own: ``PAPI_enum_event`` walks the preset table and ``PAPI_query_event``
    answers "does this CPU have it". Nothing is hardcoded per CPU because nothing can be -- the
    same preset name is available on one machine and absent on the next, which is the whole reason
    this function exists rather than a constant.
    """
    lib = initialised()
    code = ctypes.c_int(PRESET_MASK)
    name = ctypes.create_string_buffer(NAME_LEN)
    if lib.PAPI_enum_event(ctypes.byref(code), ENUM_FIRST) != PAPI_OK:
        return ()
    out: List[str] = []
    while True:
        if lib.PAPI_query_event(code.value) == PAPI_OK and lib.PAPI_event_code_to_name(code.value, name) == PAPI_OK:
            out.append(name.value.decode())
        if lib.PAPI_enum_event(ctypes.byref(code), ENUM_NEXT) != PAPI_OK:
            return tuple(out)


def hardware_counters() -> int:
    """How many events this CPU can count AT ONCE (0 when PAPI will not say).

    Reported alongside the numbers so a reader can check the premise: the moment a run asks for
    more events than this, its counts are multiplexed estimates rather than counts.
    """
    return max(0, int(initialised().PAPI_num_cmp_hwctrs(0)))


def event_name(term: str) -> str:
    """The event a candidate term names, sign stripped."""
    return term[1:] if term.startswith("-") else term


def resolve(metric: str, available: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """The first candidate expression for ``metric`` whose every event is in ``available``.

    Pure, so the fallback ladder is testable on any host: hand it the event set of a CPU that
    lacks ``PAPI_L1_DCH`` and it must return the accesses-minus-misses derivation instead.
    """
    have = frozenset(available)
    for candidate in METRICS[metric]:
        if all(event_name(term) in have for term in candidate):
            return candidate
    return None


def expression(terms: Sequence[str]) -> str:
    """The candidate as readable arithmetic -- ``'PAPI_L1_DCA - PAPI_L1_DCM'``.

    Shipped with every count. The metric name states the QUESTION; this states the quantity that
    actually answered it, which is the difference between a number and a labelled number.
    """
    parts = [event_name(terms[0])]
    parts += [f"{'-' if t.startswith('-') else '+'} {event_name(t)}" for t in terms[1:]]
    return " ".join(parts)


def combine(terms: Sequence[str], values: Sequence[int]) -> int:
    """Signed sum of one reading, so a derived metric is one number like a direct one."""
    return sum(-v if t.startswith("-") else v for t, v in zip(terms, values))


def missing(metric: str, reason: str) -> dict:
    """The "no number for this metric" payload. Always the same shape as a successful one, with
    ``count`` explicitly ``None`` -- a caller must never have to tell absence from zero."""
    return {"metric": metric, "count": None, "missing": reason}


def counting_worker(lib_path: str, binding: Binding, data: Dict, lang: str, workspace_bytes: Optional[str], metric: str,
                    reps: int, warmup: int, rep_timeout: float, memory_bytes: int) -> dict:
    """CHILD: resolve ``metric`` on this CPU and count it around the timed call. Returns the payload.

    Resolution happens HERE, before the kernel runs, so a metric this CPU cannot express costs a
    fork and not a measured run. PAPI is brought up here too, in a process that exits right after,
    which is what PAPI's own per-process initialization guidance asks for.

    The reading is taken per call and the counted value is the FASTEST measured rep's, matching
    the best-of-reps reduction the score uses, so the counts and the time describe the same rep.
    Warmup reps are dropped exactly as :func:`hpcagent_bench.harness.timing.sampled_reps` drops
    their samples -- a cold-cache first pass would otherwise dominate every miss count.
    """
    import resource  # child-local, like _native_call_worker's: nothing in the parent needs it
    terms = resolve(metric, available_events())
    if terms is None:
        tried = ", ".join(expression(c) for c in METRICS[metric])
        return missing(metric, f"no candidate is available on this CPU (tried: {tried})")

    lib = initialised()
    # Same additive RLIMIT_AS cap _native_call_worker applies: the kernel's allowance ON TOP of
    # the interpreter footprint, so a runaway allocation dies in this child instead of the box.
    if memory_bytes > 0:
        cap = _current_vmsize_bytes() + memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    codes = [ctypes.c_int(0) for _ in terms]
    for term, code in zip(terms, codes):
        demand(lib, lib.PAPI_event_name_to_code(event_name(term).encode(), ctypes.byref(code)), f"lookup {term}")
    eventset = ctypes.c_int(PAPI_NULL)
    demand(lib, lib.PAPI_create_eventset(ctypes.byref(eventset)), "PAPI_create_eventset")
    for term, code in zip(terms, codes):
        demand(lib, lib.PAPI_add_event(eventset, code), f"add {event_name(term)}")

    width = len(terms)
    before = (ctypes.c_longlong * width)()
    after = (ctypes.c_longlong * width)()
    readings: List[Tuple[int, List[int]]] = []

    def counted(fn, c_args) -> int:
        # Read-delta rather than start/stop per rep: PAPI_start arms the counters once, and a
        # pair of reads is the cheapest bracket that still isolates ONE call.
        demand(lib, lib.PAPI_read(eventset, before), "PAPI_read")
        t0 = time.perf_counter_ns()
        fn(*c_args)
        ns = time.perf_counter_ns() - t0
        demand(lib, lib.PAPI_read(eventset, after), "PAPI_read")
        readings.append((ns, [int(after[i] - before[i]) for i in range(width)]))
        return ns

    demand(lib, lib.PAPI_start(eventset), "PAPI_start")
    _call_native_impl(lib_path,
                      binding,
                      data,
                      lang,
                      workspace_bytes,
                      xp=np,
                      to_host=lambda a: a,
                      timed_call=counted,
                      reps=reps,
                      warmup=warmup,
                      rep_timeout=rep_timeout)
    # Disarm only, and unchecked: the counts are already harvested per rep, so a teardown error
    # must not be allowed to throw away good numbers.
    lib.PAPI_stop(eventset, after)

    measured = readings[warmup:] or readings
    elapsed_ns, raw = min(measured, key=lambda r: r[0])
    return {
        "metric": metric,
        "expression": expression(terms),
        "events": [event_name(t) for t in terms],
        "derived": len(terms) > 1,
        "count": combine(terms, raw),
        "elapsed_ns": elapsed_ns,
        "reps_counted": len(measured),
        "hardware_counters": hardware_counters(),
    }


def count_metric(lib_path: str,
                 binding: Binding,
                 data: Dict,
                 lang: str,
                 metric: str,
                 *,
                 workspace_bytes: Optional[str] = None,
                 reps: int = 1,
                 warmup: int = 0,
                 rep_timeout: float = 0.0,
                 memory_gb: float = 0.0) -> dict:
    """Count ONE metric over ``reps`` timed calls of ``lib_path``'s kernel, in an isolated child.

    Never raises for a measurement failure: a segfault, an OOM kill, a PAPI bring-up error or a
    timeout all come back as :func:`missing` with the reason. That is the whole point of the
    per-metric run -- losing metric *k* must cost metric *k*'s number and nothing else, and the
    parent must still be alive to run metric *k+1*.
    """
    run = run_forked(counting_worker,
                     str(lib_path),
                     binding,
                     data,
                     lang,
                     workspace_bytes,
                     metric,
                     reps,
                     warmup,
                     rep_timeout,
                     int(memory_gb * (1024**3)),
                     label=f"papi:{metric}",
                     timeout=max(1.0, rep_timeout) * (warmup + max(1, reps) + 2))
    if not run.ok:
        return missing(metric, f"counted run failed ({forked_failure_reason(run)})")
    return run.result
