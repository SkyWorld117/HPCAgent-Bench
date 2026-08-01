# DESIGN: `hpcagent_papi.h` -- region-level hardware counters for optimizing agents

STATUS: **design only. Nothing here is implemented.** Open questions at the end are unanswered.

Today an agent's only counter surface is the judge's `POST /profile` with `counters:true`, which
counts the WHOLE run from outside. That cannot answer "which of my three loop nests is missing L2"
or "did my tiling actually raise arithmetic intensity in the hot region". This header lets an agent
bracket a REGION inside its own source.

Grounded in `hpcagent_bench/harness/papi.py`, `flags.py`, `languages.py`, `harness/sandbox.py`,
`envs/compilers.yaml`, and the DaCe reference on `spcl/dace` branch `papi-fix-2`.

---

## 0. The API (the whole surface)

```c
/* hpcagent_bench/envs/hpcagent_papi.h  -- GENERATED. Do not edit.
 * Source of truth: hpcagent_bench/harness/papi_header.py (tables from harness/papi.py).
 *
 *   #define HPCAGENT_PAPI_IMPLEMENTATION   // in EXACTLY one translation unit
 *   #include <hpcagent_papi.h>
 *
 * Nothing here is on the link line. libpapi is dlopen'd at first use; absent PAPI
 * degrades to a no-op with a named cause. Never aborts, never exits, never allocates
 * inside a region, never touches the kernel's arithmetic.
 */

/* ---- lifecycle. All idempotent. All silent no-ops once degraded. ---------------- */
int          hpc_papi_init(void);        /* 0 = counting; <0 = degraded (see cause) */
const char  *hpc_papi_cause(void);       /* "" while counting, else one of papi.CAUSES */
int          hpc_papi_passes(void);      /* runs needed to cover the resolved metric set */
int          hpc_papi_finalize(void);    /* write $HPC_PAPI_OUT (default ./hpc_papi.json) */

/* ---- regions. FLAT (never nested). id is an array index, so begin/end are O(1). -- */
int          hpc_papi_region(const char *name);  /* register ONCE, outside the hot path */
void         hpc_papi_begin(int region);
void         hpc_papi_end(int region);

/* ---- sweep: the header owns the pass x repetition schedule and the reseeding ----- */
typedef void (*hpc_papi_fill_fn)(uint64_t seed, void *ctx);
typedef void (*hpc_papi_run_fn )(void *ctx);
int          hpc_papi_sweep(hpc_papi_fill_fn fill, hpc_papi_run_fn run,
                            void *ctx, int reps);   /* reps <= 0 -> $HPC_PAPI_REPS or 7 */

/* ---- deterministic input fill, so `fill` is one line and reproducible from a seed - */
void         hpc_papi_fill_f64(double  *p, size_t n, uint64_t seed);          /* [0,1) */
void         hpc_papi_fill_f32(float   *p, size_t n, uint64_t seed);
void         hpc_papi_fill_i64(int64_t *p, size_t n, uint64_t seed,
                               int64_t lo, int64_t hi);
```

Eleven symbols. Two usage shapes:

```c
/* BRACKET -- inside a kernel the harness calls. Harness owns the inputs. */
static int R_TILE = -1;
void gemm_c_auto(const double *restrict A, ..., uint8_t *restrict ws, int64_t wsn) {
  if (R_TILE < 0) { hpc_papi_init(); R_TILE = hpc_papi_region("tile_loop"); }
  hpc_papi_begin(R_TILE);
  /* ... one loop nest ... */
  hpc_papi_end(R_TILE);
}
```

```c
/* SWEEP -- the agent's OWN driver, in its own workspace. Agent owns the inputs. */
static void fill(uint64_t s, void *c) { State *st = c; hpc_papi_fill_f64(st->A, st->n, s); }
static void run (void *c)             { State *st = c; gemm_c_auto(st->A, ..., 0, 0); }
int main(void) {
  State st = ...; hpc_papi_init();
  hpc_papi_sweep(fill, run, &st, 0);   /* passes x reps, reseeding between reps */
  return hpc_papi_finalize();
}
```

`hpc_papi_region` is the only string-taking call, and it is required to be outside the hot path.
`begin`/`end` take an int, so the bracket costs an array index + a fence + `PAPI_read` per thread.

**No nesting.** A second `begin` on a thread with a region already open sets
`cause=events_unsupported` for the inner region and drops it, rather than double-counting. One
thread's counter registers hold one armed event set per event; a nested region would need a second
set on the same events, which PAPI either refuses or multiplexes, and `papi.py`'s preamble already
refuses multiplexing.

---

## 1. Counter selection = the intersection, collected whole

At `hpc_papi_init`, in order (cheap checks first, mirroring `papi.check()` / `perf_event_reason()`):

1. `__APPLE__` -> `not_linux`, stop.
2. Read `/proc/sys/kernel/perf_event_paranoid`: absent -> `no_perf_events`; `> 2` ->
   `perf_event_paranoid`. Stop. Checked BEFORE dlopen for the reason `papi.py` states: PAPI's own
   error is `PAPI_ESYS` at `PAPI_start`, which reads like a broken install.
3. `dlopen("libpapi.so")` then `libpapi.so.<N>` -> `papi_missing`.
4. `PAPI_library_init` version probe, `major<<24|minor<<16`, descending -- the range comes from
   `papi.VERSION_MAJORS` / `VERSION_MINORS` -> `papi_init_failed`.
5. `PAPI_thread_init((unsigned long (*)())omp_get_thread_num)` (see section 4).
6. `B = PAPI_num_cmp_hwctrs(0)` -- the budget. `papi.hardware_counters()` uses exactly this call.
7. For every candidate event in the generated table, `PAPI_query_named_event`. Then run
   `papi.resolve`'s ladder in C: first candidate whose every event queried OK wins; a leading `-` is
   a sign, not part of the name. A metric with no surviving candidate is recorded `count:null` +
   `missing`, never substituted -- the `PAPI_FP_OPS` / `PAPI_FP_INS` rule holds unchanged.
8. The survivors are the INTERSECTION, and EVERY metric in it is collected. No fixed event list
   ships.

The table in the header is emitted from `papi.METRICS` verbatim:

```c
/* GENERATED from hpcagent_bench.harness.papi.METRICS -- 15 metrics, candidates best-first. */
static const char *const HPC_M_cache_hits[][4] = {
  {"PAPI_L1_DCH",0,0,0}, {"PAPI_L1_DCA","-PAPI_L1_DCM",0,0},
  {"PAPI_L2_DCH",0,0,0}, {"PAPI_L2_DCA","-PAPI_L2_DCM",0,0} };
/* ... */
```

There is no second table: `papi_header.py` walks `papi.METRICS` and prints this. A test asserts the
round trip (section 9).

---

## 2. Passes and repetitions

**Partition.** The intersection normally exceeds `B` (15 metrics vs 5 registers on Zen4). Metrics
are packed into PASSES by distinct event count `<= B`, greedy first-fit in `papi.METRICS` order (a
Python dict, insertion-ordered, so the packing is deterministic across hosts). Two rules on top:

- `cycles` and `instructions` are FORCED INTO EVERY PASS. They are the denominators of nearly all of
  `papi.RATIOS`, and `derive()` takes `seconds` from `needs[0]`'s row -- a ratio whose numerator and
  denominator came from two different passes is a ratio over two different schedules. Cost: `B-2`
  free slots per pass (3 on Zen4), so 15 metrics land in ~5 passes.
- A single metric whose own candidate needs more than `B` events is dropped with
  `events_unsupported`. Multiplexing is NOT offered, matching `papi.py`.

**Repetitions.** `R = 7` by default (`$HPC_PAPI_REPS`). Odd, so the median is an observed value and
not an interpolation; small enough that a 100 ms region costs `P*R*0.1s` = about 3.5 s.

**Schedule.** Rep-outer, pass-inner:

```
for rep in 0..R-1:
    fill(seed = splitmix64(base_seed ^ rep))     # ONE randomization per rep
    for pass in 0..P-1:
        arm(pass); run(); disarm(pass)           # every pass sees the SAME input
```

This resolves the tension. Random inputs change the data-dependent branch profile and (for
sparse/iterative kernels) the amount of work; that is what makes repetition informative. But if
pass 1 of rep 1 measured `branch_mispredictions` on one input and pass 2 of rep 7 measured
`branch_instructions` on another, `branch_misprediction_rate` divides two different workloads.
Holding the input fixed across the passes of one rep makes every ratio well-formed; rerandomizing
between reps makes the spread meaningful.

**Combination: median across reps, per (region, metric).** Not sum, not mean.

- NOT SUM. With randomized inputs the sum counts a workload nobody ran -- the union of R different
  problems. It is also not comparable across metrics whenever a pass is dropped (different
  cardinalities). WITHIN a repetition the per-THREAD counts ARE summed; that is a sum over a
  partition of one workload, and it is what `counting_worker` already does.
- NOT MEAN. Counter distributions across reps are right-skewed: one page-fault storm, one migration,
  one SMT neighbour, and the mean moves while the typical run does not.
- MEDIAN. A real observation, robust to one bad rep, and the number a reader should compare
  before/after a transform.

The report ships `samples[]` (the raw per-rep vector), `min`, `max` and `cv = stdev/median` next to
`count = median`, so nothing is discarded and the spread is readable -- a high `cv` on
`branch_mispredictions` with a flat `instructions` IS the finding for a data-dependent kernel.

**Where randomization can be reached from.** Only in sweep mode. In bracket mode the harness
generates the inputs (`_data_seeded`) and calls the kernel with the same buffers each rep; the
header cannot and must not touch them. The report stamps `"randomized": false` for bracket mode and
`true` for sweep mode, and the reader prints which, because rep-to-rep spread means different things
in the two cases. In sweep mode the agent's driver owns the buffers, `hpc_papi_sweep` calls
`fill(seed)` between reps, and `hpc_papi_fill_*` are splitmix64 + a fixed transform -- no libm, no
allocation, bit-reproducible from the seed, so a counted run is replayable.

---

## 3. Header-only, and how PAPI is reached

**One definition across TUs.** stb-style: without `HPCAGENT_PAPI_IMPLEMENTATION` the header is
declarations; with it, definitions. Four lines of preprocessor. The scored/restricted path writes
exactly one source file (`Sandbox.build` -> `self.root / f"{symbol}.{ext}"`) so it never sees the
problem; the `any` path and the sweep driver are multi-TU and do. `static`-everything was rejected:
it is silently wrong in multi-TU (per-TU counters), and silent wrongness is the failure mode this
whole subsystem exists to avoid.

**No `-lpapi`, ever.** The harness composes the link line from `compilers.yaml` + `flags.py` and the
submission cannot add optimization flags; `split_build` DOES let `-lpapi` through (`-l` prefix,
`_safe_link("papi")` passes) -- but requiring it would make the build FAIL on a host with no
`libpapi.so` dev symlink, which violates section 5. So: `dlopen` at first use, mirroring
`papi.check()`'s ctypes path, and `dlsym` for ~14 entry points:

```
PAPI_library_init  PAPI_thread_init      PAPI_register_thread  PAPI_unregister_thread
PAPI_create_eventset  PAPI_destroy_eventset  PAPI_cleanup_eventset
PAPI_assign_eventset_component  PAPI_add_named_event  PAPI_query_named_event
PAPI_num_cmp_hwctrs  PAPI_start  PAPI_stop  PAPI_read  PAPI_strerror
```

Consequence: the header must NOT `#include <papi.h>` (not guaranteed installed, and including it
would reintroduce a build dependency). It declares `PAPI_OK 0`, `PAPI_NULL -1` and the
function-pointer types itself -- the same three constants `papi.py` already declares for the same
reason.

`-ldl` is not needed on the target images (glibc >= 2.34 folded `libdl` into `libc`; Ubuntu
24.04/26.04 per `containers/*.def`). If a host needs it, `-ldl` is a legal `build` token.

`-fopenmp` is already on every CPU baseline (`CPU_BASELINE_GCC/CLANG/GFORTRAN`), so `omp.h` and
`omp_get_thread_num` are available with no build change.

**Reaching the include path.** `flags.PAPI_HEADER: pathlib.Path = paths.ROOT / "hpcagent_bench" /
"envs" / "hpcagent_papi.h"` -- the exact shape of the existing `flags.VECMATH_H`, shipped via
`setup.py` `package_data` (`'envs/hpcagent_papi.h'` beside `'envs/vecmath.h'`).
`Sandbox.build(debug=True)` appends `-I{PAPI_HEADER.parent}` next to `flags.DEBUG_SYMBOLS`;
`debug=False` does not. `/shared` is left alone -- it is an empty compose volume nothing populates
today, and using it would need new bring-up code.

---

## 4. OpenMP: carried over from `papi-fix-2`, not reinvented

Reference: `git show origin/papi-fix-2:dace/runtime/include/dace/perf/papi.h`, and commits
`1a4e83ebb` (thread-safe init), `d5621eea7` (thread registration), `1af672443` (counter-name
use-after-free).

**Carried over, verbatim in intent:**

| Reference behaviour | Where it lands here |
|---|---|
| `PAPI_library_init(PAPI_VER_CURRENT)` THEN `PAPI_thread_init((unsigned long(*)())omp_get_thread_num)` | `hpc_papi_init` steps 4-5. Without the second call every thread shares one PAPI thread context. |
| One event set per thread, created inside `#pragma omp parallel`, indexed by `omp_get_thread_num()` | `hpc_papi_arm(pass)` opens `#pragma omp parallel num_threads(omp_get_max_threads())` once per pass. |
| `PAPI_register_thread` + `PAPI_create_eventset` + `PAPI_add_named_event` ALL inside `#pragma omp critical` | Exactly this. Commit `1a4e83ebb`. The symptom without it is intermittent, not a hard failure, which is why it must be structural rather than tested-for. |
| `PAPI_register_thread` / `PAPI_unregister_thread` paired per thread | Paired in arm/disarm; disarm runs inside the same critical section. |
| Memory fence before `PAPI_start` and before `PAPI_stop` | `HPC_PAPI_FENCE` before both, and before `PAPI_read` in `begin`/`end`. |
| Per-thread storage cache-line aligned | See below. False sharing between threads' counter slots corrupts the measurement it is trying to take. |
| A failing `PAPI_add_named_event` warns, drops the event, keeps going -- never aborts | See below; strengthened. |
| Never keep a `char*` into a temporary (`1af672443`) | The header has no `std::string`; region names are `static const char *const` with static storage duration, copied into a fixed buffer by the JSON writer. |

**Fence.** The reference's ladder is `__x86_64__ && __GNUC__` -> `_mm_mfence()`, `_WIN64` ->
`MemoryBarrier()`, else NOTHING. "Else nothing" includes aarch64, which is in our target set
(`_ARCH_NATIVE` branches on Apple arm64; CSCS is Neoverse). Extended:

```c
#if defined(__x86_64__) && defined(__GNUC__)
#  include <x86intrin.h>
#  define HPC_PAPI_FENCE _mm_mfence()
#elif defined(__aarch64__)
#  define HPC_PAPI_FENCE __atomic_thread_fence(__ATOMIC_SEQ_CST)
#else
#  define HPC_PAPI_FENCE ((void)0)   /* reported in caveats, not silently assumed */
#endif
```

**Alignment.** The reference ships `AlignedElement` + `AlignedContainer` (~120 lines of
placement-new, `CHECK_BOUNDS` asserts, manual offset arithmetic). Replaced by three lines:

```c
typedef struct { _Alignas(HPC_PAPI_LINE) long long v[HPC_PAPI_MAXEV]; int eventset; int registered; }
        hpc_papi_slot;      /* HPC_PAPI_LINE generated from papi.cache_line_bytes(), 64 fallback */
static hpc_papi_slot *hpc_papi_slots;   /* one per thread, allocated once at arm() */
```

Same guarantee, C11, no template machinery.

**Degradation, strengthened past the reference.** The reference decrements a shared `_papi_nevents`
from inside the critical section, so the surviving event count depends on how many threads happened
to fail, and it calls `std::exit(EXIT_FAILURE)` when `PAPI_create_eventset` fails. Here: events are
resolved ONCE at init, before any thread exists, so every thread's event set is identical by
construction; a per-thread `PAPI_add_named_event` failure is impossible by then, and if it happens
anyway the whole pass degrades to `events_unsupported` rather than producing a shorter-than-declared
vector. Nothing ever calls `exit`.

**Deliberately dropped, and why:**

- Supersections / sections / `PAPIValueStore` / the flush threshold / `OVERHEAD_REPETITIONS`
  overhead compensation / `thread_lock_context_t` + `ThreadLockReleaser` / runtime byte-movement
  accumulation. All of it exists to attribute counts to SDFG NODES across a codegen'd program and to
  stream a growing report incrementally. We have one flat region table and write once at finalize.
  This is the entire size difference.
- `getThreadID()` via `sched_getcpu`. We key slots by `omp_get_thread_num()`, which is what the array
  is indexed by anyway; `sched_getcpu` is recorded once per thread as PLACEMENT data for the
  caveats, not as an identity.
- `template<int... events> PAPIPerf`. A compile-time event pack cannot express requirement 1 -- the
  event set is discovered on the machine at runtime.
- `PAPI_multiplex_init` / `PAPI_set_multiplex`. Refused, matching `papi.py`'s preamble. A multiplexed
  number is an estimate wearing a count's clothes.
- `LogError` appending to `errors.log` in the cwd. Writes to `stderr` behind `$HPC_PAPI_VERBOSE`
  instead; a file appearing in the judge's cwd is a side effect a graded run must not have.

---

## 5. Degradation, in the existing vocabulary

The generated cause strings are `papi.CAUSES`, in order, emitted as
`static const char *const HPC_PAPI_CAUSES[]`. A test pins the equality (section 9).
`hpc_papi_cause()` returns one of them, or `""`.

| Condition | Cause | What happens |
|---|---|---|
| macOS | `not_linux` | `init` returns <0; every `begin`/`end` is a no-op; report has `regions: []` + cause. |
| `perf_event_paranoid` absent | `no_perf_events` | same |
| `perf_event_paranoid > 2` | `perf_event_paranoid` | same; message names `sysctl -w` / `--cap-add=CAP_PERFMON`, as `perf_event_reason()` does |
| `dlopen` fails | `papi_missing` | same |
| version probe exhausted | `papi_init_failed` | same |
| no metric survives the intersection | `events_unsupported` | same |
| `PAPI_start` fails on a pass | `events_unsupported` | that pass's metrics are `count:null` + `missing`; other passes keep going |
| a thread appears after arm | `threads_moved` | that pass's metrics are dropped -- a sum missing a thread is a wrong number with no symptom |
| fewer harness calls than `P*R` (bracket mode) | `no_measured_rep` | the uncovered metrics only |
| one thread burned all the cycles | `not_openmp` | on the per-thread block only, never on the counts |

Invariants, in all cases: NO `exit`, NO `abort`, NO allocation inside a region, NO change to any
floating-point value, NO new link dependency, NO nonzero process exit code. A build cannot fail
because PAPI is absent, because PAPI is never on the link line. `count: null` + `missing` is never
`0` -- the rule `papi.missing()` enforces one level down, and the report's empty shape mirrors
`papi.missing_report()` so no reader can misread absence as a balanced, fast kernel.

---

## 6. The measurement trap

**Rule: a counted build is a diagnostic build. It is never the scored submission.**
`skills/general/SKILL.md` already forbids "time inside the kernel"; this is the same prohibition with
a counter instead of a clock.

Three layers, in order of strength:

1. STRUCTURAL. The `-I` that finds the header is added only by `Sandbox.build(debug=True)` -- the
   `/profile` path. A scored build (`debug=False`) has no include path to the header.
2. TOKEN DENY. `-I` and `-D` ARE agent-suppliable through `submission.build`
   (`_COMPILE_PREFIXES = ("-I", "-D")`), so layer 1 alone is bypassable. `split_build` must drop any
   `-D` matching `HPC_PAPI*` and any `-I` naming `flags.PAPI_HEADER.parent`, the same way it already
   drops `-O3`.
3. SOURCE SCAN. In `Sandbox.build`, when `debug=False`, refuse a submission whose source contains
   `hpcagent_papi.h` or `hpc_papi_` -- a `BuildResult(ok=False, log=<the rule>)`, which the scorer
   already turns into a zero-score datum. A refusal, not a silent strip: silently stripping would
   grade source the agent did not write. About six lines in one existing function.

For `any` delivery (a prebuilt `.so`, never rebuilt) layers 1-3 do not apply; see open questions.

**And the physics, for the skill.** `PAPI_read` costs ~1-3 us per thread and `HPC_PAPI_FENCE` drains
the store buffer. Bracketing an inner loop body perturbs it by more than most transforms gain. The
rule the agent is given: BRACKET A REGION THAT RUNS >= ~10 ms; NEVER A LOOP BODY; AND NEVER COMPARE A
COUNTED BUILD'S TIME TO ANYTHING. The counted run's wall clock is not the scored run's -- the same
rule `papi.GPU_CAVEATS` already states for device counters.

---

## 7. Tooling: one formula table, in Python

**The header computes no ratios. Ever.** It emits raw counts. `papi.RATIOS`, `papi.derive`,
`papi.imbalance`, `papi.PER_THREAD_FORMULAS`, `papi.IMBALANCE_FORMULA` stay the only place a division
happens. Enforced by schema, not discipline: the per-metric row the header writes is EXACTLY
`counting_worker`'s row, which is EXACTLY `derive()`'s input.

```json
{"schema":"hpcagent_papi/1","cause":"","missing":"","randomized":true,
 "host":"nid001234","cpu_model":"AMD Ryzen ...","hardware_counters":5,"line_bytes":64,
 "smt":true,"passes":5,"reps":7,"fence":"mfence",
 "regions":[{"name":"tile_loop","elapsed_ns":12345678,
   "metrics":[{"metric":"cycles","expression":"PAPI_TOT_CYC","events":["PAPI_TOT_CYC"],
               "derived":false,"count":94100322,"elapsed_ns":12345678,"reps_counted":7,
               "hardware_counters":5,"threads_counted":8,"scope":"all_threads","smt":true,
               "samples":[...],"min":...,"max":...,"cv":0.021}],
   "threads":[{"tid":4711,"cpu":3,"cycles":11800112,"instructions":24900871}]}]}
```

`derive()` reads only `metric`, `count`, `expression`, `elapsed_ns` -- the extra keys ride along
untouched.

**The reader:** `hpcagent_bench/harness/papi_header.py`, `main()`:

```sh
python -m hpcagent_bench.harness.papi_header --emit-header            # regenerate the .h
python -m hpcagent_bench.harness.papi_header --fragments --lang c     # the four fragments
python -m hpcagent_bench.harness.papi_header --read hpc_papi.json     # counts -> ratios
```

`--read` does, per region: `papi.derive(region["metrics"])`, then `profiling.render_counters` /
`render_ratios` -- the same tables the `/profile` endpoint already prints, with the same `formula`
and `reading` strings. Per-thread: `papi.quotient` twice with `papi.PER_THREAD_FORMULAS` as the
labels, then `papi.imbalance(cycles)` for the reduction, then `papi.measurement_caveats`. Zero new
arithmetic.

`measurement_caveats` and `flags.smt_enabled()` / `papi.governor()` read THIS host, so `--read`
refuses to render caveats when `host`/`cpu_model` in the file differ from the reading machine, and
says so instead of describing the wrong box.

Agent-facing doc: `hpcagent_bench/tools/region-counters.md`, which ships automatically
(`package_data` already globs `tools/*.md`, and `prompts.py` injects them).

---

## 8. Standalone generated form

`papi_header.py` emits four fragments per language: `papi-init`, `papi-start`, `papi-stop`,
`papi-finalize`.

**C / C++** -- switch is `-DHPC_PAPI`, and when it is absent the fragments are lexically absent from
the TU:

```c
/* papi-init */      #ifdef HPC_PAPI
                     #define HPCAGENT_PAPI_IMPLEMENTATION
                     #include <hpcagent_papi.h>
                     static int hpc_r_tile = -1;
                     #define HPC_PAPI_INIT() do { hpc_papi_init(); \
                         hpc_r_tile = hpc_papi_region("tile_loop"); } while (0)
                     #else
                     #define HPC_PAPI_INIT()      ((void)0)
                     #define HPC_PAPI_START(r)    ((void)0)
                     #define HPC_PAPI_STOP(r)     ((void)0)
                     #define HPC_PAPI_FINALIZE()  ((void)0)
                     #endif
/* papi-start */     HPC_PAPI_START(hpc_r_tile);
/* papi-stop  */     HPC_PAPI_STOP(hpc_r_tile);
/* papi-finalize */  HPC_PAPI_FINALIZE();
```

Off, the TU contains four `((void)0)` expression statements and no include. Byte-identical output is
a TEST, not an assumption (section 9). `-D` is confirmed reachable: `_COMPILE_PREFIXES` includes
`-D`, so `"build": ["-DHPC_PAPI"]` reaches the compile argv -- and is exactly what section 6 layer 2
must deny on the scored path.

**Fortran** -- two findings that change the answer:

- gfortran does NOT preprocess a lowercase `.f90`, `LANG_EXT["fortran"] == "f90"`, and `-cpp` is NOT
  a legal `build` token (`split_build` keeps only `-I -D -l -L`). So in the RESTRICTED path a Fortran
  submission has no preprocessor at all.
- `Sandbox.build` writes exactly ONE source file, so a Fortran submission cannot carry the C
  implementation TU alongside it.

Therefore **Fortran instrumentation is standalone-only**. The fragments are `bind(C)` interface
blocks against the header's own C entry points, linked with a C TU the agent's own driver owns --
where the agent controls the build line and `-cpp -DHPC_PAPI` works normally.

```fortran
! papi-init
#ifdef HPC_PAPI
  use, intrinsic :: iso_c_binding
  interface
    integer(c_int) function hpc_papi_init() bind(C, name="hpc_papi_init"); end function
    integer(c_int) function hpc_papi_region(nm) bind(C, name="hpc_papi_region")
      import :: c_int, c_char; character(kind=c_char), intent(in) :: nm(*)
    end function
    subroutine hpc_papi_begin(r) bind(C, name="hpc_papi_begin")
      import :: c_int; integer(c_int), value :: r
    end subroutine
    subroutine hpc_papi_end(r) bind(C, name="hpc_papi_end")
      import :: c_int; integer(c_int), value :: r
    end subroutine
    integer(c_int) function hpc_papi_finalize() bind(C, name="hpc_papi_finalize"); end function
  end interface
  integer(c_int), save :: hpc_r_tile = -1
#endif
```

Fallback for a Fortran build that cannot pass `-cpp`: a `logical, parameter :: hpc_papi_on = .false.`
guard, `if (hpc_papi_on) then ... end if`. A `parameter` constant is compile-time, so `-O3` eliminates
the branch and the interface blocks emit no code -- same byte-identity claim, same test.

REJECTED for restricted-mode Fortran: pure-Fortran fragments declaring `PAPI_*` directly via
`bind(C)`, which would work in one source file -- at the price of `-lpapi` on the link line, which
fails the build wherever PAPI is absent and violates section 5.

---

## 9. Tests that pin it to `papi.py`

`tests/test_papi_header.py`. Most run on any host; the compile-probes gate on
`languages.resolve_compiler("gcc")`, the run-probes on `ctypes.util.find_library("papi")` -- a named
predicate, never a swallowed exception, matching `test_papi_counters.py`.

1. `test_header_is_up_to_date` -- `papi_header.header_text()` equals the tracked
   `envs/hpcagent_papi.h` byte-for-byte.
2. `test_event_table_matches_papi_metrics` -- parse the generated C table back out and assert it
   equals `papi.METRICS` exactly: metric names, candidate order, term order, leading `-`. This is the
   "no duplicated table" invariant made mechanical.
3. `test_causes_enum_matches_papi_causes` -- the generated string array equals `papi.CAUSES`, in
   order.
4. `test_no_second_ratio_table` -- the header text contains none of the `papi.RATIOS` formula
   fragments (`/ cycles`, `1000 *`, `per_1k`, `line_bytes`, ...).
5. `test_report_rows_are_derive_input` -- a canned report fixture through `papi.derive`: every
   `RATIOS` key is either computed or in `unavailable` WITH A REASON.
6. `test_off_build_is_byte_identical` -- compile a fixture with and without `-DHPC_PAPI` (and the
   Fortran `parameter` on/off), compare the objects' `.text`. `flags.baseline_flags(lang)` supplies
   the flags so the probe is at the real level.
7. `test_scored_build_refuses_instrumentation` -- `Sandbox.build(debug=False)` on a source containing
   `hpc_papi_` returns `ok=False` with the rule in the log; `debug=True` succeeds;
   `split_build(["-DHPC_PAPI"])` drops it.
8. `test_header_never_links_papi` -- no `-lpapi` anywhere in the header or fragments; the documented
   `build` list survives `split_build` with no `-l` token added.
9. `test_partition_respects_budget` -- compile a probe with `$HPC_PAPI_BUDGET=4` and assert no pass
   declares more than 4 distinct events and that `cycles` + `instructions` are in every pass.
10. `test_skill_names_only_real_metrics_and_ratios` -- every metric/ratio name in
    `skills/profiling/SKILL.md` and `tools/region-counters.md` is a key of `papi.METRICS` /
    `papi.RATIOS`.

---

## 10. What the rewritten `profiling` SKILL.md must say

Invocation is ~15 lines at the top. INTERPRETATION is the rest. The formula table is NOT restated in
the skill -- the reader tool prints `formula` + `reading` with every value, and the skill says so.
One table, in `papi.RATIOS`.

**Keep, re-pointed at regions.** The existing 8-step decision procedure stays, with every step now
read PER REGION:

1. Which region owns the time? `region.elapsed_ns / sum` -- the region equivalent of `kernel_pct`.
   Below ~30%, go elsewhere; no counter reading changes that arithmetic.
2. `imbalance.max_over_mean` PER REGION. A run can be balanced overall and 1.8x imbalanced in one
   nest.
3. `ipc` -> 4. memory / 5. branches / 6. dependence chain / 7. right work / 8. did the transform do
   what you think -- unchanged, per region.

**New interpretation the region view enables and the whole-run view cannot:**

- TWO REGIONS OF ONE RUN, SIDE BY SIDE. The nest with low `ipc` and high
  `l3_misses_per_1k_instructions` is the memory-bound one EVEN WHEN THE RUN AVERAGE LOOKS HEALTHY.
  The average of a compute-bound and a memory-bound region describes neither. This is the "which of
  my three loop nests is missing L2" question, and it is the reason the header exists.
- THE SAME REGION, BEFORE AND AFTER. The only honest way to say "my tiling raised arithmetic
  intensity" is `arithmetic_intensity_flops_per_byte` on region X before vs after, at the same input.
  IF THE `expression` STRING DIFFERS BETWEEN THE TWO RUNS, THE COMPARISON IS VOID -- a different CPU
  or a different fallback rung answered a different quantity under the same metric name. Read
  `expressions`, not just the value.
- THE REP-TO-REP SPREAD. With `randomized:true`, a high `cv` on `branch_mispredictions` with a flat
  `instructions` is data-dependent control flow: go branchless. A high `cv` on `l3_cache_misses` with
  flat `instructions` is a working set that straddles a cache level for some inputs -- the median is
  the honest number and the max is the tail latency. With `randomized:false` (bracket mode), spread
  is machine noise only and carries no information about the kernel.
- REGIONS THAT DO NOT SUM TO THE RUN. The gap is what was not instrumented. It is the region-level
  `kernel_pct`, and it is where an unexpected 40% has hidden more than once.
- A ZERO IS A MEASUREMENT; `null` IS AN ABSENCE. Carried over verbatim from the current skill
  (`fma_instructions` reads exactly 0 for gemm on Zen4 because `PAPI_FMA_INS` is derived there and
  AMD does not feed it).
- THE TRAP, STATED FIRST AND HARD: never ship the counted build; never compare a counted run's TIME;
  bracket >= 10 ms, never a loop body.

---

## Open questions -- UNANSWERED, decide before implementing

1. **`any`-delivery enforcement.** A prebuilt `.so` is never recompiled, so section 6's three layers
   do not reach it. Options: `nm -D` for `hpc_papi_` (defeated by `static`), a `dlopen` check for the
   symbol, or accept it as unenforced and rely on the correctness/verify gates.
2. **Refusal severity.** Should a scored build containing `hpc_papi_` be a BUILD FAILURE (`ok=False`,
   scores 0 via the existing path) or a scored-but-flagged result? Design assumes the former.
3. **Who writes the header where.** Design chose "tracked in `envs/`, reached by a `debug`-only
   `-I`, mirroring `VECMATH_H`", and left `/shared` alone. Confirm you do not want it installed into
   `$HPCAGENT_BENCH_SHARED_DIR/include` at container bring-up instead.
4. **`PAPI_num_cmp_hwctrs(0)` vs `PAPI_num_counters()`.** Same number (the latter is the legacy
   alias). Design used the former, for consistency with `papi.py`.
5. **`R = 7` repetitions, median.** Confirm the default and the reduction. `min` (best-of-reps) is
   the alternative; the design argues against it because a COUNT has no "best".
6. **Sweep-mode input distribution.** `hpc_papi_fill_f64` is uniform `[0,1)`. The harness generates
   inputs through `hpcagent_bench.initialize` / `_data_seeded`, which is NOT uniform for every kernel
   (index arrays, SPD matrices, sparsity patterns). For a data-dependent kernel the two produce
   different counter profiles. Should the fill mirror the harness's distributions (much bigger
   header), or should the agent be told plainly that sweep-mode counts describe uniform inputs?
7. **Fortran restricted mode.** Confirm that leaving restricted-mode Fortran uninstrumented is
   acceptable, given `-cpp` is not a legal `build` token and only one source file is written.
8. **aarch64 fence.** `__atomic_thread_fence(__ATOMIC_SEQ_CST)` vs an explicit `dmb ish`. Not
   measured on Neoverse.
9. **Reader surface.** `python -m hpcagent_bench.harness.papi_header --read` (chosen) vs a
   `hpcagent-bench counters` CLI subcommand (more discoverable, more surface).
10. **Region cap.** Design assumed a fixed `HPC_PAPI_MAXREG` (say 32) so region storage is static and
    `begin`/`end` allocate nothing. Confirm 32, or name a number.
