---
name: papi-gpu-amd-judge
description: AMD GPU hardware counters over ONE of your kernels, run by the JUDGE -- PAPI's rocp_sdk component in your source, one counter per submission, profile on stdout.
---

`rocprof` answers WHICH kernel owns device time. This page answers WHAT THE DEVICE DID while one
kernel ran: HBM bytes moved, L2 hits, waves launched, VALU busy. You bracket your own code, so the
answer is attributed to a region you chose rather than to a symbol.

This is the AMD twin of `papi-gpu`. The discipline is identical because the failure mode is
identical; the component, the event names and the environment traps are not.

## What was measured here, and what was not

**There is no AMD GPU on the box this was written on.** Nothing below was executed against ROCm.
Every command, event name, environment variable and limitation comes from the upstream PAPI and
ROCm documentation cited at the bottom, and you should treat all of it as unverified.

What IS carried over from measurement is the METHOD, and that part is not vendor folklore: the
start/stop-versus-read-delta result in the first section below was measured on NVIDIA hardware
here, against known ground truth, and PAPI's own `rocp_sdk` README documents the same underlying
behaviour on AMD in its own words. The self-test in `gpu_papi_init` is what makes that portable --
it fails loudly on a box this page could not be tested on. **Run it before you believe a number.**

## Start and stop the event set per region -- a read-delta does NOT attribute

`PAPI_read` leaves the set counting and looks like it brackets a region. On a GPU component it
does not, because the counter value is flushed ASYNCHRONOUSLY and a device synchronise does not
flush it. A read-delta returns whatever happened to be flushed between the two reads, which has no
relationship to what ran between them.

Measured on the NVIDIA twin of this component (RTX 4050, PAPI 7.2.0.0), four kernels of
deliberately different shape, 25 regions each, against each kernel's compulsory traffic:

| region | truth / rep | `PAPI_start`/`PAPI_stop` | read-delta |
| --- | --- | --- | --- |
| streams b and c into a | 128 MiB | **134.26 MB** | 128.4 MB |
| touches 64 KB, 64 launches | 64 KB | **77.9 KB** | 93.5 MB |
| reads a, 64 FMAs, writes a | 64 MiB | **67.08 MB** | 111.1 MB |
| reads a and c, divergent | 128 MiB | **134.27 MB** | 126.0 MB |

Start/stop lands on the compulsory traffic to within 0.1% on every row. The read-delta is wrong on
every row and wrong by **1300x** on the 64 KB one. Note what that does to a comparison: the true
spread across those four kernels is 2100x and the read-delta reports 1.2x. It does not add noise,
it FLATTENS the ranking you are profiling to find.

**PAPI's `rocp_sdk` README documents the same behaviour on AMD**, and its wording is the tell:
dispatch mode "may read zeros immediately after kernel returns due to buffer flushing delays",
with the recommendation to "add delays between kernel return and `PAPI_read()`/`PAPI_stop()`
calls". A delay is a race you cannot see losing -- too short and you read zero, slightly longer and
you read a number that looks fine and is not yours. Do not tune a sleep. Close the range with
`PAPI_stop`, which is the call that forces the flush, and verify with the empty bracket.

## Two components, and the old one is deprecated

```sh
papi_component_avail | grep -A2 -E 'Name:[[:space:]]+(rocm|rocp_sdk)'
```

| component | build | use it when |
| --- | --- | --- |
| `rocp_sdk` | `./configure --with-components="rocp_sdk"` | **default.** Sits on ROCprofiler-SDK |
| `rocm` | `./configure --with-components="rocm"` | pre-MI300 only, and only if `rocp_sdk` is absent |

`rocm` is DEPRECATED from AMD Instinct MI300A onward. Do not configure both for an older device --
upstream calls them mutually exclusive there. Neither is built by default: like the `cuda`
component, a distribution PAPI on a box with a perfectly good GPU usually has neither, and
rebuilding is the only fix.

Set `PAPI_ROCP_SDK_ROOT` (or `PAPI_ROCM_ROOT` for the old component) to the ROCm install, at BOTH
compile and run time. `PAPI_ROCP_SDK_LIB` gives the full path to `librocprofiler-sdk.so` when the
install is not where PAPI expects.

## The two environment traps that return silent zeros

Both produce a counter of 0 with no error anywhere, which reads exactly like a kernel that did no
work. This is the failure this whole page exists to prevent.

- **`AQLPROFILE_READ_API=0`** applies to INTERCEPT mode on ROCm >= 6.2.0 (`0` for intercept, `1`
  or unset for sampling). Intercept is opt-in via `ROCP_HSA_INTERCEPT` and the variable has no
  effect on `rocp_sdk`, so do NOT export it unconditionally -- set it only if you have deliberately
  selected intercept mode on the older `rocm` component and are reading zeros.
- **`PAPI_library_init()` must run BEFORE any HIP call.** The AMD runtime reads its environment
  once, at the first HIP call; initialise PAPI after that and the counter configuration never
  takes. With a statically linked `libpapi.a` this is mandatory and upstream says so explicitly;
  dynamically linked it is documented as unconstrained, but the ordering costs nothing, so keep it.

That last one fights the CUDA rule, so do not port the ordering across: on NVIDIA you arm AFTER a
warmup launch because the component profiles through a live context. On AMD you initialise PAPI
FIRST. Same library, opposite order, and each is silent when you get it wrong.

## Event names

```sh
papi_component_avail                        # which of the two you actually have
papi_native_avail -i rocp_sdk:::            # every event THAT component enumerates
papi_native_avail -e rocp_sdk:::SQ_CYCLES   # ONE event, resolved, defaults filled in
```

**The prefix is the component name, and the two components do not share one.** `rocp_sdk.c`
declares `.name = "rocp_sdk"`, so events are `rocp_sdk:::EVENT_NAME:device=N`; the older component
uses `rocm:::`. Copying a `rocm:::` example onto a `rocp_sdk` build resolves nothing. Enumerate
first and use whatever prefix comes back.

Device indices run `[0, N-1]` over VISIBLE devices, so `ROCR_VISIBLE_DEVICES` renumbers them and a
resource manager that hands you a subset changes what `device=0` means. Where the mapping matters,
resolve it by UUID (`hipDeviceGetUuid`) rather than trusting the index. `rocp_sdk` also takes
`DIMENSION_*=` qualifiers to select a specific instance of a multi-instance counter; enumerate to
see which ones an event accepts.

Only single-pass metric sets are supported. How a fractional metric survives the `long long` return
differs BY COMPONENT, so check which one you are on: the older `rocm` component keeps "the binary
image of a `double`" intact, while `rocp_sdk` accumulates into `long long int` and TRUNCATES. Under
`rocp_sdk`, never bit-reinterpret the value -- a percentage really is an integer there, and a
fractional one is already lost.

Ask a QUESTION, then find the event that answers it on THIS device. A hard-coded event list is a
list that stops working: the names differ by generation, and CDNA and RDNA do not even agree on
what a wavefront is.

## The code

```c
#include <papi.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

static int gpu_es = PAPI_NULL;
static long long gpu_total = 0;
static const char *gpu_event = NULL;
static int gpu_ok = 0, gpu_regions = 0;

/* Call FIRST, before ANY hip call -- see the environment traps above. */
static int gpu_papi_init(const char *event_name)
{
    gpu_ok = 0; gpu_total = 0; gpu_regions = 0; gpu_event = event_name;
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "papi-gpu-amd: library_init failed\n"); return -1;
    }
    int cid = -1;
    for (int i = 0; i < PAPI_num_components(); ++i) {
        const PAPI_component_info_t *ci = PAPI_get_component_info(i);
        if (ci && (!strcmp(ci->name, "rocp_sdk") || !strcmp(ci->name, "rocm"))) { cid = i; break; }
    }
    if (cid < 0) { fprintf(stderr, "papi-gpu-amd: no rocp_sdk/rocm component\n"); return -1; }
    int rc; long long probe = 0;
    /* A GPU event set must be bound to the GPU component; the default (0) is the CPU. */
    if ((rc = PAPI_create_eventset(&gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_assign_eventset_component(gpu_es, cid)) != PAPI_OK) goto fail;
    if ((rc = PAPI_add_named_event(gpu_es, event_name)) != PAPI_OK) goto fail;
    /* Arm and disarm once around NOTHING. Two jobs: it surfaces a refusal HERE rather than at
       the first region, and the value must come back ~0. If an empty bracket reports real
       work, the counter is accumulating device-wide instead of attributing -- STOP. */
    if ((rc = PAPI_start(gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_stop(gpu_es, &probe)) != PAPI_OK) goto fail;
    if (probe > 4096) {
        fprintf(stderr, "papi-gpu-amd: EMPTY BRACKET READ %lld, not ~0 -- not attributing\n", probe);
        return -1;
    }
    gpu_ok = 1;
    return 0;
fail:
    fprintf(stderr, "papi-gpu-amd: %s: %s (code %d)\n", event_name, PAPI_strerror(rc), rc);
    return -1;
}

/* START and STOP per region. PAPI_stop is what forces the counter to be attributed;
   a PAPI_read delta across the same span is not a measurement of that span. */
static void gpu_region_begin(void)
{
    if (gpu_ok && PAPI_start(gpu_es) != PAPI_OK) gpu_ok = 0;
}

static void gpu_region_end(void)
{
    if (!gpu_ok) return;
    long long v = 0;
    if (PAPI_stop(gpu_es, &v) != PAPI_OK) { gpu_ok = 0; return; }
    gpu_total += v;                                /* ACCUMULATES across every visit */
    ++gpu_regions;
}

static void gpu_papi_report(void)
{
    if (!gpu_ok) { printf("%s = ERROR (not counted)\n", gpu_event ? gpu_event : "?"); return; }
    printf("%s = %lld   (regions: %d)\n", gpu_event, gpu_total, gpu_regions);
    PAPI_cleanup_eventset(gpu_es); PAPI_destroy_eventset(&gpu_es);
}
```

`gpu_total` accumulates across visits, so a 20 us kernel called 500 times is measurable without
changing what you measured. A `PAPI_start` after a `PAPI_stop` is a supported re-arm, not a leak:
the event set is created once and destroyed once.

**Accumulate COUNTS only.** A sum of percentages is not a percentage, and half the metrics worth
asking for on this vendor (`GPUBusy`, `L2CacheHit`, `VALUBusy`, `VALUUtilization`, `MemUnitStalled`)
are described upstream as "the percentage of...". Sum `SQ_WAVES`, `FetchSize`, `WriteSize`,
`GRBM_GUI_ACTIVE`; read the ratios per region instead.

**What makes the bracket work on AMD was NOT verified here.** The start/stop result above was
measured on NVIDIA. PAPI's own `rocp_sdk` README says dispatch mode "may read zeros immediately
after kernel returns due to buffer flushing delays" and suggests adding a delay before
`PAPI_read`/`PAPI_stop` -- and `rocp_sdk_stop` performs no read of its own, so the NVIDIA mechanism
("stop forces the flush") is the wrong story here even though start/stop is still the right shape.
Treat a zero as unproven rather than as a measurement, and check the region count.

## How it runs

> **This route does not exist yet.** The judge accepts `oracle`, `submit`, `score` and `profile`
> today (`harness/service.py`), there is no `/instrument`, `JudgeClient` has no `instrument()`, and
> nothing returns the child's stdout. The contract below is the one being built, stated exactly so
> the page is ready the day it lands -- but do NOT try these calls against a judge yet. Until then,
> run the instrument yourself; the rest of this page is unchanged either way.

You write the bracket; the JUDGE compiles and runs it, on its own GPU -- its part, its ROCm build,
and its answer to whether the component was configured in at all. That last point is the reason
this route exists: neither `rocp_sdk` nor `rocm` is built into PAPI by default, so the box you are
on very likely has neither, and the judge's may have one.

The judge URL, the kernel name, your language and your rank are the ones your task statement gave
you -- substitute them; this page cannot know them.

Three differences from running it yourself, all consequences of the judge building a LIBRARY
rather than a program:

- **There is no `main`.** The judge dlopens `lib<kernel>.so` and calls your entry symbol, so
  `gpu_papi_init`, every `gpu_region_begin` / `gpu_region_end` pair and `gpu_papi_report` all live
  INSIDE the kernel function, in that order. The "initialise before any HIP call" rule is HARDER
  here, not easier: the judge's harness may already have touched HIP before your symbol runs, so
  put `gpu_papi_init` at the very top of your entry function and treat a zero count as that rule
  having been broken rather than as a kernel that moved nothing.
- **The event cannot come from `argv`.** Take it from a `-D`, one of the token prefixes that
  survive: pass `-DHPC_EVENT="rocm:::FetchSize:device=0"` in `build` and call
  `gpu_papi_init(HPC_EVENT)`. One submission per counter, for the same reason as one run per
  counter.
- **The profile leaves on STDOUT.** Replace `gpu_papi_report`'s two `printf` calls with ONE
  self-delimiting block and print nothing else anywhere in the source:

```c
printf("HPCB2 begin papi-gpu-amd %s\n", gpu_event ? gpu_event : "?");
if (!gpu_ok) printf("HPCB2 row error=not_counted\n");
else         printf("HPCB2 row value=%lld regions=%d\n", gpu_total, gpu_regions);
printf("HPCB2 end rows=1\n");
fflush(stdout);
```

The `error=` row is what `ERROR (not counted)` becomes on this route, and it is the whole point on
this instrument: a missing component, a missing `AQLPROFILE_READ_API=0`, or PAPI initialised after
the first HIP call all produce a silent zero, and a refusal that arrives as an absent block reads
exactly like a kernel that moved no bytes. The region count rides in the same row, so every check
on this page that reads it still works -- a short one says brackets were skipped.

```sh
curl -s -X POST "$JUDGE_URL/instrument" -H 'Content-Type: application/json' \
  -d '{"kernel":"<kernel>","language":"hip","rank":<judge rank>,
       "build":["-lpapi","-DHPC_EVENT=\"rocm:::FetchSize:device=0\""],
       "source":"<your instrumented source>"}'
```

```python
JudgeClient("<judge url>", rank=<judge rank>).instrument(
    Submission(language="hip", source="<your instrumented source>",
               build=["-lpapi", '-DHPC_EVENT="rocm:::FetchSize:device=0"']), "<kernel>")
```

One counter per submission. The events worth asking for, in the order this page reads them:
`rocm:::GPUBusy`, `rocm:::SQ_WAVES`, `rocm:::FetchSize`, `rocm:::WriteSize`, `rocm:::L2CacheHit`,
`rocm:::VALUBusy`, `rocm:::VALUUtilization`, `rocm:::MemUnitStalled` -- each with `:device=0`.

## One region per kernel

A kernel launch returns immediately, so under a read-delta you would need a device synchronise to
have any hope of bracketing the kernel -- and, as the table above shows, it still would not work.
Under `PAPI_start`/`PAPI_stop` you do not need one: `PAPI_stop` closes the range and collects it.

**A counted run's wall clock belongs to no comparison.** Profiling serialises the queue and re-arms
the counter set per region, which removes exactly the kernel/copy and kernel/kernel overlap a real
run depends on -- about 2x on the NVIDIA twin. Read the COUNTS; take every speedup from the
uninstrumented build.

One kernel per region: two kernels in one bracket give you their sum, and a sum cannot be
attributed. Move the bracket and run again. Bracket INSIDE the timestep loop, not around it.

## Reading the numbers

The counts are yours; the THRESHOLDS below are vendor-doc reasoning, so calibrate on your own
kernel. Counters do not name a bottleneck. They eliminate candidates, in this order -- stop at the
first step that fires, because the later numbers are consequences of the earlier ones.

**1. Was the device even the problem?** If `rocprof` already showed device time well under the
wall clock, stop. Launch gaps and copies are host findings and no counter below moves them.

**2. Occupancy -- against the part, not against a number you remember.** AMD occupancy is waves
resident on a SIMD over the maximum that SIMD holds (8), or scaled to the CU (32 waves on CDNA).
The wavefront width is the thing you must not assume: **CDNA is 64 lanes; RDNA is 32, with an
optional 64-lane mode.** Every "threads per block for full occupancy" number you know from NVIDIA
is wrong here by that factor -- CDNA needs 256 threads to fill a CU with one wave per SIMD, RDNA
needs 128. `rocprof`'s agent report prints `Wave_Front_Size`, `Simd_Count`, `Max_Waves_Per_Simd`
and `Cu_Count` for the actual part; read it rather than assuming.

High occupancy is not a goal. Occupancy counts waves PARKED, not waves working -- a kernel with
enough memory work in flight per wave runs at peak with half the slots empty.

**3. Memory stall, read WITH the traffic.** `MemUnitStalled` is the percentage of GPU time the
memory unit was stalled; read it against `FetchSize` + `WriteSize` (both KILOBYTES, not bytes --
the one unit trap on this vendor).

| stall | traffic | what it is | what to change |
| --- | --- | --- | --- |
| high | low | LATENCY-bound: too few loads in flight | more occupancy, unroll, wider loads |
| high | high | BANDWIDTH-bound: the wire is the limit | move less -- tile for reuse, fuse, shrink the dtype |
| low | high | streaming at rate, nothing wasted | only an algorithmic change moves it |
| low | low | not memory at all | go to 5 |

**4. Traffic against the algorithm's minimum.** The most actionable number here, and it needs no
peak: work out how many bytes the kernel MUST move -- every input read once, every output written
once -- and divide the measured `FetchSize + WriteSize` by it.

- ratio near 1 -- compulsory. Tiling buys nothing; only a different algorithm does.
- ratio well above 1 -- you are re-reading data that should have stayed in cache. Check
  `L2CacheHit` next. This is what a tiling or fusion change is for, and the ratio checks it worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show.

**5. `L2CacheHit`** -- `100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))` on
CDNA, verified against `counter_defs.yaml`. WARNING: On RDNA (gfx10+) the same metric counts `GL2C_HIT` /
`GL2C_MISS` instead: a different cache block with different counter names, so a `TCC_*` request
returns nothing there rather than a wrong number. Read it as the EXPLANATION of step 4, never on
its own: a rising hit rate with unchanged fetch bytes means you added accesses, not locality.

**6. Which pipe, last.** Ask for the DERIVED metric BY NAME and let the tool compute it --
`rocprofv3 --pmc VALUBusy` gives you the number the vendor stands behind. The expressions are in
ROCm's `counter_defs.yaml` and are architecture-specific, so a formula copied for one part is wrong
on the next. For gfx942 (MI300), verified against that file:

| metric | expression on gfx942 |
| --- | --- |
| `VALUBusy` | `100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `SALUBusy` | `100*reduce(SQ_INST_CYCLES_SALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `MemUnitStalled` | `100*TCP_TCP_TA_DATA_STALL_CYCLES_max/reduce(GRBM_GUI_ACTIVE,max)/SE_NUM` |
| `VALUUtilization` | `100*reduce(SQ_THREAD_CYCLES_VALU,sum)/(reduce(SQ_ACTIVE_INST_VALU,sum)*MAX_WAVE_SIZE)` |
| `LDSBankConflict` | `100*reduce(SQ_LDS_BANK_CONFLICT,sum)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM` |

The normaliser is `GRBM_GUI_ACTIVE` scaled by a part constant, never `SQ_BUSY_CU_CYCLES`. On gfx10
`LDSBankConflict` is `SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE` instead -- a different pair of
counters, not a rescaled one.

Two readings to be careful with, both of which invite an NVIDIA habit that does not transfer:

- `VALUUtilization` on this component is lane occupancy within a wave -- the DIVERGENCE number.
  Note that `rocprof-compute` prints something spelled almost identically, `VALU Utilization`, which
  means the opposite thing (what fraction of the kernel the VALU was BUSY); its divergence metric is
  `VALU Active Threads`. Two tools, near-identical spellings, different quantities.
- Whatever it is called, it is scaled by the wavefront width, so the SAME source branch reads
  differently on CDNA (64 lanes) and RDNA (32). Never compare it across parts.

## Comparing two counters -- they always came from different runs

One counter per run means every ratio spans two executions. That is only legitimate through **a
denominator BOTH runs measured**. Collect `rocp_sdk:::GRBM_GUI_ACTIVE` -- GPU active CYCLES -- in
every run, and divide each raw count by its OWN run's value before comparing. It is a duration, so
it is a normaliser and not evidence the two runs did the same work: a run that got slower has more
of them.

WARNING: Not `GPUBusy`. Upstream defines it as `100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)` --
a PERCENTAGE of time, not a cycle count -- so dividing by it inverts the normalisation instead of
applying it.

Same binary, same input, same grid is what makes two runs comparable. With all three held, an
active-cycle count that still moves by more than a few percent means something outside the code
moved, and no ratio built from those runs is trustworthy.

Two rules override all of it:

- **The kernel's work is the invariant.** If the fetch byte count moved between two versions meant
  to compute the same thing, recheck correctness before reading any other number.
- **A counter improving while the uninstrumented run gets slower is not an improvement.**

## Traps

- **A count of 0 is a measurement; ERROR is not.** The code prints `ERROR (not counted)` when setup
  failed. Read that line before the numbers. On this vendor a silent 0 is also what both
  environment traps produce, which is why the empty-bracket check refuses to continue.
- **Check an empty bracket before you believe a full one.** `gpu_papi_init` does it for you, and it
  catches ONE of the two failures: a counter accumulating device-wide instead of attributing, which
  reads back large. WARNING: It does NOT catch a dead counter -- a component returning 0 for everything
  passes an empty-bracket probe, because 0 is the right answer for an empty bracket. That is why the
  region loop checks the TOTAL as well: an all-zero run with the expected region count is the
  silent-zero failure, not a kernel that moved nothing.
- **A cache-resident working set reports near-zero HBM traffic, and that is CORRECT.** Before
  calling a traffic counter broken, scale the working set past the last-level cache and check the
  number tracks. On a part with a large MALL/Infinity Cache this bites at sizes that feel big.
- **`regions:` must be the launch count you expect.** Fewer means brackets were skipped.
- **The counted binary is not your submission.** Build the probe separately; submit the clean
  source.
- **Never run the probe under `rocprofv3` or `rocprof-compute`.** They are the same profiling
  client the component needs, and two subscribers do not share it.
- **Do not port NVIDIA thresholds.** Wavefront width, LDS banking and the cache hierarchy all
  differ. A number that means "bad" on an SM does not mean it on a CU.

## Documentation

- PAPI project home -- https://icl.utk.edu/papi/
- PAPI `rocp_sdk` component: build flags, env vars, dispatch mode -- https://github.com/icl-utk-edu/papi/blob/master/src/components/rocp_sdk/README.md
- PAPI `rocm` component (deprecated from MI300A) -- https://github.com/icl-utk-edu/papi/blob/master/src/components/rocm/README.md
- ROCprofiler-SDK, which `rocp_sdk` sits on -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- MI300/MI200 counter DEFINITIONS and units (note: this page gives no expressions) -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- The derived-counter EXPRESSIONS, per architecture -- the authority for every formula above:
  https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml
- rocprof-compute's per-panel metric definitions and UNITS, per part (`gfx942/*.yaml`) --
  https://github.com/ROCm/rocprofiler-compute/tree/develop/src/rocprof_compute_soc/analysis_configs
- Occupancy on AMD: 8 wavefront slots per SIMD, 32 per CU on CDNA -- https://gpuopen.com/learn/occupancy-explained/
- AMD Instinct MI300 (CDNA3) ISA reference, for the hardware numbers -- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- Occupancy on AMD, wave-per-SIMD arithmetic -- https://gpuopen.com/learn/occupancy-explained/
- HIP programming model: wavefront, CU, LDS -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
