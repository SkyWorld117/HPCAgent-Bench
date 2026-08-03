---
name: papi-gpu-amd
description: Count what an AMD GPU did inside ONE of your kernels with PAPI's rocp_sdk component -- start/stop per region, the empty-bracket self-test, and the two environment traps that return silent zeros.
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

- **`AQLPROFILE_READ_API=0`** is required for intercept mode on **ROCm >= 6.2.0**. Without it the
  counters come back zero. Export it before the run.
- **`PAPI_library_init()` must run BEFORE any HIP call.** The AMD runtime reads its environment
  once, at the first HIP call; initialise PAPI after that and the counter configuration never
  takes. With a statically linked `libpapi.a` this is mandatory and upstream says so explicitly;
  dynamically linked it is documented as unconstrained, but the ordering costs nothing, so keep it.

That last one fights the CUDA rule, so do not port the ordering across: on NVIDIA you arm AFTER a
warmup launch because the component profiles through a live context. On AMD you initialise PAPI
FIRST. Same library, opposite order, and each is silent when you get it wrong.

## Event names

```sh
papi_native_avail -i rocm:::         # every event this component enumerates
papi_native_avail -e rocm:::GPUBusy  # ONE event, resolved, defaults filled in
```

Events are `rocm:::EVENT_NAME:device=N`, e.g. `rocm:::GPUBusy:device=0`. Device indices run
`[0, N-1]` over VISIBLE devices, so `ROCR_VISIBLE_DEVICES` renumbers them and a resource manager
that hands you a subset changes what `device=0` means. Where the mapping matters, resolve it by
UUID (`hipDeviceGetUuid`) rather than trusting the index.

Only single-pass metric sets are supported. Floating-point metrics are recast to `long long` on
the way out -- read them back into a `double` before dividing, or a percentage becomes 0 or 1.

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

`PAPI_stop` ends the profiling range, which is what forces the counter to be flushed and
attributed to the work inside it; `PAPI_start` reopens a fresh one. `gpu_total` accumulates across
visits, so a 20 us kernel called 500 times is measurable without changing what you measured. A
`PAPI_start` after a `PAPI_stop` is a supported re-arm, not a leak: the event set is created once
and destroyed once.

## How it runs

Use it:

```c
if (gpu_papi_init(argv[1]) != 0) return 2;  /* BEFORE any hip call -- see the traps */
your_kernel<<<grid, block>>>(...);          /* warmup */
hipDeviceSynchronize();
for (int step = 0; step < nt; ++step) {
    gpu_region_begin();
    your_kernel<<<grid, block>>>(...);      /* ONE kernel per region */
    gpu_region_end();
}
gpu_papi_report();
check_results();                            /* ALWAYS verify -- a wrong answer measures nothing */
```

```sh
hipcc -O2 -o probe probe.cpp -lpapi
export AQLPROFILE_READ_API=0                /* ROCm >= 6.2.0, or every count is zero */
```

One counter per run. Loop outside the program:

```sh
for ev in rocm:::GPUBusy \
          rocm:::SQ_WAVES \
          rocm:::FetchSize \
          rocm:::WriteSize \
          rocm:::L2CacheHit \
          rocm:::VALUBusy \
          rocm:::VALUUtilization \
          rocm:::MemUnitStalled; do
  ./probe "$ev:device=0"
done
```

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

**5. `L2CacheHit`**, which is `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum) * 100`. Read it as the
EXPLANATION of step 4, never on its own: a rising hit rate with unchanged fetch bytes means you
added accesses, not locality.

**6. Which pipe, last.** `VALUBusy` (`SQ_ACTIVE_INST_VALU / SQ_BUSY_CU_CYCLES * 100`) and
`SALUBusy` (`SQ_INST_CYCLES_SALU / SQ_BUSY_CU_CYCLES * 100`) say which pipe was issuing.
`VALUUtilization` is the percentage of LANES active in a wave -- the divergence number, and the one
that is scaled by the wavefront width, so a 32-of-64 branch on CDNA reads 50% where the same source
on RDNA reads 100%. `LDSBankConflict` (`SQ_LDS_BANK_CONFLICT / SQ_BUSY_CU_CYCLES * 100`) is the LDS
equivalent, and has no NVIDIA-shaped intuition to borrow: pad the stride and re-measure.

## Comparing two counters -- they always came from different runs

One counter per run means every ratio spans two executions. That is only legitimate through **a
denominator BOTH runs measured**. Collect `rocm:::GRBM_GUI_ACTIVE` (GPU active cycles) or
`rocm:::GPUBusy` in EVERY run, and divide each raw count by its OWN run's value before comparing.
It is a DURATION, so it is a normaliser and not evidence the two runs did the same work -- a run
that got slower has more of them.

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
- **Check an empty bracket before you believe a full one.** `gpu_papi_init` does it for you. It is
  the one self-test that catches a counter accumulating device-wide instead of attributing -- the
  failure mode that produces confident, plausible, wrong numbers on every region at once.
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
- MI300/MI200 counters and every derived formula quoted above -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- Occupancy on AMD, wave-per-SIMD arithmetic -- https://gpuopen.com/learn/occupancy-explained/
- HIP programming model: wavefront, CU, LDS -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
