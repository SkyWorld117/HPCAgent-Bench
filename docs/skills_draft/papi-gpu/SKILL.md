---
name: papi-gpu
description: Count what the GPU did inside ONE of your kernels with PAPI's cuda component -- explicit :stat= roll-up, device sync on both sides, one counter per run.
---

`nsys` answers WHICH kernel owns device time. This page answers WHAT THE DEVICE DID while one
kernel ran: DRAM bytes moved, warps stalled on memory, sectors hit. You bracket your own code, so
the answer is attributed to a region you chose rather than to a symbol.

Everything you need is here. Paste the code into your `.cu`, compile with `-lpapi -lcudart`, run
it. Run `nsys` first anyway -- a counter on the wrong kernel is a perfectly measured 4% of the run.

## What on this page was run, and what was not

The box this was written on has the NVIDIA profiling gate ON and no root, so `PAPI_start` returns
-14 and NO COUNTER VALUE was ever produced here. Verified here: the component list, the compile
line, every event name and qualifier below (`PAPI_add_named_event` runs before `PAPI_start`, so
name resolution IS testable under the gate), every error code, and the gate's own behaviour. NOT
verified here: any counter value, any delta, and every threshold in "Reading the numbers" -- those
come from the vendor docs at the bottom. Treat them as untested.

## Two checks before you write any code

```sh
papi_component_avail | grep -A2 'Name:   cuda'
grep -E 'RestrictProfilingToAdminUsers|RmProfilingAdminOnly' /proc/driver/nvidia/params
```

The first asks whether this PAPI has a `cuda` component AT ALL. It is a BUILD option, not a
package: a distribution PAPI on a box with a perfectly good GPU usually has none, and rebuilding
is the only fix -- `./configure --with-components="cuda"` with `PAPI_CUDA_ROOT` set.

The second is the permission gate, the failure you are most likely to hit: `: 1` while you are not
root means every count below returns nothing -- see "When it counts nothing". Grep BOTH spellings.
Older drivers echo `NVreg_RestrictProfilingToAdminUsers`; the open kernel module publishes the
internal name `RmProfilingAdminOnly` instead, and matching only the documented one reports "no
gate" on a gated box -- measured here on driver 595.84.

## Write :stat= yourself -- the default roll-up is the wrong number

```sh
papi_native_avail -e cuda:::dram__bytes_read
# Event name:  cuda:::dram__bytes_read:stat=avg:device=0
```

The bare name is not the event you want. `:stat=` and `:device=` are Mandatory qualifiers that
PAPI fills in for you. `:device=0` is fine. `:stat=avg` is not: in NVIDIA's metric scheme `avg` is
the AVERAGE across hardware unit instances and `sum` is the total, so bare
`cuda:::dram__bytes_read` is bytes per DRAM partition -- low by the instance count, and nothing in
the output says so. Write `:stat=sum` on every count.

Rate events take a different qualifier set, and their default is worse than wrong: bare
`cuda:::l1tex__t_sector_hit_rate` resolves to `:stat=max_rate` and is then REJECTED at
`PAPI_add_named_event` with -14 -- the same code the permission gate returns. `:stat=pct` and
`:stat=ratio` both add. `papi_native_avail -e` prints the legal set: `[avg, max, min, sum]` for
counts, `[max_rate, pct, ratio]` for hit rates.

A ratio across two events needs `:stat=sum` on BOTH: the `avg` defaults average over different
instance counts at different levels (`sm__`, `smsp__`, `dram__`), so a ratio of two defaults is
off by the ratio of those counts.

## The code

```c
#include <papi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

static int gpu_es = PAPI_NULL;
static long long gpu_total = 0, gpu_before = 0;
static const char *gpu_event = NULL;
static int gpu_ok = 0, gpu_regions = 0;

/* Call ONCE, AFTER a warmup launch: the component profiles through a live CUDA context. */
static int gpu_papi_init(const char *event_name)
{
    gpu_ok = 0; gpu_total = 0; gpu_regions = 0; gpu_event = event_name;
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "papi-gpu: library_init failed\n"); return -1;
    }
    int cid = -1;
    for (int i = 0; i < PAPI_num_components(); ++i) {
        const PAPI_component_info_t *ci = PAPI_get_component_info(i);
        if (ci && !strcmp(ci->name, "cuda")) { cid = i; break; }
    }
    if (cid < 0) { fprintf(stderr, "papi-gpu: PAPI has no 'cuda' component\n"); return -1; }
    int rc;
    /* A GPU event set must be bound to the cuda component; the default (0) is the CPU. */
    if ((rc = PAPI_create_eventset(&gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_assign_eventset_component(gpu_es, cid)) != PAPI_OK) goto fail;
    if ((rc = PAPI_add_named_event(gpu_es, event_name)) != PAPI_OK) goto fail;
    if ((rc = PAPI_start(gpu_es)) != PAPI_OK) goto fail;
    gpu_ok = 1;
    return 0;
fail:
    fprintf(stderr, "papi-gpu: %s: %s (code %d)\n", event_name, PAPI_strerror(rc), rc);
    return -1;
}

/* The syncs are the measurement. A launch is ASYNCHRONOUS: without them you count the launch. */
static void gpu_region_begin(void)
{
    if (!gpu_ok) return;
    cudaDeviceSynchronize();                       /* drain EARLIER work out of the delta */
    if (PAPI_read(gpu_es, &gpu_before) != PAPI_OK) gpu_ok = 0;
}

static void gpu_region_end(void)
{
    if (!gpu_ok) return;
    cudaDeviceSynchronize();                       /* the launch returned; the kernel may not have */
    long long after = 0;
    if (PAPI_read(gpu_es, &after) != PAPI_OK) { gpu_ok = 0; return; }
    gpu_total += after - gpu_before;               /* ACCUMULATES across every visit */
    ++gpu_regions;
}

static void gpu_papi_report(void)
{
    if (!gpu_ok) { printf("%s = ERROR (not counted)\n", gpu_event ? gpu_event : "?"); return; }
    printf("%s = %lld   (regions: %d)\n", gpu_event, gpu_total, gpu_regions);
    long long sink = 0;
    PAPI_stop(gpu_es, &sink);
    PAPI_cleanup_eventset(gpu_es); PAPI_destroy_eventset(&gpu_es);
}
```

Arm ONCE, then read a delta per region: `PAPI_read` copies the counters and leaves them counting,
so consecutive reads bracket a region. `PAPI_start`/`PAPI_stop` per launch re-arms the CUPTI event
set every time, which is instrumentation cost landing inside the region you are measuring.

## How it runs

Use it:

```c
your_kernel<<<grid, block>>>(...);          /* warmup: this is what creates the context */
cudaDeviceSynchronize();
if (gpu_papi_init(argv[1]) != 0) return 2;  /* the event name comes from the shell loop below */
for (int step = 0; step < nt; ++step) {
    gpu_region_begin();
    your_kernel<<<grid, block>>>(...);      /* ONE kernel per region */
    gpu_region_end();
}
gpu_papi_report();
check_results();                            /* ALWAYS verify -- a wrong answer measures nothing */
```

```sh
nvcc -O2 -arch=native -o probe probe.cu -lpapi -lcudart
```

One counter per run, for the reason below. Loop outside the program:

```sh
for ev in cuda:::sm__cycles_elapsed:stat=sum \
          cuda:::dram__bytes_read:stat=sum \
          cuda:::dram__bytes_write:stat=sum \
          cuda:::smsp__warps_issue_stalled_long_scoreboard:stat=sum \
          cuda:::smsp__warps_active:stat=sum \
          cuda:::l1tex__t_sector_hit_rate:stat=pct \
          cuda:::lts__t_sectors_lookup_hit:stat=sum \
          cuda:::lts__t_sectors:stat=sum; do
  ./probe "$ev"
done
```

## One region per kernel, synced on both sides

A kernel launch returns immediately. A bracket without a device synchronise measures the LAUNCH:
the read after `your_kernel<<<>>>` lands while the kernel is still running. The two syncs do
different jobs. The one BEFORE the first read drains earlier work out of your delta; the one
BEFORE the second read is what makes the delta the kernel's.

**The syncs are part of the measurement, not neutral scaffolding.** A synchronised run removes
exactly the kernel/copy and kernel/kernel overlap a real run depends on. So a counted run's wall
clock belongs to no comparison at all -- not to a timed run, not to another counted run. Read the
COUNTS; take every speedup from the uninstrumented build.

One kernel per region: two kernels in one bracket give you their sum, and a sum cannot be
attributed. Move the bracket and run again.

Bracket INSIDE the timestep loop, not around it. The delta accumulates, so a 20 us kernel called
500 times becomes measurable without changing what you measured.

## One counter per run

Not a hardware limit: the cuda component reports 30 counters (`papi_component_avail`) and accepted
ten single-pass events in one event set here. It is a blast-radius choice. CUPTI REPLAYS a kernel
when a set needs more than one pass, and a set that was fine event-by-event can tip over the pass
budget as a whole; whether ten survive `PAPI_start` together was not testable here. Until you check
on your own box, collect one event per run.

`PAPI_add_named_event` is the check that matters: it returns -27 for an event this device cannot
count in one pass, before you spend a run. Refused here:
`cuda:::sm__throughput.pct_of_peak_sustained_elapsed` (Numpass=6) and
`cuda:::lts__t_sector_hit_rate` (Numpass=2) -- which is why the L2 hit rate above is built from
`lts__t_sectors_lookup_hit / lts__t_sectors` instead of asked for directly.

## Enumerate what THIS device has -- never assume a list

Event names are matched against what the component ENUMERATES; they cannot be built from a
template. Nsight Compute's spelling of the same metric is rejected -- `cuda:::dram__bytes_read`
resolves, `cuda:::dram__bytes_read.sum` comes back `Invalid argument`, because in this component
the roll-up is the `:stat=` qualifier and not a `.sum` suffix. The event set also depends on the
PART, so a name that works on one GPU is absent on the next.

```sh
papi_native_avail -i dram__bytes_read          # every matching event, with units and Numpass
papi_native_avail -e cuda:::dram__bytes_read   # ONE event, resolved, defaults filled in
```

```c
/* The in-program form: PAPI_enum_cmp_event walks one component's native events. */
int code = PAPI_NATIVE_MASK; char name[PAPI_HUGE_STR_LEN];
if (PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, cid) == PAPI_OK) do {
    if (PAPI_event_code_to_name(code, name) == PAPI_OK && strstr(name, argv[1])) puts(name);
} while (PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, cid) == PAPI_OK);
```

That walked 53782 events here in 0.6 s, so run it and grep rather than guess.

Ask a QUESTION, then find the event that answers it on THIS device. "How much DRAM traffic" is a
different name on every vendor and often on every generation, so a hard-coded event list is a list
that stops working. NVIDIA events come through the `cuda` component; AMD through `rocm`.

## Reading the numbers -- none of this was measured here

The gate meant no value was ever collected here, so the order below and every number in it are
vendor-doc reasoning, not observation. Calibrate on your own kernel before trusting a threshold.
Counters do not name a bottleneck. They eliminate candidates, in this order -- stop at the first
step that fires, because the later numbers are consequences of the earlier ones.

**1. Was the device even the problem?** If `nsys` already showed device time well under the wall
clock, stop. Launch gaps and copies are host findings and no counter below moves them.

**2. Occupancy -- but only against the grid.** `smsp__warps_active:stat=sum` over
`sm__cycles_elapsed:stat=sum` is the resident-warp count; the ceiling is per-part, so read it as a
trend across your own versions, not against an absolute. Low occupancy has two causes the number
alone cannot separate: fewer blocks than SMs (fix the decomposition -- one element per thread, not
one row; split the reduction), or a full grid still capped by registers or shared memory per block
(`-maxrregcount`, `__launch_bounds__`, a smaller tile). The geometry that tells them apart is
`nsys`'s `cuda_gpu_trace`; this instrument does not measure it.

High occupancy is not a goal. A kernel with enough in-flight memory work per thread runs at peak
with half the warp slots empty. Occupancy matters only when something else says the SMs stalled.

**3. Memory stall, read WITH the DRAM traffic.** The stall event is
`smsp__warps_issue_stalled_long_scoreboard:stat=sum` -- warps waiting on an L1TEX dependency --
read as a fraction of `smsp__warps_active:stat=sum`. This is the one pairing that separates the
two memory bottlenecks, and neither event answers it alone:

| stall | DRAM | what it is | what to change |
| --- | --- | --- | --- |
| high | low | LATENCY-bound: too few loads in flight | more occupancy, unroll, wider loads (`float4`) |
| high | high | BANDWIDTH-bound: the wire is the limit | move less -- tile for reuse, fuse, shrink the dtype |
| low | high | streaming at rate, nothing wasted | only an algorithmic change moves it |
| low | low | not memory at all | compute- or divergence-bound; go to 5 |

**4. DRAM bytes against the algorithm's minimum.** The most actionable number on the page, and it
needs no peak: work out how many bytes the kernel MUST move -- every input read once, every output
written once -- and divide the measured `dram__bytes_read + dram__bytes_write` (`:stat=sum`, or the
ratio is nonsense) by it.

- ratio near 1 -- the traffic is compulsory. Tiling buys nothing; only a different algorithm does.
- ratio well above 1 -- you are re-reading data that should have stayed in cache. Check the hit
  rates next. This is what a tiling or fusion change is for, and the ratio is how you check it
  worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show. Coalescing is a layout change (SoA, padding), not a scheduling one.

**5. Hit rates, L1 then L2.** L1 is where a tiling change shows up first; L2 is what did NOT become
DRAM traffic. Read them as the EXPLANATION of step 4, never on their own: a rising hit rate with
unchanged DRAM bytes means you added accesses, not locality. Check the unit before believing a
number -- `papi_native_avail -e` prints it, and `:stat=ratio` arrives in 0..1 while `:stat=pct`
arrives in 0..100.

**6. Throughput against peak, last.** The component enumerates one roofline coordinate directly,
already normalised, so no timing is involved:
`cuda:::gpu__dram_throughput.pct_of_peak_sustained_elapsed:stat=avg` (`Units=(percent)`,
`Numpass=1`, adds here). The SM-side equivalent is `Numpass=6` here and cannot be counted on this
part -- check yours. As a rule of thumb, not a measurement: above ~80% of DRAM peak, stop tuning
instructions and cut traffic; below ~20% on both units, neither is the limit and you are latency-
or occupancy-bound, so go back to 2.

## Comparing two counters -- they always came from different runs

One counter per run means every ratio you want spans two executions. That is only legitimate
through **a denominator BOTH runs measured**.

- Collect `cuda:::sm__cycles_elapsed:stat=sum` in EVERY run. It is `# of cycles elapsed on SM`, a
  DURATION -- so it is a NORMALISER, not evidence the two runs did the same work. A run that got
  slower has MORE of them.
- Divide each raw count by its OWN run's elapsed cycles before comparing. Bytes per SM cycle from
  run A against bytes per SM cycle from run B is a comparison; bytes from A against bytes from B
  compares two schedules.
- Same binary, same input, same grid is what makes two runs comparable. With all three held, an
  elapsed-cycle count that still moves by more than a few percent means something outside the code
  moved -- clocks, another process -- and no ratio built from those runs is trustworthy.

The same rule buys you a metric no single event provides. Warp lane efficiency is
`sm__sass_thread_inst_executed:stat=sum / (smsp__inst_executed:stat=sum * 32)` -- thread
instructions over warp instructions times the warp width. Both enumerate here at `Numpass=1`, and
`:stat=sum` on BOTH is what makes the `sm__` and `smsp__` levels comparable. Well under 1 is
divergent control flow or a partial last warp: wasted issue slots, not wasted bandwidth.

Two rules override all of it:

- **The kernel's work is the invariant.** If the DRAM byte count moved between two versions meant
  to compute the same thing, recheck correctness before reading any other number.
- **A counter improving while the uninstrumented run gets slower is not an improvement.**

## When it counts nothing

A counter that was never collected reads exactly like a kernel that did no work. Four failures,
four different fixes, all reproduced here:

| code | where it fires | what it means |
| --- | --- | --- |
| `-1` `PAPI_EINVAL`, "Invalid argument" | `PAPI_add_named_event` | not a name this component enumerates: a typo, or ncu's `.sum` spelling |
| `-27` `PAPI_EMULPASS`, "multiple passes required" | `PAPI_add_named_event` | `Numpass > 1` on this part; pick another event |
| `-14` `PAPI_EMISC`, "Unknown error code" | `PAPI_add_named_event` | a `:stat=` this event does not accept -- including its own default |
| `-14` `PAPI_EMISC`, "Unknown error code" | `PAPI_start` | the permission gate |

**`PAPI_EMISC` at `PAPI_start` is the gate.** PAPI's error table does not cover what a component
returns, so the driver's real complaint never reaches you. Get it from a tool that prints it:

```sh
ncu --metrics dram__bytes_read.sum ./probe
# ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
#           Performance Counters on the target device 0.
```

The gate is on counters, not on kernel tracing. Under the same gate, `nsys profile --trace=cuda`
reported this probe's kernel durations here while `PAPI_start` returned -14. A run that hands you
kernel timings and refuses every counter is this, not a broken toolkit. NVIDIA's wording covers
"Performance Counters or the Hardware Event System", so device-scope HW tracing
(`nsys --gpu-metrics`) is gated too.

The fix, as root:

```sh
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
# reload the nvidia module or reboot; in a container pass --cap-add=SYS_ADMIN
```

Otherwise run the counted binary as root or with `CAP_SYS_ADMIN` (`CAP_PERFMON` also works from
driver R565). No code change works around it.

## Traps

- **A count of 0 is a measurement; ERROR is not.** The code above prints `ERROR (not counted)` when
  setup failed, and stops counting if a read fails mid-run. Read that line before the numbers.
- **`regions:` must be the launch count you expect.** Fewer means brackets were skipped and the
  total is short.
- **The counted binary is not your submission.** `cudaDeviceSynchronize` inside a graded region
  perturbs exactly what is being graded. Build the probe separately; submit the clean source.
- **Never run the probe under `ncu` or `nsys`.** CUPTI's profiling APIs take ONE client --
  multi-subscriber support is Activity-API only. Under either tool the enumeration walk above
  returned 0 events here instead of 53782, so the probe finds no event to add and reports ERROR.
- **Arm after the context exists.** The component profiles through a context made by `cuCtxCreate`
  or a primary context activated by `cudaSetDevice`, so `gpu_papi_init` must run after the warmup
  launch. What it returns with no context could not be checked here: the gate returns -14 to
  everything.
- **One event set counts ONE device.** `:device=` is a Mandatory qualifier and PAPI defaults it to
  `:device=0`, so multi-GPU needs `:device=N` and one event set per device. The device and thread
  binding of a running set was not testable here.

## Documentation

- PAPI project home -- https://icl.utk.edu/papi/
- PAPI cuda component, build flags and context requirement -- https://github.com/icl-utk-edu/papi/blob/master/src/components/cuda/README.md
- NVIDIA CUPTI, which the cuda component sits on -- https://docs.nvidia.com/cupti/main/main.html
- Nsight Compute profiling guide: metric naming, `.sum`/`.avg` roll-ups, kernel replay -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- The profiling permission gate and how to lift it -- https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
