---
name: papi-gpu
description: Count what the GPU did inside ONE of your kernels with PAPI's cuda component -- explicit :stat= roll-up, start/stop per region, one counter per run.
---

`nsys` answers WHICH kernel owns device time. This page answers WHAT THE DEVICE DID while one
kernel ran: DRAM bytes moved, warps stalled on memory, sectors hit. You bracket your own code, so
the answer is attributed to a region you chose rather than to a symbol.

Everything you need is here. Paste the code into your `.cu`, compile with `-lpapi -lcudart`, run
it. Run `nsys` first anyway -- a counter on the wrong kernel is a perfectly measured 4% of the run.

## Start and stop the event set per region -- a read-delta does NOT attribute

This is the whole page. `PAPI_read` leaves the set counting and looks like it brackets a region; on
the cuda component it does not. From the component source, `cuptip_ctx_read` pops the CUPTI range,
ends the pass, flushes, evaluates, ACCUMULATES into a running total and pushes a fresh range. So a
"delta" is the difference of two accumulations sliced at host-side range boundaries, and no call on
that path synchronises the device -- whatever had not finished at `pop_range` is charged to the next
range. `PAPI_stop` reaches the counter through that same read, then additionally runs
`EndPass`/`FlushCounterData`/`UnsetConfig`/`EndSession`, and the next `PAPI_start` rebuilds config
image, counter-data image and session from scratch. That teardown is the ordering the read path
never gets. Whether a `cudaDeviceSynchronize` before every read recovers it was NOT tested here; the
table below is what was.

Measured here, RTX 4050 / driver 595.84 / PAPI 7.2.0.0, four kernels of deliberately different
shape, `cuda:::dram__bytes_read:stat=sum`, 25 regions each. "Truth" is the algorithm's compulsory
traffic -- every input read once:

| region | truth / rep | `PAPI_start`/`PAPI_stop` | read-delta |
| --- | --- | --- | --- |
| streams b and c into a | 128 MiB | **134.26 MB** | 128.4 MB |
| touches 64 KB, 64 launches | 64 KB | **77.9 KB** | 93.5 MB |
| reads a, 64 FMAs, writes a | 64 MiB | **67.08 MB** | 111.1 MB |
| reads a and c, divergent | 128 MiB | **134.27 MB** | 126.0 MB |

Start/stop lands on the compulsory traffic to within **0.05% at MiB scale**; the 64 KB row reads
77.9 KB against 64 KB, which is +18.9% and is the launch overhead of 64 separate dispatches showing
up at a scale where it is no longer negligible. The read-delta is wrong on every row and wrong by
**1300x** on that same 64 KB one -- and note what that does to a comparison: the
true spread across these four kernels is 2100x, and the read-delta reports 1.2x. It does not merely
add noise, it FLATTENS the ranking you are profiling to find.

The same holds on the SM side: `cuda:::smsp__inst_executed:stat=sum` start/stop gives 22528 for the
64 KB kernel and 161480704 for the FMA chain, a ratio of **7168x**, matching 512 warps x 11
instructions against 524288 x 77 exactly. The read-delta reports those two as 1.66x apart.

Start/stop costs about 2x wall clock here (2.37 s against 1.22 s over 20 regions) -- re-arming the
CUPTI set per region is real. Spend it. You are reading COUNTS, and a counted run's wall clock
already belongs to no comparison (see below), so the only thing that cost buys back is a number
that means what it says.

## Two checks before you write any code

```sh
papi_component_avail | grep -A2 'Name:   cuda'
grep -E 'RestrictProfilingToAdminUsers|RmProfilingAdminOnly' /proc/driver/nvidia/params
```

The first asks whether this PAPI has a `cuda` component AT ALL. It is a BUILD option, not a
package: a distribution PAPI on a box with a perfectly good GPU usually has none, and rebuilding is
the only fix. It needs no root and takes a few minutes -- next section.

The second is the permission gate, the failure you are most likely to hit: `: 1` while you are not
root means every count below returns nothing -- see "When it counts nothing". Grep BOTH spellings.
Older drivers echo `NVreg_RestrictProfilingToAdminUsers`; the open kernel module publishes the
internal name `RmProfilingAdminOnly` instead, and matching only the documented one reports "no
gate" on a gated box -- measured here on driver 595.84.

## Building a PAPI that has the cuda component

`configure` lives in `src/`, not at the tarball root, and `--with-components` takes ONE quoted
space-separated list. No root anywhere; nothing is written outside `--prefix`.

```sh
curl -LO https://github.com/icl-utk-edu/papi/releases/download/papi-7-2-0-t/papi-7.2.0.tar.gz
tar xf papi-7.2.0.tar.gz && cd papi-7.2.0/src        # configure is HERE
export PAPI_CUDA_ROOT=/usr/local/cuda
./configure --prefix=$HOME/papi --with-components="cuda"
make -j8 && make install
```

Set `PAPI_CUDA_ROOT` EXPLICITLY. `Rules.cuda` otherwise derives it from `which nvcc`, which on any
box carrying an HPC SDK lands on `.../<ver>/compilers` -- no `cupti.h` under it by either include
path (measured here). It is a RUN-time variable as well: everything CUDA is `dlopen`ed, `ldd
libpapi.so` here shows `libpfm`, `libc`, the loader and no CUPTI at all, so swapping toolkits is a
repoint and not a rebuild.

The two products lay CUPTI out differently and BOTH work, because PAPI passes both `-I` paths and
searches both lib subdirectories (measured here):

| | CUDA Toolkit 13.3, `/usr/local/cuda` | HPC SDK 26.3, `.../26.3/cuda/13.1` |
| --- | --- | --- |
| `cupti.h`, `nvperf_host.h` | `include/` | `extras/CUPTI/include/` |
| `libcupti.so`, `libnvperf_host.so` | `lib64/` | `extras/CUPTI/lib64/` |
| `extras/CUPTI/{include,lib64}` | ABSENT (`doc`, `samples` only) | present |

For the HPC SDK the root is the VERSIONED directory -- `.../26.3/cuda` has `nvcc` and symlinks but no
`extras/` and no `cupti.h`. Nothing is missing on a modern toolkit either; do not symlink an
`extras/CUPTI/lib64` into existence.

Then walk the ladder and stop at the first step that fails:

1. `$HOME/papi/bin/papi_component_avail` -- `cuda` in **Compiled-in** but not in **Active** means it
   built and then failed init. The reason prints beneath as `\-> Disabled: <reason>`;
   `\-> Partially disabled:` on a mixed-compute-capability box is by design, not a fault.
2. `Native: 0` -- component up, nothing enumerated: no visible device, or the wrong API for this CC.
3. `papi_native_avail -e cuda:::dram__bytes_read` -- one event, resolved, defaults filled in.
4. The permission grep from the previous section.
5. `make -C components/cuda/tests && ./components/cuda/tests/HelloWorld` -- it creates a real
   context, which `papi_command_line` does not. `HelloWorld_noCuCtx` covers the primary-context path.

**The PAPI utilities are STATICALLY linked, so `papi_component_avail` reports its OWN build.**
Measured here: `ldd $(command -v papi_component_avail)` is libc and the loader, `nm -D` finds zero
`PAPI_*` symbols. `LD_LIBRARY_PATH` cannot move it, so a distro copy earlier on `PATH` will keep
reporting "no cuda component" whatever you export -- call yours by absolute path. The same trap runs
the other way at link time: a probe built without an rpath loads whichever `libpapi.so` `ld.so` finds
first and then prints `PAPI has no 'cuda' component` after compiling cleanly against your headers.

```sh
nvcc -O2 -arch=native -o probe probe.cu -I$HOME/papi/include -L$HOME/papi/lib -lpapi \
     -Xlinker -rpath -Xlinker $HOME/papi/lib -lcudart
# or PKG_CONFIG_PATH=$HOME/papi/lib/pkgconfig plus  $(pkg-config --cflags --libs papi)
ldd ./probe | grep papi        # must be YOUR prefix
```

Headers from one prefix against a library from another surfaces only as `PAPI_library_init`
returning something other than `PAPI_VER_CURRENT`.

Compute capability decides which API you get, and the boundary is now a wall. PerfWorks is CC >= 7.0
on toolkits 11.4.0-13.0.0; the Legacy Event/Metric APIs are CC <= 7.0, were REMOVED in CUDA 13.0.0,
and driver branches >= 580 are incompatible with them -- so a P100 or V100 needs a toolkit <= 12.9.1
AND a driver < 580, and `PAPI_CUDA_API=LEGACY` is a no-op anywhere else. CC exactly 7.0 is the only
overlap and defaults to PerfWorks. Upstream's stated PerfWorks ceiling is toolkit 13.0.0; measured
here, PAPI 7.2.0 against CUDA 13.3 enumerates 53782 events and counts correctly -- works, untested
upstream, and the first thing to back off if the component misbehaves.

| symptom, exact string | fix |
| --- | --- |
| `cuda` in NEITHER list | configure never got the flag -- check `src/config.log` line 7 |
| `\-> Disabled: Unable to load CUDA library functions.` | export `PAPI_CUDA_ROOT`; or `PAPI_CUDA_RUNTIME=/full/path/libcudart.so` |
| `\-> Disabled: CUDA configuration not supported.` | CC and toolkit map to no API -- typically CC <= 7.0 on toolkit >= 13.0 or driver >= 580 |
| `\-> Disabled: PAPI not built with NVIDIA profiler API support.` | built against a root with no `nvperf_host.h` -- rebuild against one that has it |
| build: `cupti.h: No such file or directory` | `PAPI_CUDA_ROOT` unset, or auto-derived to a compilers-only dir |
| component active, every PerfWorks event fails at run time | `export PAPI_CUDA_PERFWORKS=/full/path/libnvperf_host.so` |
| `papi_native_avail` looks hung | 140,000+ counters per GPU, up to 2 minutes each, worse when redirected to a file |
| `PAPI_start` returns -14 and `ncu` prints `ERR_NVGPUCTRPERM` | the gate -- ASK THE USER to run the `modprobe.d` line in "When it counts nothing" as root. Never sudo yourself |

## Write :stat= yourself -- the default roll-up is the wrong number

The whole `:stat=` qualifier system is **PAPI 7.2.0 or newer** -- the release notes carry it as a
"statistics qualifier added to CUDA events". Run `papi_version` before porting any spelling on this
page onto an older install.

```sh
papi_native_avail -e cuda:::dram__bytes_read
# Event name:  cuda:::dram__bytes_read:stat=avg:device=0
```

The bare name is not the event you want. `:stat=` and `:device=` are Mandatory qualifiers that
PAPI fills in for you. `:device=0` is fine. `:stat=avg` is not: in NVIDIA's metric scheme `avg` is
the AVERAGE across hardware unit instances and `sum` is the total, so bare
`cuda:::dram__bytes_read` is bytes per DRAM partition -- low by the instance count, and nothing in
the output says so. Write `:stat=sum` on every count.

Measured on the same region here: `:stat=sum` 537,323,136 against `:stat=avg` 179,049,812, a ratio
of **3.001**. This part has a 96-bit bus, which is 3 x 32-bit partitions -- so the instance count
is exactly the number you would have to already know to spot that the default was wrong. The bare
name returned 179,004,500, confirming it resolves to `avg`. `min` and `max` came back at 179.0M
too, i.e. the partitions are evenly loaded, which is why nothing in the number itself looks off.

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
static long long gpu_total = 0;
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
    int rc; long long probe = 0;
    /* A GPU event set must be bound to the cuda component; the default (0) is the CPU. */
    if ((rc = PAPI_create_eventset(&gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_assign_eventset_component(gpu_es, cid)) != PAPI_OK) goto fail;
    if ((rc = PAPI_add_named_event(gpu_es, event_name)) != PAPI_OK) goto fail;
    /* Arm and disarm once around NOTHING. Two jobs: it surfaces the permission gate here
       instead of at the first region, and the value it returns must be ~0. If an empty
       bracket reports real traffic, the counter is not attributing -- stop and read below. */
    if ((rc = PAPI_start(gpu_es)) != PAPI_OK) goto fail;
    if ((rc = PAPI_stop(gpu_es, &probe)) != PAPI_OK) goto fail;
    if (probe > 4096) {
        fprintf(stderr, "papi-gpu: EMPTY BRACKET READ %lld, not ~0 -- not attributing\n", probe);
        return -1;
    }
    gpu_ok = 1;
    return 0;
fail:
    fprintf(stderr, "papi-gpu: %s: %s (code %d)\n", event_name, PAPI_strerror(rc), rc);
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

`PAPI_stop` is the call that makes the number yours. It ends the profiling SESSION -- pop range, end
pass, flush, unset config -- and the next `PAPI_start` builds config image, counter-data image and
session again from scratch. That teardown and rebuild is the boundary a read does not have.
`gpu_total` accumulates across visits, so a 20 us kernel called 500 times is measurable without
changing what you measured.

Verified here at 25 regions per kernel, and note that `PAPI_start` after a `PAPI_stop` is a
supported re-arm, not a leak -- the event set is created once and destroyed once.

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

One counter per run, for the reason below. The last three exist because the reading steps below
CONSUME them: step 6 divides by `gpu__dram_throughput...`, and the lane-efficiency ratio needs
`sm__sass_thread_inst_executed` over `smsp__inst_executed`. Collect a counter a later step needs or
that step has nothing to read. All eleven verified to resolve here with `papi_native_avail -e`.
Loop outside the program:

```sh
for ev in cuda:::sm__cycles_elapsed:stat=sum \
          cuda:::dram__bytes_read:stat=sum \
          cuda:::dram__bytes_write:stat=sum \
          cuda:::smsp__warps_issue_stalled_long_scoreboard:stat=sum \
          cuda:::smsp__warps_active:stat=sum \
          cuda:::l1tex__t_sector_hit_rate:stat=pct \
          cuda:::lts__t_sectors_lookup_hit:stat=sum \
          cuda:::lts__t_sectors:stat=sum \
          cuda:::gpu__dram_throughput.pct_of_peak_sustained_elapsed:stat=avg \
          cuda:::sm__sass_thread_inst_executed:stat=sum \
          cuda:::smsp__inst_executed:stat=sum; do
  ./probe "$ev"
done
# the coalescing pair in step 4 was NOT verified on this box -- resolve it before you spend runs:
#   cuda:::l1tex__t_sectors_pipe_lsu_mem_global_op_ld:stat=sum
#   cuda:::l1tex__t_requests_pipe_lsu_mem_global_op_ld:stat=sum
```

## One region per kernel, and no sync of your own

A kernel launch returns immediately, so a read-delta would need a device synchronise even to line up
with the kernel -- and the table above is what a read-delta does without one. Under
`PAPI_start`/`PAPI_stop` you need none: `PAPI_stop` ends the session and flushes before it
evaluates. Adding `cudaDeviceSynchronize` on both sides changed the answer here by
**0.008%** (536,976,512 against 536,934,656 bytes), which is to say it did nothing. Leave it out;
it is a line that looks load-bearing and is not.

**A counted run's wall clock still belongs to no comparison.** Profiling serialises the queue and
re-arms the CUPTI set per region, which removes exactly the kernel/copy and kernel/kernel overlap a
real run depends on -- about 2x here. Read the COUNTS; take every speedup from the uninstrumented
build.

One kernel per region: two kernels in one bracket give you their sum, and a sum cannot be
attributed. The session the component opens declares `maxLaunchesPerPass = 1` (component source), so
a second launch inside one range is outside what the configuration says it expects -- what CUPTI does
with it was not tested here, and a sum is reason enough. Move the bracket and run again. Bracket
INSIDE the timestep loop, not around it.

## One counter per run

Not a hardware limit: the cuda component reports 30 counters (`papi_component_avail`) and accepted
ten single-pass events in one event set here. It is a blast-radius choice, and the component source
says the blast is real. The pass check at `PAPI_add_named_event` is PER EVENT -- it asks the pass
count one metric name at a time -- while `PAPI_start` demands the SET fit one pass
(`beginPassGroupParams.maxPassCount = 1`) and returns -14 if it does not. Ten events that each added
can still refuse to start, one line of error for all ten. The session is `CUPTI_UserRange` +
`CUPTI_UserReplay` with `maxRangesPerPass = 1` and `maxLaunchesPerPass = 1`: the APPLICATION is the
replayer and PAPI runs the body once, so a second pass is not slow, it is never collected. One event
per run and the question does not arise.

`PAPI_add_named_event` is still the check worth having: it returns -27 for an event this device
cannot count in one pass, before you spend a run. Refused here:
`cuda:::sm__throughput.pct_of_peak_sustained_elapsed` (Numpass=6) and
`cuda:::lts__t_sector_hit_rate` (Numpass=2) -- which is why the L2 hit rate above is built from
`lts__t_sectors_lookup_hit / lts__t_sectors` instead of asked for directly. Note what that implies
for the two halves: the rate is Numpass=2 BECAUSE it is those two counters, so an event set holding
both is a candidate for the -14 at start. Separate runs.

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

**The name is a grammar, so you can build a candidate to grep for.** From the component source and
NVIDIA's metric structure: a PerfWorks base name is
`unit__(subunit)_(pipestage)_quantity_(qualifiers)`, and a full ncu name is base + rollup +
submetric (`smsp__warps_launched.sum.per_second`). The unit prefix is WHERE the counter sits and
fixes the instance count you are rolling up over -- `gpu__` 1, `sm__` per SM, `smsp__` 4 per SM,
`l1tex__` one per SM, `lts__` per L2 slice, `dram__` per DRAM partition.

There are exactly five quantities, and mixing them is the commonest reading error:

| quantity | NVIDIA's words |
| --- | --- |
| `instruction` | "An assembly (SASS) instruction. Each executed instruction may generate zero or more requests." |
| `request` | "A command into a HW unit to perform some action ... Each request accesses one or more sectors." |
| `sector` | "Aligned 32 byte-chunk of memory in a cache line or device memory." An L1 or L2 line is FOUR of them |
| `tag` | "Unique key to a cache line. A request may look up multiple tags" |
| `wavefront` | "Unique 'work package' generated at the end of the processing stage for requests" -- CYCLES of unit occupancy, not bytes touched |

The metric TYPE fixes which qualifiers are legal, which is why `:stat=` has two sets and not one.
Counter: rollup MANDATORY (`avg sum min max`), submetric optional. Ratio: rollup FORBIDDEN, only
`pct`, `ratio`, `max_rate`. Throughput: BOTH mandatory, submetric restricted to
`.pct_of_peak_sustained_active` or `.pct_of_peak_sustained_elapsed`.

PAPI splits the ncu name down that seam: the ROLLUP becomes `:stat=`, the submetric stays a `.`
suffix.

```
ncu   gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
PAPI  cuda:::gpu__dram_throughput.pct_of_peak_sustained_elapsed:stat=avg
ncu   dram__bytes_read.sum            ->  cuda:::dram__bytes_read:stat=sum
ncu   l1tex__t_sector_hit_rate.pct    ->  cuda:::l1tex__t_sector_hit_rate:stat=pct
```

The grammar tells you what to try; `papi_native_avail -e` still decides.

PAPI's only cuda PRESETs are the FLOP counters and they exist only for GA100 and GH100
(`papi_events.csv`). Everywhere else ask for the native spelling directly:
`cuda:::sm__sass_thread_inst_executed_op_ffma_pred_on:stat=sum`, with `_dfma_` for FP64 and `_hfma_`
for FP16 -- x2 for FLOPs, and note the preset cannot separate BF16 from FP16 because both map to
`_hfma_`.

Ask a QUESTION, then find the event that answers it on THIS device. "How much DRAM traffic" is a
different name on every vendor and often on every generation, so a hard-coded event list is a list
that stops working. NVIDIA events come through the `cuda` component; AMD through `rocp_sdk`, or the
deprecated `rocm` on pre-MI300 parts.

## Reading the numbers

The counts above were measured here; the THRESHOLDS below still come from the vendor docs, so
calibrate them on your own kernel before trusting one. Counters do not name a bottleneck. They
eliminate candidates, in this order -- stop at the first step that fires, because the later numbers
are consequences of the earlier ones.

**1. Was the device even the problem?** If `nsys` already showed device time well under the wall
clock, stop. Launch gaps and copies are host findings and no counter below moves them.

**2. Occupancy -- but only against the grid.** `smsp__warps_active:stat=sum` over
`sm__cycles_elapsed:stat=sum` is the mean resident WARPS PER SM -- a warp count, not a fraction. For
achieved occupancy divide by the part's ceiling, `maxThreadsPerMultiProcessor / 32` read from the
device with `cudaDeviceGetAttribute` and never from memory: 48 on Ada, 64 on GA100/GH100, 32 on
Turing. Read it as a trend across your own versions; the absolute is now computable, not a guess.
Low occupancy has two causes the number alone cannot separate: fewer blocks than SMs (fix the
decomposition -- one element per thread, not one row; split the reduction), or a full grid still
capped by registers or shared memory per block
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

Coalescing has a FLOOR and it is not 1. Measure it as sectors per request:
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld:stat=sum` over
`l1tex__t_requests_pipe_lsu_mem_global_op_ld:stat=sum` (`:stat=sum` on both, or the levels do not
divide; `_op_st` for stores). NVIDIA's optimum is per access size -- **4 for 32-bit, 8 for 64-bit, 16
for 128-bit** -- because 32 lanes x 4 B is 128 B is 4 sectors. So 4 on a `float` load is PERFECT and
reading it as "should be 1" is the trap. Above the floor is uncoalesced, 32 being every lane in its
own sector; below it is broadcast or overlap inside one line.
`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld:stat=ratio` is the same number in
one event and one run -- check it enumerates on your part.

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

**Split the counters before you compare anything.** At fixed algorithm, input and grid, some are an
IDENTITY check between the two versions and the rest are the explanation. Do not read the second row
until the first row matches.

| | counters | what a change means |
| --- | --- | --- |
| INVARIANT | `sm__sass_thread_inst_executed`, `smsp__inst_executed`, `sm__sass_thread_inst_executed_op_{f,d,h}fma_pred_on`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld` | the code does different work -- a correctness bug, unless you changed unrolling or vector width, which move warp-instruction counts by design |
| SCHEDULE-DEPENDENT | every `smsp__warps_issue_stalled_*`, `l1tex__t_sector_hit_rate`, `lts__t_sectors*`, measured `dram__bytes*` above the compulsory floor, `smsp__warps_active`, `sm__cycles_elapsed` | this is the subject of the comparison |

- **Never divide a `dram__` count by an `sm__` cycle count and call it a rate.** Different clock
  domains -- `<unit>__cycles_elapsed` is in that unit's own domain. Bytes per SM cycle is a legal
  unitless comparison KEY between two runs of one binary on one box; it is not a bandwidth, and it
  quietly changes meaning if the two runs boosted differently.
- **The noise floor is about 7%.** NVIDIA, on two counters that should be algebraically consistent
  disagreeing inside ONE ncu report: "the small variations are due to the multiple replay passes
  which each collect different metrics and are not always identical." Under one counter per run,
  EVERY pair you divide is in that regime. Two significant figures on a ratio, and a sub-10% delta
  in a ratio metric is unresolved until you have repeats.
- **`ncu --baseline` is a per-metric percent delta and nothing more** -- `(current - baseline) /
  baseline`, no normalisation, no cross-metric score. The translation here is the same shape: a
  delta table normalised per element of FIXED work, with `sm__cycles_elapsed` beside it as the
  duration normaliser it already is. Say which of the two a number is.

The same rule buys you a metric no single event provides. Warp lane efficiency is
`sm__sass_thread_inst_executed:stat=sum / (smsp__inst_executed:stat=sum * 32)` -- thread
instructions over warp instructions times the warp width. Both enumerate here at `Numpass=1`, and
`:stat=sum` on BOTH is what makes the `sm__` and `smsp__` levels comparable. Well under 1 is
divergent control flow or a partial last warp: wasted issue slots, not wasted bandwidth.

Two rules override all of it:

- **The kernel's work is the invariant.** If a counter from the INVARIANT row moved between two
  versions meant to compute the same thing, recheck correctness before reading any other number.
- **A counter improving while the uninstrumented run gets slower is not an improvement.**

## When it counts nothing

A counter that was never collected reads exactly like a kernel that did no work. Four rows, all
reproduced here; the last code covers a second cause that comes from the component source:

| code | where it fires | what it means |
| --- | --- | --- |
| `-1` `PAPI_EINVAL`, "Invalid argument" | `PAPI_add_named_event` | not a name this component enumerates: a typo, or ncu's `.sum` spelling |
| `-27` `PAPI_EMULPASS`, "multiple passes required" | `PAPI_add_named_event` | `Numpass > 1` on this part; pick another event |
| `-14` `PAPI_EMISC`, "Unknown error code" | `PAPI_add_named_event` | a `:stat=` this event does not accept -- including its own default |
| `-14` `PAPI_EMISC`, "Unknown error code" | `PAPI_start` | the permission gate -- OR the combined set does not fit one pass |

**`PAPI_EMISC` at `PAPI_start` has TWO causes and one code.** PAPI's error table does not cover what
a component returns, so the real complaint reaches you in neither case. Tell them apart by the shape
of the failure: ONE event that will not start is the gate; a SET whose members each added fine and
which will not start together is the pass budget -- `get_config_image` sets
`beginPassGroupParams.maxPassCount = 1` over the COMBINED set and returns `PAPI_EMISC` when
`NVPW_RawMetricsConfig_AddMetrics` cannot schedule it, and the per-event -27 check never modelled
the set (component source). Bisect the set before you go near `/proc/driver/nvidia/params`. For the
gate, get the driver's own wording from a tool that prints it:

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

The fix needs root, so ASK THE USER to run it -- do not sudo yourself:

```sh
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
# reload the nvidia module or reboot; in a container pass --cap-add=SYS_ADMIN
```

Otherwise run the counted binary as root or with `CAP_SYS_ADMIN` (`CAP_PERFMON` also works from
driver R565). No code change works around it.

## Traps

- **A count of 0 is a measurement; ERROR is not.** The code above prints `ERROR (not counted)` when
  setup failed, and stops counting if a stop fails mid-run. Read that line before the numbers.
- **Check an empty bracket before you believe a full one.** `gpu_papi_init` does this for you and
  refuses to run if it fails. It is the one self-test that catches a counter which is accumulating
  device-wide instead of attributing -- the failure mode that produces confident, plausible, wrong
  numbers on every region at once, with no error anywhere.
- **A cache-resident working set reports near-zero DRAM traffic, and that is CORRECT.** This part
  has 24 MB of L2; a 6 MB buffer set never reaches DRAM, and `dram__bytes_read` duly returned 640
  bytes for a kernel touching 4 MB. Before calling a DRAM counter broken, scale the working set
  past L2 (`cudaDeviceGetAttribute` with `cudaDevAttrL2CacheSize`) and check the number tracks.
- **L2 is NOT flushed between regions.** There is no cache control on this path at all, so region N
  runs warmed by region N-1 and the first region is the only cold one. Discard it, or warm every
  region deliberately, and say which you did.
- **`lts__t_sectors` is not "L1 misses".** It counts L2 tag-stage sector lookups from ALL source
  units -- copy engine, other GPCs, the sysmem and peer apertures. Ask for the `srcunit_tex` variants
  if you want what L1 sent to L2. And `lts__t_sectors_lookup_miss x 32` is NOT `dram__bytes_read`:
  dirty-line writeback, write-through, memory compression and sysmem/peer misses all sit between the
  two. DRAM traffic from `dram__*`, L2 behaviour from `lts__*`, neither derived from the other.
- **The L1 hit rate excludes shared memory.** `l1tex__t_sector_hit_rate` covers local, global,
  surface and texture through the tag stage, so a kernel restaged through `__shared__` moves traffic
  OUT of that stage and the rate can go either way while DRAM bytes fall hard. Judge a tiling change
  on `dram__bytes` and `lts__t_sectors`; the hit rate explains, it does not decide.
- **`sm__sass_thread_inst_executed` without `_pred_on` counts predicated-OFF lanes.** The suffix is
  what restricts a thread-instruction counter to lanes that did the work, so the lane-efficiency
  ratio above measures DIVERGENCE and not PREDICATION -- a branch-to-predication rewrite scores near
  1.0 while wasting the same lanes. Use the `_pred_on` pair if that is the question.
- **`smsp__warps_issue_stalled_not_selected` going UP is healthy.** It means the scheduler had more
  eligible warps than issue slots. It is the one stall where more is fine, and a blanket "fewer
  stalls is better" optimises away your own latency hiding.
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
  `:device=0`, so multi-GPU needs `:device=N` and one event set per device. Both limits are enforced
  with their own errors in the component source: `PAPI_EISRUN` for a second running cuda event set,
  and "Profiling same gpu from multiple event sets not allowed."
- **The component sits on an API NVIDIA deprecated in CUDA 13.0** -- the CUPTI Profiler API, tracked
  as PAPI issue #307 -- with two live bugs that land on events this page recommends: #542, `lts__*`
  metrics unqueryable on Blackwell, and #594, a segfault on toolkits 13.0 through 13.2. Check the
  tracker before blaming your kernel.
- **`PAPI_reset` is broken here, and the multi-read `.avg` path has an off-by-index bug.**
  `cuptip_ctx_reset` loops over the READ count instead of the event count, and the `.avg` aggregation
  indexes by read rather than by event (component source). The start/stop-per-region, one-event-per-
  run shape above calls neither. That is a reason to keep the shape, not a coincidence.

## Documentation

- PAPI project home -- https://icl.utk.edu/papi/
- PAPI cuda component, build flags and context requirement -- https://github.com/icl-utk-edu/papi/blob/master/src/components/cuda/README.md
- The component source quoted above: pass budget, read/stop paths, `:stat=` rebuild, broken reset --
  https://github.com/icl-utk-edu/papi/blob/master/src/components/cuda/cupti_profiler.c
- cuda component support matrix: compute capability, API, toolkit bounds -- https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-Cuda-Component
- PAPI release notes, where `:stat=` arrived -- https://github.com/icl-utk-edu/papi/blob/master/RELEASENOTES.txt
- PAPI issues #307 (deprecated CUPTI Profiler API), #542 (`lts__*` on Blackwell), #594 (CUDA 13.0-13.2 segfault) -- https://github.com/icl-utk-edu/papi/issues
- NVIDIA CUPTI, which the cuda component sits on -- https://docs.nvidia.com/cupti/main/main.html
- Nsight Compute profiling guide: metric naming, `.sum`/`.avg` roll-ups, kernel replay -- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- The profiling permission gate and how to lift it -- https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
