---
name: rocprofv3
description: Trace an AMD GPU run with rocprofv3 -- which kernel, which copy, which gap -- rank by total_ns not mean_ns, and know when only rocprof-compute can answer.
---

The device half of a profile, on AMD. `perf` samples a host call stack; a HIP launch is
ASYNCHRONOUS, so a host profile of a HIP kernel shows the synchronisation the host waited in and
nothing about the kernel. What the DEVICE did is recorded instead, one record per dispatch and per
copy.

This is the AMD counterpart of `nsys`. It answers WHICH kernel and WHICH copy. It does not answer
why a kernel is slow -- that is `rocprof-compute`.

## What was measured here, and what was not

**There is no AMD GPU on the box this was written on.** No command below was executed. Every flag,
file name and column comes from the upstream ROCm documentation cited at the bottom, and from the
CSV readers this repo already ships. Treat the tool behaviour as unverified; check `rocprofv3
--help` before building a plan on a flag.

The READING RULE in "rank by the right column" is not vendor folklore -- it was measured on the
NVIDIA twin of this page, where the fixture's launch-bound kernel owns **67.3%** of device time by
total and ranks **DEAD LAST** by mean. That arithmetic is vendor-independent.

## The name changed twice

| you may see | current name | what it is |
| --- | --- | --- |
| `rocprof`, `rocprofv2` | `rocprofv3` | THIS page: dispatch trace, `--pmc` counters |
| `omniperf` | `rocprof-compute` | kernel-level analysis, SOL, roofline |
| `omnitrace` | `rocprof-sys` | whole-application CPU+GPU timeline |

Older tuning guides use the left column throughout. A host with only the deprecated v1 takes a
DIFFERENT command line and produces a different schema -- see the bottom of this page.

## How it runs

```sh
rocprofv3 --kernel-trace --memory-copy-trace --stats --output-format csv \
          --output-directory prof --output-file run -- ./your_app <args>
```

`--` separates the tool's flags from the application's. It matters: without it the first
application argument is parsed as a tool flag.

## The four reports

They answer different questions:

| report | file | what it answers |
| --- | --- | --- |
| kernel stats | `*_kernel_stats.csv` | per kernel: `Calls`, `TotalDurationNs`, `AverageNs`, `MinNs`, `MaxNs`, `Percentage` |
| memory copy stats | `*_memory_copy_stats.csv` | per operation: how long H2D / D2H took. **NO byte volume** |
| kernel trace | `*_kernel_trace.csv` | per dispatch: `Workgroup_Size_*`, `Grid_Size_*` (in WORK-ITEMS), `LDS_Block_Size` (LDS bytes, rounded up to the allocation granule) |
| agent info | `*_agent_info.csv` | the PART: `Wave_Front_Size`, `Num_Xcc`, `Cu_Count`, `Simd_Count`, `Max_Waves_Per_Simd`, `Lds_Size_In_Kb` |
| domain stats | `*_domain_stats.csv` | per API/dispatch DOMAIN totals -- the top-level split before you rank within one |

Find them RECURSIVELY. Some ROCm releases write them flat in the output directory, others under
`<hostname>/<pid>/`, and a glob that assumes one layout silently finds nothing on the other.

**Read `*_agent_info.csv` first.** It is the part's geometry, measured, and it is what makes every
occupancy sentence arithmetic instead of folklore. `Grid_Size_*` is in WORK-ITEMS, not workgroups
-- divide by `Workgroup_Size_*` to get the block count, or every occupancy number you derive is
wrong by the block size.

## Rank by the right column

`TotalDurationNs`, not `AverageNs`. The kernel worth working on is the one that owns the most
device time in aggregate, and the two columns disagree exactly when it matters: a trivial kernel
launched thousands of times can own most of the run while ranking last by mean. On the NVIDIA
fixture built for this, the 64-launches-per-rep kernel owns 67.3% of device time and has the
smallest mean of the four. Sorting that table by mean picks the wrong kernel with a real number.

`Percentage` is that ranking already done for you. Use it, then check `Calls` -- a high percentage
with a high call count is a LAUNCH problem (batch, fuse, or use a graph), and a high percentage
with a low call count is a KERNEL problem (go to `rocprof-compute`).

## Was the device busy at all?

Sum `TotalDurationNs` across kernels and divide by the wall clock of the same run. This is the
first number to compute and the one that decides whether any of the rest matters.

- **Device percentage low** -- the GPU is idle most of the run. The finding is on the HOST: launch
  gaps, synchronous copies, a `hipDeviceSynchronize` in the timestep loop, or work that never got
  offloaded. No kernel-level tool will help; fix the gaps first.
- **Device percentage high, one kernel dominant** -- go to `rocprof-compute` for that kernel.
- **Device percentage high, time spread evenly** -- an algorithmic or fusion question, not a
  per-kernel one.

**Exclude one-time setup from the wall clock before you divide.** On the NVIDIA twin this exact
recipe read **0.04% against a truth of 6.01%** -- a 150x error -- because a JIT compile sat inside
the span being divided by. AMD has the same hazard in a different place: the first dispatch of a
code object pays a load, and `hipMalloc` of a large buffer is not free. Time the STEADY-STATE
reps, not the process.

## Copies: the byte volume is there, in the right format

The memory-copy STATS report gives durations only. The byte count exists -- the buffer-tracing
record carries a `bytes` field, emitted in the Perfetto, rocpd and JSON outputs -- so ask for a
format that carries it rather than reconstructing transfer sizes from your source:

```sh
rocprofv3 --memory-copy-trace --output-format csv json -- ./your_app
```

Divide bytes by the reported time to get the achieved rate, then compare against the link: a
PCIe-attached part and an Infinity-Fabric-attached one differ by an order of magnitude, so "is this
copy slow" has no answer without knowing which one you are on.

The actionable findings are almost always structural rather than rate-related: a copy inside the
timestep loop that could be hoisted, a H2D of data the device already had, or pageable host memory
where pinned would let the copy overlap.

## Counters, when the trace has done its job

`--pmc` collects hardware counters per dispatch. It is the raw form of what `rocprof-compute`
packages, and it is the right tool when you want ONE number rather than a whole analysis.

```sh
rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE TCC_HIT_sum TCC_MISS_sum -- ./your_app
```

Results land in `pmc_<n>/<pid>_counter_collection.csv`, one directory per pass. The file is PID-prefixed, so glob (`pmc_*/*_counter_collection.csv`) rather than naming it.

**The counter budget is hardware, and exceeding it costs runs.** Too many counters in one row and
the kernel is executed multiple times to collect them all.

**Repeating `--pmc` does NOT give you two passes -- it silently DISCARDS the first.** The option
is declared `nargs="*"` with no `append` action, so the second occurrence overwrites the first and
only the survivor is collected. Nothing warns. Multi-pass comes from an INPUT FILE with one `pmc`
row per pass:

```
pmc: SQ_WAVES SQ_BUSY_CU_CYCLES
pmc: TCC_HIT_sum TCC_MISS_sum
```

```sh
rocprofv3 -i counters.txt -- ./your_app
```

Which means the same rule as every other counter instrument: **two counters from two different
passes came from two different executions of your kernel.** A ratio across passes is only
legitimate through a denominator both passes measured (`GRBM_GUI_ACTIVE` is the usual one), and it
is only meaningful at all if the application is deterministic.

Name a counter without a dimension specifier (`TCC_MISS`, not a per-channel form) and rocprofv3
aggregates across all instances for you -- the AMD equivalent of the `:stat=sum` problem on
NVIDIA, resolved in the opposite direction: here the aggregate is the default.

## The deprecated v1, if that is all the host has

```sh
rocprof --stats --timestamp on -o prof/run.csv ./your_app
```

No `--` (its wrapper stops at the first non-option token), and one `*.stats.csv` with no per-kernel
min/max and no memory report. It DOES print launch geometry (`grd`, `wgr`, `lds`, `scr`,
`arch_vgpr`, `sgpr`, `wave_size`), so that column survives the fallback even though most do not. If
half the fields above are missing, this is why -- check which binary you actually ran before
concluding the data is broken.

## Traps

- **`--` before the application.** Missing it turns your app's first argument into a tool flag.
- **Find the CSVs recursively.** Flat or `<hostname>/<pid>/`, depending on the release.
- **`Grid_Size_*` is WORK-ITEMS.** Divide by workgroup size for blocks.
- **A traced run's wall clock is not a timed run's.** Take every speed-up from an uninstrumented
  build.
- **One profiling client at a time.** `rocprofv3`, `rocprof-compute` and a PAPI GPU component all
  want the same subscriber; nested, one of them silently gets nothing.
- **The build gets no extra flags.** Kernel names come from the code object, and the device-debug
  switch would disable device optimisation -- so the traced binary is the one you timed.
- **Which device is measured.** `ROCR_VISIBLE_DEVICES` renumbers devices, so `device 0` in the
  report is not necessarily the one you think. Check `*_agent_info.csv` against the part you meant.
- **Verify the answer.** A kernel that got faster and wrong measures nothing.

## Documentation

- Application tracing and profiling with rocprofv3 -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCprofiler-SDK -- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/
- MI300/MI200 counters, for the `--pmc` names -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- AMD's profiling walkthrough -- https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html
- ROCm Compute Profiler, where a slow kernel goes next -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- HIP programming model: wavefront, CU, LDS -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
