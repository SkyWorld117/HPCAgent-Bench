---
name: rocprof-compute
description: Kernel-level analysis on AMD with rocprof-compute -- Speed-of-Light first, then the memory chart, then the pipe. The ncu-shaped question, answered with CU-shaped numbers.
---

`rocprof` answers WHICH kernel owns device time. This page answers WHY THAT KERNEL IS SLOW: which
hardware block is at its limit, how far the kernel is from the roof, and which pipe was issuing.
It is the AMD counterpart of `ncu`, and the ladder below is the same ladder -- the numbers are not.

Run `rocprof` first anyway. A perfectly analysed kernel that owns 4% of the run is 4%.

## What was measured here, and what was not

**There is no AMD GPU on the box this was written on.** No command below was executed, no number
below was observed. Every flag, file name, metric and formula comes from the upstream ROCm
documentation cited at the bottom. Treat all of it as unverified and check the first command
against your own `--help` before building a plan on it.

What is NOT vendor folklore is the reading ORDER, and the reason to trust it here is a measured
one: on the NVIDIA twin of this page, following the ladder in order produced a **47.4x** kernel
speed-up, beating three of the vendor's own shipped recommendation blocks -- because the vendor's
blocks each argue for their own chapter and the ladder decides which chapter to be in. That part
ports. The thresholds do not.

## The name changed twice

| you may see | current name | what it is |
| --- | --- | --- |
| `omniperf` | `rocprof-compute` | THIS page: kernel-level counters, SOL, roofline |
| `omnitrace` | `rocprof-sys` | whole-application trace, CPU+GPU timeline |
| `rocprof` / `rocprofv2` | `rocprofv3` | the dispatch trace and raw `--pmc` collection |

Search results and older tuning guides are full of the left column. They describe the same tools.
If `rocprof-compute` is not found, try `omniperf` before concluding the tool is absent.

## How it runs

```sh
rocprof-compute profile --name <workload> -- ./your_app <args>   # collect
rocprof-compute analyze -p workloads/<workload>/<gpu_model>/     # read
```

## The two-command shape

Profiling writes a WORKLOAD DIRECTORY, and analysis reads it back. That split is the point: you
collect once and then ask many questions of the same data, so do not re-profile to change a
question.

```
workloads/<name>/<gpu_model>/
  log.txt
  perfmon/               counter_def_*.yaml, pmc_perf_*.yaml -- what was asked for
  pmc_perf.csv           the merged counter results
  profiling_config.yaml
  roofline.csv           absent if you passed --no-roof
  sysinfo.csv            the PART. read this first
```

`sysinfo.csv` is the part's geometry, measured. It is what turns every occupancy sentence below
into arithmetic instead of folklore, and it is the file to open first.

## It REPLAYS your kernel, and that is the cost

`rocprof-compute` collects all available counters for the part, and no GPU has enough counter
hardware to do that in one pass. It acquires them by **application replay** -- running the
application repeatedly, a different counter set each time. Three consequences, all of them
practical:

- **It is slow.** Expect many multiples of one run. Cut the work before you profile, not after.
- **The application must be deterministic and re-runnable.** A run whose output depends on wall
  clock, RNG without a fixed seed, or a file it consumes-and-deletes will produce counter rows
  from runs that did different things, and nothing in the merged CSV says so.
- **Roofline is a second collection stage** on top of the first: it runs the part's micro-
  benchmarks to find the achievable roofs. `--no-roof` skips it, and is the first flag to reach
  for while you are iterating. Roofline is unavailable pre-MI200 regardless.

Narrow before you widen. `-k <kernel-substr>` filters to one kernel by name; `-d <id>` picks
dispatches (1-based) so you profile the steady-state iteration and not the cold first one; and
`-b <block>` collects only the hardware blocks you asked about.

```sh
rocprof-compute profile --name vcopy --no-roof -k vecCopy -d 3:8 -- ./vcopy -n 1048576
```

## Read it in this order

Stop at the first step that fires. The later numbers are consequences of the earlier ones, so a
number read out of order will send you to the wrong chapter with real evidence for it.

**1. System Speed-of-Light.** One panel, every major block as a percentage of its own peak. This
is the whole triage: the block nearest its roof is the one to work on, and every other panel in
the tool is an explanation of that one number. If nothing is near a roof, the kernel is
latency-bound and you are in step 2, not step 4.

**2. Wavefront launch and occupancy -- against the PART.** The wavefront width is the thing you
must not carry over: **CDNA is 64 lanes, RDNA is 32** with an optional 64-lane mode. Occupancy is
waves resident per SIMD over the 8 that SIMD holds, or 32 waves scaled to the CU on CDNA. So a CU
is filled by 256 threads on CDNA and 128 on RDNA, and every "use 256 threads" habit from NVIDIA is
wrong here by exactly that factor.

Low occupancy has two causes this number cannot separate: too few workgroups for the CUs (fix the
decomposition), or a full grid capped by VGPRs or LDS per workgroup (fix the resource use). The
Wavefront Launch panel has the register and LDS figures that tell them apart.

Occupancy counts waves PARKED, not waves working. It matters only once something else says the CUs
stalled.

**3. The memory chart.** The one panel with no NVIDIA analogue worth borrowing: it lays out the
whole hierarchy -- vector L1D, scalar L1D, LDS, L2 (TCC), and the fabric out to HBM -- with the
traffic on each link. Read it as a flow. The level where the numbers stop shrinking is the level
your working set does not fit in, and that is the level to tile for.

`L2CacheHit` = `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum) * 100`. Read it as the EXPLANATION of
the traffic, never on its own: a rising hit rate with unchanged HBM bytes means you added
accesses, not locality.

**4. Traffic against the algorithm's minimum.** Needs no peak and no roofline. Count the bytes the
kernel MUST move -- every input read once, every output written once -- and divide the measured
`FetchSize + WriteSize` by it. **Both are KILOBYTES on this vendor**, which is the unit trap that
turns a correct ratio into a 1000x wrong one.

- near 1 -- compulsory traffic. Tiling buys nothing; only a different algorithm does.
- well above 1 -- you are re-reading what should have stayed in cache. This is what tiling and
  fusion are for, and the ratio is how you check it worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show.

**5. Which pipe.** Only once memory is excluded.

| metric | formula | what it says |
| --- | --- | --- |
| `VALUBusy` | `SQ_ACTIVE_INST_VALU / SQ_BUSY_CU_CYCLES * 100` | the vector ALU was issuing |
| `SALUBusy` | `SQ_INST_CYCLES_SALU / SQ_BUSY_CU_CYCLES * 100` | scalar work -- high here is usually address arithmetic that should be hoisted |
| `MemUnitStalled` | `SQ_WAIT_INST_ANY / SQ_BUSY_CU_CYCLES * 100` | the memory unit was stalled |
| `VALUUtilization` | active LANES in a wave, percent | divergence |
| `LDSBankConflict` | `SQ_LDS_BANK_CONFLICT / SQ_BUSY_CU_CYCLES * 100` | LDS stride collides |

`VALUUtilization` is scaled by the wavefront width, so the SAME source branch reads 50% on CDNA
(32 of 64 lanes) and 100% on RDNA in wave32. Do not compare it across parts, and do not compare it
to an NVIDIA warp-efficiency number.

Matrix work rides a separate pipe: on CDNA the MFMA units are not counted by `VALUBusy`, so a
GEMM-shaped kernel showing a low `VALUBusy` is not idle, it is on the pipe you did not look at.

**6. Roofline, last.** It tells you which side of the ridge point you are on and therefore which of
the steps above can pay at all -- it does not tell you what to change. Memory-bound kernels sit
left of the crossover, compute-bound right, and a kernel sitting far BELOW both curves is neither:
it is latency-bound, and the fix is occupancy or more work in flight, not traffic and not flops.

## What each finding costs the next

| pair | the conflict |
| --- | --- |
| occupancy -> registers | raising waves per SIMD means fewer VGPRs each; past a point the kernel spills to scratch and the extra waves are slower than the spill |
| tiling -> LDS | a bigger tile is more LDS per workgroup, which is itself an occupancy cap. The two settle together |
| LDS -> bank conflicts | the padding that fixes a conflict also changes the tile's LDS footprint, so re-read occupancy after |
| wave64 -> divergence | a 64-lane wave serialises a branch across twice the lanes of a 32-lane one, so the same source diverges harder on CDNA |
| replay -> trust | every counter row came from a DIFFERENT run of your app. Non-determinism does not show up as an error, it shows up as a number |

## Traps

- **`sysinfo.csv` before anything else.** Every occupancy and width sentence above depends on the
  part, and the part is in that file.
- **Do not port NVIDIA thresholds.** Wavefront width, LDS banking, the cache hierarchy and the
  matrix pipe all differ. A number meaning "bad" on an SM does not mean it on a CU.
- **A profiled run's wall clock belongs to no comparison.** Replay alone makes it meaningless.
  Read the COUNTERS; take every speed-up from an uninstrumented build.
- **Verify the answer.** A kernel that got faster and wrong measures nothing. This is not a
  formality on AMD: the fastest paths here often involve changing the wave width or the LDS
  layout, and both can change a reduction's summation order.
- **One profiling client at a time.** `rocprof-compute`, `rocprofv3` and a PAPI GPU component all
  want the same subscriber. Nest them and one of them silently gets nothing.
- **`--no-roof` while iterating.** Then one final run with the roofline when you want the picture.

## Documentation

- ROCm Compute Profiler (rocprof-compute), formerly Omniperf -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/
- Profile mode: every flag quoted above -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
- The performance model: SOL, memory chart, the per-block panels -- https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/performance-model.html
- MI300/MI200 counters and every derived formula quoted above -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- Occupancy on AMD, wave-per-SIMD arithmetic -- https://gpuopen.com/learn/occupancy-explained/
- AMD's own profiling walkthrough, roofline reading -- https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html
- HIP programming model: wavefront, CU, LDS, XCD -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
