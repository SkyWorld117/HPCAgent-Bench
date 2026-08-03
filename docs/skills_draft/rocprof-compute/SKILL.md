---
name: rocprof-compute
description: Kernel-level analysis on AMD with rocprof-compute -- Speed-of-Light first, then the memory chart, then the pipe. The ncu-shaped question, answered with CU-shaped numbers.
---

`rocprof` answers WHICH kernel owns device time. This page answers WHY THAT KERNEL IS SLOW: which
hardware block is at its limit, how far the kernel is from the roof, and which pipe was issuing.
It is the AMD counterpart of `ncu`, and the ladder below is the same ladder -- the numbers are not.

Run `rocprof` first anyway. A perfectly analysed kernel that owns 4% of the run is 4%.

## What was measured here, and what was not

**No command below was executed and no number below was observed.** Every flag, file name, metric
and formula comes from the upstream ROCm documentation cited at the bottom. Treat all of it as
unverified and check the first command against your own `--help` before building a plan on it.

What WAS established, on a Radeon 780M with ROCm 7.2.4: `rocprof-compute` is INSTALLED by the
distro ROCm packages and still refuses to run, because it pins Python dependencies the system
Python does not satisfy. Every subcommand -- including `--help` -- exits after printing:

```
[ERROR] the 'astunparse==1.6.2' distribution does not meet version requirements to use rocprofiler-compute.
  --> version installed : 1.6.3
[ERROR] The 'plotext' package was not found in the current execution environment.
[ERROR] The 'dash>=3.0.0' package was not found in the current execution environment.
   ... 11 packages in total
```

Note it exits **0**, so a wrapper that checks the return code concludes the profile succeeded and
finds no output. The pin is exact (`==1.6.2`) and the installed version is NEWER, so this does not
resolve by upgrading; build a venv from
`<rocm-root>/libexec/rocprofiler-compute/requirements.txt`. Confirm `rocprof-compute --help`
actually prints its usage before assuming the tool is available on any host.

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
- **Dispatches are SERIALIZED**, independently of replay: kernels that would overlap across HIP
  streams on the same GPU do not while profiling. So a counted run's concurrency is not your run's
  concurrency, and this distorts wall clock even on a single pass.
- **Replay breaks MPI.** Running the application repeatedly means repeated `MPI_Init` /
  `MPI_Finalize`, which fails. Use `--iteration-multiplexing` for MPI workloads.
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
must not carry over: **CDNA is 64 lanes, RDNA is 32** with an optional 64-lane mode.

Occupancy is waves resident per SIMD over the slots that SIMD holds. On CDNA that is 8 per SIMD and
**32 wavefront slots per CU**, so filling a CU means 32 x 64 = **2048 work-items**; RDNA3 has 16
slots per SIMD, so 1024. Those are the numbers to size a launch against -- read `sysinfo.csv` for
the actual part rather than either figure, because this is exactly the arithmetic that differs by
generation.

Low occupancy has two causes this number cannot separate: too few workgroups for the CUs (fix the
decomposition), or a full grid capped by VGPRs or LDS per workgroup (fix the resource use). The
Wavefront Launch panel has the register and LDS figures that tell them apart.

Occupancy counts waves PARKED, not waves working. It matters only once something else says the CUs
stalled.

**3. The memory chart.** The one panel with no NVIDIA analogue worth borrowing: it lays out the
whole hierarchy -- vector L1D, scalar L1D, LDS, L2 (TCC), and the fabric out to HBM -- with the
traffic on each link. Read it as a flow. The level where the numbers stop shrinking is the level
your working set does not fit in, and that is the level to tile for.

The L2 panel prints `Hit Rate` as a percentage; the underlying metric is
`100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))` on CDNA, and counts
`GL2C_HIT`/`GL2C_MISS` on RDNA. Read it as the EXPLANATION of the traffic, never on its own: a
rising hit rate with unchanged HBM bytes means you added accesses, not locality.

**4. Traffic against the algorithm's minimum.** Needs no peak and no roofline. Count the bytes the
kernel MUST move -- every input read once, every output written once -- and divide the measured
traffic by it.

**Check the UNIT on the panel in front of you; the two tools disagree.** Verified in the
sources: `rocprof-compute`'s gfx942 L2 panel declares `Read BW` with `unit: (Bytes + $normUnit)`,
while ROCm's counter reference defines `FetchSize` as "The total kilobytes fetched from the video
memory". So the same physical quantity arrives in **BYTES** from one tool and **KILOBYTES** from the
other. Importing one page's habit into the other tool is a 1024x error in the one number this step
exists to produce. The same panel also prints `L2-Fabric Read BW` in `GB/s` -- a RATE, not a volume,
and not interchangeable with either.

- near 1 -- compulsory traffic. Tiling buys nothing; only a different algorithm does.
- well above 1 -- you are re-reading what should have stayed in cache. This is what tiling and
  fusion are for, and the ratio is how you check it worked.
- write bytes far above the output size -- uncoalesced stores, or a read-modify-write the source
  does not show.

**5. Which pipe.** Only once memory is excluded.

**The names differ between the two AMD tools, and one pair means opposite things.** Read the Read the
column for the tool you are actually running:

| what you want to know | `rocprof-compute` prints | `rocprofv3 --pmc` name |
| --- | --- | --- |
| was the vector ALU busy | `VALU Utilization` | `VALUBusy` |
| how many LANES were active (DIVERGENCE) | `VALU Active Threads` (work-items) | `VALUUtilization` |
| scalar pipe busy | `SALU Utilization` | `SALUBusy` |
| memory unit stalled | `Mem Unit Stalled` | `MemUnitStalled` |
| LDS bank conflicts | `LDS Bank Conflict` | `LDSBankConflict` |

`VALUUtilization` and `VALU Utilization` are the trap: near-identical spellings, different
quantities. On `rocprof-compute` the divergence number is **`VALU Active Threads`**, whose unit is
work-items -- against the wavefront width, so read 32/64 on CDNA rather than a percentage.

The expressions, read out of ROCm's `counter_defs.yaml` for **gfx942** (MI300). They are
ARCHITECTURE-SPECIFIC -- `LDSBankConflict` uses `SQC_LDS_BANK_CONFLICT / SQC_LDS_IDX_ACTIVE` on
gfx10, and `L2CacheHit` counts `GL2C_HIT`/`GL2C_MISS` there instead of `TCC_*` -- so ask the tool
for the metric BY NAME and let it pick, rather than hand-computing from a formula for the wrong
part:

| metric | expression on gfx942 |
| --- | --- |
| `VALUBusy` | `100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `SALUBusy` | `100*reduce(SQ_INST_CYCLES_SALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)` |
| `MemUnitStalled` | `100*TCP_TCP_TA_DATA_STALL_CYCLES_max/reduce(GRBM_GUI_ACTIVE,max)/SE_NUM` |
| `VALUUtilization` | `100*reduce(SQ_THREAD_CYCLES_VALU,sum)/(reduce(SQ_ACTIVE_INST_VALU,sum)*MAX_WAVE_SIZE)` |
| `LDSBankConflict` | `100*reduce(SQ_LDS_BANK_CONFLICT,sum)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM` |
| `L2CacheHit` | `100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))` |
| `GPUBusy` | `100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)` |

Note what those denominators are NOT: none of them is `SQ_BUSY_CU_CYCLES`. The normaliser is
`GRBM_GUI_ACTIVE` (GPU active cycles) scaled by a part constant (`CU_NUM`, `SE_NUM`), and
`VALUUtilization` alone divides by `MAX_WAVE_SIZE`, which is why it is the one that is a lane
fraction rather than a time fraction.

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
- MI300/MI200 counter DEFINITIONS and units (note: this page gives no expressions) -- https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- The derived-counter EXPRESSIONS, per architecture -- the authority for every formula above:
  https://github.com/ROCm/rocprofiler-sdk/blob/amd-staging/source/share/rocprofiler-sdk/counter_defs.yaml
- rocprof-compute's per-panel metric definitions and UNITS, per part (`gfx942/*.yaml`) --
  https://github.com/ROCm/rocprofiler-compute/tree/develop/src/rocprof_compute_soc/analysis_configs
- Occupancy on AMD: 8 wavefront slots per SIMD, 32 per CU on CDNA -- https://gpuopen.com/learn/occupancy-explained/
- AMD Instinct MI300 (CDNA3) ISA reference, for the hardware numbers -- https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- Occupancy on AMD, wave-per-SIMD arithmetic -- https://gpuopen.com/learn/occupancy-explained/
- AMD's own profiling walkthrough, roofline reading -- https://rocm.blogs.amd.com/software-tools-optimization/profiling-guide/novice/README.html
- HIP programming model: wavefront, CU, LDS, XCD -- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
