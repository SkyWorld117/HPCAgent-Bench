---
name: nsys
description: Trace a CUDA submission with Nsight Systems -- which kernel, which copy, which gap -- and know when only ncu can answer.
---

A GPU has no call stack to sample. The host thread launches asynchronously and then waits, so a
`perf` profile of a CUDA kernel shows one synchronization call and nothing about the device. What
the device did is RECORDED instead: CUPTI hands `nsys` one activity record per kernel launch and per
memory operation, and the profile is those records, not samples.

That changes what the tool can tell you. `nsys` answers **which kernel, how many times, and when**.
It cannot answer **why that kernel is slow** -- read "nsys or ncu" below before you spend a run on
the wrong instrument.

This is the device half of `profiling`, on NVIDIA. The host instruments (`perf`, PAPI CPU counters)
are that skill; a HIP submission is the `rocprof` skill's, because `nsys` traces CUDA and cannot see
an AMD queue.

## The invocation, exactly

`POST /profile` with `language: "cuda"` routes to the GPU path automatically -- the dispatch is the
LANGUAGE, so you ask the one route the same way whatever you submitted. Knobs that apply:
`reps`, `min_percent` (default 1.0), `residency` (`host` or `device`). `threads` does not apply, and
`counters: true` is a 503 `counters_unsupported` naming `ncu` -- PAPI counts HOST events, which say
nothing about a device kernel.

What runs underneath, and what to type if you trace by hand:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
     --force-overwrite=true --output gpu-profile -- ./app input
nsys stats --format csv --force-export=true --output - \
     --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
     --report cuda_gpu_mem_size_sum --report cuda_gpu_trace gpu-profile.nsys-rep
```

Every part of that line is a decision:

- **`--trace=cuda,nvtx`** and nothing else. `osrt`, `cublas` and `cudnn` each add interception
  overhead to the run you are measuring. `nvtx` is free and is the one lever you have over the
  timeline: bracket your own phases with `nvtxRangePush`/`nvtxRangePop` and they come back as named
  ranges, which is how a gap gets attributed to a phase instead of to "somewhere".
- **`--sample=none`**. CPU sampling answers the host path's question, and it is the one part of
  `nsys` that would drag `kernel.perf_event_paranoid` into a GPU profile -- a device measurement
  failing for a host reason.
- **No `-g`, no `-G`.** Kernel names come from CUPTI, which reads them out of the fatbinary, so
  there is nothing for DWARF to add; and `-G` disables device optimization, which would profile a
  program nobody runs. The traced `.so` is byte-identical to the one the judge times.
- **One `nsys stats` invocation for all four reports.** The first use exports the recording to
  SQLite; asking four times pays that export four times.
- **CSV, not the pretty summary.** The human format right-aligns and thousands-separates numbers a
  parser then has to un-format.
- The recording is `.nsys-rep` on nsys 2021.4+ and `.qdrep` on older builds. These four report names
  need nsys >= 2022.1; older builds spell them `gpukernsum`, `gpumemtimesum`, `gpumemsizesum`,
  `gputrace`, and asking for the new names there returns nothing at all.

Two reports the harness does NOT request, worth adding when you run `nsys` yourself:
`--report cuda_api_sum` (host-side time in `cudaMemcpy`, `cudaLaunchKernel`,
`cudaDeviceSynchronize` -- the gap's own accounting) and the NVTX summary (`nvtx_sum` on current
builds) once you have bracketed your phases.

## `cuda_gpu_kern_sum` -- which kernel, and what to do about it

Per kernel: `instances` (launches), `total_ns`, `mean_ns`, `min_ns`, `max_ns`, `time_pct`.

**`mean_ns` is the number to optimize against.** `total_ns` is a launch-count artifact: change the
rep count and it moves without the kernel changing. Kernels below `min_percent` of device time are
dropped and COUNTED (`kernels_omitted`), so a short list is a short list, never a truncated one.

The summary answers "fewer launches, a bigger grid, or a different algorithm" and they are three
different findings:

| what the rows show | the finding | the change |
| --- | --- | --- |
| one kernel at 80%+ of device time, `mean_ns` >= ~100 us | the kernel BODY is the cost | nothing here says why: `ncu` on that kernel |
| many `instances`, `mean_ns` under ~10 us | launch-bound: more time being told what to do than doing it | fuse, do more per launch, or capture a CUDA graph |
| `blocks` below the SM count (108 on A100, 132 on H100/GH200) | the grid does not fill the device | one element per thread, not one row; split the reduction |
| `mean_ns` flat as the input grows | fixed overhead, not the kernel | read the copies and the gap instead |
| `mean_ns` growing faster than the input | an algorithmic term | no geometry or launch change reaches it; change the algorithm |
| two or three kernels at ~30% each | no single hotspot | fusing them beats tuning any one of them |
| `max_ns` far above `mean_ns`, `min_ns` near it | one slow launch: JIT/module load, clock ramp, another tenant | check warmup covered it before believing the mean |

`device_pct` frames all of it: traced device time per rep against the measured host time per rep.
Below ~50% the kernel is NOT what costs -- the launches and the copies are, and a faster kernel
moves the total by less than the number says.

## The copies -- the single most common finding

`cuda_gpu_mem_time_sum` (how long) and `cuda_gpu_mem_size_sum` (how much) are separate reports,
joined here per operation into `memory[]`: `direction` (`h2d`, `d2h`, `d2d`, `memset`, `other`),
`count`, `total_ns`, `mean_ns`, `total` + `unit`.

The volume keeps nsys's OWN unit rather than being converted to bytes: releases disagree on whether
their `MB` is 10^6 or 2^20, and picking one invents a precision the recording does not have. Do the
bandwidth division yourself and carry the unit with it.

**`total_ns` in `memory[]` is summed over every rep the child ran -- warmup included.** Divide by
`reps + warmup` before you put it next to `elapsed_ns`, which is one rep. Comparing the two raw is
the commonest arithmetic error on this payload; `device_ns_per_rep` is already divided, the memory
rows are not.

What the numbers mean:

- **Transfer time near or above kernel time** -- the transfer IS the problem. No kernel change can
  reach it. Ask first whether the data changes between reps: if it does not, the copy is pure
  overhead and belongs outside the timed region entirely (that is what `residency: "device"` times).
- **Achieved bandwidth near the link's practical rate** (~12 GB/s per direction on PCIe gen3 x16,
  ~25 on gen4, ~50 on gen5, hundreds on GH200's NVLink-C2C) -- the copy is running as fast as the
  wire allows. The only remaining lever is moving LESS: keep buffers resident, transfer once and
  loop on the device, send fp32 where fp64 is not needed, or overlap with streams -- which HIDES
  the transfer behind compute but does not remove it.
- **Achieved bandwidth far below the link with large copies** -- pageable host memory, staged
  through the driver's bounce buffer. `cudaHostAlloc`/`cudaMallocHost` typically doubles it.
- **High `count`, tiny `mean_ns`** -- per-copy latency (a few microseconds each) dominates the
  volume. Batch them into one transfer of a packed buffer.
- **`memset` rows are work too.** A `cudaMemset` per rep is device time and device bandwidth; fold
  it into the kernel that was about to overwrite the buffer anyway.
- **`d2d` traffic you did not write** is usually a library staging a layout change.

## Gaps -- and idle is not the opposite of saturated

The gap is what the arithmetic leaves over:

```
gap_per_rep = elapsed_ns - device_ns_per_rep - (sum of memory total_ns) / (reps + warmup)
```

Where it goes:

- **Launch overhead** -- a few microseconds per launch, host and device side. Against
  `launch_count`, that is a bound you can check in one multiplication: 5000 launches at ~5 us is
  25 ms of nothing, and it will not shrink by making the kernel faster.
- **Synchronization stalls** -- a `cudaDeviceSynchronize` or a synchronous `cudaMemcpy` per rep
  turns an asynchronous pipeline into a round trip. `cuda_api_sum` names which call held the host.
- **Host-side work between launches** -- index math, allocation, a Python frame. The device is idle
  and no device-side change touches it.
- **Context creation** -- the first CUDA call costs 100 ms or more. It belongs in warmup; if it
  lands in a measured rep, the mean is fiction.

**The distinction that matters most: `nsys` cannot tell you whether the GPU was SATURATED.** It
records that a kernel was RESIDENT. A kernel occupying 100% of the timeline while using 3% of the
SMs looks exactly like a kernel at peak -- same rows, same `device_pct`, same "the GPU is busy"
reading. Resident is not busy. The device-side utilization question is answered by counters
(`occupancy`, `device_utilization` below) or by `ncu`, never by the timeline.

Conversely a low `device_pct` IS conclusive: the device really was idle for that fraction, and the
fix is on the host or in the copies.

## `cuda_gpu_trace` -- launch geometry bounds occupancy, it does not measure it

Distinct geometries, most-launched first: `grid` (blocks), `block`, `threads_per_block`,
`warps_per_block` (threads / 32), `blocks`, `registers_per_thread`, `shared_memory` + its unit.

Read it as a set of caps on how many blocks can be resident per SM:

- `blocks` below the SM count -- most of the device never gets work, whatever the kernel does.
- `registers_per_thread * threads_per_block` against the SM's 65536 registers -- 64 registers on a
  256-thread block is 16384, so at most 4 such blocks are resident, i.e. 32 of the 64 warp slots.
- `shared_memory` per block against the SM's shared-memory budget -- the same arithmetic, the other
  resource.
- `threads_per_block` not a multiple of 32 -- a 100-thread block is 4 warps with 28 lanes idle in
  the last one, on every block, on every SM.

**Achieved occupancy is not here and is not inferable from here.** It is a per-SM counter that
Nsight Compute reads:

```sh
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active -- ./app input
```

An occupancy number derived from geometry would be indistinguishable from a measured one, so the
payload ships the note instead of the number.

## nsys or ncu -- what each one cannot answer

`nsys` answers **which kernel and when**: the ranked kernels, the launch count, the copies, the
gaps, the timeline. It CANNOT tell you why any of them is slow -- it records activity, not
counters, so there is no achieved occupancy, no stall reason, no memory throughput, no cache hit
rate anywhere in it. One traced run, small overhead.

`ncu` answers **why this kernel is slow**: stalls by reason, achieved occupancy, DRAM and L1/L2
throughput, divergence, register and spill pressure. It CANNOT tell you how often the kernel ran,
what ran around it, where the host waited, what the copies cost, or whether two kernels overlapped.
It REPLAYS each kernel many times and serialises them, so its wall clock is not your program's.

**Always `nsys` first.** `ncu` on the wrong kernel is a perfectly analysed 4% of the run. Once the
summary has named the kernel:

```sh
ncu --set full -- ./app input                        # everything, slow
ncu --kernel-name regex:gemm --launch-count 1 -- ./app input   # one launch of one kernel
```

And never quote an `ncu` timing as a speed. Replay makes its numbers per-kernel counts, not
durations you can compare to anything.

## The PAPI `cuda` / `nvml` path

Device counters through PAPI are a LIBRARY call, not a judge route: `/profile` with
`counters: true` on a `cuda` submission is a 503 `counters_unsupported`. Use
`hpcagent_bench.harness.papi` directly -- `gpu_feature_set()` to ask what this machine can count
before running anything, then `count_gpu_metric(...)` or `count_gpu_group(..., group=...)`.

Two components, two different questions:

- **`cuda`** (CUPTI): kernel counters -- `occupancy`, `dram_read_bytes`, `dram_write_bytes`,
  `memory_stall`, `l1_hit_rate`, `l2_hit_rate`. This is the "why is the kernel slow" half.
- **`nvml`**: device state -- `power`, `core_clock`, `temperature`, `device_utilization`. This is
  the "was the machine the same machine" half, and it is what catches a sweep whose later reps ran
  at a lower clock.

Groups, so you ask a question rather than an event: `occupancy`, `memory`, `cache`, `power`, `all`.
Cost is one measured run per metric in the group.

Three constraints ship with every device count, and each one is a way to be wrong:

1. **Counter collection SERIALISES kernels and REPLAYS multi-pass metric sets.** A counted run's
   wall clock is not the plain run's. Read the counts, never the time -- and never put a counted
   run's milliseconds next to a timed run's. This is the same reason `ncu` timings are not speeds.
2. **CUPTI changed profiling APIs at Volta.** Pre-Volta parts answer through the event-group names
   (`achieved_occupancy`, `inst_executed`), Volta+ through PerfWorks (`sm__warps_active...`,
   `dram__bytes_read`). Different namespaces, so the event is resolved against what this install
   ENUMERATES rather than built from a template -- which is why a metric can be absent here and
   present on the next box.
3. **One event set counts ONE device through ONE context.** A second GPU needs a second event set,
   and work on another device or in another context is simply not counted -- which looks exactly
   like a kernel that did nothing.

A metric this machine cannot express comes back with a REASON, never as a zero. On a GPU, a missing
number and a zero counter are the two things a reader most reliably confuses, and only one of them
is a finding.

## The permission gate -- what an empty profile actually means

NVIDIA's driver can be configured to serve profiling to root only. When it is, CUPTI-based tools
refuse with **`ERR_NVGPUCTRPERM`** -- a message about administrators, from a library you never
named -- and PAPI's `cuda` component answers `PAPI_EMISC` at `PAPI_start`. What you SEE is an empty
profile, which is exactly what a fast kernel looks like.

The gate is on COUNTERS, so it does not fail everything equally: plain CUDA activity tracing (this
skill's four reports) usually survives it, while `ncu`, PAPI's `cuda` component and
`nsys --gpu-metrics-device` do not. A run that gives you kernel durations but refuses every counter
is this gate, not a broken toolkit.

The harness classifies this as `insufficient_permissions` rather than as a failed trace. Recognise
it yourself by:

- `ERR_NVGPUCTRPERM` anywhere in stderr;
- `nsys`/`ncu` complaining about `CAP_SYS_ADMIN` or administrator privileges;
- a recording that exists but whose kernel summary is empty on a submission you know launches.

The fix is one of:

```sh
grep -E 'RestrictProfilingToAdminUsers|RmProfilingAdminOnly' /proc/driver/nvidia/params
# then, as root:
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > /etc/modprobe.d/nvidia-profiling.conf
# reload the module or reboot; in a container, add --cap-add=CAP_SYS_ADMIN
```

**Grep for both spellings.** The module option is `NVreg_RestrictProfilingToAdminUsers` and older
drivers echo it back, but the open kernel module publishes the INTERNAL name
`RmProfilingAdminOnly` instead. Matching only the documented one reports "no gate" on a gated box --
measured here on driver 595.84, which publishes `RmProfilingAdminOnly: 1` while every CUPTI tool on
it refuses.

## When the profiler says nothing

A profiler that reports nothing must never read as a fast kernel. The harness refuses with a named
cause instead of an empty profile, and each one has a different fix:

| cause | what it is | fix |
| --- | --- | --- |
| `nsys_missing` | Nsight Systems not on PATH | `nsight-systems-cli` from NVIDIA's CUDA repo; `nvidia-cuda-toolkit` lacks it |
| `no_gpu` | `/dev/nvidiactl` absent: no GPU here | `--gpus all` (docker), `--device nvidia.com/gpu=all` (podman), `--nv` |
| `insufficient_permissions` | the gate above | `NVreg_RestrictProfilingToAdminUsers=0`, or `--cap-add=CAP_SYS_ADMIN` |
| `nsys_failed` | no recording, another reason | read its stderr, which the error carries verbatim |
| `nsys_report_missing` | a recording, four empty reports | nsys older than 2022.1: upgrade, or use the old spellings |
| `no_kernels` | 0 GPU kernels traced | it ran on the host, or the launch failed: check its error code |
| `counters_unsupported` | `counters: true` on a GPU submission | use `ncu`, or the PAPI GPU path above |
| `rocprof_unsupported` | a `hip` submission | nothing to fix: `nsys` cannot see an AMD queue, `rocprofv3` answers |

If you run `nsys` by hand, apply the same rule: an empty `cuda_gpu_kern_sum` is a finding about your
environment, not about your kernel.

## Traps

- **The trace covers warmup reps too.** `device_ns_per_rep` divides by `reps + warmup` for exactly
  that reason. Any number you divide yourself must use the same denominator.
- **Kernel names arrive demangled and long.** A C++ template kernel comes back as its full
  signature; the rendered text truncates at 44 characters, the JSON does not. Match on the JSON.
- **Tracing is not free**, only cheap. Compare a traced run against a traced run; take speedups from
  the graded measurement.
- **`nsys` traces the whole child process tree.** A submission that spawns workers gets all of their
  device activity in one summary, which is what you want for totals and not what you want when
  attributing a kernel to a rank.
- **The judge has no `ncu` route.** Everything past "which kernel" you run yourself, on your own
  build, with the kernel name this profile gave you.
