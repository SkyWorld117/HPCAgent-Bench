---
name: profiling
description: Find the bottleneck with perf and hardware counters before you edit anything.
---

Measure before you edit, and again after. A change you cannot measure is a change you cannot
defend. Three questions, in order -- each one narrows what the next has to look at.

| question | tool | what you get |
| --- | --- | --- |
| where does the time go? | `perf record` + `perf report` | a ranked call graph |
| what is the machine doing there? | PAPI counters (`/profile` `counters:true`), `perf stat` | instructions, misses, flops |
| why does *this loop* behave that way? | `objdump -d`, cachegrind, the compiler's vector report | the emitted code |

Never start at question 3. A perfectly analysed loop that owns 4% of the run is 4% of a win.

## Build for profiling

Add `-g`. Nothing else.

`-g` emits DWARF beside the code; it changes no instruction, so a profiled build times
identically to the scored one and its hotspots are the scored run's hotspots. Without it perf
names addresses, and an address-only profile is unreadable.

Do **not** add `-fno-omit-frame-pointer`. It costs a general-purpose register in every function
-- real slowdown on a register-hungry inner loop -- and buys nothing here, because a
frame-pointer unwind is only correct when *every* frame kept its frame pointer, which CPython
and the BLAS libraries do not. Unwind with `--call-graph=dwarf` instead: it reads `.eh_frame`,
works on untouched release builds, and costs only a bigger `perf.data`.

Keep the release optimization level. Profiling a `-O0` build tells you about a program nobody
runs.

## Read a call graph

```sh
perf record -g -e cycles:u -F 999 --call-graph=dwarf -- ./app input
perf report --stdio
```

`cycles:u` is user-space only -- kernel samples need a lower `perf_event_paranoid` and answer a
different question. `-F 999` rather than 1000 so the sampler cannot phase-lock onto a kernel
whose own period is a round number of milliseconds.

Two columns, two different findings:

- **self%** -- time in this frame's own instructions. Ranks *what to optimize*.
- **children%/total%** -- time in this frame and everything it called. Traces *who is
  responsible*.

A frame with high total% and near-zero self% is a caller, not a bottleneck; walk down. A leaf
with high self% inside `libopenblas` is not your kernel, it is a library call -- your decision is
about the call, not the loop inside it. Walk up from the hottest leaf until you reach the first
frame whose body *is* the algorithm rather than dispatching, packing or reducing. That frame is
what you optimize.

Watch for `[unknown]`. It is unattributed time, kept in the tree on purpose: a dropped frame
silently re-parents its callees and invents a call path that never happened.

Profile more than one thread count. The function whose **self% RISES with threads** is the serial
fraction; it caps the whole kernel no matter what you do to the parallel part. That is a
different finding from "the hottest function", and usually a more valuable one.

## Read counters

One run per metric, deliberately. A CPU has a handful of counter registers (5 on a Ryzen 8845HS,
4-8 typical); ask for more events at once and PAPI or perf will multiplex -- time-slice the
events and scale the partial counts back up. What comes back looks exactly like a count and is
an extrapolation. So `counters:true` costs one extra measured run per metric on top of the
thread sweep. Turn it on after the call graph has named the loop, not before.

Which events exist is a property of *this CPU*, discovered at run time and never assumed:
`PAPI_L1_DCM` is available on a Zen4 while `PAPI_L1_ICM`, `PAPI_L3_DCM` and `PAPI_L1_TCM` are
not. Read the `expression` field, not just the metric name -- the metric names the question,
`PAPI_L1_DCA - PAPI_L1_DCM` names the quantity that answered it. `count:null` with a `missing`
reason means this CPU cannot express that metric; it never means zero, and nothing else is
substituted under the name.

Raw counts are almost useless. Ratios are the whole point:

| ratio | compute | reading |
| --- | --- | --- |
| IPC | instructions / cycles | < 1 stalled; 2-4 healthy; near issue width = compute-bound |
| misses per 1k instructions | 1000 x L1 misses / instructions | < 10 cache-friendly; > 50 memory-bound |
| hit rate | hits / (hits + misses) | falls off a cliff when the working set crosses a cache level |
| flops per cycle | fp ops / cycles | against the machine's peak: 1/8th of peak is not compute-bound |
| ops per instruction | fp ops / fp instructions | ~1 is scalar code; 4-8 means it vectorized |

What the combinations mean:

- **Low IPC + high miss rate** -> memory-bound. Tile, fuse, change layout, fix access order. More
  arithmetic per byte is free here; more FLOPs are not the problem.
- **Low IPC + low miss rate** -> dependence stalls or branch misses, not memory. Look for a
  loop-carried dependence, a serial reduction, or an unpredictable branch in the inner loop.
- **High IPC + low fp-op count** -> the machine is busy doing something other than the math:
  index arithmetic, bounds checks, conversions, a scalar tail. A "compute-bound" kernel whose
  fp-op count is far below its instruction count is not compute-bound, it is overhead-bound.
- **fp ops unchanged after a transform that should have vectorized** -> it did not vectorize.
  Nothing about the transform mattered; go look at the emitted code.
- **Instruction-cache misses that matter at all** -> unrolled or inlined too far. Rare in
  numerical kernels; when it appears it is self-inflicted.

`fma_instructions` and `integer_instructions` count *instructions*, not operations -- one packed
AVX-512 FMA is one instruction and thirty-two operations. Do not multiply them by a vector width
you have not confirmed in the disassembly.

Counted runs are single-threaded, because PAPI counts the calling thread: a threaded count would
report the master thread's share under the whole kernel's name. Counts describe the WORK; the
scaling table describes the PARALLELISM. Do not read one for the other.

## Two rules that save the most time

1. **Compare like with like.** Same shapes, same thread count, same build flags, same host. Run
   every configuration more than once -- a single timing on a shared box is noise, and so is a
   single counter reading.
2. **Attribute the win.** One change at a time. If two transforms land together and the kernel
   got slower, you cannot tell which to revert -- and the profile of the pair does not decompose.

## Everything else

| question | tool |
| --- | --- |
| exact cache behaviour of one nest (slow, simulated, deterministic) | `valgrind --tool=cachegrind` |
| exact call counts and call paths | `valgrind --tool=callgrind`, `pprof` (gperftools) |
| achieved memory bandwidth | `likwid-perfctr`, PAPI |
| where are the allocations | `heaptrack` |
| did it actually vectorize | `objdump -d` on the symbol (look for `%zmm`/`%ymm`), or the compiler's vector report |
