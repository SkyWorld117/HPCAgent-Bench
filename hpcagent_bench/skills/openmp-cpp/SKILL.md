---
name: openmp-cpp
description: "OpenMP in C++: the four loop bins, the sharing clauses, and the build errors that cost a turn."
---

# openmp-cpp

Grading is MULTI-CORE, baseline SERIAL, `-fopenmp` always on. Never hardcode a thread count --
the grading machine presets `OMP_NUM_THREADS`, so read `omp_get_max_threads()` (needs
`#include <omp.h>`). Classify the loop first, then thread it: misfiling returns
`correct: false` and costs a round trip.
(`<execution>` policies are the other threading spelling -- the lang-cpp page; one spelling per
loop.)

## Before every directive: three lines, written down

A pragma is a claim about dependences. Write the claim as a comment ABOVE the directive, every
time, using the loop you are about to thread:

```cpp
// carried by: j      -- axis whose index appears at -1/+1 in a read of what this loop writes
// unit stride: i     -- axis that is the LAST term of the subscript (C++ is row-major)
// threading: i       -- must differ from "carried by", and should be the loop outside "unit stride"
#pragma omp parallel for simd
```

Two rules decide the outcome, and both are mechanical:

- **threading == carried by** is a RACE. Not slow -- wrong. Thread another axis, fission the
  statement out, or leave it serial.
- **unit stride is not the innermost loop** means ~1.00x however many cores you use. Interchange
  first (`loop-transformations-cpp` has the legality test), then thread.

`a[i*N + j] = a[i*N + (j-1)]` is carried by `j`, unit stride `j`: the only free axis is `i`.
`a[j*N + i] = a[(j-1)*N + i]` is carried by `j`, unit stride `i`: thread `i`, and put it innermost.

**A directive on a recurrence, with the comment saying so.** A pragma is an assertion,
not a request: `simd` on a carried dependence claims lanes are independent while the line above
proves they are not:

```cpp
// the recurrence relation requires sequential processing   <- correct diagnosis
#pragma omp simd                                            <- contradicts it
for (std::int64_t i = 2; i < n; i++)
    a[i] = a[i-2] + x[i];
```

That stride-2 chain is two independent chains (even `i`, odd `i`): split the index space along an
axis the dependence does not cross, or use the scan form below. If the comment you are about to
write names a dependence, the directive is the wrong one.

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`parallel for simd` on the OUTERMOST such loop: threads across cores, lanes within each.

```cpp
#pragma omp parallel for simd
for (int64_t i = 0; i < n; i++)
    y[i] = a[i] * x[i] + b[i];
```

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(op:acc)`. Never a shared scalar, never hand-built per-thread arrays. The clause
also authorizes the FP reassociation the compiler refuses on its own.

```cpp
double s = 0.0;
#pragma omp parallel for simd reduction(+:s)
for (int64_t i = 0; i < n; i++)
    s += w[i] * v[i];
```

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript: `x[i-1]`, in-place
stencil, wavefront. Threading it is WRONG, not slow. Fission the independent statements into their
own threaded loop and keep the chain serial; or thread a dimension the chain does not cross; or
give a stencil a separate output. Prefix sum is the one recurrence with a parallel spelling:
`inclusive_scan` / `exclusive_scan` (lang-cpp page), or the directive form:

```cpp
double s = 0.0;
#pragma omp parallel for simd reduction(inscan,+:s)
for (int64_t i = 0; i < n; i++) {
    s += a[i];
    #pragma omp scan inclusive(s)
    out[i] = s;
}
```

`exclusive(s)` is the value-before-this-iteration variant. Scans reassociate; tolerance applies.

**SCATTER** -- writes through an index array, `a[idx[i]]`. If the task guarantees distinct indices
it is PARALLEL, no atomics. Only DUPLICATE indices collide: then per-thread copies merged after the
loop (usually fastest), or `omp atomic` on the update (often slower than serial).

```cpp
#pragma omp parallel for
for (int64_t i = 0; i < m; i++) {
    #pragma omp atomic
    hist[bin[i]] += 1.0;      // correct; per-thread copies usually faster
}
```

## Clauses

| clause | means |
|---|---|
| `shared(x)` | one object, all threads. Right for input/output arrays. Two threads writing one shared scalar is a RACE. |
| `private(x)` | own UNINITIALIZED copy per thread. Every scratch scalar the body writes. |
| `firstprivate(x)` | private, initialized from the value on entry. |
| `lastprivate(x)` | private, sequentially-last value copied back out. |
| `reduction(op:x)` | per-thread copy at `op`'s identity, combined at the end. What a sum/max/count wants. |

Getting sharing wrong is a RACE, not a build error: it compiles, runs, and returns a different
answer under load. Induction variables are already private.

## Worth one line each

- `collapse(n)` when one loop is too short to fill cores -- exactly n PERFECTLY nested loops,
  nothing between the headers.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `#pragma omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel for` on the outer loop, `simd`
  alone on the unit-stride inner one.

## Build errors that cost a turn

- **`aligned(p:32|64)` or `assume_aligned` on an ABI input pointer is UB and SIGSEGVs at vector
  width**. ABI pointers carry natural alignment only; storage you
  allocate yourself and the 256B `workspace` are fair game.
- **Skip `default(none)`.** The one variable you miss is always the accumulator -- which belongs in
  `reduction(...)` anyway.
- **Nothing between the directive and its loop, and the loop must be canonical.** One induction
  variable (`int64_t`, like every subscript), initialized IN the header, bound known at entry;
  `for (; i >= 0; i -= 4)` is a build error.
- **`simd` is part of the directive NAME**: `parallel for simd schedule(static)`, never
  `parallel for schedule(static) simd`.
- **No `break` / `return` / `throw` out of a threaded loop.** A search loop keeps its trip count
  and reduces instead: `reduction(min:first)` over a per-iteration candidate.
- `nowait` does not exist on a combined `parallel for`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
