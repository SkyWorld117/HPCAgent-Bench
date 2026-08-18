---
name: loop-deps-cpp
description: Read this BEFORE adding omp parallel/simd or par execution policies to any C++ loop -- spotting loop-carried dependences by reading the code, and the legal remedy for each pattern.
---

A loop is parallel only when no iteration touches data another iteration touches, except
reads of the same location. Decide by READING the loop, not by trying pragmas or
`std::execution::par` until the grade fails: find every container element the loop WRITES
and ask whether a different iteration can read or write that same element.

Three dependence kinds. Flow (write in iteration i, read in i+k) is the hard one. Anti
(read-then-write) and output (write-write) disappear by giving each iteration its own
storage; flow does not.

**Pattern 1 -- in-place stencil: the loop reads neighbors of the element it writes.**

    for (size_t i = 1; i + 1 < n; ++i)
        u[i] = 0.5 * (u[i-1] + u[i+1]);   // u[i-1] was ALREADY overwritten

Iteration i reads `u[i-1]` written by iteration i-1: the serial loop already computes a
sweep, and threading it computes something else entirely. Remedy: write into a separate
output buffer (ping-pong), or if the sweep order IS the algorithm (Gauss-Seidel), keep it
serial -- do not "fix" it into a different algorithm unless the tolerance proves it.

**Pattern 2 -- running state across iterations: exclusive scan / running extremum.**

    T run = 0;
    for (size_t i = 0; i < n; ++i) { out[i] = run; run += in[i]; }

`run` carries the whole prefix: this is an exclusive scan, not a reduction -- a
`reduction` clause gives the total but every `out[i]` is wrong. Remedy: a two-pass
block-scan (per-thread partial sums, then offset pass), or `std::exclusive_scan`; a plain
running maximum with its position is a max-location reduction (per-thread best, merged
after). A scalar that only accumulates a total is the easy case: `reduction(+:run)`,
minding the reassociated sum against the tolerance.

**Pattern 3 -- scatter through an index vector: subscripts that can collide.**

    for (size_t e = 0; e < edges.size(); ++e)
        force[edges[e].dst] += contrib(e);   // many e share one dst

Two iterations with the same `dst` race on `force[dst]`. Remedies, fastest first:
per-thread `std::vector<T>` accumulators merged afterwards; `#pragma omp atomic` on the
update (correct, often slower than serial); sort/color the edges so no bucket collides.
References are the C++-specific trap: `T &acc = out[f(i)];` hides an indirect write, and
two ranges passed as plain pointers can alias -- prove them distinct or mark them
`__restrict__` before vectorizing.

**When only ONE statement carries the dependence: split the loop (fission).** A loop
mixing an independent update with a recurrent one parallelizes after fission -- first
loop threads/vectorizes, second stays serial or becomes `std::exclusive_scan`. Statement
order decides the easy cases: `a[i] = a[i+1]` reads the OLD neighbor and is safe as
written; `a[i+1] = a[i]` propagates one value through the array and is not. Saving the
overwritten values into a temporary first (node splitting) often breaks the cycle
outright.

Checklist before any `parallel for`, `simd`, or `par` policy: every written element
indexed only by the loop index (or provably unique)? every carried scalar private, a
declared reduction, or restructured as a scan? no iteration reading an element another
iteration writes (in-place stencils!)? possibly-aliasing pointers/references proven
distinct? If any answer is no, name the pattern above and apply its remedy -- or leave
the loop serial and win elsewhere.
