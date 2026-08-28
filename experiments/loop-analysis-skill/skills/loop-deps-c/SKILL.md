---
name: loop-deps-c
description: Read this BEFORE adding omp parallel/simd to any C loop -- spotting loop-carried dependences by reading the code, and the legal remedy for each pattern.
---

A loop is parallel only when no iteration touches data another iteration touches, except
reads of the same location. Decide by READING the loop, not by trying pragmas until the
grade fails: find every array the loop WRITES, and check whether any subscript of that
array read or written elsewhere in the body can equal it for a different iteration.

Three dependence kinds. Flow (write in iteration i, read in i+k) is the hard one. Anti
(read then later write) and output (two writes) disappear by privatizing or renaming the
storage; flow does not.

**Pattern 1 -- recurrence: the written element feeds the next iteration.**

    for (int i = 1; i < n; i++)
        x[i] = a[i] * x[i-1] + b[i];

`x[i]` needs `x[i-1]` from the PREVIOUS iteration: threading this computes with stale
values and the answer is wrong, not slow. No pragma fixes it. Legal moves: keep it serial
and optimize the body, or recognize a prefix-sum/scan shape and rewrite as a two-pass
scan. If you cannot name the transformation, do not thread the loop.

**Pattern 2 -- carried scalar: an accumulator or running state crosses iterations.**

    double s = 0.0; int best = -1;
    for (int i = 0; i < n; i++) {
        s += w[i] * v[i];
        if (v[i] > vmax) { vmax = v[i]; best = i; }
    }

`s` is a reduction: legal ONLY as `reduction(+:s)` -- a shared accumulator without the
clause races, and the clause reorders the sum, so check the tolerance still holds.
`vmax/best` is a max-with-index reduction: needs a user-declared reduction or a per-thread
best merged after the loop. A plain scalar that just carries state (e.g. `prev = cur;`)
makes the loop a recurrence -- pattern 1.

**Pattern 3 -- indirect or overlapping writes: the subscript is not the loop index.**

    for (int i = 0; i < m; i++)
        hist[bin[i]] += 1.0;             /* two i can share bin[i] */

Duplicate `bin[i]` values mean two iterations write the same element: a race. Remedies,
fastest first: per-thread copies of `hist` merged afterwards; `#pragma omp atomic` on the
update (correct but often slower than serial); keep it serial. The same trap hides in
`a[i] = a[i+k]` in-place shifts and stencils updating the array they read -- when the
read range and write range overlap across iterations, use a separate output array.

**When only ONE statement carries the dependence: split the loop (fission).** Most real
kernels mix independent statements with one recurrent one, e.g.

    for (int i = 1; i < n; i++) {
        a[i] += c[i] * d[i];             /* independent -- vectorizes */
        b[i] = b[i-1] + a[i] + d[i];     /* recurrence  -- serial     */
    }

Split into two loops: the first parallelizes/vectorizes, the second stays serial (or
becomes a scan). Statement ORDER decides the easy cases too: `a[i] = a[i+1]` reads the
OLD neighbor and is safe as written; `a[i+1] = a[i]` propagates one value and is not --
when a swap of two statements or a saved copy of the old values (node splitting) breaks
the cycle, that beats any pragma.

Checklist before any `parallel for` / `simd`: every written array indexed only by the
loop index (or provably unique subscripts)? every carried scalar either private or a
declared reduction? reads and writes of the same array non-overlapping across
iterations? pointers that could alias marked `restrict` or proven distinct? If any
answer is no, name the pattern above and apply its remedy -- or leave the loop serial
and win elsewhere.
