---
name: loop-deps-fortran
description: Read this BEFORE adding omp parallel/simd or DO CONCURRENT to any Fortran loop -- spotting loop-carried dependences by reading the code, and the legal remedy for each pattern.
---

A loop is parallel only when no iteration touches data another iteration touches, except
reads of the same location. Decide by READING the loop, not by trying directives until
the grade fails -- and remember `DO CONCURRENT` is a PROMISE you make to the compiler
that iterations are independent, not a request: on a dependence-carrying loop it
miscompiles silently instead of failing.

Three dependence kinds. Flow (write in iteration i, read in i+k) is the hard one. Anti
(read-then-write) and output (write-write) disappear by privatizing the storage; flow
does not.

**Pattern 1 -- first-order linear recurrence.**

    do i = 2, n
        x(i) = a(i) * x(i-1) + b(i)
    end do

`x(i)` needs the `x(i-1)` computed one iteration earlier. Neither `!$omp parallel do`
nor `do concurrent` is legal -- both compute with stale values and the answer is WRONG,
not slow. Legal moves: keep it serial and optimize the body, or rewrite a genuine
prefix-sum shape as a two-pass scan. Array syntax does not save you:
`x(2:n) = a(2:n)*x(1:n-1) + b(2:n)` uses the OLD x values by semantics -- which is a
DIFFERENT result from the loop above, so check which one the reference computes.

**Pattern 2 -- carried scalar state: running sums, extrema, wrap-around neighbors.**

    do j = 1, n
        t     = s(j) + q(j)
        r(j)  = t - tprev        ! tprev carries across iterations
        tprev = t
    end do

`tprev` makes iteration j depend on j-1: a recurrence in scalar form. Privatizing
`tprev` gives wrong answers; the fix is restructuring (here: `r(j) = s(j)+q(j) -
(s(j-1)+q(j-1))` reads only OLD data and parallelizes cleanly). A scalar that only
accumulates (`total = total + w(j)`) is a reduction: legal ONLY as
`reduction(+:total)` -- mind the reassociated sum against the tolerance, and note
`maxval`/`sum` intrinsics are the cleaner spelling where the whole loop IS the
reduction.

**Pattern 3 -- indirect writes through an index array.**

    do k = 1, m
        acc(node(k)) = acc(node(k)) + flux(k)   ! node(k) values repeat
    end do

Duplicate `node(k)` entries mean two iterations update the same element: a race under
any parallel directive. Remedies, fastest first: per-thread copies of `acc` merged
afterwards; `!$omp atomic update` on the accumulation (correct, often slower than
serial); reorder/color k so no two concurrent iterations share a node. The same trap in
disguise: dummy arguments that overlap at the call site -- Fortran assumes dummies do
not alias, so passing overlapping sections is already illegal; inside the kernel you may
assume distinct dummies are distinct.

**When only ONE statement carries the dependence: split the loop (fission).**

    do i = 2, n
        a(i) = a(i) + c(i)*d(i)      ! independent -- vectorizes after the split
        b(i) = b(i-1) + a(i) + d(i)  ! recurrence  -- stays serial
    end do

as two loops: the first takes `parallel do`/`simd`, the second stays serial (or becomes
a scan). Statement order decides the easy cases too: `a(i) = a(i+1)` reads the OLD
neighbor and is safe as written; `a(i+1) = a(i)` propagates one value and is not --
saving the old values to a temporary first (node splitting) often breaks the cycle.

Checklist before `parallel do`, `simd`, or `do concurrent`: every written array indexed
only by the loop index (or provably unique subscripts)? every carried scalar private, a
declared reduction, or eliminated by restructuring? no iteration reading an element
another one writes? array-syntax rewrites checked against the loop's OLD-vs-NEW value
semantics? If any answer is no, name the pattern above and apply its remedy -- or leave
the loop serial and win elsewhere.
