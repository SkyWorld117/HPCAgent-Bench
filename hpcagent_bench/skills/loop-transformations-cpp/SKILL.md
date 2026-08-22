---
name: loop-transformations-cpp
description: "Reshaping a C++ loop nest so it can be threaded: permutation, distribution and wavefront skewing, each with its legality test."
---

# loop-transformations-cpp

A nest that resists `parallel for` usually needs its SHAPE changed first. Three rewrites cover
almost all of it. Each has a mechanical legality test -- run it, do not guess. C++ is row-major:
the LAST subscript must run innermost.

## Dependence vectors

For two iterations touching the same element, write (later index - earlier index), one component
per loop, outermost first. `a[i][j] = a[i-1][j] + a[i][j-1]` carries `(1,0)` and `(0,1)`. A vector
is POSITIVE when its first non-zero component is positive; the original nest always is, and a
rewrite is legal exactly when every vector still is afterwards. A loop is PARALLEL when no vector
has a non-zero at its position with all outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector's components the same way leaves them all positive. For a
2-deep nest: illegal exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or moves a parallel axis outward. When one
axis carries the dependence and the other is free AND unit-stride, you get both:

```c
for (std::int64_t j = 1; j < n; j++)       // carries the dependence: serial
    #pragma omp parallel for simd          // free and unit stride
    for (std::int64_t i = 0; i < n; i++)
        aa[j*n + i] = aa[(j-1)*n + i] + bb[j*n + i];
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; the resulting loops run in
topological order of the statement graph.

**Pays** on a body mixing a recurrence with independent work -- fused, the whole loop is serial:

```c
for (std::int64_t i = 1; i < n; i++) { s[i] = s[i-1] + x[i];  y[i] = 2.0 * x[i]; }
/* becomes */
for (std::int64_t i = 1; i < n; i++) s[i] = s[i-1] + x[i];    // chain stays serial
#pragma omp parallel for simd
for (std::int64_t i = 1; i < n; i++) y[i] = 2.0 * x[i];
```

Costs an extra pass, so a memory-bound body can come out slower. Fusion is the inverse: legal
when no dependence between the bodies is reversed, pays when the second re-reads the first.

## Wavefront -- when every loop carries a dependence

Iterations on an anti-diagonal are independent: with `(1,0)` and `(0,1)`, both advance `i+j` by
one, so no two points sharing `i+j` can depend on each other. Skewing is always legal -- it
renumbers without reordering -- and exists to make the following interchange legal.

```c
for (std::int64_t t = 2; t <= (n-1) + (m-1); t++) {          // serial across diagonals
    std::int64_t lo = t - (m-1) > 1 ? t - (m-1) : 1;
    std::int64_t hi = t - 1 < n-1 ? t - 1 : n-1;
    #pragma omp parallel for                            // parallel within one
    for (std::int64_t i = lo; i <= hi; i++)
        a[i*m + (t-i)] = a[(i-1)*m + (t-i)] + a[i*m + (t-i-1)];
}
```

Bounds come from keeping `j = t - i` inside `1..m-1` while `i` stays in `1..n-1`. A diagonal
walks with a stride, so this buys parallelism with locality: it wins only on long diagonals with
real work per point, and re-forks per diagonal. Skew over TILES rather than points to restore
unit stride inside a block and cut synchronisations to the number of block diagonals.

Cheaper exits first: a dependence in one dimension only needs permutation, and a body mixing a
chain with independent statements needs distribution -- what is left may already be parallel.
