---
name: loop-transformations-fortran
description: "Reshaping a Fortran loop nest so it can be threaded: permutation, distribution and wavefront skewing, each with its legality test."
---

# loop-transformations-fortran

A nest that resists `parallel do` usually needs its SHAPE changed first. Three rewrites cover
almost all of it. Each has a mechanical legality test -- run it, do not guess. Fortran is
COLUMN-major: the FIRST subscript must run innermost.

## Dependence vectors

For two iterations touching the same element, write (later index - earlier index), one component
per loop, outermost first. `a(i,j) = a(i,j-1) + a(i-1,j)` carries `(0,1)` and `(1,0)`. A vector
is POSITIVE when its first non-zero component is positive; the original nest always is, and a
rewrite is legal exactly when every vector still is afterwards. A loop is PARALLEL when no vector
has a non-zero at its position with all outer components zero.

## Permutation -- swap two loops

**Legal** when permuting every vector's components the same way leaves them all positive. For a
2-deep nest: illegal exactly when some dependence is `(+,-)`.

**Pays** when it puts the unit-stride axis innermost, or moves a parallel axis outward. When one
axis carries the dependence and the other is free AND unit-stride, you get both:

```fortran
do j = 2, n                            ! carries the dependence: serial
  !$omp parallel do simd               ! free and unit stride
  do i = 1, n
    aa(i, j) = aa(i, j - 1) + bb(i, j)
  end do
end do
```

## Distribution -- split one loop into several

**Legal** when statements on a dependence CYCLE stay together; the resulting loops run in
topological order of the statement graph.

**Pays** on a body mixing a recurrence with independent work -- fused, the whole loop is serial:

```fortran
do i = 2, n                            ! chain stays serial
  s(i) = s(i - 1) + x(i)
end do
!$omp parallel do simd
do i = 2, n
  y(i) = 2.0d0 * x(i)
end do
```

Costs an extra pass, so a memory-bound body can come out slower. Fusion is the inverse: legal
when no dependence between the bodies is reversed, pays when the second re-reads the first.

## Wavefront -- when every loop carries a dependence

Iterations on an anti-diagonal are independent: with `(0,1)` and `(1,0)`, both advance `i+j` by
one, so no two points sharing `i+j` can depend on each other. Skewing is always legal -- it
renumbers without reordering -- and exists to make the following interchange legal.

```fortran
do t = 3, n + m                        ! serial across diagonals
  !$omp parallel do                    ! parallel within one
  do i = max(2, t - m), min(n, t - 2)
    a(i, t - i) = a(i - 1, t - i) + a(i, t - i - 1)
  end do
end do
```

Bounds come from keeping `j = t - i` inside `2..m` while `i` stays in `2..n`. A diagonal walks
with a stride, so this buys parallelism with locality: it wins only on long diagonals with real
work per point, and re-forks per diagonal. Skew over TILES rather than points to restore unit
stride inside a block and cut synchronisations to the number of block diagonals.

Cheaper exits first: a dependence in one dimension only needs permutation, and a body mixing a
chain with independent statements needs distribution -- what is left may already be parallel.

`do concurrent` asserts independence too, so every legality test above applies to it unchanged.
