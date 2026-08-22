---
name: openmp-fortran
description: "OpenMP in Fortran: the four loop bins, the sharing clauses, and the build errors that cost a turn."
---

# openmp-fortran

Grading is MULTI-CORE: the timed run owns 24 physical cores (no SMT), `OMP_NUM_THREADS` preset,
baseline SERIAL. `-fopenmp` is always on. Classify the loop first, then thread it. Misfiling returns `correct: false` and costs a round
trip. (`do concurrent` is the other threading spelling -- the lang-fortran page; one spelling per
loop, `!$omp simd` cannot sit on a `do concurrent`.)

Threading is the LAST step, not the first. Cores add arithmetic; they do not add bandwidth. A
`parallel do` whose inner loop strides by a column is memory-bound in every lane, so more threads
buy nothing -- get the unit-stride axis innermost, THEN thread the loop outside it. Fortran is
COLUMN-major, so the FIRST subscript is the one that must run innermost. Both of the following
compile, run, and return `correct: true`, and both waste the speedup available.

**Wrong 1 -- threaded the outer loop and left the inner one striding.** The dependence runs along
`j`, so `i` is the free axis AND the unit-stride one. Threading `i` from the outside puts the
strided walk on the inside, and every lane misses cache:

```fortran
!$omp parallel do                        ! correct, and pointless
do i = 1, n
  do j = 2, n                            ! strides by a column
    aa(i, j) = aa(i, j - 1) + bb(i, j)
  end do
end do
```

Interchange so the unit-stride axis is innermost, THEN thread the loop that carries no dependence
-- here the inner one, because the outer one does carry it:

```fortran
do j = 2, n                              ! carries the dependence: stays serial
  !$omp parallel do simd                 ! free AND unit stride
  do i = 1, n
    aa(i, j) = aa(i, j - 1) + bb(i, j)
  end do
end do
```

**Wrong 2 -- put a directive on a recurrence and said so in the comment.** A directive is an
assertion, not a request. `simd` on a carried dependence claims lanes are independent when the
line above proves they are not:

```fortran
! the recurrence relation requires sequential processing   <- correct diagnosis
!$omp simd                                                 <- contradicts it
do i = 3, n
  a(i) = a(i - 2) + x(i)
end do
```

That stride-2 chain is really two independent chains (even `i`, odd `i`). Split the index space
along the axis the dependence does NOT cross, or use the scan form below. If the comment you are
about to write names a dependence, the directive you are about to write is the wrong one.

## The four bins

**PARALLEL** -- every write lands at this iteration's own subscript, no scalar carries state.
`parallel do simd` on the OUTERMOST such loop: threads across cores, lanes within each.

```fortran
!$omp parallel do simd
do i = 1, n
  a(i) = c(i) * x(i) + d(i)
end do
```

**REDUCTION** -- the only carried state is an accumulator (sum, max, min, count). Same directive
plus `reduction(op:s)`. Never a shared scalar, never hand-built per-thread arrays. The clause also
authorizes the FP reassociation the compiler refuses on its own.

```fortran
s = 0.0d0
!$omp parallel do simd reduction(+:s)
do i = 1, n
  s = s + c(i) * d(i)
end do
```

**RECURRENCE** -- the written array is read at ANOTHER iteration's subscript: `x(i-1)`, in-place
stencil, wavefront. Threading it is WRONG, not slow. Fission the independent statements into their
own threaded loop and keep the chain serial; or thread a dimension the chain does not cross; or
give a stencil a separate output.

```fortran
do i = 2, n
  a(i) = a(i) + c(i) * d(i)   ! independent -- fission out, thread it
  b(i) = b(i-1) + a(i)        ! chain -- stays serial
end do
```

Prefix sum is the one recurrence with a directive; statements before the `scan` feed the sum,
statements after read the scanned value:

```fortran
s = 0.0d0
!$omp parallel do simd reduction(inscan, +:s)
do i = 1, n
  s = s + c(i)
  !$omp scan inclusive(s)
  x(i) = s
end do
```

`exclusive(s)` is the value-before-this-iteration variant. Scans reassociate, so tolerance
still applies.

**SCATTER** -- writes through an index array, `a(idx(i))`. If the task guarantees distinct indices
it is PARALLEL, no atomics. Only DUPLICATE indices collide: then per-thread copies merged after the
loop (usually fastest), or `!$omp atomic` on the update (often slower than serial).

```fortran
subroutine hist(a, bin, n) bind(C)
  use iso_c_binding
  integer(c_int64_t), value, intent(in) :: n
  real(c_double), intent(inout) :: a(n)
  integer(c_int64_t), intent(in) :: bin(n)
  integer(c_int64_t) :: i
  !$omp parallel do
  do i = 1, n
    !$omp atomic
    a(bin(i)) = a(bin(i)) + 1.0d0
  end do
end subroutine hist
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
answer under load. Loop indices are private already.

## Worth one line each

- `collapse(n)` when one loop is too short to fill cores -- exactly n PERFECTLY nested loops,
  nothing between the `do` statements.
- `declare simd` on a helper called from the hot loop, else the call is a vectorization barrier.
- `!$omp unroll partial(4)` on the INNER loop of a nest you already thread -- never `full`:
  it deletes the loop the worksharing directive above needs.
- Split the construct when the shape demands it: `parallel do` on the outer loop, `simd`
  alone on the unit-stride inner one.

## Build errors that cost a turn

- **End the loop with `end do` and write nothing after it.** The closing directive is optional and
  omitting it is always safe; if you write one it must name the SAME construct token for token:
  opening `!$omp parallel do simd` and closing `!$omp end parallel do` drops the `simd` and is a
  BUILD ERROR -- and gfortran blames the closing line.
- **`aligned(...)` is unavailable on ABI dummies** (*must be POINTER, ALLOCATABLE, Cray pointer or
  C_PTR*) -- rejected outright.
- **`!$omp workshare` does NOT thread on gcc** (gfortran lowers it to `single`). Rewrite array
  syntax as an explicit loop under `parallel do`.
- **Skip `default(none)`.** The one variable you miss is always the accumulator -- which belongs
  in `reduction(...)` anyway. Leaving it off removes a whole class of build failure.
- **No `exit` / `cycle` to an outer loop, no `return`, out of a threaded loop.** A search loop
  keeps its trip count and reduces instead: `reduction(min:first)` over a per-iteration candidate.
- `nowait` does not exist on a combined `parallel do`; `schedule` is worksharing-only (on a bare
  `simd` it is a build error).
