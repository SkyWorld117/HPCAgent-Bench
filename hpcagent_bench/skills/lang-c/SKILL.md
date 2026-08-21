---
name: lang-c
description: "C23 here: the ABI, the mistakes that cost most of the recorded turns, and writing fast C."
---

# lang-c

Score = MULTI-CORE speedup vs a SERIAL same-toolchain C baseline. Threading: the openmp-c page.

## Harness facts

- Flags (fixed, also shown in the main prompt): `gcc -std=c23 -O3 -march=native -fopenmp` plus the
  strict-FP set. `-ffast-math` is never on -- reassociation is yours to authorize per loop
  (`reduction`) and must stay inside tolerance. LLVM 22 (`clang`) via the submission's `compiler`
  field.
- The ABI already spells restrict: `void k(const double *restrict a, double *restrict out,
  int64_t n)`. Symbols are `int64_t`. `workspace` may be NULL and `workspace_size` 0 unless you
  asked via `workspace_bytes` -- check both. It is the only 256B-aligned buffer you get.
- C23 is parsed: `constexpr`, `typeof`, `nullptr`, bare `bool` -- and compile-time extents arrive
  declared at the top of your stub as `constexpr int64_t`.

## The expensive mistakes

1. **Dropping the stub's include block.** The file opens with `<stdint.h> ... <omp.h>` and the
   signature is spelled in `int64_t`. Pasting back only the function loses the headers and fails
   on the signature itself -- the LARGEST build failure on record (168 of 185 across two C arms).
   **Edit in place. Never replace the whole file.**
2. **Claiming alignment on an ABI pointer.** `__builtin_assume_aligned` or an OpenMP
   `aligned(p:...)` clause on a judge input pointer is UB and SIGSEGVs at vector width -- the #1
   crash on record. Inputs carry NATURAL alignment only; the workspace and your own
   `aligned_alloc` storage are fair game.
3. **Rewriting a loop must not change WHICH elements it writes.** A hand-unrolled
   `for (i = 0; i < n - 3; i += 4)` body stops at the last whole group on purpose; rerolling it to
   `i < n` writes elements the reference does not. Sizes are fuzzed, so `n % 4 != 0` is the normal
   case.

## Writing fast C

- **restrict is part of the type**: a local or helper pointer declared without it drops the ABI's
  non-aliasing promise. One pointer, one object, whole loop; no type punning.
- **Scalars over length-1 arrays**: accumulate in a scalar, store once.
- **`int64_t` for every induction variable and subscript**; no `int`/`size_t` mixing.
- **Row-major**: innermost loop runs over the LAST index. Prefer SoA over AoS.
- **Plain countable loop shape**: one induction variable, affine subscripts, trip count known on
  entry, no `break`/`return`/`goto` out of the body.
- **`x * x`, not `pow(x, 2.0)`**; `sqrt`/`fabs`/`fmin`/`fmax` are single instructions here.
- `const` on read-only data and invariant locals.

## Workflow

- Compile locally with the judge's own build line (main prompt) and READ the errors and warnings;
  iterate until clean before spending a judge call. `syntax_check` is the free in-turn variant.
- Iterate with `score`; `submit` every correct improvement.
- Profiling is its own tool (`profile`, see its page): reach for it when a correct version stops
  improving, not before.
- Your context is finite and the kernel is under 100 lines: do NOT re-read the file after an edit
  that reported success.
