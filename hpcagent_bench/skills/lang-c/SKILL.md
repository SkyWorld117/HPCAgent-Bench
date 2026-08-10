---
name: lang-c
description: "Writing correct C17 for this harness: explicit casts, const/restrict, and the six gates that check it."
---

# lang-c

Two jobs: (A) quality-check a C file through six gates; (B) write correct C17 for this
harness. `<file>.c` is the placeholder for the target file. This harness compiles at
`-std=c17` (`languages.py::std_flag("c")` is the source of truth) -- not C23; C23-only
spellings (`constexpr`, `nullptr`, `typeof`, `_BitInt`, `auto`, `unreachable()`,
`[[...]]`) are a compile error here, not a nicer kernel.

## Workflow

Run `syntax_check` (free, instant, `-fsyntax-only -fopenmp -Wall`, same turn) on every
source file before `score` or `submit` -- a grade that dies on a compile error burns a
full judge round-trip for less information than syntax_check already gave you. Iterate
with `score`; `submit` finalizes one build against both seeds. Submit a working,
already-scored version well before the wall-clock limit -- an unsubmitted improvement
scores zero.

## Harness facts

- The judge always compiles with `-fopenmp`; it is never something you add or remove.
  Under single-core grading, `OMP_NUM_THREADS=1` is pinned, so a `#pragma omp` loop must
  stay correct but shows no speedup until multi-core mode actually schedules it.
- `-ffast-math` is never on. Do not depend on reassociation or reciprocal rewrites for
  correctness or speed.
- You are scored against a SERIAL same-toolchain baseline, not an arbitrary reference.

## A. The six gates (run in this order)

1. **clang-format** -- project `.clang-format` if present, else
   `--style='{BasedOnStyle: LLVM, ColumnLimit: 120}'`.
2. **clang-tidy**, hand-written C:
   `--checks='-*,bugprone-*,cert-*,clang-analyzer-*,performance-*,portability-*,readability-*' --header-filter='.*' --warnings-as-errors='*' <file>.c -- -std=c17 -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast`.
   Machine-generated code: narrow to `--checks='-*,clang-analyzer-*' --header-filter='$^'`
   -- style/bugprone checks are near-100% false positives on emitted code.
3. **cppcheck**:
   `--enable=warning,performance,portability,style --std=c17 --language=c --inline-suppr --error-exitcode=1 --quiet --suppress=preprocessorErrorDirective --suppress=missingIncludeSystem --suppress='*:*/external/*' <file>.c`.
4. **gcc `-fanalyzer`** (syntax-only, no build): same warning flags as gate 2 plus
   `-fsyntax-only -fanalyzer`. Every `-Wanalyzer-*` hit is a real defect.
5. **ASan, build and RUN**:
   `gcc -std=c17 -fsanitize=address -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_asan && ASAN_OPTIONS=detect_leaks=1 /tmp/cq_asan`.
6. **UBSan, build and RUN**:
   `gcc -std=c17 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 <file>.c -o /tmp/cq_ubsan && UBSAN_OPTIONS=halt_on_error=1 /tmp/cq_ubsan`.

Warnings are errors on all six. Fix at the source; the cppcheck suppressions above cover
third-party/system noise only, never your own bugs. "Clean" = zero output on all six.

## B. Writing C17

- **No silent conversions -- cast explicitly.** `-Wconversion -Wsign-conversion
  -Wfloat-conversion -Wdouble-promotion -Wbad-function-cast` fail the build on any
  implicit narrowing/signed/float conversion; fix with a `(type)` cast at the source.
- **`enum { CAP = 256 };`** for integer compile-time constants usable in array bounds /
  `case` labels; `static const double X = ...;` for non-integers (not a constant
  expression in C -- cannot size a file-scope array with it).
- **`_Static_assert`**, `<stdbool.h>` for `bool`/`true`/`false` (macros, not keywords --
  do not `#undef` them), `NULL` from `<stddef.h>` (there is no `nullptr` here).
- **`__attribute__((warn_unused_result))`** on must-check returns (`malloc`, parse/IO
  results) -- `[[nodiscard]]` is C23 syntax and does not compile at `-std=c17`.
- **`sizeof(*ptr)` in allocations**: `p = malloc(n * sizeof(*p));`, not the type name.
- **`const`/`restrict`** on non-written / non-aliasing pointer params in hot paths.
- Designated initializers with `= {0}`; `static inline` over function-like macros; check
  every `malloc`/`realloc`/`fopen`/`snprintf` return; no VLAs in public interfaces;
  declare at first use, `static` for anything not exported.

After writing or modernizing, run the six gates in section A on the result.
