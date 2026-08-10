---
name: lang-cpp
description: "Writing correct C++23 for this harness: static_cast over silent conversion, and the six gates that check it."
---

# lang-cpp

Two jobs: (A) quality-check a C++ file through six gates; (B) write correct C++23 for
this harness. `<file>.cpp` is the placeholder for the target file.

## Workflow

Run `syntax_check` (free, instant, `-fsyntax-only -fopenmp -Wall`, same turn) on every
source file before `score` or `submit` -- a grade that dies on a compile error burns a
full judge round-trip for less information than syntax_check already gave you. Iterate
with `score`; `submit` finalizes one build against both seeds. Submit a working,
already-scored version well before the wall-clock limit -- an unsubmitted improvement
scores zero.

## Harness facts

- `-fopenmp` is always on in the judge build; it is never something you add or remove.
  Under single-core grading `OMP_NUM_THREADS=1` is pinned, so an OpenMP loop must stay
  correct but shows no speedup until multi-core mode runs it.
- `<execution>` policies (`std::execution::par`, `par_unseq`) on `std::transform` /
  `std::for_each` / `std::reduce` and friends dispatch into oneTBB; the judge links
  `-ltbb` automatically, nothing to declare yourself. OpenMP and stdpar are both real
  parallel paths here -- use whichever fits the loop.
- `-ffast-math` is never on. Do not depend on it for correctness or speed.
- You are scored against a SERIAL same-toolchain baseline, not an arbitrary reference.

## Self-written vs generated

Full six gates + Section B apply to code you write by hand. Code emitted by a tool you
don't hand-edit gets the narrow clang-tidy set only (`clang-analyzer-*`, everything else
off -- style/bugprone are near-100% false positives on emitted code) plus the sanitizer
runs; do not "modernize" generated output, fix its generator instead.

## A. The six gates (run in this order)

1. **clang-format** -- project `.clang-format` if present, else
   `--style='{BasedOnStyle: LLVM, Standard: c++23, ColumnLimit: 120}'`.
2. **clang-tidy**, hand-written:
   `--checks='-*,bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,portability-*,readability-*,clang-analyzer-*' --header-filter='.*' --warnings-as-errors='*' <file>.cpp -- -std=c++23 -Wall -Wextra -Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wold-style-cast`.
   Generated code: `--checks='-*,clang-analyzer-*' --header-filter='$^'`.
3. **cppcheck**:
   `--enable=warning,performance,portability,style --std=c++23 --language=c++ --inline-suppr --error-exitcode=1 --quiet --suppress=preprocessorErrorDirective --suppress=missingIncludeSystem --suppress='*:*/external/*' <file>.cpp`.
4. **g++ `-fanalyzer`** (syntax-only, no build): same warning flags as gate 2 plus
   `-fsyntax-only -fanalyzer`. Every `-Wanalyzer-*` hit is a real defect.
5. **ASan, build and RUN**:
   `g++ -std=c++23 -fsanitize=address -fno-omit-frame-pointer -g -O1 <file>.cpp -o /tmp/cppq_asan && ASAN_OPTIONS=detect_leaks=1 /tmp/cppq_asan`.
6. **UBSan, build and RUN**:
   `g++ -std=c++23 -fsanitize=undefined -fno-omit-frame-pointer -g -O1 <file>.cpp -o /tmp/cppq_ubsan && UBSAN_OPTIONS=halt_on_error=1 /tmp/cppq_ubsan`.

Warnings are errors on all six; fix at the source. "Clean" = zero output on all six.

## B. Writing modern C++23

- **No implicit conversions -- make every cast explicit.** `static_cast<T>` (never a
  C-style or functional cast, never `const_cast`/`reinterpret_cast` unless truly
  unavoidable), brace-init (`T x{expr};`) so a narrowing conversion is a compile error.
  `-Wconversion -Wsign-conversion -Wfloat-conversion -Wdouble-promotion -Wold-style-cast`
  fail the build on anything implicit.
- **Concepts** over SFINAE/`enable_if`; **`if constexpr`** over tag dispatch;
  **`constexpr`/`consteval`** plus `static_assert` to lock in compile-time invariants.
- **No macros** -- `constexpr` values instead of `#define` constants,
  `constexpr`/`consteval`/`inline` functions instead of function-like macros.
- **`std::ranges`**, `std::span`, `std::string_view`, `std::format`, `std::expected`.
- **Value semantics + RAII.** Raw pointers are fine for non-owning references and
  perf-sensitive interfaces -- don't force `unique_ptr`/`shared_ptr` where a raw
  pointer/reference is clearer. No manual `new`/`delete` leaks.
- `auto`, range-`for`, `enum class`, `[[nodiscard]]`, `noexcept` where it holds.

After writing or modernizing, run the six gates in section A on the result.
