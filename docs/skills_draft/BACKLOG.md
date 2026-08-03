# Skill backlog

## 1. static-analysis -- LLVM and GNU only, grounded in what they REALLY emit

Scope decision: **only the LLVM and GNU toolchains.** No PVS-Studio, no Infer, no commercial tool.
Two compilers, two analyzers, done. That matches `dace_fortran/codegen_check.py`'s rule that deep
analysis FOLLOWS THE COMPILER -- gcc builds get `-fanalyzer`, clang builds get the LLVM static
analyzer -- so the analysis always matches the toolchain that produced the binary.

A first version is being written now from `dace_fortran/codegen_check.py` (`CRITICAL_WARNINGS`,
`CLANG_TIDY_CHECKS`, `CPPCHECK_SUPPRESSIONS`) and `hpcagent_bench/languages.py`. That version is
grounded in what those two repos already decided. It is NOT yet grounded in what the tools emit for
THIS corpus.

**The measurement that has to follow, and it is the point of the task:**

- Take **40 kernels sampled across the corpus** -- not 40 easy ones. Stratify by track (foundation
  / hpc / ml) and by dwarf so the sample is not all stencils.
- Compile each at **-O3 with the analyzer and report flags on**, for BOTH gcc and clang.
- **Mine the diagnostics.** Rank by frequency: which warnings actually fire on real generated HPC
  code, which fire constantly and mean nothing here, which fire rarely and mean something every
  time.
- The output is the part a doc cannot give you: a frequency-ordered list of what an agent will
  really see, so the skill can say "this one you will see on every kernel and it is noise" and
  "this one is rare and it is always a bug". Right now the skill can only repeat the tool's own
  documentation, which does not rank anything.

Also search the official GNU and LLVM documentation for the diagnostic INVENTORY -- what each tool
can report at all -- so the mined list can be checked against it: a warning class that never fired
in 40 kernels but exists is worth one line; one that does not exist is worth none.

Feeds back into `dace_fortran`'s `CRITICAL_WARNINGS` if the mining turns up a UB-class warning that
list does not carry.

## 2. opt-reports -- extend the EXISTING skill, do not write a new one

`hpcagent_bench/skills/opt-reports/SKILL.md` already exists: 173 lines, shipped, and pinned by
about four assertions in `tests/test_skill_content.py`
(`test_the_opt_report_skill_quotes_every_report_flag_the_harness_can_pass`,
`..._names_the_compilers_with_no_report_channel`, `..._names_every_capture_kind_and_where_it_lands`,
`..._separates_a_legality_refusal_from_a_cost_model_one`). Its sections are: Get one (GCC / Clang /
the rest), What the harness captures on its own, Read one: a refusal is not one thing, Diagnostic ->
change, The limits.

So the ask -- "the report gives the LINE where vectorization failed and WHY" -- is partly there
already: it separates a LEGALITY refusal from a COST-MODEL one and quotes the compilers' own
wording verbatim, because a reader matches those strings against real stderr.

**What is missing, and what the task should actually deliver:**

- **The line number is the deliverable and the page does not lead with it.** An opt report's value
  is `file.c:LINE:COL: remark: ...` pointing at the exact loop. The page should open with "read the
  line number, go to that loop" and only then explain the taxonomy.
- **Worked examples from THIS corpus.** Same 40-kernel sweep as task 1: compile at -O3 with
  `-fopt-info-vec-missed` (GNU) and `-Rpass-missed=loop-vectorize` (LLVM), collect the real
  remarks, and rank them. The page currently teaches the categories; it should teach the five
  messages an agent will actually meet, in frequency order, each with the fix.
- **The two compilers disagree**, and the page should say where. GCC's `-fopt-info-vec-missed` and
  LLVM's `-Rpass-analysis=loop-vectorize` do not report the same misses on the same loop; a reader
  who checks only one concludes the other's finding does not exist.
- **What a SILENT loop means.** No remark is not "it vectorized" -- it can mean the loop was never
  considered. The page's "The limits" section should carry this at the top, not the bottom.

## 2b. The harness only captures VECTORIZATION reports -- that is the real gap

Measured in `hpcagent_bench/flags.py`:

```
GCC_OPT_REPORT   = "-fopt-info-vec-optimized -fopt-info-vec-missed"
CLANG_OPT_REPORT = "-Rpass=loop-vectorize|slp-vectorizer -Rpass-missed=loop-vectorize|slp-vectorizer ..."
```

Vectorization and nothing else. Every other optimization decision the compiler makes is invisible
to the harness and therefore to the skill. `test_skill_content.py` pins the skill's flag strings
AGAINST `flags.py`, so widening the skill means widening the flags in the same change.

The classes worth capturing, and why each matters for an HPC kernel:

| class | GNU | LLVM | why it decides something |
|---|---|---|---|
| loop vectorization | `-fopt-info-vec-*` | `-Rpass*=loop-vectorize` | already captured |
| SLP vectorization | (folded into vec) | `-Rpass*=slp-vectorizer` | already captured; straight-line code, not loops |
| **inlining** | `-fopt-info-inline-*` | `-Rpass*=inline` | a hot call left out of line is often the whole gap, and the report says WHY it was refused (cost, size, recursion) |
| **unrolling** | `-fopt-info-loop-*` | `-Rpass*=loop-unroll` | decides whether accumulators stay in registers |
| **LICM** | `-fopt-info-loop-*` | `-Rpass*=licm` | a load the compiler could NOT hoist usually means it could not prove non-aliasing -- the finding is really an aliasing finding |
| **loop distribute / idiom** | `-fopt-info-loop-*` | `-Rpass*=loop-distribute`, `loop-idiom` | tells you the compiler already did the fission you were about to hand-write |
| **IPA** (const-prop, cloning) | `-fopt-info-ipa-*` | `-Rpass*=ipsccp`, `-Rpass*=inline` | why a symbolic bound stayed symbolic |
| **OpenMP** | `-fopt-info-omp-*` | (limited) | whether a `parallel for` was actually parallelized |
| **register allocation** | -- | `-Rpass-missed=regalloc` | spills in the inner loop, which no other report names |

Two ways to widen, and the second is better:

- Enumerate more flags. Explicit, but the list grows and each compiler spells things differently.
- **`-fsave-optimization-record`** (clang, already mentioned once in the skill) emits EVERY remark
  as structured YAML with source line + column + pass name, and `-fopt-info-all` is the GNU
  equivalent to stderr. One flag, all passes, machine-readable -- which also makes the 40-kernel
  frequency mining trivial instead of a grep exercise per class.

Preferred plan: capture with `-fsave-optimization-record` / `-fopt-info-all`, mine the YAML for the
frequency table, and let the SKILL teach the handful of remark kinds that actually fire -- rather
than teaching a flag list that will drift from what the compilers emit.

Caveat to check during the sweep: `-fopt-info-all` and the full optimization record are VERBOSE.
Measure the volume on a real kernel before wiring either into every build; if it is large, capture
it only on the `/profile` path the way `DEBUG_SYMBOLS` is handled.

## Shared work between the two tasks

Both need the same thing built once: **a sweep that compiles N corpus kernels at -O3 with every
report and analyzer flag on, for gcc and clang, and collects the diagnostics into a frequency
table.** Build it once, mine it twice. It is also reusable as a regression check -- a new warning
class appearing in the corpus is a signal on its own.

Cost note: this is a compile sweep over 40 kernels x 2 compilers, so it is a background job, not an
interactive one. `-j1` per the machine rules, and it should write its table to a file rather than
holding it in an agent's context.

## 3. Non-agentic frameworks must work in CONTAINER mode, not only native

DaCe, TVM, Triton, JAX and Pluto currently have containerless job-submission scripts. The native
path is worth keeping -- it lets a user measure an optimizer on their own machine without the
contention of several containers -- but it should be a CHOICE, not the only route.

Why it matters beyond symmetry: the agentic track already runs in containers, so today a DaCe
number and an agent number are produced under different toolchains, different library versions and
different CPU visibility. Comparing them is comparing two environments as much as two optimizers.
Containerising the non-agentic frameworks is what makes that comparison mean one thing.

Work: a container per framework family (or one image carrying all five), the same submission shape
the agentic track uses, and a test that the same kernel through the same framework gives the same
answer native and containerised -- otherwise the two modes silently diverge and nobody notices
which number a paper quoted.
