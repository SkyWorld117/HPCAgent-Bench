# Merge landing 2026-08-05 -- what closed, and the two items that did not

Filed against `main` @ `29180570`, updated at session end @ `9b53130a`. This session merged the five
outstanding worktree branches, fixed the CI reds the merge exposed, landed one translator gap, and
worked the A-J backlog down to two residual items. Everything that RESOLVED is compressed to its root
cause below -- the verification prose is in the git history, not here.

Companion docs: [`BACKLOG_dace_frontend_and_gpu_20260805.md`](BACKLOG_dace_frontend_and_gpu_20260805.md)
(all open), [`BACKLOG_skill_page_review_20260803.md`](BACKLOG_skill_page_review_20260803.md) (all
open), [`BACKLOG_ablations_tagging_and_plots.md`](BACKLOG_ablations_tagging_and_plots.md) (all open).

## Shipped

| Commit | What |
|---|---|
| 5 merges | squeeze, mean, numeric-semantics (NEP 50 + fp32 accumulators), ABI-`dim` runtime-axis dispatch, isopar `par_unseq` |
| `ea59881c` | warnings ratchet reports the warning TEXT + toolchain, not just a count; `libtbb-dev` in all six native CI jobs |
| `79a8e6e8` | the emitter under test is built through `__init__`, not `__new__` |
| `29180570` | `_FullCallHoister`: a nested `np.full` is spilled so `np.triu` can lower |
| `5867cdd7` | PIC asserted as a property of the flags every shared-library build runs with |
| `fcdefffe` | `lang-cuda` / `lang-hip` skill pages; cuda and hip stop routing at `lang-cpp` |
| `559456fb` | A -- helper return buffers follow the source array's dtype |
| `0c7bc5a9` | B -- Phase 6 split into its own `integration` job |
| `0233838c` | PR #13 merged (18 tests) |
| `9b53130a` | the DaCe-frontend / GPU-lane findings doc |

`ad0dd5e9` was DROPPED as a strict duplicate of `aedb957c` -- proven by identical
`git patch-id --stable` (`84107e649d25`, `f9367d7c48a8`), not by reading the diffs. Merging both
would have applied the runtime-axis dispatch twice.

---

## Still open

### I. The null-workspace protocol is unexercised

The determinism half of I is done (`tests/test_determinism_gate.py`, 7 tests, runs on any CPU
runner). What is NOT done is everything else the disabled `gpu` job would cover, and the specific
protocol the cuda/hip pages warn about: a null workspace makes the second CUB/rocPRIM call re-query
the size and perform NO reduction, leaving the output as found -- which on fresh device memory reads
as a clean all-zero array with `cudaSuccess`.

⛔ The gate that would catch it is the ORACLE leg of scoring, not the determinism leg: an all-zero
result reproduces perfectly. Whether any corpus kernel's correct answer is all-zeros (or close enough
at the grading tolerance) is the open question, and it decides whether this is a coverage gap or a
real hole.

⛔ **Do not treat "the GPU pages are green" as evidence the GPU track works.** Both GPU lanes now pass
on a real RTX 4050 (see the frontend/GPU doc), which is evidence about the LANES, not about this.

### J. HPTT clone 403 -- mitigated, not eliminated

`containers/build-hptt.sh` fetches the pinned SHA `942538649b51ff14403a0c73a35d9825eab2d7de` and
retries 4 times with exponential backoff, failing with a message that names the dependency and prints
the `git ls-remote` command that settles throttle-vs-missing. Both container recipes call it.

⛔ **Retry raises the odds, it does not remove the dependency.** A sustained throttle still red-lines
the whole container track, because the step is early in the image and `test_container_launch.py`
never runs. Vendoring is the only option that cannot fail this way; it was not taken because it costs
a vendored-source update path.

### Run summaries do not distinguish a failure from a non-failure

Three of the four reds on 2026-08-04 were not failures: a step timeout, a preemption
(`exit code 143`, "runner has received a shutdown signal"), and a throttled clone. A reader cannot
tell them apart from the run page. Worth having the summary say **failed** vs **timed out** vs
**cancelled** vs **could not fetch a dependency**, so the count means what it appears to mean.

---

## Closed A-J, one line of root cause each

* **A. fp32 kernels emitted `double` helper-return buffers** -- `_ctor_dtype_tag` did not chase a
  `dtype=x.dtype` kwarg through the alias walk that already resolved the shape. Deeper defect:
  **every emitter's dtype table fell back to `double` on a miss**, so an unvalidated token silently
  picked a type. `ArrayDesc.__post_init__` now canonicalises on store and raises on an unknown token
  -- the contract `ScalarDesc` already honoured.
* **B. The `unit` job died on a 25-MINUTE STEP ceiling**, not the 1h30m job ceiling, and the
  `coverage` red was purely downstream. Two things had to move with the split: Phase 7's HF export
  (`if: success()` covered Phase 6 inside the old job, so leaving it in `unit` would have quietly
  narrowed the guarantee) and `coverage`'s `needs` list (nothing pins it to the set of uploading
  jobs, so a missing entry races and reports a plausible total from fewer jobs).
* **C. Five kernels refused to lower -- TWO causes, not one.** (1) Two vocabularies for one extent:
  a kernel names dimensions off a parameter (`batch, channels, h, w = x.shape`) while `init.shapes`
  keeps the symbol, so a string `==` declined `channels` against `embed_dim`. Replaced with
  `lib_nodes.dims_agree` -- literal, then alias-substituted to a fixpoint, then symbolic; cheapest
  rung first, and **unresolvable answers FALSE**, because a wrong True contracts over two different
  extents (a miscompile) while a wrong False only declines. (2) `relu_self_attention`'s
  `np.maximum(scores, 0.0) @ v` never reached the shape comparison at all -- a Call operand, now
  spilled to a temp. `KNOWN_NON_LOWERING` is empty: all 631 kernels lower.
* **D. `cpp_isopar` capability gate** -- libstdc++ picks its `<execution>` backend PER TRANSLATION
  UNIT via `__has_include(<tbb/tbb.h>)`, so `par_unseq` degrades to sequential with correct answers,
  no link error and no diagnostic. `languages.isopar_capability()` compiles one `par_unseq` call at
  the harness's real C++ flags and reads `nm` for a TBB runtime call. ⛔ `cpp_isopar` is NOT a scored
  column -- no `FRAMEWORK_LANG` entry, only correctness oracles consume it -- so the exposure is zero
  and the work keeps it zero via `preflight.AUTOPAR_PROBES`.
* **E. The warnings ratchet** -- never reproduced locally under gcc 12/14/15/16, clang++ 20/21/22 or
  gfortran; the count is a property of the compiler, not the sources. It counts 0 on CI (run
  `31001878513`, integration job `92296893031`: `76 passed, 7 skipped`, all 7 skips polycc-absent)
  and not vacuously, since the test asserts `total_builds >= 20`. Two rot-guards landed with it:
  `verify_toolchain.py` now checks `clang++` (the ratchet skips its ENTIRE count if any of the five
  drivers is missing), and CI moved to **LLVM 21 from apt.llvm.org** because ubuntu-latest ships 18.
  ⛔ The `/usr/local/bin` symlinks for unversioned `clang`/`clang++`/`flang` are load-bearing:
  `flags.polly_capability` probes bare `shutil.which("clang")`, so a distro clang would keep winning.
* **F. `_assert_ok` accepted a skip**, so an op-oracle test could pass with no backend having run it.
  Skips stay accepted; it now also asserts at least one of c/cpp/fortran reported `ok` and prints all
  three when none did.
* **G. Large-N fp32 reductions** now accumulate the innermost axis in blocks of 128 (numpy's own
  pairwise cutoff): measured error at n = 2**22 drops from 1.09e+02 to 4.00e+00. ⛔ The test passes
  `dtypes=` explicitly -- without it `run_op` emits float64, whose naive error is ~1e-11, and the test
  goes green whatever the accumulation does. Scope is the FULL reduction's innermost axis only, so an
  emitted nest still looks like a nest to the parallelism and isopar recognisers.
* **H. Objects built with `__new__` in tests** -- a hand-built stand-in silently re-asserts a private
  attribute list the test does not care about, and broke twice for that reason. Build through
  `__init__`; an empty kernel is enough for a scalar-only call.
* **Also fixed in passing:** `test_gate_is_a_no_op_for_ungated_frameworks` still listed `pluto` after
  it joined `AUTOPAR_GATED`, so it asserted the opposite of the tree and passed or failed by accident
  of the host's clang.

Two defects surfaced while building coverage rather than while hunting bugs:

* ⭐ **The determinism gate false-failed legitimate NaN.** `np.array_equal` reports `NaN != NaN`, so a
  bit-for-bit identical rerun of a kernel with a masked cell or a log of zero scored NONDETERMINISTIC
  -- unfixable by the agent, since run 2 was a copy of run 1. `equal_nan=True` now; reproducibility
  and validity are different questions, and `compare_arrays` was already NaN-aware, so the two legs
  had disagreed on what NaN means.
* ⭐ **`cumsum_exclusive`'s symbolic axis** (closed 08-04): `dim` reaches the ABI, so the manifest's
  value is not the one the harness passes. The kernel carries one loop nest per axis and picks at run
  time, each branch declaring and freeing its OWN scratch. ⛔ No trailing `else` -- making the last
  axis the fallthrough is the silent clamp.

## Method notes worth keeping

- **`git patch-id --stable` settles "are these the same work" across different SHAs and fork points.**
  It answered the `ad0dd5e9` / `aedb957c` duplicate question in one command, where diff-reading had
  produced a wrong answer earlier.
- **A verification inherited from an agent is not a verification.** Both agents this session branched
  from `a4796377`, two merges behind, so their measurements describe a translator that is not the one
  shipping.
- **Print the raw status dict, not the assertion result** (finding F).
- **A pipeline hides the exit code you care about**: `pytest ... | tail -40` returns `tail`'s status.
- **A 403 on an anonymous clone from a CI runner is throttling, not a missing repo** --
  `git ls-remote` settles it in one command, and it makes the red intermittent rather than permanent.
