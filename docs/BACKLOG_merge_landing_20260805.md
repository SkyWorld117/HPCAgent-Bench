# Merge landing 2026-08-05 -- what shipped, and what is still open

Filed against `main` @ `29180570`. This session merged the five outstanding worktree branches, fixed
the two CI reds that merge exposed, and landed one translator gap. Everything below the fold is
**open**, with the evidence that found it, so the next session starts from findings rather than from
a re-triage.

Companion docs: [`BACKLOG_ci_reds_20260804.md`](BACKLOG_ci_reds_20260804.md) (CI causes, one item
updated below), [`BACKLOG_skill_page_review_20260803.md`](BACKLOG_skill_page_review_20260803.md)
(skill pages).

## Shipped

| Commit | What |
|---|---|
| 5 merges | squeeze, mean, numeric-semantics (NEP 50 + fp32 accumulators), ABI-`dim` runtime-axis dispatch, isopar `par_unseq` |
| `ea59881c` | warnings ratchet reports the warning TEXT + toolchain, not just a count |
| `79a8e6e8` | the emitter under test is built through `__init__`, not `__new__` |
| `29180570` | `_FullCallHoister`: a nested `np.full` is spilled so `np.triu` can lower |
| (in `ea59881c`) | `libtbb-dev` installed in all six native CI jobs |
| `5867cdd7` | PIC asserted as a property of the flags every shared-library build runs with |
| `fcdefffe` | `lang-cuda` / `lang-hip` skill pages; cuda and hip stop routing at `lang-cpp` |

`ad0dd5e9` was DROPPED as a strict duplicate of `aedb957c` -- proven by identical
`git patch-id --stable` on both commits (`84107e649d25`, `f9367d7c48a8`), not by reading the diffs.
Merging both would have applied the runtime-axis dispatch twice.

Gate state after the merge: full translator op suite **1491 passed, 7 skipped** (the skips are
pre-existing -- 6 `jax-in-parent`, 1 no-cupy runtime); ABI corpus agreement **5 passed**; warnings
ratchet **1 passed**.

---

## A. A helper's array-return buffer is typed float64 inside an fp32 kernel -- RESOLVED `559456fb`

**RESOLVED 2026-08-05.** The worktree patch was re-applied onto the merged tree (its author branched
from `a4796377`, two merges stale, so its own verification described a translator that is not the one
shipping) and put through the gate this entry asked for:

- ABI corpus agreement, whole registry: **5 passed in 668.78 s**.
- Full translator op suite: **1495 passed, 6 skipped**, zero failures.
- Both named kernels re-checked at fp32 specifically. They now emit
  `static void _instance_norm(float *restrict __hret_0, const float *restrict x, ...)` -- `float`,
  not `double`, in both.

No kernel started raising from the new `__post_init__` refusal, so nothing in the corpus was
depending on the double fallback.

The fix landed at two levels rather than one, because the defect had two: `_ctor_dtype_tag` resolves
a `dtype=x.dtype` kwarg by chasing `x` through the alias walk that already resolves the shape (the
dtype must FOLLOW the source array), and `ArrayDesc.__post_init__` canonicalises on store and raises
on an unknown token -- the contract `ScalarDesc` already honoured. The original entry below is kept
because the reasoning is what made the gate the right one.

---

**Original entry, 2026-08-05:**

A non-inlinable helper's array-return buffer is emitted `double` while an fp32 caller passes
`float*`. Breaks `conv2d_instance_norm_divide` and `conv3d_multiply_instance_norm_clamp_multiply_max`
at fp32 in all three native backends. Same family as the accumulator defect fixed on 2026-08-04,
different code path (`_helper_return_ctype` / helper return buffers).

The deeper defect underneath it: **every emitter's dtype table falls back to `double` on a miss**, so
an unvalidated dtype token does not fail -- it silently picks a type. `dtype=x.dtype` was once read as
the literal string `"dtype"` and emitted a double buffer with nothing raised.

Written but NOT landed, sitting uncommitted in `.claude/worktrees/agent-abab318792c8de406`
(+138/-4 across `frontend.py`, `ir.py`, `test_array_return_helpers.py`). The shape of it:

```python
def __post_init__(self) -> None:
    try:
        self.dtype = dtypes.canonical(self.dtype)
    except KeyError:
        raise ValueError(f"array {self.name!r}: {self.dtype!r} is not a known dtype") from None
```

Canonicalise once where the dtype is STORED, and refuse a token that is not a dtype at all -- the
same contract `ScalarDesc` already honours.

**Why it did not land:** it touches EVERY array descriptor in the corpus, and the agent that wrote it
died before verifying anything. Unit tests are not the gate for a change with that blast radius.

**To land it:** apply, then run the ABI corpus agreement gate (~692 s, whole registry) plus the full
translator op suite, and re-check the two named fp32 kernels specifically. A `ValueError` from
`__post_init__` on a kernel that previously emitted is a finding, not a regression to work around --
it means that kernel was choosing `double` by fallback.

## B. The `unit` job's Phase 6 dies on a 25-MINUTE STEP ceiling -- RESOLVED `0c7bc5a9`

**RESOLVED 2026-08-05.** Phase 6 is now its own `integration` job with the whole job budget, and its
step cap is 70 minutes -- a hang detector rather than the budget it had become. It deliberately does
NOT `needs: [unit]`, so a Phase 1 unit failure cannot hide what the integration tests would have said.

Two consequences that had to be handled with it, neither of which is visible from the timeout alone:

- **Phase 7's gate would have silently narrowed.** The HF export is `if: success()` meaning "never
  publish from a run where a phase went red", and inside the old single job that covered Phase 6.
  Left as a step of `unit` it would have begun publishing on runs whose integration tests failed. It
  moved to an `hf-export` job with `needs: [unit, integration]`, which restores the guarantee across
  the split instead of quietly reducing it to `unit`.
- **`coverage` needed the new job in its `needs` list.** Nothing pins that list to the set of jobs
  uploading coverage -- the combine counts whatever it downloaded -- so a missing entry would not
  fail, it would race and report a plausible total built from fewer jobs. That is the same defect the
  per-artifact subdirectories were added to fix.

`tests/test_ci_coverage.py` green (11 passed) against the split.

---

**Original entry.** Updates item 2 of `BACKLOG_ci_reds_20260804.md`, which recorded the **1h30m JOB**
ceiling. That is a different limit, and the fix was more urgent because the step ceiling bites sooner:

```
##[error]The action 'Phase 6 -- integration-marked tests (build/run real artifacts)' has timed out after 25 minutes.
```

The `coverage (combined total)` red is purely downstream of this -- it has no failure of its own.

So a run page currently shows three reds where there is one cause and zero test failures. Fix is
unchanged: **split Phase 6 into its own job.** It is the only phase that builds and runs real
artifacts, so it has a different cost profile from the rest of the job and no reason to share a
budget with it. Do not fix it by dropping integration tests from the sweep.

## C. Shape-token aliasing keeps five kernels refusing -- RESOLVED, and it was TWO causes

`KNOWN_NON_LOWERING` is now **empty**: all 631 kernels lower. The entry above said the five were
ONE root cause. Instrumenting `_matmul_result_shape` at each decline showed **two**, and the second
was invisible from the refusal message because it never reached that function at all.

**Cause 1 -- two vocabularies for one extent (4 kernels).** A kernel names its own dimensions off a
parameter (`batch, channels, h, w = x.shape`), which the tuple desugar folds to `batch = batch_size`
/ `channels = embed_dim`. The `init.shapes` side keeps the symbol. So the contraction dim arrived
spelled `channels` on one operand and `embed_dim` on the other, and a string `==` declined it. The
`batch` vs `batch_size` case named above was real but was only netvlad; vision_attention was
`channels` vs `embed_dim`, and swin's inlined stages spell theirs as a read off a LOCAL array
(`__inl91_c = __inl8_y.shape[3]`) that no alias resolves without the shape table.

Fixed by replacing the token `==` with `lib_nodes.dims_agree`: literal equality, then equality after
substituting dimension aliases to a fixpoint, then a symbolic compare -- cheapest rung first,
because the first settles nearly every call, and the last is what recognises swin's
`4 * (4 * embed_dim)` against `16 * embed_dim`. Unresolvable answers FALSE, never True: a wrong True
contracts over two different extents, which is a miscompile, while a wrong False only declines.
`lowering.collect_dim_aliases` builds the map, filtering out the ARRAY expressions
`_collect_inlined_scalar_defs` over-collects (a BinOp of Names looks structurally like a dimension).

**Cause 2 -- a call-valued operand (1 kernel).** `relu_self_attention` writes
`np.maximum(scores, 0.0) @ v`. The extent is well defined but the loop nest indexes its operands by
NAME, so the hoister bailed before any shape comparison. `_MatmulHoister` now spills a `Call`
operand to a temp via `_materialise_dense_operand`, generalised from 1-D to any rank (the sparse
callers pass `max_rank=1` to keep failing loudly on the SpMM-with-sliced-RHS shape they do not
support). `Subscript` operands are untouched -- they have their own slice-aware path.

The refusal guard `lowering._refuse_scalarising_a_contraction` is **unchanged**. It is simply never
reached now.

Gates: ABI corpus **5 passed / 555.85 s** with the ratchet empty in both directions, and all five
kernels verified numerically against numpy on c/cpp/fortran at preset S -- `{'c': 'ok', 'cpp':
'ok', 'fortran': 'ok'}` for four of them. swin's fortran is `FAIL:compile-timeout` (twelve inlined
stages), not a numeric disagreement; c and cpp both agree.

## D. `cpp_isopar` capability gate -- RESOLVED

**RESOLVED.** `languages.isopar_capability()` compiles one `std::execution::par_unseq` call at the
harness's real C++ flags and reads `nm` for a TBB runtime call, returning the same three-way
`AutoparVerdict` every other column uses. Measured on this box: `ok`, `runtime_calls=12`.

It lives in `languages`, not beside `flags.polly_capability`, because only that module can name the
cpp block's compiler and its `-std=`; `flags.probe_autopar` grew `runtime_pattern` + `suffix` so the
one probe engine serves an OpenMP column and a TBB one instead of forking into two.

⛔ **`cpp_isopar` is NOT a scored column today.** The entry above implies a performance column exists
to protect; there is none. `FRAMEWORK_LANG` has no `cpp_isopar` entry, and the only consumers are
correctness oracles (`tests/numerical_oracle.py`'s `ISOPAR`, `test_cpp_isopar_emit.py`), where a
serial backend is slow rather than wrong. So the exposure today is zero and the work is to keep it
zero: the probe is registered in `preflight.AUTOPAR_PROBES`, so the column cannot be added ungated.

The negative case is REAL, not monkeypatched: `-D_GLIBCXX_USE_TBB_PAR_BACKEND=0` is exactly what a
runner without libtbb-dev compiles (libstdc++ defines that macro AS the `__has_include`). Same
source, same exit code, right answers -- 12 TBB references down to 0, object 22088 B down to 1256 B.
`test_isopar_probe_discriminates_a_serial_execution_backend` asserts BOTH halves in one test so it
cannot pass by measuring nothing, and `test_isopar_capability_agrees_with_the_link_decision` pins
that the header question (which decides `-ltbb`) and the `nm` evidence never disagree.

Also fixed in passing: `test_gate_is_a_no_op_for_ungated_frameworks` still listed `pluto` as
ungated after it joined `AUTOPAR_GATED`, so it asserted the opposite of what the tree does and
passed or failed according to whether the host's clang honours an OpenMP pragma.

### Original entry


libstdc++ chooses its parallel `<execution>` backend **per translation unit** via
`__has_include(<tbb/tbb.h>)`. With no TBB header, `par_unseq` degrades silently to sequential --
correct answers, no link error, no diagnostic, no test failure. A `cpp_isopar` performance column
measured on such a runner reports sequential numbers under a parallel-looking name.

`libtbb-dev` is now installed in all six native CI jobs, and both container recipes
(`containers/hpcagent_bench.Dockerfile`, `containers/cpu.def`) already had it -- so today the
backend is genuinely parallel everywhere. **That is a configuration fact, not a guarantee.** A runner
that loses the package silently reverts to measuring sequential work under a parallel name.

This repo already engineered against exactly this failure for Polly (`flags.polly_capability`,
`AutoparVerdict`, VACUOUS -> `NotSupportedByFramework`); the equivalent for isopar does not exist.
`languages.stdpar_link_flags()` already returns `()` precisely when the backend is serial, so the
gate is a short step from what is there.

## E. The warnings ratchet -- RESOLVED, the premise no longer holds

Never reproduced locally: 0 warnings under gcc 12, 14, 15, 16, clang++ 20, 21, 22, and gfortran
(local, 14, 15, 16). The count is a property of the compiler, not of the sources.

`ea59881c` made the failure print the warning TEXT and `toolchain_versions()` instead of only a
number, so the next runner failure names the compiler and the exact diagnostics.

**RESOLVED -- it no longer counts 20 on CI.** Checked against run `31001878513` (green), integration
job `92296893031`: `76 passed, 7 skipped`, and all 7 skips are `test_opt_reports_e2e.py`'s
polycc-absent cases. `test_warnings_ratchet` is therefore among the 76 that PASSED, at
`_KNOWN_BAD_COUNT = 0`. Nor did it pass vacuously: the test asserts `total_builds >= _MIN_BUILDS`
(20), so a green run is 20+ real (kernel, flavor) builds emitting zero warnings on the runner. The
three preceding red runs (`30994638862`, `30990017840`, `30952093069`) all have a **successful**
integration job -- their failures were elsewhere.

Two things landed so this cannot rot back into an unfalsifiable green:

* `scripts/verify_toolchain.py` now verifies `clang++`, not just `clang`. The ratchet skips its
  ENTIRE count when any of gcc/g++/clang/clang++/gfortran is missing, so a runner with clang but no
  clang++ would have turned the ratchet into a silent no-op while the toolchain gate stayed green.
* **CI moved to LLVM 21 from apt.llvm.org** (`.github/actions/setup`). ubuntu-latest ships 18 --
  a CI-only major nobody develops against is precisely how a count reads 0 on a dev box and nonzero
  on a runner with nothing in the log to say the toolchains differed. The unversioned
  `clang`/`clang++`/`flang` are symlinked at `/usr/local/bin`, because `resolve_compiler`'s
  highest-`<name>-<major>` fallback does NOT help here: `flags.polly_capability` probes bare
  `shutil.which("clang")` and `compilers.yaml` names the drivers unversioned, so a distro clang left
  on the box would keep winning. `toolset.yaml`'s discovery list gained clang-22/21.

## F. `_assert_ok` accepts a skip, so an op-oracle test can pass vacuously

`test_contraction_indexing_ops.py`:

```python
def _assert_ok(status, label):
    fails = {b: s for b, s in status.items() if s.startswith("FAIL")}
    assert not fails, f"{label}: {fails}"
```

A backend that reports "unsupported" is not a `FAIL`, so it passes. This is deliberate (a backend
that cannot lower an op should not fail the op's test) and it is also how a green test can mean *no
backend ran it*. It bit this session: the `np.triu` e2e case looked green before the raw status dict
was printed, and only the printout established that all six backends actually ran and matched.

**RESOLVED.** Skips stay accepted; `_assert_ok` now also asserts that at least one of `c` / `cpp` /
`fortran` reported `ok`, and prints all three when none did. A case can no longer go green with
nothing behind it, and the failure message names what actually happened per backend rather than
leaving the reader to print the dict by hand.

## G. Large-N fp32 reductions want pairwise summation -- RESOLVED as BLOCKED summation

A full float `sum` / `mean` now accumulates the innermost axis in blocks of 128 (numpy's own
pairwise cutoff) instead of one serial chain -- the reassociation a vectorizing compiler performs
once it is allowed to, written into the source so every backend gets it.

**Measured A/B through the op oracle**, n = 2**22, emitted float32, gcc `-O2` (which does NOT
reassociate on its own), seeded data. Difference from numpy's own pairwise sum:

```
one accumulator   |d| = 1.09e+02   (5.2e-05 relative)
blocked, 128      |d| = 4.00e+00   (1.9e-06 relative)
```

`test_large_fp32_sum_agrees_with_numpy_pairwise` pins that at `rtol=1e-5`, a factor ~5 either side,
so a regression to a single accumulator fails rather than drifting. ⛔ The test passes `dtypes=`
explicitly: without it `run_op` emits **float64**, whose naive error is ~1e-11, and the test goes
green whatever the accumulation does. That footgun cost a full A/B cycle here -- the first version
of this test had no teeth and looked fine.

**Scope, deliberately:** the FULL reduction only, and only its innermost axis -- that is the one
long dependence chain. Outer axes keep plain loops so an emitted nest still looks like a nest to
the parallelism and isopar recognisers. Axis-wise reductions and integer sums are untouched
(integer addition is exact and associative, so blocking it is pure code growth).

**Perspective on urgency, since the entry called it not urgent:** the fp32 grading tolerance is
`rtol=1e-3`, so even the naive form had ~20x margin at n = 2**22. This lands because the reference
lowering should not be the least accurate thing in the comparison, not because a gate was red.

## H. Objects built with `__new__` in tests

Fixed once this session and worth a standing note, because it broke twice for the same reason.
`test_variadic_minmax_folds_to_nested_2arg` constructed `_CBodyEmitter` via `__new__` and hand-set
only the attributes that one call happened to read. Each new attribute the emitter grew raised
`AttributeError` from inside emit -- first `kir`, then `isopar_param_dtypes` once the isopar work
merged. A hand-built stand-in silently re-asserts a private attribute list the test does not care
about.

A sweep found no other `__new__` bypasses in the translator tests. If one appears, build through
`__init__`: an empty kernel is enough for a scalar-only call.

## I. The determinism gate is now exercised without a GPU -- PARTLY RESOLVED

**The determinism half is RESOLVED.** `tests/test_determinism_gate.py` runs the gate the GPU pages
promise, on every CPU runner. It needs no device: `_determinism_check` is host Python over two output
dicts, and "two runs of a float-atomic reduction" is fully characterised by what those dicts hold --
values equal to the last ulp, unequal in their bits. Built with `nextafter` rather than a
re-summation in another order, because numpy's pairwise sum may reassociate to identical bits and the
fixture would then silently test nothing.

Six tests, paired so none can pass vacuously: the ulp-apart pair is REJECTED under `bitwise=True`
and **accepted** under `bitwise=False` (so the rejection is the bitwise leg's work, not a fixture
whose numbers are simply far apart); a reproducing run is accepted; a run that reproduces a WRONG
answer is still rejected (the oracle leg); and the default is pinned off `inspect.signature`, not
off a call site's line number.

⭐ **Found while pinning it: the gate false-failed any kernel whose output legitimately holds NaN.**
`np.array_equal` reports `NaN != NaN`, so a bit-for-bit identical rerun of a kernel with a masked
cell or a log of zero was scored NONDETERMINISTIC -- unfixable by the agent, since the second run was
a copy of the first. `_determinism_check` now passes `equal_nan=True`. Reproducibility and validity
are different questions: whether the NaN BELONGS there is the oracle leg's, and `compare_arrays` was
already NaN/±Inf-aware, so the two legs had disagreed on what NaN means.
`test_a_nan_that_appears_in_only_one_run_is_still_caught` pins that `equal_nan` did not become
"NaN matches anything".

**STILL OPEN:** the null-workspace protocol the pages warn about, and everything else the disabled
`gpu` job would cover. **Do not treat "the GPU pages are green" as evidence the GPU track works.**

### Original entry


`fcdefffe` landed `lang-cuda` and `lang-hip` and routed cuda/hip at them instead of
`lang-cpp`. What the pages CLAIM is pinned by tests -- neither may name a `-std=` the harness does
not pass, and a cuda/hip task must get its own page plus `lang-cpp` and no other language page. What
the pages DESCRIBE is not exercised at all: the `gpu (tvm-gpu / triton codegen)` job is
`[disabled -- no GPU runner]`, so `scoring._determinism_check` has never rejected a real float-atomic
reduction on this repo's CI, and the null-workspace protocol the pages warn about has never been hit
by a scored submission here.

That is the correct order to have done it in -- a page that is wrong about the gate is worse than no
page -- but it means the pages are currently reviewed prose plus two structural invariants, not
measured behaviour. **Do not treat "the GPU pages are green" as evidence the GPU track works.**

Two consequences worth carrying forward:

- The determinism claim is checkable WITHOUT a GPU: `_determinism_check` is host Python, so a
  fixture that returns deliberately non-bitwise-identical arrays pins "no float-atomic reduction
  passes" at the harness level rather than at the prose level.
- An `any`-mode prompt now inlines six language pages instead of four, ~18 KB more in every such
  prompt. That is the pre-existing `any` rule ("withholding a page withholds the rules for a
  language the agent may pick") applied to two more languages, not a new policy -- cuda and hip were
  already selectable, they just had no rules. Worth measuring if `any` prompts approach a context
  budget.

## J. HPTT clone 403 -- MITIGATED (pinned + retried), not eliminated

See `BACKLOG_ci_reds_20260804.md` item 4 for the diagnosis: the repository is public and answering,
so the 403 is an unauthenticated clone from a shared runner egress pool being throttled.

`containers/build-hptt.sh` now (a) fetches the pinned SHA
`942538649b51ff14403a0c73a35d9825eab2d7de` instead of a floating `master`, so the image's contents
stop being a function of the day it was built, and (b) retries the fetch 4 times with exponential
backoff, failing with a message that names the dependency, says a 403 here usually means throttling
rather than a missing repository, and prints the `git ls-remote` command that settles which. Both
container recipes call the same script (`hpcagent_bench.Dockerfile:99`, `cpu.def:37`), so both are
covered.

⛔ **Retry raises the odds, it does not remove the dependency.** A sustained throttle still red-lines
the whole container track, because this step is early in the image and
`tests/test_container_launch.py` never runs. Vendoring remains the only option that cannot fail this
way; it was not taken here because it costs a vendored-source update path.

`--branch` takes a branch or a tag and never a SHA, which is why the pin is `git init` + `fetch
<sha>` + `checkout FETCH_HEAD` rather than a one-line clone. Verified against the live remote.

---

## Method notes worth keeping

- **`git patch-id --stable` settles "are these the same work" across different SHAs and fork points.**
  It answered the `ad0dd5e9` / `aedb957c` duplicate question in one command, where diff-reading had
  produced a wrong answer earlier.
- **A verification inherited from an agent is not a verification.** Both agents this session branched
  from `a4796377`, two merges behind. Their measurements describe a translator that is not the one
  shipping. `np.triu` was re-run on the merged tree before landing; the numbers happened to hold, but
  that was the check, not the assumption.
- **Print the raw status dict, not the assertion result.** See finding F.
- **A pipeline hides the exit code you care about**: `pytest ... | tail -40` returns `tail`'s status.
  Capture the exit code directly, and never report an exit code read through a pipe.
