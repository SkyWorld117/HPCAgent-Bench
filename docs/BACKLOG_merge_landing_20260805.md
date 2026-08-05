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

## C. Shape-token aliasing keeps five kernels refusing -- the root fix

`KNOWN_NON_LOWERING` in `test_abi_corpus_agreement.py` holds exactly five kernels, and they are ONE
root cause, not five:

```
ml/netvlad_no_ghost_clusters      matmul '__cb8 @ x' reaches slice fusion un-hoisted
ml/netvlad_with_ghost_clusters    matmul '__cb8 @ x' reaches slice fusion un-hoisted
ml/relu_self_attention            matmul 'fmax(scores, 0.0) @ v' reaches slice fusion un-hoisted
ml/swin_transformer_v2            matmul '__hcall43 @ __cb34' reaches slice fusion un-hoisted
ml/vision_attention               matmul 'tokens @ __cb3' reaches slice fusion un-hoisted
```

Read the list as five kernels that USED to emit silently-wrong elementwise products and now decline
instead. The refusal (`lowering._refuse_scalarising_a_contraction`) is correct and must not be
relaxed; it fires at the one point where the difference between a contraction and a product is still
visible, because the fusion rewrite replaces both operands with scalar subscripts and emits `*`.

**The fix is one rung down:** `_matmul_result_shape` compares shape TOKENS as strings, so a batch
dimension spelled `batch` in one operand and `batch_size` in the other reads as a mismatch and the
hoister declines a shape it should accept. Teach the hoister those shapes and all five lower.

Corpus survey for context: **5 refusals of 631 kernels, zero non-refusal errors.**

## D. `cpp_isopar` still has no capability gate

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

## E. The warnings ratchet counts 20 on the CI runner and 0 on every local toolchain

Never reproduced locally: 0 warnings under gcc 12, 14, 15, 16, clang++ 20, 21, 22, and gfortran
(local, 14, 15, 16). The count is a property of the compiler, not of the sources.

`ea59881c` made the failure print the warning TEXT and `toolchain_versions()` instead of only a
number, so the next runner failure names the compiler and the exact diagnostics. **Nothing to fix
until that output exists** -- and the report change was the fix for "the CI log had nothing
actionable in it", which was the real problem.

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

**Suggested shape:** keep skips accepted, but assert a floor -- at minimum that the native backends
(`c`, `cpp`, `fortran`) did not all skip -- so a case cannot go green with nothing behind it.

## G. Large-N fp32 reductions want pairwise summation

At preset S the naive-vs-pairwise gap does not bite. At `n = 262144` it is ~1e-05. A future large-N
fp32 kernel will need pairwise summation to keep agreeing with numpy, which reduces pairwise itself.
Not urgent; record so it is not re-discovered as a mystery disagreement.

## H. Objects built with `__new__` in tests

Fixed once this session and worth a standing note, because it broke twice for the same reason.
`test_variadic_minmax_folds_to_nested_2arg` constructed `_CBodyEmitter` via `__new__` and hand-set
only the attributes that one call happened to read. Each new attribute the emitter grew raised
`AttributeError` from inside emit -- first `kir`, then `isopar_param_dtypes` once the isopar work
merged. A hand-built stand-in silently re-asserts a private attribute list the test does not care
about.

A sweep found no other `__new__` bypasses in the translator tests. If one appears, build through
`__init__`: an empty kernel is enough for a scalar-only call.

## I. The GPU skill pages ship, but nothing in CI ever scores a GPU submission

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

## J. HPTT clone 403 -- unchanged, still intermittent

See `BACKLOG_ci_reds_20260804.md` item 4. Repeating only the part that decides scheduling: the
repository is public and answering (`git ls-remote` returns
`942538649b51ff14403a0c73a35d9825eab2d7de`), so the 403 is an unauthenticated clone from a shared
runner egress pool being throttled. **A re-run may pass, and a green run does not prove it fixed.**

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
