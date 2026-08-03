# Ablation studies, run tagging, and the speed-up plot -- OPEN

Filed 2026-08-03. Seven items, none started. They interlock: the two ablations (1, 2) are the
CONSUMERS of the tagging work (5) and the plot work (7), so tagging lands first or the ablation
results are unseparable from ordinary runs.

## 1. Ablation: does a repo PR-and-merge framing improve agent performance?

Build the sample. The question is whether presenting a code-optimization task as a repository pull
request the agent opens and merges -- rather than as a bare kernel and a submission -- changes the
score. Same kernels, same budget, two framings.

CONTAINERIZED launch, not native. The framing is the independent variable and the environment has
to be held fixed, which the native path does not guarantee across two long runs.

## 2. Ablation: do profiling tools help?

Two arms, same kernels, same budget:

- **bare** -- an agent with NO profiling skills in its prompt at all
- **instrumented** -- an agent with the profiling skills: `linuxperf`, `papi-cpu`, `papi-gpu`,
  `nsys`, `ncu`, and the AMD set (`rocprofv3`, `rocprof-compute`, `papi-gpu-amd`)

CONTAINERIZED launch, same reason as 1.

This is the ablation the whole skills programme is aimed at, so it is worth stating what would
make it honest: the bare arm must not be handicapped by anything OTHER than the missing skills.
`prompt.profiling_guidance` already gates whether the instrument bodies are inlined (that gate cut
a prompt from 1373 to 284 lines), so the two arms differ by exactly those bodies and nothing else.
Check the token counts of both arms before believing a result -- if the instrumented arm is also
the larger-context arm, context length is a confound.

## 3. `samples/scripts`: how to install PAPI 7.2.0 with NVIDIA or AMD support

Neither GPU component is built by default, and a distribution PAPI on a box with a perfectly good
GPU usually has neither -- which is the single most common reason a `papi-gpu` page produces
nothing. Two scripted recipes:

- **NVIDIA** -- `./configure --with-components="cuda"` with `PAPI_CUDA_ROOT` set. Verified on this
  box: PAPI 7.2.0.0 at `/usr/local`, cuda component active, 53782 native events, 30 counters.
- **AMD** -- `./configure --with-components="rocp_sdk"` with `PAPI_ROCP_SDK_ROOT` set. Note that
  the older `rocm` component is DEPRECATED from MI300A onward and that the two are mutually
  exclusive on older parts. Also `AQLPROFILE_READ_API=0` for ROCm >= 6.2.0 or every count is zero.

Include the verification step in each, not just the build: `papi_component_avail` plus one real
counted region. A build that links and counts nothing is the failure mode.

## 4. README: document the tag system

Users should be able to register and add tags. For now the one tag that must exist is `npbench`.

## 5. DB: a tag on the run, defaulting to `None`

Store it per run. The consuming rule is the point of the feature: **plotting must never mix two
run tags.** A plot takes a STUDY (run) tag and shows only that study. Without this, an ablation's
two arms and every unrelated run in the database land on one chart.

Default `None` so existing rows keep working.

## 6. (not filed)

## 7. Speed-up plot: default OFF, and a better one when it is on

The current speed-up table ships on by default; it should not. Replace the plot with a
median-speed-up chart:

- **X axis: kernels.**
- **Y axis: signed relative change, not a ratio.** 1.0x (no change) sits at **0**. A kernel 100%
  faster (2x) is **+1**; 200% faster (3x) is **+2**. Slow-downs go NEGATIVE. This is the part that
  matters -- a raw ratio axis puts every slow-down in the 0..1 sliver and every speed-up in an
  unbounded tail, so the eye reads a 0.5x regression as smaller than a 1.5x win when they are the
  same magnitude.
- **Three INDEPENDENT y axes by order of magnitude**, so one 100x outlier cannot flatten the rest:
  - `> 10x`
  - `2x .. 10x` (and the mirrored slow-down band)
  - `-2x .. 2x`
- Ship it as a new plotting script.
- Then generate a SIMPLIFIED single-order-of-magnitude variant for SVG.

## 8. Follow npbench's 0-init change

https://github.com/spcl/npbench/pull/47 -- `np.empty` / `np.empty_like` -> `np.zeros` /
`np.zeros_like`, 256 sites across 132 files, every backend (numpy, dace, numba, cupy, dpnp,
pythran, legate, jax).

The reason is a correctness one and it applies to this corpus verbatim: **a kernel that writes an
`empty` buffer only PARTIALLY leaves stale memory in its output.** Those kernels are read-
nondeterministic -- the same input yields different outputs on different runs and the comparison
against the reference flakes. This repo grades BITWISE, so it is worse here than upstream: a stale
byte is not a tolerance question, it is a failed verification that looks like a flake.

Two things ride along in that PR and are worth taking together:

- **Gram-Schmidt input conditioning.** The reject-sampling loop (`while np.linalg.matrix_rank(A) <
  N`) is replaced by deterministic diagonal dominance, `A[:N, :N] += N * np.eye(N, dtype=datatype)`,
  giving `cond(A) ~= 1.5` every time. A plain random matrix is only full-rank probabilistically and
  can be conditioned badly enough that harmless FMA contraction changes the answer -- which reads
  as a translator bug and is not one.
- **`dace_canonicalize_cpu` / `dace_canonicalize_gpu`** framework variants, exercising the
  canonicalize pipeline with WCR array reductions enabled.

Audit this corpus for the same pattern rather than porting the diff: find every `np.empty` /
`np.empty_like` in `hpcagent_bench/benchmarks/` whose buffer is not fully written before it is
read, and check the CNF `declare-then-fill` invariant already covers the rest.

## 9. Unit-test the KernelBench NumPy ports against PyTorch

The 239 ports under `hpcagent_bench/benchmarks/ml/` are translations of upstream PyTorch models, and
**nothing currently checks that they compute the same thing.** `scripts/collect_reference_sources.py`
resolves each port to its upstream file for PROVENANCE only -- the collected original "is never
imported (it needs torch) and never graded", per `handle_kernelbench`. So the mapping is verified and
the semantics are not.

What is needed: per port, run the upstream PyTorch model and the NumPy port on the same inputs and
compare. Points to settle when doing it:

- **Where torch lives.** It is deliberately not a harness dependency, so this is an opt-in suite --
  its own marker and its own CI job, skipped (loudly, with a reason) where torch is absent. It must
  never silently pass by not running; see the three-case skip table rule.
- **Tolerance, not bitwise.** This is the one comparison in the repo that CANNOT be bitwise: torch
  and numpy differ in accumulation order, and torch may use different BLAS. Pick per-dtype
  tolerances and state them.
- **Feature coverage, not just outputs.** The ask is the ports' FEATURES too: conv stride/padding,
  pooling, batchnorm in train vs eval, softmax axis, broadcasting, and the depthwise/grouped conv
  cases. A port that matches on one input shape can still have the stride wired wrong -- which is
  exactly the class of bug the structural slice-step fold just touched, where two convs in one
  kernel take different strides.
- **Ordering.** Do this AFTER the emit gap is closed (see below), or a passing NumPy-vs-torch test
  still says nothing about what the translator produces.

Related open defect found 2026-08-03: `efficientnet_mb_conv` and `resnet_basic_block` now PARSE and
LOWER (commit `6efe7665`) but still fail at EMIT with `NotImplementedError: expression Tuple`, from a
local `np.zeros((n, hidden, h, w))` whose dims come from tuple-unpacked `.shape`. Separate gap,
downstream of the fold.

## 10. Port the WHOLE of KernelBench

239 of KernelBench's kernels are in the corpus today: all 100 of level1, all 100 of level2, and 39
of level3. The vendored submodule (`third_party/KernelBench` @ `423217d9`) holds **level1 100,
level2 100, level3 50, level4 20** -- so the remaining work is **11 level3 networks and the 20
level4 entries**, and the target is all 270.

Two things to settle before starting, because they are the reason the tree stopped where it did:

- **level4 is a different KIND of entry.** It holds HuggingFace model + batch + seq configs
  (`16_gpt2_bs1_seq1023.py`), not self-contained kernels. Decide whether those become corpus
  kernels at all, or whether the subtrack is level1-3 by definition. `collect_reference_sources.py`
  currently excludes level4 deliberately, with that reason recorded.
- **A port that does not EMIT is not done.** `efficientnet_mb_conv` and `resnet_basic_block` parse
  and lower after `6efe7665` and still fail at emit (`NotImplementedError: expression Tuple`). The
  translation ratchet accepts non-emitting ports, so "ported" and "usable" are different states and
  the count should track both.

Each new batch has to move `KERNELBENCH_PORT_COUNT` in `tests/corpus_counts.py` (one constant, four
consumers), pass the CNF invariants in `docs/canonical_numpy_form.md`, and resolve 1:1 to an
upstream file. Interacts with item 9: the PyTorch-agreement tests are what make a port trustworthy,
so grow the two together rather than landing 31 more unverified translations.

## Order to do them in

5 before 1 and 2 (an untagged ablation run cannot be separated afterwards). 7 before 1 and 2 as
well, or the results get read off the plot that misleads. 3 and 4 are independent.
