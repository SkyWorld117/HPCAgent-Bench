# Ablation studies, run tagging, and the speed-up plot -- OPEN

Filed 2026-08-03, pruned 2026-08-05 (levels 1-3 of KernelBench are ported; the rocprofv3 schema
fixes landed). Items 1-5 and 7-9 are unstarted. They interlock: the two ablations (1, 2) are the
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

### The AMD PROFILER install is the harder half, and needs its own sample

NVIDIA needs no sample: install the CUDA Toolkit and you have `nsys`, `ncu` and CUPTI. AMD does
not work that way, and every step below cost time on 2026-08-03 that a sample would have saved.
Write it as a runnable script plus a preflight check, not prose:

- **Ubuntu already packages ROCm** (7.2.4 as of writing). Do NOT send people to
  `amdgpu-install`: the URL is version-pinned and 404s, and `repo.radeon.com` has no directory for
  a recent Ubuntu codename. `apt install rocminfo rocm-smi hip-runtime-amd hipcc-rocm
  rocprofiler-sdk rocprofiler-compute hsa-amd-aqlprofile` is the whole thing -- the last one is not
  a dependency of any of the others and `rocprofv3` requires it, so an install line without it
  reproduces the exact failure this section exists to prevent (item 11 has the symptom, which does
  not look like a missing package).
- **The tools install to `/opt/rocm/bin` and are NOT on PATH.** `rocprofv3: command not found`
  while `apt` reports the package as newest is the confusing first symptom.
- **`rocprof-compute` has pinned Python deps** (`astunparse==1.6.2` against a system 1.6.3, plus
  `plotext`, `dash`, `colorlover`, `kaleido`, `plotille`, `textual` absent). Ubuntu's python3 is
  PEP-668 externally managed, so the sample should build a
  `python3 -m venv --system-site-packages` from
  `/opt/rocm/libexec/rocprofiler-compute/requirements.txt` rather than fighting pip.
- **Two ROCm toolchains coexist and do not interoperate.** `/usr/bin/hipcc` links against
  `/usr/lib/rocm/llvm` and fails with `undefined symbol: __hipUnregisterFatBinary`;
  `/opt/rocm/bin/amdclang++` has no device bitcode. The build that works crosses them:

  ```sh
  HIPCLANG=/usr/lib/rocm/llvm/bin/clang++
  $HIPCLANG --driver-mode=g++ -O2 -x hip --offload-arch=<gfx> \
    --hip-device-lib-path="$($HIPCLANG --print-resource-dir)/amdgcn/bitcode" \
    -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib
  ```

  DERIVE the bitcode path, never write `.../clang/20/...`: the major version is whatever that
  install happens to ship, and on any other one the directory is absent and the compile fails with
  a missing-bitcode error that looks exactly like the toolchain mismatch this bullet is about.

- **An unsupported target needs an override, and it only covers the RUNTIME.** gfx1103 (Radeon
  780M) is not on ROCm's official list; `HSA_OVERRIDE_GFX_VERSION=11.0.0` plus
  `--offload-arch=gfx1100` runs HIP code fine. It does NOT reach rocprofiler: measured, `--pmc`
  under the override still aborts and still names gfx1103, because counter enumeration reads the
  real hardware ID. `rocm_agent_enumerator` prints the real target and should be the sample's
  first line.

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

The 250 ports under `hpcagent_bench/benchmarks/ml/` are translations of upstream PyTorch models, and
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

## 10. level4 -- decide whether it belongs in the corpus at all

Levels 1-3 are DONE: 250 kernels (100 + 100 + 50), `KERNELBENCH_PORT_COUNT = 250`. The vendored
submodule (`third_party/KernelBench` @ `423217d9`) holds one more level, and it is the only thing
left of "port the whole of KernelBench".

**level4 is a different KIND of entry.** It holds HuggingFace model + batch + seq configs
(`16_gpt2_bs1_seq1023.py`), not self-contained kernels. Decide whether those become corpus kernels
at all, or whether the subtrack is level1-3 by definition. `collect_reference_sources.py` currently
excludes level4 deliberately, with that reason recorded.

**A port that does not EMIT is not done**, and the translation ratchet accepts non-emitting ports --
so "ported" and "usable" are different states and the count should track both. Any level4 batch has
to move `KERNELBENCH_PORT_COUNT` (one constant, four consumers), pass the CNF invariants in
`docs/canonical_numpy_form.md`, and resolve 1:1 to an upstream file. Interacts with item 9: the
PyTorch-agreement tests are what make a port trustworthy, so grow the two together.

## Order to do them in

5 before 1 and 2 (an untagged ablation run cannot be separated afterwards). 7 before 1 and 2 as
well, or the results get read off the plot that misleads. 3 and 4 are independent.

## 11. rocprofv3 reader -- two open pieces, the schema fixes are done

The `LDS_Block_Size` / `VGPR_Count` schema drift is fixed and pinned in fixtures (both generations).
⛔ The trap that found it is worth carrying: `column()` returns `""` for an unmatched prefix and
`number("")` is `0.0`, so an unread 16 KB workgroup came back as `shared_memory: 0.0` -- **a
measurement saying the LDS budget was free**, not a `null`. An agent then sizes a tile against a
budget it has already spent.

Still open, both found on real hardware (Radeon 780M / gfx1103, ROCm 7.2.4, rocprofiler-sdk 1.1.0):

- `*_kernel_stats.csv` carries a **`StdDev` column the reader does not surface.** Run-to-run spread
  per kernel is exactly what a "did this change anything" question needs.
- **A `rocprofv3` preflight check in the backend.** It requires `hsa-amd-aqlprofile` and does not
  depend on it; without the package the run dies with `error while loading shared libraries:
  libhsa-amd-aqlprofile64.so.1` prefixed with the CHILD's name, so it reads as a bug in the profiled
  program. The AMD sample in item 3 names the package, but the message the harness quotes is still
  the child's.

Two things NOT to re-learn: `*_memory_copy_trace.csv` has no size field on this version (so the
`total`/`unit` nulls are correct for CSV -- the buffer-tracing record does define `bytes`, so a JSON
/ rocpd / pftrace emitter may carry it), and the output layout is FLAT
(`<dir>/<prefix>_kernel_stats.csv`), so keep the recursive glob but do not assume the nested form.
