# Campaign results so far

Written 2026-08-14, before `hpcagent-bench-runs/` was cleared for a re-run. Everything below is
what survives that deletion; the per-run transcripts and `.db` files it summarises are gone.

Track: `loop_level_reasoning` (llr, renamed from `foundation` on 08-05, hard rename, no aliases).
242 vectorization-puzzle kernels per arm. Agent = Claude Code CLI in a container talking to vLLM
over `/v1/messages` (the litellm proxy is bypassed -- it is broken), one MCP server `optarena`
exposing 6 tools: `task`, `syntax_check`, `profile`, `score`, `submit`, `search`.

---

## Headline: skills pages RAISE speedup, and one line CRATERED solve rate

**Ablation 3** (588130-135 + 588883), gpt-oss-120b, 2x3 = skills on/off x c/cpp/fortran.
Crediting: last submission with score-fallback synthesis, held fixed across arms. REVISED 08-11 so
fallback credits only score calls at the grading default preset `fuzzed` -- a preset-S/M/XL score
measures a different problem size, and the unrevised numbers were contaminated by that.
Stats: Wilcoxon on log-speedup of paired kernels, BH-FDR q, McNemar on solve.

| arm | solved | median | gmean | HL on-vs-off | q |
|---|---|---|---|---|---|
| c-on | 227/242 (93.8%) | 1.000 | 1.163 | **+4.4%** | 0.0046 |
| c-off | 226/242 (93.4%) | 0.991 | 1.009 | | |
| cpp-on | 209/242 (86.4%) | 1.014 | 1.155 | **+10.6%** | 6.1e-6 |
| cpp-off | 217/242 (89.7%) | 0.988 | 0.964 | | |
| fortran-on | 160/242 (66.1%) | 1.019 | 1.202 | +4.0% | 2.2e-4 |
| fortran-off | 219/242 (90.5%) | 0.998 | 1.004 | | |

Speedup is up in all three languages. But the fortran page's own "iterate with preset S, it's
cheap" line **cut solve rate from 90.5% to 66.1%** (McNemar p=3e-10): 752 preset-S score calls in
fortran-on, and 82 agents never produced a correct default-preset grade at all. c and cpp solve
rates are neutral (p=1.0, p=0.31). That line is removed for v3.

fortran-on's higher gmean is survivorship -- only kernels with a correct default-preset grade
credit at all. The solve-rate crater is the real result.

**Ablation 2** (earlier, skills v1): null to negative. Censoring was 0% exhaustion. This is what
motivated the v2 page rewrite that ablation 3 measured.

## Mechanism, from mining all 1452 transcripts

- The skills win is largely **redirection under the old single-thread regime**: `omp parallel`
  usage 36-41% in the off arms vs 0-1% in the on arms. Adding `omp parallel for` helped 16/332
  attempts and hurt or broke 136. Grading went multi-core at `e0de00a0`, so this evidence
  describes the OLD contract and does not transfer.
- `omp simd` on the unit-stride inner loop is the reliable lever (136 better / 33 worse).
  `restrict` re-spelling and reduction clauses follow.
- Both 24x wins came from DELETING deliberately silly reference structure and writing the plain loop.
- Failure taxes, largest first: preset shopping (fixed); alignment lies via OpenMP `aligned()`
  (c-on segfaulted 71 runs vs c-off 23 -- caused by lang-c wording, fixed); no revert-to-best
  discipline (30-55 kernels/arm credited >5% below their own best); context exhaustion (~25% of
  runs); phantom reference hunts; judge-side submit-500 on 11 kernels (open infra bug); sub-us
  noise chased as signal.

---

## Infrastructure root causes (the expensive lessons)

These cost more wall-clock than the science did. Recording them so they are not re-learned.

- **`VLLM_DISABLE_PYNCCL=1` was the root cause of BOTH the ~20x slow inference and the RCCL
  hangs** (`run_cluster.sh:144`). Without PyNCCL every collective goes through
  `ProcessGroupNCCL`, which is not graph-capturable on vLLM's path, so capture stalled,
  `--enforce-eager` went on every arm, and kimi decoded at **1.4 tok/s per request against 16.8
  measured on the same TP=4/PP=4/4-node shape**. It also owns the watchdog hangs. Now deliberately
  not defaulted.
- **XL preset was capped at 1024 on the timed path** -- every performance number taken before the
  08-12 fix is void. Fuzz is now always `[XL*0.85, XL*1.15]`.
- **Baseline is serial**: 0/631 manifests declare `baseline:`, so speedup is divided by the
  NumpyToX C column, not by a parallel reference.
- **CXI/RCCL cross-node is intermittently wrong, not just slow** -- roughly 2 in 3 runs produced
  SILENT WRONG SUMS. `NCCL_NET_GDR_LEVEL=0` (unset is NOT off) took it from 1/5 to 4/5 clean.
  `NCCL_PROTO` and `split_group` are dead ends. The earlier "occupancy threshold" was an n=1
  artifact.
- **Grading is multi-core as of `e0de00a0`**: the timed child gets the slot's 24 physical cores,
  pinned, no SMT, same affinity for omp/tbb/dc. This is a regime break -- no result from the 588*
  series is comparable to anything measured after it.
- **DaCe never unifies a promoted runtime scalar**: `k=K` mints `__sym_k_0`, which is the
  132-kernel IndexError class.

## Startup cost, measured (592283, 08-13)

The reason a re-run is expensive. On 4 nodes, TP=4/PP=4, Kimi-K2.7-Code (~1T total / ~32B active,
INT4 group-quantised experts, 555 GB on disk):

```
Loading safetensors shards:        100% | 64/64   [~1:35:00]
Capturing CUDA graphs (PIECEWISE): 100% | 147/147 [3:13:33, 79 s/it]
Capturing CUDA graphs (decode, FULL):  0% | 0/19  <-- OOM killed here
```

Two findings. The FULL-decode capture is a second phase nobody had budgeted for. And the OOM is
host RAM: MI300A is an APU where CPU and GPU share one 128 GB HBM pool, so
`--gpu-memory-utilization 0.85` starves the host. The run never served, so there is still no
measured tok/s on this shape.

`VLLM_CACHE_ROOT` is unset and `HOME` is not mounted into the inference container, so vLLM's
torch.compile cache lands in the container's ephemeral layer and is destroyed every run -- that
3 h 13 m of capture is recompiled from scratch on every single start.

## Expected throughput, for calibration

Single-stream decode on this shape is **tens of tok/s**, not thousands: every token reads the whole
active set, MoE decode is latency-bound on small GEMMs and expert gather/scatter, and PP=4 leaves
three quarters of the pipeline idle for one request. 16.8 tok/s is consistent; 1.49 is broken.
Thousands of tok/s is an **aggregate** figure across concurrent requests -- MoE batches well, and
`--max-num-seqs 128` with 40 agents in flight is that regime.

---

## Where things stand

Nothing is currently running. The llr4 campaign (18 arms, 590351-62/79-84/89) was cancelled at
~10 h; the two K2.7 smoke attempts both died in startup (591899 on a 7200 s readiness wait,
592283 on the OOM above). No tok/s measurement exists yet on the current stack.

Open before the next launch:

- `VLLM_CACHE_ROOT` on a persistent filesystem, weights and HF cache staged to `/iopsstor`.
- Lower `--gpu-memory-utilization` for the MI300A shared-memory OOM.
- Six llr4 kimi27code arms still carry `AGENT_READY_TIMEOUT_SECONDS=7200`, the value measured
  fatal in 591899.
- Aggregate multi-agent tok/s measurement (single-stream probe already landed).
