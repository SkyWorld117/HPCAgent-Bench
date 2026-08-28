# Submitting on beverin (AMD MI300A)

What the submission scripts do, how to launch the three campaign models, and the configurations
that have actually served versus the ones that have not. Every claim below names the job that
produced it -- a config with no job id behind it has not been measured.

Node facts: 192 cores, 4 GPU dies, ~513 GB per node, partition `mi300`. **Never pass `--account`:**
beverin schedules root, a-g200 and a-g34 identically, so `-A` only picks a billing line nobody
chose. **House ceiling is 36 nodes in flight.**

## The stack

```
submit-llr8.sh          picks arms, sizes nodes, chains dependencies
  -> beverin.sbatch     one allocation; splits it into inference / agent / judge roles
    -> run_cluster.sh   re-entered INSIDE each role's container; builds the serve command
```

`CLUSTER_ENV_FILE` selects the arm and is the only thing that differs between arms, so several
arms run in parallel from one script. The env file is `. `-sourced, meaning **a duplicate
assignment silently shadows the earlier one** -- edit the line, never append a second.

`beverin.sbatch` refuses an allocation that does not equal `INFERENCE_NODES + AGENT_NODES +
JUDGE_NODES` exactly. `arm_nodes.sh` reads those three from the same file the launcher reads, so
the submitter and the launcher cannot disagree; always size with it rather than a literal.

One judge NODE is four ranks, one per socket (`GRADE_CPUS` = cores-per-socket, from
`run_cluster.sh:88`). One judge node covers 40 agents with headroom, so `JUDGE_NODES=1` is right
for every arm here.

## Submitting the three models

`submit-llr8.sh` runs one model, both legs (base and skills), languages chained so only one holds
nodes at a time. Extra arguments pass through to `sbatch` verbatim.

```bash
cd containers/cluster/example-script

MODEL=oss120b       ./submit-llr8.sh          # vLLM,   3 nodes/arm
MODEL=qwen38        ./submit-llr8.sh          # SGLang, 3 nodes/arm
MODEL=kimi27sglang  ./submit-llr8.sh          # SGLang, 6 nodes/arm (4 inference)

MODEL=oss120b LEGS=2 ./submit-llr8.sh         # skills leg only
MODEL=qwen38  LANGS=c ./submit-llr8.sh        # one language
```

A single arm, when you want exactly one:

```bash
CLUSTER_ENV_FILE=.env.llr8-oss120b-c sbatch \
    --nodes="$(. ./arm_nodes.sh; arm_nodes .env.llr8-oss120b-c)" \
    --time=08:00:00 --partition=mi300 --job-name=llr8-oss120b-c beverin.sbatch
```

### Kimi needs waves; submit-llr8.sh does not do them

Kimi runs 12 agents against 40 kernels, so a single arm cannot hold the tag. The 40 are split into
two 20-kernel halves with their own lists and arm labels, submitted as separate jobs:

```bash
for arm in c-a c-b c-skills-a c-skills-b; do
    CLUSTER_ENV_FILE=.env.llr8-kimi27sglang-$arm sbatch \
        --nodes=6 --time=24:00:00 --partition=mi300 \
        --job-name=llr8-kimi27-$arm beverin.sbatch
done
```

Halves are disjoint and their union is the full tag -- verify that before trusting a wave, because
nothing enforces it. `-c-r1` / `-c-skills-r1` are retry arms holding only the kernels an earlier
wave never submitted.

## What worked

| config | evidence |
|---|---|
| oss120b on vLLM, **aiter off** | 20 COMPLETED arms, 600516 through 609359 |
| Kimi K2.7 on SGLang, `--attention-backend triton` + `SGLANG_USE_AITER=1` | 610247/610249/610250 COMPLETED, 53-68 submissions per arm |
| qwen3.8 on SGLang, same attention config | 610229: 163.0 tok/s, 9/9 accuracy incl. 51,200-token cases, 96 requests, 0 errors |
| `JUDGE_NODES=1` (4 ranks) for a 40-agent arm | wave-1 arms graded 354-502 scoring calls without a judge backlog |
| Text-only serving (`--language-only`) | campaign is always text-only; a vision stack only costs KV |
| Weights on iopsstor | 9.45 GB/s at 16 readers vs capstor 0.83 (593523) |

SGLang needs **both** `--reasoning-parser` and `--tool-call-parser` named. With one missing, turn-1
tool calls are swallowed and the run is logged as a success that submitted nothing.

## What did not work

| config | what happened |
|---|---|
| **aiter master switch on, vLLM path** | 0 for 9. Kernels JIT-build on the FIRST REQUEST behind a baton lock; the build outlives the engine's RPC deadline. 610251/610252 died at 38 min with `step_counter=0` -- not one token decoded, zero submissions, 6 nodes lost |
| qwen3.8 on vLLM, any leg | best of seven legs decoded 8.5 tok/s; `mtp`, `fp8kv+mtp` and all three aiter legs DID NOT SERVE (610203 on 0.27.1, 610204 on 0.23.0) |
| qwen3.8 `fp8kv` on vLLM | served, then decoded 0.0 tok/s |
| aiter MLA on gfx942 | `fmha_v3_varlen_fwd invalid argument` (600662) |
| `INFERENCE_ENGINE=sglang` with a vLLM `INFERENCE_CE_ENV` | `/opt/venv/bin/python3` in that image has neither sglang nor huggingface_hub; 610646/610647 died in 31 s resolving the model path |
| `AGENTS_PER_NODE=120` | a coding agent spends most of its wall clock in tools or a compile, so the batch never filled |

The aiter row is one mechanism, not nine bugs. A job that dies mid-build **leaves its lock**, and
every later server on that cache root then blocks on a baton nobody holds; a 0-byte lock names
nobody at all. Before any run that enables aiter, sweep them:

```bash
find "${SCRATCH}/.jit-cache" -name 'lock' -o -name 'lock_*'   # inspect, then delete if no job is running
```

On SGLang aiter is fine and stays on -- it imports a prebuilt `module_aiter_core` and serves. It is
the vLLM path that builds on first request and dies.

## Traps that cost whole runs

- **The image moves with the engine.** Changing `INFERENCE_ENGINE` without changing
  `INFERENCE_CE_ENV` gets you an interpreter with neither the engine nor its dependencies. The
  judge's uvicorn logs `Application startup complete` before the model server dies, so the log
  looks healthy.
- **A job's exit code is not its result.** An arm is FAILED when *every* agent exits nonzero and
  COMPLETED when one does not -- 610248 was FAILED and still produced 53 submissions across 15
  kernels, while 610247 was COMPLETED with 17 of 20 agents nonzero. Read the judge DBs, not sacct.
- **`verdicts:` in the run report is utilization advice, not scoring.** `verdicts: none` means the
  monitor had no sizing complaint.
- **`launch failed requeued held` does not restart.** Slurm holds the job; `scontrol release <id>`.
- **Results live in per-rank SQLite**, `<run>/judge/rank-*/hpcagent_bench*.db`, tables
  `submissions` / `attempts` / `calls`. Join on the bare kernel name -- the problems lists carry a
  full path and the DB stores the basename.
- **Agents are cut off by `AGENT_TIMEOUT_SECONDS`, not job wall clock.** In wave 1 that killed 12 of
  20 agents at 8h each. A shorter kernel list does not buy an agent more time.
- **Never edit an env file or a launcher script while jobs run** -- roles re-source them.

## Agents

`AGENTS_PER_NODE` x `AGENT_NODES` is a ROLLING pool -- one
`ThreadPoolExecutor(max_workers=workers)` with a submit per problem, so that many start and each
finisher launches the next. It is not a barrier.

| model | agents | kernels | passes | `AGENT_TIMEOUT_SECONDS` |
|---|---|---|---|---|
| oss120b | 40 | 40 | 1 | 14400 (4h) |
| qwen3.8 | 20 | 40 | 2 | 14400 (4h) |
| Kimi K2.7 | 12 | 20 per half | 2 halves | 28800 (8h) |

Kimi cannot hold 40 kernels in one arm at 12 agents, which is why C is split into `-a` / `-b`.

## Effort levels -- per model, not a shared dial

| model | value | why |
|---|---|---|
| oss120b | `high` | ladder is low/medium/high, and the template renders `Reasoning: <v>` VERBATIM with no guard -- a wrong value is pasted into the system prompt rather than refused |
| qwen3.8 | `xhigh` | ladder is low/medium/xhigh; the template RAISES on anything else |
| Kimi K2.7 | *(empty)* | no ladder at all -- `reasoning_effort` is a K3-only field |

Set it EMPTY, never delete the line: `agent_driver.py` defaults a MISSING `AGENT_EFFORT` to
`xhigh`, which would raise on oss120b and silently mislabel Kimi.

## Problem lists

Regenerate whenever a SKILL.md changes -- a `-skills` list INLINES the packet and goes stale
silently. Lists are gitignored generated artifacts; the as-run copies for recorded experiments live
in `ICLR26Reproducibility/paper_artifacts/problems/`.

```bash
V=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python3
cd <repo root>
PYTHONPATH=$PWD $V containers/cluster/example-script/make_problems.py \
    --track loop_level_reasoning --language c --tag llr-focus40 \
    > containers/cluster/example-script/problems-llr6-c.jsonl
# skills leg: the same command plus --skills

# Kimi halves, disjoint and covering the tag:
head -20 problems-llr6-c.jsonl > problems-llr8kimi-c-a.jsonl
tail -20 problems-llr6-c.jsonl > problems-llr8kimi-c-b.jsonl
```

`hints-and-triggers.md` is NOT checked in -- `materialize_shared.sh` builds it at launch from
`containers/agent/hints.md` plus `skill-triggers.md`, so it always tracks the repo.

## Results and watching

`RUN_ROOT` in the .env decides where a run lands; point a new campaign at a new folder rather than
mixing waves. Scores are in `<RUN_ROOT>/<jobid>/judge/rank-*/hpcagent_bench*.db`.

```bash
squeue -u $USER -o "%.10i %.28j %.2t %.10M %.6D %R"
tail -f results/beverin-services-<jobid>.out
scontrol release <jobid>          # for launch failed requeued held
```

## Python

`/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314` (3.14.7, pyenv global). The repo is
MOUNTED, never pip-installed, so put it on `PYTHONPATH`. Rebuild with
`tools/session-038c86bd/rebuild_venv.sh`. Keep caches off HOME -- that quota is INODES, not bytes.
Note that `pre-commit`'s format hook needs the venv on `PATH` or it reports `missing formatter(s):
yapf` even when yapf is installed.
