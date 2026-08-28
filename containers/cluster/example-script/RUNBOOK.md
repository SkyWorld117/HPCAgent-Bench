# Running a campaign arm yourself

## One arm

```bash
cd containers/cluster/example-script
. ./arm_nodes.sh
sbatch --nodes="$(arm_nodes .env.llr8-oss120b-c)" --time=08:00:00 \
       --job-name=llr8w1-oss120b-c \
       --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.llr8-oss120b-c" beverin.sbatch
```

`CLUSTER_ENV_FILE` picks the arm, so arms run in parallel. Never `--account`.

## Both legs of one model

```bash
MODEL=oss120b LANGS=c ./submit-llr8.sh      # leg 1 = base, leg 2 = +skills
MODEL=oss120b LANGS=c LEGS=2 ./submit-llr8.sh   # skills leg only
```

## Node sizing

An arm is `INFERENCE_NODES + AGENT_NODES + JUDGE_NODES`; `arm_nodes` reads the
same .env the launcher does, so pass it rather than a literal.

| model | inference | agent | judge | total |
|-------|-----------|-------|-------|-------|
| kimi27sglang | 4 (TP4 x PP4, one endpoint) | 1 | 1 | 6 |
| oss120b      | 1 | 1 | 1 | 3 |

One judge NODE is 4 ranks = 4 concurrent grades
(`HPCAGENT_BENCH_JUDGE_GPUS_PER_NODE`). Both legs of an A/B must be sized
IDENTICALLY -- unequal judge capacity confounds the comparison.

Beverin: 24 h max walltime, no `--account`, max 36 nodes at a time by house rule.

## Agents

`AGENTS_PER_NODE` x `AGENT_NODES` is a ROLLING pool
(`ThreadPoolExecutor(max_workers=workers)`, one submit per problem): that many
start, and each one that finishes launches the next. It is not a barrier.

Kimi is 12; 40 kernels through 12 rolling agents at `AGENT_TIMEOUT_SECONDS`
36000 (10 h) does not fit a 24 h job, which is why C is split into 20-kernel
halves (`-a`, `-b`). oss120b is 40 agents for 40 kernels: one pass.

## Problem lists

Regenerate whenever a SKILL.md changes -- the `-skills` list INLINES the packet
and goes stale silently. Lists are gitignored (generated artifacts).

```bash
V=/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314/bin/python3
cd <repo root>
PYTHONPATH=$PWD $V containers/cluster/example-script/make_problems.py \
    --track loop_level_reasoning --language c --tag llr-focus40 \
    > containers/cluster/example-script/problems-llr6-c.jsonl
# skills leg: same plus --skills
```

Split for the Kimi halves:

```bash
head -20 problems-llr6-c.jsonl > problems-llr8kimi-c-a.jsonl
tail -20 problems-llr6-c.jsonl > problems-llr8kimi-c-b.jsonl
```

`hints-and-triggers.md` is NOT checked in: `materialize_shared.sh` builds it at
launch from `containers/agent/hints.md` + `skill-triggers.md`, so it always
tracks the repo.

## Where results land

`RUN_ROOT` in the .env. Wave 1 uses
`$SCRATCH/hpcagent-bench-runs/llr8w1-20260827`. Point a new campaign at a new
folder rather than mixing runs.

## Effort levels -- per model, not a shared dial

| model | value | why |
|-------|-------|-----|
| oss120b | `high` | ladder is low/medium/high; renders `Reasoning: <v>` VERBATIM with no guard, so a wrong value is silently pasted into the system prompt |
| qwen3.8 | `xhigh` | ladder is low/medium/xhigh; the template RAISES on anything else |
| kimi K2.7 | *(empty)* | no ladder at all; `reasoning_effort` is a K3-only field |

Set it EMPTY, never delete the line: `agent_driver.py` defaults a missing
`AGENT_EFFORT` to `xhigh`.

## Serving

Kimi/qwen38 on SGLang want `--attention-backend triton` WITH
`SGLANG_USE_AITER=1`. Unpinned, SGLang selects aiter, whose attention kernels
JIT-compile on the FIRST request behind a baton lock and outlive SGLang's hard
600 s warmup timeout -- the server is killed before serving a token.

Before any aiter run, sweep stale batons; a lock is orphaned if it is 0 bytes or
its node runs none of your jobs:

```bash
find $SCRATCH/.jit-cache -name 'lock_*' -exec sh -c \
  'echo "$1 -> $(tr -d "\0" < "$1" | tr "\n" " ")"' _ {} \;
```

Never run two jobs sharing a `.jit-cache/<image>` root at once -- serialize with
`--dependency=afterany:<jobid>`.

## Watching a run

```bash
squeue -u $USER -o "%.10i %.28j %.2t %.10M %.6D %R"
tail -f results/<jobid>.out
scontrol release <jobid>      # user_env_retrieval_failed_requeued_held (transient here)
```

## Python

`/capstor/scratch/cscs/ybudanaz/x86_64/venv-optarena-314` (3.14.7, from pyenv
global). The repo is MOUNTED, never pip-installed, so put it on `PYTHONPATH`.
Rebuild with `tools/session-038c86bd/rebuild_venv.sh`. Keep caches off HOME --
that quota is INODES, not bytes.
