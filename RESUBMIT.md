# Resubmitting the campaigns

Quick reference for putting every campaign arm back on the machine.
`SUBMITTING.md` is the full runbook -- traps, watching a run, cancelling, infrastructure
jobs. This page is only "how do I launch them all again".

Everything runs from `containers/cluster/example-script`.

## The arms

One submitter, `submit-llr4.sh`. `MODEL` names the family and therefore the env files
(`.env.llr4-<MODEL>-<lang>[-skills]`).

| MODEL | langs | arms | nodes/arm | all arms |
|---|---|---|---|---|
| `qwen30b` | c, cpp, fortran | off + skills | 10 | 6 |
| `oss120b` | c, cpp, fortran | off + skills | 8 | 6 |
| `kimi27code` | c, cpp, fortran | off + skills | 6 | 6 |
| `qwen3next` | c, cpp, fortran | off + skills | 10 | 6 |

24 arms total. The ceiling is **32 nodes at any instant**, so they never all run at once --
`MODEL=qwen30b ./submit-llr4.sh` alone is 60 nodes and will sit in the queue.

Every arm runs **80 agents** against its engine. That is not a throughput knob, it is what
keeps the vLLM decode batch above 32: an agent sits in tools or a compile most of its wall
clock, so 48 agents held a mean batch of only 20 (600514/600515). Lowering it starves the
engine rather than relieving it.

No `--account` on beverin. `root`, `a-g200` and `a-g34` are scheduling-identical there, so
the submitter passes `-A` only if you set `ACCOUNT=` yourself.

## Resubmit everything

One language at a time, both sides of each skills pair chained so the pair costs wall clock
instead of nodes. Each `for` body is 24 nodes at peak (10 + 8 + 6).

```bash
cd containers/cluster/example-script

for lang in c cpp fortran; do
    for model in qwen30b oss120b kimi27code; do
        off=$(LANGS="$lang" ARMS=off MODEL="$model" ./submit-llr4.sh \
                  | tee /dev/stderr | grep -oP 'Submitted batch job \K[0-9]+')
        [[ -n "$off" ]] || { echo "sbatch refused ${model}/${lang} -- stopping" >&2; break 2; }
        LANGS="$lang" ARMS=skills MODEL="$model" ./submit-llr4.sh --dependency="afterany:${off}"
    done
done

# qwen3next is a 4th 10-node family, so give it its own wave rather than adding it
# to the loop above -- 10 + 8 + 6 + 10 would be 34 nodes.
for lang in c cpp fortran; do
    off=$(LANGS="$lang" ARMS=off MODEL=qwen3next ./submit-llr4.sh \
              | grep -oP 'Submitted batch job \K[0-9]+')
    LANGS="$lang" ARMS=skills MODEL=qwen3next ./submit-llr4.sh --dependency="afterany:${off}"
done
```

That queues all 24. Slurm releases each wave as the previous drains; nothing here exceeds the
ceiling on its own.

To resubmit **one** arm:

```bash
LANGS=c ARMS=skills MODEL=kimi27code ./submit-llr4.sh
```

## Before you resubmit

- `afterany`, never `afterok` -- a died arm still leaves a usable judge DB and you want the
  pair either way.
- A dependency can only be added while the job is **PENDING**. After it starts,
  `scontrol update ... dependency=` answers *"Job is no longer pending execution"* and the two
  run concurrently.
- Arms are only comparable if the serve config is identical. Changing an `.env.llr4-*` file
  splits the A/B, and a **queued** arm reads the file when it starts, not when you submit it.
- Every arm serves on vLLM 0.27.1. Image patches (aiter, MORI) rewrite that image **in place**,
  so never let one land while arms are queued against it.

## Checking what came back

```bash
squeue -u $USER -o '%.8i %.26j %.2t %.10M %.3D'
sacct -u $USER -S <date> -X -o JobID,JobName%26,State,Elapsed,NNodes
```

Completion counts matter: an arm cut off by wall clock is `COMPLETED` but did not finish all
242 problems. Check the counts before treating an arm as done.
