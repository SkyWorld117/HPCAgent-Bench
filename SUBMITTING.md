# Submitting the llr4 campaign on Beverin

Every command below runs from `containers/cluster/example-script/`. The submitters derive
`--nodes` from the arm's own `.env` (`arm_nodes.sh`), so never pass `--nodes` yourself -- an
allocation that disagrees with `INFERENCE_NODES + AGENT_NODES + JUDGE_NODES` makes
`beverin.sbatch` exit 2 before the run starts.

```bash
cd containers/cluster/example-script
```

## Node budget

Hard ceiling: **32 nodes in flight**, agreed with the team sharing the machine. Builds may take
2 more (34 total).

| model | nodes per arm | why |
|---|---|---|
| qwen30b | 10 | 1 inference + 1 agent + 8 judge |
| oss120b | 10 | 1 inference + 1 agent + 8 judge |
| kimi27code | 6 | 4 inference (pp=4) + 1 agent + 1 judge |

Three arms at once is the practical shape: `10 + 10 + 6 = 26`, or `10 + 10 + 10 = 30`.

## Before you submit anything

1. **The skills packet is FROZEN INTO the problems file.** `make_problems.py` inlines the
   `SKILL.md` bodies at generation time; a running arm never re-reads the pages. Edit a page and
   the jsonl is stale until you regenerate it:

   ```bash
   for lang in c cpp fortran; do
       PYTHONHASHSEED=0 python3 make_problems.py \
           --track loop_level_reasoning --language "$lang" --skills \
           > "problems-llr4-${lang}-skills.jsonl"
   done
   ```

   The off arms read `problems-llr2-<lang>.jsonl`, which carries no skill text and does not need
   regenerating when a page changes.

2. **Check the pair is intact** -- 242 problems each, same kernels, same ids:

   ```bash
   wc -l problems-llr4-*-skills.jsonl problems-llr2-*.jsonl
   ```

3. **Confirm nothing is already running:**

   ```bash
   squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D"
   ```

## The submitters

Three scripts, one per model. Each takes two knobs and passes any extra argument through to
`sbatch` verbatim.

| knob | values | default |
|---|---|---|
| `LANGS` | `c` `cpp` `fortran`, space separated | all three |
| `ARMS` | `off` `skills` | both |
| `TIME` | wall clock | 20 h (qwen/oss), 24 h (kimi) |
| `ACCOUNT` | Slurm account | `a-g200` |

```bash
./submit-llr4-qwen30b.sh          # 6 arms x 10 nodes -- WAY over budget, always narrow it
```

So in practice always name the slice:

```bash
LANGS=c ARMS=off    ./submit-llr4-qwen30b.sh      # 1 arm, 10 nodes
LANGS=c ARMS=skills ./submit-llr4-oss120b.sh      # 1 arm, 10 nodes
LANGS=c             ./submit-llr4-kimi27code.sh   # 2 arms (off + skills), 6 nodes each
```

## A full wave, inside the budget

C first, both sides of the skills pair, all three models. 26 nodes:

```bash
LANGS=c ARMS=off    ./submit-llr4-qwen30b.sh
LANGS=c ARMS=off    ./submit-llr4-oss120b.sh
LANGS=c ARMS=off    ./submit-llr4-kimi27code.sh
```

Then chain the skills side behind it rather than doubling the footprint -- see below.

## Chaining with a dependency

The skills arm is the same 242 kernels as the off arm, so running them back to back costs wall
clock but not nodes. `afterany` (not `afterok`) because an arm that dies still leaves a usable
judge DB, and you want the pair either way.

```bash
# 1. submit the off side and capture its job id.  `tee /dev/stderr` keeps the submitter's own
#    "submitted <arm>" confirmation visible -- a bare $( ) would swallow it with the id.
off_id=$(LANGS=c ARMS=off ./submit-llr4-qwen30b.sh | tee /dev/stderr \
             | grep -oP 'Submitted batch job \K[0-9]+')
echo "off arm: ${off_id}"

# 2. submit the skills side held behind it
LANGS=c ARMS=skills ./submit-llr4-qwen30b.sh --dependency="afterany:${off_id}"
```

The extra `--dependency=...` lands on the `sbatch` line unchanged, which is what `"$@"` in the
submitter is for.

For all three models chained in pairs (26 nodes at any instant, six arms total):

```bash
for model in qwen30b oss120b kimi27code; do
    off_id=$(LANGS=c ARMS=off "./submit-llr4-${model}.sh" | grep -oP 'Submitted batch job \K[0-9]+')
    [[ -n "${off_id}" ]] || { echo "no job id from ${model} -- sbatch refused, stopping" >&2; break; }
    LANGS=c ARMS=skills "./submit-llr4-${model}.sh" --dependency="afterany:${off_id}"
done
```

**A dependency can only be added while the job is still PENDING.** Once it starts,
`scontrol update jobid=<id> dependency=afterany:<other>` answers *"Job is no longer pending
execution"* and the two run concurrently.

## Later waves: cpp and fortran

Same commands with `LANGS` changed. Run them only once the C wave has drained, so the
32-node ceiling holds:

```bash
LANGS=cpp     ARMS=off ./submit-llr4-qwen30b.sh
LANGS=fortran ARMS=off ./submit-llr4-qwen30b.sh
```

## The regression smoke

Narrow gate: only the 69 kernels the skills arms previously got wrong. Cheap, and the right
thing to run before committing a full wave to a freshly edited packet.

```bash
MODELS="oss120b qwen30b" LANGS="c cpp fortran" ./submit-regress-smoke.sh
```

It needs `problems-regress-<lang>-skills.jsonl`; the script prints the exact regeneration
command if one is missing.

## Watching a run

```bash
squeue -u "$USER" -o "%.10i %.30j %.9T %.10M %.5D %R"
sacct -j <jobid> -o JobID,JobName%30,State,Elapsed,ExitCode --parsable2

# the job's own logs
tail -f containers/cluster/example-script/results/beverin-services-<jobid>.err

# is the engine actually decoding?  zero of these after requests arrive = a wedged engine
grep -c 'Avg generation throughput' results/beverin-services-<jobid>.out

# outcome counts, live, from the per-rank judge shards
python3 - <<'PY'
import glob, sqlite3, collections
counts = collections.Counter()
for shard in glob.glob('<RUN_ROOT>/<jobid>/judge/rank-*/*.db'):
    con = sqlite3.connect(f'file:{shard}?mode=ro', uri=True)
    for route, status, n in con.execute('select route, status, count(*) from calls group by route, status'):
        counts[(route, status)] += n
    con.close()
for key in sorted(counts):
    print(key, counts[key])
PY
```

`RUN_ROOT` is `$SCRATCH/hpcagent-bench-runs` (see the arm's `.env`).

## Cancelling

```bash
scancel <jobid> [<jobid> ...]
scancel -u "$USER" --name=llr4-qwen30b-c     # by arm name, since every arm is --job-name'd
```

Judge shards written before the cancel survive under `<RUN_ROOT>/<jobid>/judge/`, so a cancelled
arm still carries partial results.

## Known traps

- **kimi27code is `INFERENCE_MODE=pp`** (4 nodes, one engine). The other two are
  `INFERENCE_MODE=replicas` (one engine per node behind LiteLLM). Only the pp topology has the
  pipeline process group, which is why only kimi has ever wedged on it.
- **An arm that logs requests but zero `Avg generation throughput` is wedged, not slow.** It will
  burn its whole wall clock. Kill it.
- **Never edit a file a running arm reads.** The problems jsonl is snapshotted at submit time, but
  `run_cluster.sh` and the repo tree are read live.
- **Regenerate the problems files after ANY skill page edit**, or the arm measures the old pages
  while the tree shows the new ones.
