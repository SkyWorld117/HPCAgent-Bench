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
| qwen3next | 10 | 1 inference + 1 agent + 8 judge |
| oss120b | 8 | 1 inference + 1 agent + 6 judge |
| kimi27code | 6 | 4 inference (pp=4) + 1 agent + 1 judge |

Three arms at once is the practical shape: `10 + 8 + 6 = 24`, or `10 + 10 + 8 = 28`.

Every arm carries **80 agents** (`AGENT_NODES=1`, `AGENTS_PER_NODE=80`). It is a batch-size
knob, not a throughput knob: agents are in tools or a compile most of their wall clock, so 48
of them held a mean vLLM batch of 20 and 40 split across 3 oss120b replicas held 1.4
(600514/600515/600516). A 384-expert MoE reads every expert weight per step regardless of how
few tokens ride along, so a small batch wastes the engine outright. Do not lower it, and do not
add replicas to a family without scaling agents with them.

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

One script, `MODEL` names the family. Each takes two knobs and passes any extra argument through to
`sbatch` verbatim.

| knob | values | default |
|---|---|---|
| `LANGS` | `c` `cpp` `fortran`, space separated | all three |
| `ARMS` | `off` `skills` | both |
| `TIME` | wall clock | 20 h (qwen/oss), 24 h (kimi) |
| `ACCOUNT` | Slurm account | unset -- **no `--account` on beverin** |

```bash
MODEL=qwen30b ./submit-llr4.sh          # 6 arms x 10 nodes -- WAY over budget, always narrow it
```

So in practice always name the slice:

```bash
LANGS=c ARMS=off    MODEL=qwen30b ./submit-llr4.sh      # 1 arm, 10 nodes
LANGS=c ARMS=skills MODEL=oss120b ./submit-llr4.sh      # 1 arm, 10 nodes
LANGS=c             MODEL=kimi27code ./submit-llr4.sh   # 2 arms (off + skills), 6 nodes each
```

## A full wave, inside the budget

C first, both sides of the skills pair, all three models. 24 nodes:

```bash
LANGS=c ARMS=off    MODEL=qwen30b ./submit-llr4.sh
LANGS=c ARMS=off    MODEL=oss120b ./submit-llr4.sh
LANGS=c ARMS=off    MODEL=kimi27code ./submit-llr4.sh
```

Then chain the skills side behind it rather than doubling the footprint -- see below.

## Chaining with a dependency

The skills arm is the same 242 kernels as the off arm, so running them back to back costs wall
clock but not nodes. `afterany` (not `afterok`) because an arm that dies still leaves a usable
judge DB, and you want the pair either way.

```bash
# 1. submit the off side and capture its job id.  `tee /dev/stderr` keeps the submitter's own
#    "submitted <arm>" confirmation visible -- a bare $( ) would swallow it with the id.
off_id=$(LANGS=c ARMS=off MODEL=qwen30b ./submit-llr4.sh | tee /dev/stderr \
             | grep -oP 'Submitted batch job \K[0-9]+')
echo "off arm: ${off_id}"

# 2. submit the skills side held behind it
LANGS=c ARMS=skills MODEL=qwen30b ./submit-llr4.sh --dependency="afterany:${off_id}"
```

The extra `--dependency=...` lands on the `sbatch` line unchanged, which is what `"$@"` in the
submitter is for.

For all three models chained in pairs (24 nodes at any instant, six arms total):

```bash
for model in qwen30b oss120b kimi27code; do
    off_id=$(LANGS=c ARMS=off MODEL="${model}" ./submit-llr4.sh | grep -oP 'Submitted batch job \K[0-9]+')
    [[ -n "${off_id}" ]] || { echo "no job id from ${model} -- sbatch refused, stopping" >&2; break; }
    LANGS=c ARMS=skills MODEL="${model}" ./submit-llr4.sh --dependency="afterany:${off_id}"
done
```

**A dependency can only be added while the job is still PENDING.** Once it starts,
`scontrol update jobid=<id> dependency=afterany:<other>` answers *"Job is no longer pending
execution"* and the two run concurrently.

## Later waves: cpp and fortran

Same commands with `LANGS` changed. Run them only once the C wave has drained, so the
32-node ceiling holds:

```bash
LANGS=cpp     ARMS=off MODEL=qwen30b ./submit-llr4.sh
LANGS=fortran ARMS=off MODEL=qwen30b ./submit-llr4.sh
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

## Infrastructure jobs (images, gates, weights)

These are not campaign arms. They take one to four nodes and are what you submit when the
question is "can the campaign move", not "how did the model score".

No `--account` on beverin. The user default association is `root`, and `root`, `a-g200` and
`a-g34` all carry QOS `normal` with no GrpTRES, MaxJobs, MaxSubmit, MaxTRES or priority set --
measured, and a 2-node accountless job allocated and ran (600940). Omitting the flag therefore
costs nothing in scheduling, and it stops jobs from silently splitting across two project
accounts depending on which command line was typed.

```bash
cd containers/cluster/ce-images

# Agent image. Compilers are pinned by MAJOR version only (gcc 16, LLVM 22) because the PPA
# serves 16.0.1, not a fixed point release; the build records what it actually resolved to in
# /usr/local/share/toolchain-provenance. Lands as ...-v5-candidate.sqsh, never over v4.
sbatch amd/build-agent-image.sbatch

# vLLM image, parameterised by version. The artefact name carries the version, so a new build
# lands BESIDE the one every measured arm ran on rather than over it.
VLLM_VERSION=0.27.1 sbatch \
    --export=ALL,VLLM_BUILD_ROOT=$PWD/inference \
    inference/build/build-vllm023-pt211.sbatch

# 2-node RCCL/CXI check. OPTARENA_REPO is REQUIRED -- Slurm spools the script, so it cannot
# find its own driver.
sbatch --export=ALL,OPTARENA_REPO=<repo root> \
    ../example-script/test-rccl-ofi-2node.sbatch

# 0.27.1 serving-surface gate: serve-arg parity, tool/reasoning parser choices, the tuned-MoE
# env var, and the internal API the pp collective split depends on. One node, ~2 minutes,
# no weights. Run it BEFORE spending a 4-node hour on a decode gate.
sbatch inference/gate-0271-serving-surface.sbatch

# aiter into a freshly built image. NOT optional on 0.27.1: its Kimi ViT patch-embed imports
# aiter unconditionally during the multimodal dummy profile_run, so an aiter-less 0.27.1 dies
# at startup before the API binds even on a text-only campaign (600649). Patches in place with
# a .before-aiter backup, so a decode gate must be re-run after it.
VLLM_VERSION=0.27.1 sbatch --export=ALL,VLLM_VERSION=0.27.1 \
    inference/build/add-aiter-pt211.sbatch

# aiter JIT cache, then the decode gate, both chained behind it. aiter ships no prebuilt .so,
# so an unprimed image builds module_aiter_core inside the serving job and can miss the API
# timeout outright (598021). Give each vLLM version its OWN AITER_JIT_DIR -- the pin is
# discovered from the image, so two versions sharing one cache is a silent kernel mismatch.
JIT=/iopsstor/scratch/cscs/$USER/aiter-jit-0271
sbatch --dependency=afterok:<aiter job> \
    --export=ALL,INFERENCE_EDF=rocm723-vllm-0.27.1-pytorch211-ofi,AITER_JIT_DIR=$JIT \
    inference/prebuild-aiter-jit.sbatch
sbatch --dependency=afterok:<prebuild job> \
    --export=ALL,INFERENCE_EDF=rocm723-vllm-0.27.1-pytorch211-ofi,AITER_JIT_DIR=$JIT \
    inference/smoke-kimi-eager-pg.sbatch
```

Promoting a candidate image is a rename, and only when nothing has the old one mounted:

```bash
squeue -u "$USER" -o "%.9i %.24j"          # must show no arm using the image you are replacing
mv $SCRATCH/ce-images/optarena-ce-amd-mi300-v5-candidate.sqsh \
   $SCRATCH/ce-images/optarena-ce-amd-mi300-v5.sqsh
```

### Weights: iopsstor and striping (already done -- verify, do not redo)

`run_cluster.sh` puts `HF_HOME` and `VLLM_CACHE_ROOT` on iopsstor (9.45 GB/s at 16 readers vs
capstor's 0.83) and sets a PFL default on the hub dir: narrow below 64 MiB, 16 OSTs at 4 MiB
above. Verified 2026-08-19 -- every large blob of all five models is striped 16. Re-check with:

```bash
lfs getstripe -c <blob> | head -1     # head, NOT tail: getstripe prints a trailing blank line
```

Only if that ever reports a narrow count, and only while NOTHING is reading the model:

```bash
lfs migrate -c 16 -S 4M <blob>
```

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
