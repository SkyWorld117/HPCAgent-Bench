# Sample job submissions

**Two** modes, two samples. They are concrete examples, not templates: edit the node counts and
kernel selection at the top and submit. Each one only sets env knobs and `exec`s the real script in
`scripts/`, so a sample can never drift from the launcher it demonstrates.

There are two modes because there are two *deployments*, distinguished by what a rank number means.

| | `agentic_container.sbatch` | `deterministic_kernels_to_ranks.sbatch` |
|---|---|---|
| a rank is a | **role** (inference / judge / driver) | **kernel shard** |
| script | `scripts/submit_launch.sbatch` | `scripts/submit_deterministic.sbatch` |
| optimizer | an LLM agent | numpy / polly / dace_cpu / … |
| inference | vLLM endpoints on their own nodes | none |
| judge | dedicated judge node(s) | none |
| container | optional (`EDF=`) | no |
| result of a rerun | may differ (sampling) | identical artifact |

A **deterministic optimizer that still wants to be judged** is not a third mode — it is the
role-placed launcher with nothing to serve. `INFERENCE_ENDPOINTS=0 OPTIMIZER_NODES=O JUDGE_NODES=J
sbatch -N $((O+J)) scripts/submit_launch.sbatch` swaps the inference ranks for optimizer ranks and
keeps the judge, the settle protocol and the teardown identical. Use the sample here instead when you
only want timings and no scoring.

**Why the second mode shards by kernel and not by framework.** Kernel cost spans orders of magnitude
while the framework list is short and fixed, so a framework-per-rank split leaves most ranks idle
behind the slowest column. Each rank takes `kernels[rank::nranks]` and runs *every* framework over
its own kernels. Round-robin rather than contiguous blocks because neighbours in the sorted name list
tend to be similar sizes.

## Results and the DB

Every rank writes its **own** `hpcagent_bench<rank>.db` in the repo directory. That is not a
workaround for SQLite's locking: WAL needs a `-shm` mapping that Lustre/NFS/GPFS do not provide, so
one shared file across ranks is not an option. The shards are persistent artifacts, never scratch, and
never on memory-backed storage (`recording.base_db_path` refuses a tmpfs path outright).

Merging is automatic — no step to forget:

* a reader (`plot`, `plot-dist`) calls `recording.ensure_aggregated`, which builds the aggregate if it
  is missing *or older than a shard*;
* `run-framework --summarize` merges as part of closing the run;
* `hpcagent-bench aggregate-db` forces it now, for archiving or copying one file off the cluster.

The aggregate is always rebuilt from scratch, so merging twice cannot double the rows.

## Submitting

    sbatch -A <account> samples/agentic_container.sbatch
    sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch

Both write under `results/`. The deterministic job's exit status is the merged failure count across
shards, so a shard whose kernels stopped compiling (or silently miscompiled) fails the job instead of
disappearing into one rank's log.
