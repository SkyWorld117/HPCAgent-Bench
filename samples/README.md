# Sample job submissions

Two ready-to-submit jobs, one per mode. They are **concrete examples**, not templates: edit the
account, node counts and kernel selection at the top and submit. The generic, fully-parameterised
versions live in `scripts/`.

The two modes place different roles. Pick by what the run needs, not by node count.

| | `agentic_container.sbatch` | `deterministic_kernels_to_ranks.sbatch` |
|---|---|---|
| optimizer | an LLM agent | a deterministic optimizer (numpy / polly / dace_cpu / ...) |
| inference | vLLM endpoints on their own nodes | none |
| judge | dedicated judge node(s) | none |
| container | yes (Container Engine EDF) | no |
| how nodes are used | rank -> ROLE | rank -> KERNEL SHARD |
| result of a rerun | may differ (sampling) | identical artifact |

**Why the second mode shards by kernel and not by framework.** Kernel cost spans orders of
magnitude while the framework list is short and fixed, so a framework-per-rank split leaves most
ranks idle behind the slowest column. Each rank takes `kernels[rank::nranks]` and runs *every*
framework over its own kernels. Round-robin rather than contiguous blocks because neighbours in the
sorted name list tend to be similar sizes.

There is a third mode in `scripts/submit_traditional_distributed.sbatch`: a deterministic optimizer
that still wants the agent track's optimizer-node + judge-node split (`--inference-endpoints 0`, no
vLLM). Use it when the run must be *scored by the judge*; use the sample here when you only want
timings.

## Submitting

    sbatch -A <account> samples/agentic_container.sbatch
    sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch

Both write under `results/`. The deterministic job's exit status is the merged failure count across
shards, so a shard whose kernels stopped compiling (or silently miscompiled) fails the job instead
of disappearing into one rank's log.
