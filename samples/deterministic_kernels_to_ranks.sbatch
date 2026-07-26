#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SAMPLE -- mode 2 of 2: DETERMINISTIC optimizers, kernels distributed to ranks.
#
#   8 ranks -> the foundation selection is round-robin sharded across them, and EACH rank runs
#   every framework in FRAMEWORKS over its own kernels.
#
# No vLLM, no judge, no agent, no container: a deterministic optimizer emits the same artifact
# every run, so there is nothing to serve and nothing for an LLM to score. For the agentic track
# see samples/agentic_container.sbatch.
#
#   sbatch -A <account> samples/deterministic_kernels_to_ranks.sbatch
#
# This is a filled-in example. The parameterised version is
# scripts/submit_deterministic_distributed.sbatch -- prefer that one for sweeps.

#SBATCH --job-name=hpcagent-det-sample
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1         # one rank per node; rank -> kernel shard
#SBATCH --exclusive                 # whole node -> full-core timing, no co-runners
#SBATCH --time=04:00:00
#SBATCH --output=results/sample-deterministic-%j.out

set -euo pipefail

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

FRAMEWORKS="numpy,polly,dace_cpu"
BENCH="foundation"
PRESET="L"
RANKS="${SLURM_JOB_NUM_NODES:-8}"
RUNROOT="${REPO}/results/sample-deterministic-${SLURM_JOB_ID:-local}"
mkdir -p "${RUNROOT}"

# The CPU ``autoopt`` pipeline is the fork's canonicalize + finalize pair, which only exists on
# spcl/dace@extended. Fail here, where the message names the cause, rather than 220 kernels deep
# with the dace column silently scored on the weaker auto_optimize.
python -c 'from dace.transformation.passes.canonicalize.pipeline import canonicalize' || {
    echo "installed dace has no canonicalize pipeline: HPCAgent-Bench needs spcl/dace@extended" >&2
    exit 1
}

echo "ranks=${RANKS} frameworks=${FRAMEWORKS} bench=${BENCH} preset=${PRESET} out=${RUNROOT}"

srun --ntasks="${RANKS}" --ntasks-per-node=1 --cpu-bind=none bash -c '
set -euo pipefail
rank="${SLURM_PROCID}"
ranks="'"${RANKS}"'"
mkdir -p "'"${RUNROOT}"'/rank-${rank}" && cd "'"${RUNROOT}"'/rank-${rank}"

# hwloc GL connects to the X11 socket during MPI/UCX probing and never returns, so a dace import --
# which pulls mpi4py transitively -- hangs before running a line.
export UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl

# One .so per SDFG variant per kernel: node-local scratch, per rank. A shared parallel FS turns
# thousands of small compiler writes into the bottleneck, and /tmp is tmpfs on many nodes.
export DACE_default_build_folder="${SCRATCH:-'"${REPO}"'}/dacecache-${SLURM_JOB_ID:-local}-${rank}"
mkdir -p "${DACE_default_build_folder}"

# Thread counts from flags.cpu_env -- the same source the harness documents, so this cannot drift.
eval "$(python -c "
from hpcagent_bench import flags
from hpcagent_bench.flags import Mode
for name, value in flags.cpu_env(Mode.MULTI_CORE).items():
    print(f\"export {name}={value}\")
")"

# THIS rank takes kernels [rank::ranks] and runs every framework over them, in order.
IFS="," read -r -a fws <<< "'"${FRAMEWORKS}"'"
for fw in "${fws[@]}"; do
    echo "[rank ${rank}/${ranks}] framework=${fw} cores=${OMP_NUM_THREADS}"
    hpcagent-bench run-framework \
        --benchmark "'"${BENCH}"'" --framework "${fw}" --preset "'"${PRESET}"'" \
        --repeat 10 --timeout 200 \
        --shard "${rank}/${ranks}" \
        --csv "'"${RUNROOT}"'/shard-${rank}.csv" \
        --validate --ignore-errors
done
'

# Merge every shard; the failure count IS the job status, so a shard that stopped compiling fails
# the job rather than hiding in one rank's log.
echo "=== merged ==="
hpcagent-bench run-framework --summarize "${RUNROOT}"/shard-*.csv
