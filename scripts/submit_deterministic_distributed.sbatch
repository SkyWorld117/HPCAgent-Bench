#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# DETERMINISTIC optimizers over a kernel selection, spread across X nodes. The simple mode:
#
#   X ranks -> the kernel selection is round-robin sharded across the ranks, and each rank runs
#   EVERY framework in FRAMEWORKS over its own kernels.
#
# That is the whole placement rule. No vLLM, no judge, no agent -- a deterministic optimizer emits
# the same artifact every run, so there is no model in the loop to serve and nothing to score with
# an LLM. For the agent track use scripts/launch.sbatch (or scripts/cscs/launch_alps.sbatch on
# Alps), which place inference and judge roles this job has no allocation for.
#
# Sharding by KERNEL rather than by framework is deliberate: kernel cost spans orders of magnitude
# while the framework list is short and fixed, so a framework-per-rank split would leave most ranks
# idle behind the slowest column. Round-robin rather than contiguous blocks because neighbours in
# the sorted name list tend to be similar sizes.
#
#   NODES=8 sbatch scripts/submit_deterministic_distributed.sbatch
#   NODES=4 FRAMEWORKS="numpy,dace_cpu" BENCH=foundation@lvl2 PRESET=M \
#       sbatch scripts/submit_deterministic_distributed.sbatch
#
# Overridable: NODES, FRAMEWORKS, BENCH, PRESET, REPEAT, TIMEOUT, DATATYPE.
#
# Each rank writes its OWN hpcagent_bench.db and its own CSV -- SQLite locking is not safe to share
# across ranks on a parallel FS. Rank 0 merges the CSVs at the end and the job's exit status is the
# merged failure count, so a shard whose kernels stopped compiling (or silently miscompiled) fails
# the job instead of disappearing into a per-rank log.

#SBATCH --job-name=hpcagent_bench-deterministic-dist
#SBATCH --ntasks-per-node=1         # one rank per node (rank -> kernel shard)
#SBATCH --exclusive                 # whole node -> full-core timing, no co-runners
#SBATCH --time=04:00:00
#SBATCH --output=results/deterministic-dist-%j.out

set -euo pipefail

NODES="${NODES:-4}"
FRAMEWORKS="${FRAMEWORKS:-numpy,polly,dace_cpu}"
BENCH="${BENCH:-foundation}"
PRESET="${PRESET:-L}"
REPEAT="${REPEAT:-10}"
TIMEOUT="${TIMEOUT:-200}"
DATATYPE="${DATATYPE:-}"

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

# Deterministic = same artifact every run, no sampling and no model in the loop. Refuse an agent
# column here rather than silently scoring it without the inference/judge roles it needs.
IFS=',' read -r -a FRAMEWORK_LIST <<< "${FRAMEWORKS}"
for fw in "${FRAMEWORK_LIST[@]}"; do
    case "${fw}" in
        numpy|polly|pluto|cc|cc_autopar|llvm|fortran|fortran_autopar|flang|dace_cpu) ;;
        *) echo "FRAMEWORK=${fw} is not a deterministic optimizer" >&2; exit 1 ;;
    esac
done

# The CPU ``autoopt`` pipeline is the fork's canonicalize + finalize pair, which only exists on
# spcl/dace@extended. Fail here, where the message names the cause, rather than 220 kernels deep
# with every dace column silently scored on the weaker auto_optimize.
if [[ ",${FRAMEWORKS}," == *,dace_* ]]; then
    python -c 'from dace.transformation.passes.canonicalize.pipeline import canonicalize' || {
        echo "installed dace has no canonicalize pipeline: HPCAgent-Bench needs spcl/dace@extended" >&2
        exit 1
    }
fi

RUNROOT="${REPO}/results/deterministic-dist-${SLURM_JOB_ID:-local}"
mkdir -p "${RUNROOT}"

RANKS="${SLURM_JOB_NUM_NODES:-${NODES}}"
echo "ranks=${RANKS} frameworks=${FRAMEWORKS} bench=${BENCH} preset=${PRESET} out=${RUNROOT}"

# One rank per node. --cpu-bind=none so each task keeps its whole node's cores; run-framework forks
# EACH kernel, which is what contains a crashing or hanging kernel to its own child.
srun --ntasks="${RANKS}" --ntasks-per-node=1 --cpu-bind=none bash -c '
set -euo pipefail
rank="${SLURM_PROCID}"
ranks="'"${RANKS}"'"
rundir="'"${RUNROOT}"'/rank-${rank}"
mkdir -p "${rundir}"
cd "${rundir}"

# hwloc GL connects to the X11 socket during MPI/UCX probing and never returns, so a dace import --
# which pulls mpi4py transitively -- hangs before running a line.
export UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl

# DaCe builds one .so per SDFG variant per kernel. Keep that on node-local scratch: a shared
# parallel FS turns thousands of small compiler writes into the bottleneck, and /tmp is tmpfs on
# many nodes (the build would then compete with the run for RAM). Per-rank so ranks cannot collide.
export DACE_default_build_folder="${DACE_BUILD_ROOT:-${SCRATCH:-'"${REPO}"'}}/dacecache-${SLURM_JOB_ID:-local}-${rank}"
mkdir -p "${DACE_default_build_folder}"

# Thread counts come from flags.cpu_env, the same source the harness documents, so this cannot
# drift from it. Sound per-rank because the allocation is --exclusive --ntasks-per-node=1.
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
        --benchmark "'"${BENCH}"'" \
        --framework "${fw}" \
        --preset "'"${PRESET}"'" \
        --repeat "'"${REPEAT}"'" \
        --timeout "'"${TIMEOUT}"'" \
        --shard "${rank}/${ranks}" \
        --csv "'"${RUNROOT}"'/shard-${rank}.csv" \
        --validate --ignore-errors \
        '"${DATATYPE:+--datatype "${DATATYPE}"}"'
done
'

# Merge every shard and let the failure count BE the job status, so a shard that stopped compiling
# (or silently miscompiled) fails the job rather than hiding in one rank's log.
echo "=== merged ==="
hpcagent-bench run-framework --summarize "${RUNROOT}"/shard-*.csv
