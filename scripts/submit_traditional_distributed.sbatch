#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# TRADITIONAL track, distributed: the SAME optimizer-node + judge-node split the agent track uses,
# but with NO vLLM -- a deterministic optimizer emits the same artifact every run, so there is no
# model to serve. `hpcagent-bench launch --inference-endpoints 0` selects it, and the allocation is
#
#   ranks [0, O)      -- optimizer: runs the optimizer itself (rank 0 is ALSO the driver)
#   ranks [O, O + J)  -- judge: one `hpcagent-bench serve` each
#
# so the allocation must be exactly  N = optimizer_nodes + judge_nodes.
#
#   sbatch scripts/submit_traditional_distributed.sbatch
#   NODES=9 OPTIMIZER_NODES=8 JUDGE_NODES=1 sbatch scripts/submit_traditional_distributed.sbatch
#
# THREE DISTINCT SUBMISSION MODES -- do not merge them, they place different roles:
#   * scripts/launch.sbatch                        container + agentic: vLLM endpoints + judge.
#   * THIS script                                  traditional: optimizer nodes + judge, no vLLM.
#   * scripts/submit_deterministic_distributed.sbatch
#                                                  deterministic: kernels round-robin to X ranks,
#                                                  every framework per rank, no judge and no driver.
# Pick by what the run needs scored, not by node count.

#SBATCH --job-name=hpcagent_bench-traditional-dist
#SBATCH --ntasks-per-node=1       # one launcher rank per node (MPI rank -> role)
#SBATCH --exclusive               # whole node -> full-core timing, no co-runners
#SBATCH --time=04:00:00
#SBATCH --output=results/traditional-dist-%j.out

set -euo pipefail

OPTIMIZER_NODES="${OPTIMIZER_NODES:-2}"
JUDGE_NODES="${JUDGE_NODES:-1}"
AGENT="${AGENT:-stub}"
KERNELS="${KERNELS:-gemm,gesummv}"
PRESET="${PRESET:-S}"
BASELINE="${BASELINE:-auto}"

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

NEED=$((OPTIMIZER_NODES + JUDGE_NODES))
RANKS="${SLURM_JOB_NUM_NODES:-${NEED}}"
if [[ "${RANKS}" -ne "${NEED}" ]]; then
    echo "allocation has ${RANKS} nodes but optimizer(${OPTIMIZER_NODES}) + judge(${JUDGE_NODES}) = ${NEED};" \
         "submit with -N ${NEED}" >&2
    exit 1
fi

# hwloc GL connects to the X11 socket during MPI/UCX probing and never returns, so a dace import --
# which pulls mpi4py transitively -- hangs before running a line.
export UCX_VFS_ENABLE=n HWLOC_COMPONENTS=-gl

# DaCe builds one .so per SDFG variant per kernel: node-local scratch, per job. A shared parallel FS
# turns thousands of small compiler writes into the bottleneck, and /tmp is tmpfs on many nodes.
export DACE_default_build_folder="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/dacecache-${SLURM_JOB_ID:-local}"
mkdir -p "${DACE_default_build_folder}"

RUNDIR="${REPO}/results/traditional-dist-${SLURM_JOB_ID:-local}"
mkdir -p "${RUNDIR}"

echo "optimizer_nodes=${OPTIMIZER_NODES} judge_nodes=${JUDGE_NODES} agent=${AGENT} preset=${PRESET}"

# --mpi=pmix so mpi4py's COMM_WORLD spans the allocation; one rank per node, rank decides the role.
# --inference-endpoints 0 is what selects the traditional plan (no vLLM); --model is unused there.
srun --mpi=pmix --ntasks="${RANKS}" --ntasks-per-node=1 \
    hpcagent-bench launch "${AGENT}" \
        --inference-endpoints 0 \
        --optimizer-nodes "${OPTIMIZER_NODES}" \
        --judge-nodes "${JUDGE_NODES}" \
        --kernels "${KERNELS}" \
        --baseline "${BASELINE}" --preset "${PRESET}" \
        --output "${RUNDIR}/traditional_launch-${SLURM_JOB_ID:-local}.jsonl"
