#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# scripts/launch.sbatch for CSCS Alps (Daint/Clariden/Santis). Same rank->role placement, but
# every srun runs inside a Container Engine image selected by an EDF (see env.toml.example):
# Alps ships no site-wide vLLM module, so without --environment the launcher runs bare-metal
# and vllm is simply absent.
#
#   cp scripts/cscs/env.toml.example scripts/cscs/env.toml   # then fill in image + paths
#   sbatch -A <account> scripts/cscs/launch_alps.sbatch
#
# Allocation must be exactly  N = INFERENCE_ENDPOINTS*NODES_PER_VLLM + JUDGE_NODES; the preflight
# below refuses to start otherwise rather than hanging on a rank that never gets a role.

#SBATCH --job-name=hpcagent_bench-alps
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=results/launch-alps-%j.out

set -euo pipefail

HPCAGENT_BENCH_REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EDF="${HPCAGENT_BENCH_EDF:-${HPCAGENT_BENCH_REPO}/scripts/cscs/env.toml}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
INFERENCE_ENDPOINTS="${INFERENCE_ENDPOINTS:-2}"
NODES_PER_VLLM="${NODES_PER_VLLM:-1}"
JUDGE_NODES="${JUDGE_NODES:-1}"
KERNELS="${KERNELS:-gemm,gesummv}"
PRESET="${PRESET:-S}"

export HPCAGENT_BENCH_REPO

need="$((INFERENCE_ENDPOINTS * NODES_PER_VLLM + JUDGE_NODES))"
if [[ "${SLURM_JOB_NUM_NODES}" != "${need}" ]]; then
    echo "allocation is ${SLURM_JOB_NUM_NODES} nodes, roles need ${need}" \
         "(${INFERENCE_ENDPOINTS}*${NODES_PER_VLLM} inference + ${JUDGE_NODES} judge)" >&2
    exit 1
fi
[[ -f "${EDF}" ]] || { echo "no EDF at ${EDF}; copy scripts/cscs/env.toml.example" >&2; exit 1; }

# Compute nodes are GH200 -> aarch64. An x86_64 image loads and then dies on the first exec, deep
# inside the launcher, so check the SquashFS arch here where the message is still readable.
image="$(sed -n 's/^[[:space:]]*image[[:space:]]*=[[:space:]]*"\(.*\)"/\1/p' "${EDF}" | head -1)"
image="$(eval echo "${image}")"
if [[ -f "${image}" ]]; then
    file -b "${image}" | grep -qi squashfs || echo "warning: ${image} is not a SquashFS" >&2
elif [[ "${image}" != *:* ]]; then
    echo "EDF image ${image} is neither a readable file nor an OCI reference" >&2
    exit 1
fi

mkdir -p results

# --mpi=pmix so mpi4py's COMM_WORLD spans the allocation; one rank per node, rank decides role.
srun --environment="${EDF}" --mpi=pmix \
    --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 \
    hpcagent-bench launch openai \
        --model "${MODEL}" \
        --inference-endpoints "${INFERENCE_ENDPOINTS}" \
        --nodes-per-vllm "${NODES_PER_VLLM}" \
        --judge-nodes "${JUDGE_NODES}" \
        --gpus-per-node 4 \
        --kernels "${KERNELS}" \
        --baseline auto --preset "${PRESET}" \
        --output "results/agent_launch-${SLURM_JOB_ID}.jsonl"
