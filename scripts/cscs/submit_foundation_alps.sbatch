#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# FOUNDATION TRACK on CSCS Alps (GH200, aarch64), under the Alps Container Engine.
#
# This is scripts/submit_deterministic.sbatch's Alps sibling: same rank -> kernel shard, same
# deterministic optimizers, but every step carries the container-engine flag (`--environment=$EDF`,
# read from container_backends.txt) because on Alps a step without it runs OUTSIDE the image.
# Deterministic means no vLLM and no judge: a deterministic optimizer emits the same artifact
# every run, so there is nothing to serve and nothing to score. A judged run is a different
# deployment (scripts/submit_launch.sbatch).
#
#   EDF=$SCRATCH/foundation.toml sbatch -A <account> -N 8 scripts/cscs/submit_foundation_alps.sbatch
#
# `-A <account>` is mandatory on Alps (or export SBATCH_ACCOUNT first).
#
# IMAGE SETUP. None is required: put a registry reference in the EDF's `image` field and the
# container engine pulls it, converts it to SquashFS and caches it. Pre-importing only skips
# that per-job pull:
#
#   enroot import -o $SCRATCH/ce-images/hpcagent_bench-aarch64.sqsh docker://<registry>#hpcagent_bench:cpu
#
# GH200 nodes are aarch64. One tag serves both architectures when it is a multi-arch manifest
# list -- build it once on a build box, never on the login node:
#
#   docker buildx build --platform linux/amd64,linux/arm64 --push \
#          -f containers/hpcagent_bench.Dockerfile --build-arg HW=cpu -t <registry>/hpcagent_bench:cpu .
#
# An amd64-only image will not run on a GH200 node, and the failure arrives as an exec-format
# error inside the first step rather than at submission.
#
# Overridable: FRAMEWORKS, BENCH, PRESET, REPEAT, TIMEOUT, DATATYPE.
#
#   BENCH=foundation@lvl2 PRESET=M FRAMEWORKS=dace_cpu,polly sbatch ... this script
#
# UNVERIFIED AGAINST THE SITE'S OWN SUBMISSION SCRIPTS. The Alps specifics here come from this
# repo's own notes (scripts/cscs/env.toml.example, docs/LAUNCH.md), which state that node
# allocation and srun submission are owned externally. The partition name, the account, the
# scratch layout and the reservation policy are the three things most likely to need a local
# edit before this runs as-is.

#SBATCH --job-name=hpcagent_bench-foundation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive                 # whole node -> full-core timing, no co-runners
#SBATCH --time=04:00:00
#SBATCH --output=results/foundation-alps-%j.out

set -euo pipefail

FRAMEWORKS="${FRAMEWORKS:-${FRAMEWORK:-dace_cpu}}"
BENCH="${BENCH:-foundation}"
PRESET="${PRESET:-L}"
REPEAT="${REPEAT:-10}"
TIMEOUT="${TIMEOUT:-200}"
DATATYPE="${DATATYPE:-}"
RANKS="${SLURM_JOB_NUM_NODES:-1}"

REPO="${HPCAGENT_BENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# The EDF is what puts a step inside the image. Without it the step silently runs on the bare
# node, where the toolchain and the pinned dace are absent -- so refuse now, by name, rather
# than let the whole allocation produce a log full of import errors.
if [[ -z "${EDF:-}" ]]; then
    echo "EDF is unset: point it at an Alps container-engine .toml" >&2
    echo "  cp scripts/cscs/foundation.toml.example \$SCRATCH/foundation.toml   # then edit 'image'" >&2
    echo "  EDF=\$SCRATCH/foundation.toml sbatch -A <account> $0" >&2
    exit 2
fi
if [[ ! -f "${EDF}" ]]; then
    echo "EDF=${EDF} does not exist" >&2
    exit 2
fi
export HPCAGENT_BENCH_REPO="${REPO}"   # the EDF interpolates this for its bind mount and workdir

# The flag that selects the container is read from the ONE spelling file the Python factory and
# the local bash launcher also read (hpcagent_bench/container_backends.txt, row `ce.srun_flag`),
# so Alps cannot drift from the rest of the harness on how CE is invoked.
CE_FLAG="$(sed -n 's/^ce\.srun_flag=//p' "${REPO}/hpcagent_bench/container_backends.txt")"
if [[ -z "${CE_FLAG}" ]]; then
    echo "container_backends.txt declares no ce.srun_flag; cannot select the Alps container" >&2
    exit 2
fi
CE=("${CE_FLAG}=${EDF}")

# Only deterministic optimizers belong in this job: an agent column needs the inference and judge
# roles this allocation does not have.
IFS=',' read -r -a FRAMEWORK_LIST <<< "${FRAMEWORKS}"
for fw in "${FRAMEWORK_LIST[@]}"; do
    case "${fw}" in
        numpy|polly|pluto|cc|cc_autopar|llvm|fortran|fortran_autopar|flang|dace_cpu) ;;
        *) echo "FRAMEWORKS entry '${fw}' is not a deterministic optimizer" >&2; exit 1 ;;
    esac
done

# DaCe writes one .so per SDFG variant per kernel. Keep that off the parallel FS, which turns
# thousands of small compiler writes into the job's bottleneck, and off /tmp, which is tmpfs on
# these nodes (the build would then compete with the run for RAM).
export DACE_default_build_folder="${DACE_BUILD_ROOT:-${SCRATCH:-${REPO}}}/dacecache-${SLURM_JOB_ID:-local}"
mkdir -p "${DACE_default_build_folder}"

# Results must outlive the allocation, so they live in the REPO, never node-local or memory-backed
# (recording.base_db_path refuses a tmpfs path outright). Each rank writes its OWN shard: WAL needs
# a -shm mapping no parallel FS provides, so one shared file across ranks is not an option.
RUNDIR="${REPO}/results/foundation-alps-${SLURM_JOB_ID:-local}"
mkdir -p "${RUNDIR}"

# The CPU autoopt pipeline is the fork's canonicalize + finalize pair, which exists only on
# spcl/dace@extended. Check it INSIDE the image -- the login node's python is not the one that
# will run the sweep -- and fail here, where the message names the cause, rather than hundreds of
# kernels deep with every dace column silently scored on the weaker auto_optimize.
if [[ "${FRAMEWORKS}" == *dace_* ]]; then
    srun "${CE[@]}" --nodes=1 --ntasks=1 \
        python -c 'from dace.transformation.passes.canonicalize.pipeline import canonicalize' || {
        echo "the image's dace has no canonicalize pipeline: HPCAgent-Bench needs spcl/dace@extended" >&2
        exit 1
    }
fi

# Thread counts come from flags.cpu_env, the same source the harness documents, so this script
# cannot drift from it. Resolved inside the image, because that is where the core count that
# matters is visible. Sound only because every rank is --exclusive on its own node, so the srun
# steps below inherit the identical whole-node affinity.
eval "$(srun "${CE[@]}" --nodes=1 --ntasks=1 python -c "
from hpcagent_bench import flags
from hpcagent_bench.flags import Mode
for name, value in flags.cpu_env(Mode.MULTI_CORE).items():
    print(f'export {name}={value}')
")"

echo "ranks=${RANKS} frameworks=${FRAMEWORKS} bench=${BENCH} preset=${PRESET} cores=${OMP_NUM_THREADS}"
echo "edf=${EDF} rundir=${RUNDIR}"

# Sharded by KERNEL, not by framework: kernel cost spans orders of magnitude while the framework
# list is short and fixed, so a framework-per-rank split parks most ranks behind the slowest
# column. Round-robin rather than contiguous blocks because neighbours in the sorted name list
# tend to be similar sizes. --cpu-bind=none so each task keeps its whole node's cores;
# run-framework forks EACH kernel, which is what contains a crashing or hanging kernel.
srun "${CE[@]}" --ntasks="${RANKS}" --ntasks-per-node=1 --cpu-bind=none bash -c '
set -euo pipefail
rank="${SLURM_PROCID:-0}"
IFS="," read -r -a fws <<< "'"${FRAMEWORKS}"'"
for fw in "${fws[@]}"; do
    echo "[rank ${rank}/'"${RANKS}"'] framework=${fw} cores=${OMP_NUM_THREADS}"
    hpcagent-bench run-framework \
        --benchmark "'"${BENCH}"'" --framework "${fw}" --preset "'"${PRESET}"'" \
        --repeat "'"${REPEAT}"'" --timeout "'"${TIMEOUT}"'" \
        --shard "${rank}/'"${RANKS}"'" \
        --csv "'"${RUNDIR}"'/shard-${rank}.csv" \
        --validate --ignore-errors '"${DATATYPE:+--datatype ${DATATYPE}}"'
done
'

# The merged failure count IS the job status, so a shard that stopped compiling fails the job
# instead of hiding in one rank's log.
echo "=== merged ==="
srun "${CE[@]}" --nodes=1 --ntasks=1 \
    hpcagent-bench run-framework --summarize "${RUNDIR}"/shard-*.csv
