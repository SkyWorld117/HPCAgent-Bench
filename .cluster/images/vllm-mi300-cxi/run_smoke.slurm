#!/usr/bin/env bash
#SBATCH --job-name=rccl-ofi-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=rccl-ofi-smoke-%j.out
# Replace these with the Opus project/vCluster values:
#SBATCH --account=REPLACE_ME
#SBATCH --partition=REPLACE_ME
# Add the Opus-specific GPU allocation directive if required.

set -euo pipefail
ulimit -c 0

: "${SCRATCH:?SCRATCH is not set}"
export VLLM_MI300_SQSH="${VLLM_MI300_SQSH:-$SCRATCH/ce-images/vllm-mi300-rocm6.sqsh}"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"

EDF="${EDF:-$SLURM_SUBMIT_DIR/vllm-mi300-rocm6.toml}"

# Only SCRATCH is mounted by the EDF, so stage the test there rather than
# assuming that SLURM_SUBMIT_DIR or all of HOME is visible in the container.
TEST_DIR="$SCRATCH/vllm-mi300-smoke-$SLURM_JOB_ID"
mkdir -p "$TEST_DIR"
cp "$SLURM_SUBMIT_DIR/rccl_smoke.py" "$TEST_DIR/rccl_smoke.py"
export RCCL_SMOKE_TEST="$TEST_DIR/rccl_smoke.py"

printf 'MASTER_ADDR=%s MASTER_PORT=%s\n' "$MASTER_ADDR" "$MASTER_PORT"
printf 'EDF=%s\nTEST=%s\n' "$EDF" "$RCCL_SMOKE_TEST"

# First validate the officially documented CXI discovery command, then execute
# a cross-node collective.  PyTorch calls the backend "nccl" on ROCm, while the
# implementation is RCCL underneath.
srun -ul --kill-on-bad-exit=1 --environment="$EDF" \
  bash -lc 'set -euo pipefail; fi_info -p cxi; python3 "$RCCL_SMOKE_TEST"'
