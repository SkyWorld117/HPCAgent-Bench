#!/usr/bin/env bash
# Tune the fused_moe Triton config kimi K2.7 is missing on MI300A.
#
# 595040/595049 both decode at ~90-100 tok/s aggregate at 64 concurrent and both warn
# "Using default MoE config ... E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json
# not found". aiter does not cover the int4 W4A16 expert path (595049 proved it: warning still
# printed, 89.8 tok/s), so the fix is vLLM's own tuner: benchmark_moe.py --tune derives the expert
# shape from the model config and emits exactly that JSON. Serving picks it up via
# VLLM_TUNED_CONFIG_FOLDER -- no image change.
#
# The tuner parallelizes the search over every GPU ray sees, hence a full node.
#SBATCH --job-name=tune-moe-int4
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=logs/tune-moe-int4-%j.out
#SBATCH --error=logs/tune-moe-int4-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export TP_SIZE="${TP_SIZE:-4}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.23.0-pytorch211-ofi}"
EXPECTED_JSON="${EXPECTED_JSON:-E=384,N=512,device_name=AMD_Instinct_MI300A,dtype=int4_w4a16.json}"

mkdir -p "${RUN_DIR}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" bash -c '
    set -Eeuo pipefail
    export TOKENIZERS_PARALLELISM=false

    MODEL_PATH="$(python3 - <<PY | tail -n 1
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True))
PY
)"
    test -d "${MODEL_PATH}"
    echo "model: ${MODEL_PATH}"

    cd "${RUN_DIR}"
    python3 /opt/vllm-src/benchmarks/kernels/benchmark_moe.py \
        --model "${MODEL_PATH}" \
        --tp-size "${TP_SIZE}" \
        --dtype int4_w4a16 \
        --trust-remote-code \
        --tune \
        --save-dir "${RUN_DIR}"
'

echo "===== emitted configs ====="
ls -l "${RUN_DIR}"/*.json
test -s "${RUN_DIR}/${EXPECTED_JSON}"

echo "===== TUNE-MOE PASSED: ${RUN_DIR}/${EXPECTED_JSON} ====="
