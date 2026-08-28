#!/usr/bin/env bash
# WHERE is the engine blocked during the ~200s ramp?
#
# 599301 pinned the symptom: at concurrency 20 all prompt tokens are consumed in a single 10s
# window (prompt=32.8 tok/s for 20 x ~16 tokens), then decode does not start for ~200 seconds
# while all 20 requests sit resident and the engine loop stops logging entirely. Already
# excluded: KV pressure (1.1% used, zero preemptions), Triton JIT (2 compiles, both in warmup),
# CUDA graphs (FULL decode capture completes), request shape (completions / chat / chat+stream /
# chat+stream+tools all within 5%), and RCCL routing (NET/OFI, the fast fabric).
#
# Log reading cannot go further, so ask the processes. py-spy is not in the image but
# faulthandler is, and the eager-PG sitecustomize already runs in every worker: with
# DUMP_STACKS_ON_SIGUSR1 set it registers SIGUSR1 to dump every thread's Python stack to stderr,
# landing in that rank's vllm log. Trigger the stall, signal every worker while it is stuck.
#SBATCH --job-name=probe-kimi-stacks
#SBATCH --partition=mi300
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe-kimi-stacks-%j.out
#SBATCH --error=logs/probe-kimi-stacks-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-4}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
export READY_TIMEOUT="${READY_TIMEOUT:-5400}"
export STALL_CONCURRENCY="${STALL_CONCURRENCY:-20}"
export DUMP_AT="${DUMP_AT:-45 90 150}"
export DUMP_STACKS_ON_SIGUSR1=1

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PG_PATCH_REL="containers/cluster/ce-images/inference/external-eager-pg-patch"
for cand in "${PG_PATCH_DIR:-}" "${SUBMIT_DIR}/external-eager-pg-patch" "${SUBMIT_DIR}/${PG_PATCH_REL}"; do
    if [[ -n "${cand}" && -f "${cand}/sitecustomize.py" ]]; then PG_PATCH_DIR="${cand}"; break; fi
done
test -f "${PG_PATCH_DIR:-}/sitecustomize.py" || { echo "FAIL: no sitecustomize.py" >&2; exit 2; }
MOE_DIR="${PG_PATCH_DIR%/external-eager-pg-patch}/moe-configs"
[[ -d "${MOE_DIR}" ]] && export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${MOE_DIR}}"
export PG_PATCH_DIR

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd,)"
MASTER="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export NODELIST MASTER
DONE_FLAG="${RUN_DIR}/stacks.done"

mkdir -p "${RUN_DIR}"
stage() { printf '\n===== stage: %s =====\n' "$1"; }
printf 'job:   %s\nmodel: %s tp=%s pp=%s\nlogs:  %s\n' \
    "${SLURM_JOB_ID}" "${MODEL_REPO}" "${TP_SIZE}" "${PP_SIZE}" "${RUN_DIR}"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
cat >"${SERVE_SCRIPT}" <<'SERVE'
set -Eeuo pipefail
rank="${SLURM_PROCID:-0}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="${PG_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
unset VLLM_PORT

MODEL_PATH="$(python3 - <<'PY' | tail -n 1
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True))
PY
)"
test -d "${MODEL_PATH}"

ARGS=("${MODEL_PATH}" --served-model-name "${SERVED_MODEL}"
      --tensor-parallel-size "${TP_SIZE}" --pipeline-parallel-size "${PP_SIZE}"
      --distributed-executor-backend mp --nnodes "${SLURM_NNODES}" --node-rank "${rank}"
      --master-addr "${MASTER}" --master-port "${MASTER_PORT_BASE}"
      --distributed-timeout-seconds 7200 --load-format safetensors --generation-config vllm
      --trust-remote-code --max-model-len 65536 --gpu-memory-utilization 0.70 --max-num-seqs 128)
if (( rank == 0 )); then ARGS+=(--host 0.0.0.0 --port "${API_PORT}"); else ARGS+=(--headless); fi

vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-rank${rank}.log" 2>&1 &
SERVER_PID=$!
echo "rank ${rank} on $(hostname) pid ${SERVER_PID}"

while [[ ! -e "${RUN_DIR}/stacks.done" ]]; do
    for round in 1 2 3; do
        flag="${RUN_DIR}/dump-${round}.flag"
        stampfile="${RUN_DIR}/dumped-${round}-$(hostname)"
        if [[ -e "${flag}" && ! -e "${stampfile}" ]]; then
            pids="$(pgrep -u "$(id -u)" -f 'VLLM::|vllm serve|multiprocessing' || true)"
            echo "round ${round}: signalling $(echo ${pids} | wc -w) procs on $(hostname)"
            for p in ${pids}; do kill -USR1 "${p}" 2>/dev/null || true; done
            touch "${stampfile}"
        fi
    done
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "rank ${rank}: vLLM exited" >&2; exit 1; }
    sleep 2
done
SERVE
chmod +x "${SERVE_SCRIPT}"
rm -f "${DONE_FLAG}" "${RUN_DIR}"/dump-*.flag "${RUN_DIR}"/dumped-*

stage "serving tp=${TP_SIZE} pp=${PP_SIZE}"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODELIST}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!
cleanup() { touch "${DONE_FLAG}"; sleep 5; kill "${SERVER_STEP_PID}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

stage "waiting for the API"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 10 "http://${MASTER}:${API_PORT}/v1/models" -o /dev/null; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "FAIL: vLLM died" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "FAIL: not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
    sleep 15
done
echo "ready"

stage "warming at concurrency 1 (that path ramps in ~10s, so it is not what we are measuring)"
curl -fsS --max-time 900 "http://${MASTER}:${API_PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write one line of C.\",\"max_tokens\":64}" -o /dev/null || true
sleep 30

stage "triggering the stall with ${STALL_CONCURRENCY} concurrent requests"
for ((i = 0; i < STALL_CONCURRENCY; i++)); do
    curl -fsS --max-time 600 "http://${MASTER}:${API_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Explain loop tiling, variant ${i}.\",\"max_tokens\":4096,\"ignore_eos\":true}" \
        >/dev/null 2>&1 &
done
T0=$SECONDS
round=0
for at in ${DUMP_AT}; do
    round=$((round + 1))
    while (( SECONDS - T0 < at )); do sleep 1; done
    gen="$( { curl -fsS --max-time 10 "http://${MASTER}:${API_PORT}/metrics" || true; } |
        awk '/^vllm:generation_tokens_total/{print $2; exit}')"
    echo "t=+${at}s generation_tokens_total=${gen:-?} -- dumping stacks (round ${round})"
    touch "${RUN_DIR}/dump-${round}.flag"
    sleep 5
done
wait 2>/dev/null || true

touch "${DONE_FLAG}"
sleep 10
stage "blocked frames per rank"
for f in "${RUN_DIR}"/vllm-rank*.log; do
    echo "--- $(basename "${f}") ---"
    grep -A40 "Current thread" "${f}" 2>/dev/null | grep -E "vllm/|torch/|distributed|File \"" | tail -18
done
printf '\n===== STACKS CAPTURED =====\nlogs: %s\n' "${RUN_DIR}"
