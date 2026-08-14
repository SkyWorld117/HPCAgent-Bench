#!/usr/bin/env bash
# Does Kimi K2.7 at tp=4 pp=4 actually DECODE, with eager process-group init?
#
# 594541-543 all reached a live API and then produced nothing: engine up, /v1/models answering,
# 71 requests accepted, GPUs pegged at 100%, and generation_tokens_total frozen at 8 across two
# separate sampling windows. The signature is 24 of these, on the PP ranks only:
#
#   ProcessGroupNCCL.cpp:4094] An unbatched P2P op (send/recv) was called on this ProcessGroup
#   with size 4. In lazy initialization mode, this will result in a new 2-rank NCCL communicator
#   to be created.
#
# gpt-oss (pp=1) logs ZERO of them and serves fine, so the split is pipeline parallelism: every
# unbatched pipeline send/recv mints a fresh 2-rank communicator, and that is the CXI resource
# exhaustion. external-eager-pg-patch/sitecustomize.py injects device_id= into
# init_process_group/new_group so the PGs initialise EAGERLY and the per-pair communicators are
# never minted. ("eager" here is eager PG INIT -- the opposite knob from --enforce-eager.)
#
# The patch has been in the tree since 2026-08-07 and is referenced by nothing. This job is the
# first thing that actually loads it.
#
# Every previous kimi smoke asserted the API answered. That is exactly what a wedged engine does,
# which is why three arms burned 80 minutes each looking healthy. This one gates on DECODE
# PROGRESS and nothing else, then measures the saturated ceiling.
#
#   cd containers/cluster/ce-images/inference
#   mkdir -p logs && sbatch --account=a-g200 smoke-kimi-eager-pg.sbatch
#
#SBATCH --job-name=smoke-kimi-eager-pg
#SBATCH --partition=mi300
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke-kimi-eager-pg-%j.out
#SBATCH --error=logs/smoke-kimi-eager-pg-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
export HF_HOME="${SCRATCH}/hf"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export PG_PATCH_DIR="${PG_PATCH_DIR:-${SUBMIT_DIR}/external-eager-pg-patch}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.23.0-pytorch211-ofi}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
export SERVED_MODEL="${SERVED_MODEL:-optarena-vllm}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"
PROBE_STREAMS="${PROBE_STREAMS:-64}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-512}"

test -f "${PG_PATCH_DIR}/sitecustomize.py" || {
    echo "FAIL: no sitecustomize.py under ${PG_PATCH_DIR}" >&2
    exit 2
}
mkdir -p "${RUN_DIR}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#NODES[@]} == 4 )) || { echo "FAIL: need 4 nodes, have ${#NODES[@]}" >&2; exit 2; }
export NODE0="${NODES[0]}"

SERVE_SCRIPT="${RUN_DIR}/serve.sh"
SMOKE_DONE="${RUN_DIR}/smoke.done"
stage() { printf '\n===== stage: %s =====\n' "$*"; }

cat <<EOF
job:    ${SLURM_JOB_ID}
model:  ${MODEL_REPO} tp=4 pp=4 on ${NODE0}..${NODES[3]}
patch:  ${PG_PATCH_DIR}
logs:   ${RUN_DIR}
EOF

cat >"${SERVE_SCRIPT}" <<'SERVE'
#!/usr/bin/env bash
set -Eeuo pipefail
rank="${SLURM_PROCID:-0}"

export TOKENIZERS_PARALLELISM=false
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_FILE="${RUN_DIR}/nccl.%h.%p.log"
# THE POINT OF THIS JOB. sitecustomize is imported automatically by the interpreter at startup
# for every entry on sys.path, so putting the patch dir on PYTHONPATH reaches every vLLM worker
# process without touching the image or the launcher.
export PYTHONPATH="${PG_PATCH_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# NOT set: VLLM_DISABLE_PYNCCL. It is the superseded workaround that costs ~20x and blocks
# graph capture; run_cluster.sh deliberately stopped defaulting it on 2026-08-13.
# vLLM reads env VLLM_PORT as the BASE for internal ZMQ ports, not the HTTP port (589170).
unset VLLM_PORT

MODEL_PATH="$(python3 - <<'PY' | tail -n 1
import os
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_REPO"]
try:
    print(snapshot_download(repo_id=repo, local_files_only=True))
except Exception:
    print(snapshot_download(repo_id=repo, max_workers=8))
PY
)"
test -d "${MODEL_PATH}"
echo "rank ${rank} host $(hostname) model ${MODEL_PATH} PYTHONPATH=${PYTHONPATH}"

ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size 4
    --pipeline-parallel-size 4
    --distributed-executor-backend mp
    --nnodes 4
    --node-rank "${rank}"
    --master-addr "${NODE0}"
    --master-port "${MASTER_PORT}"
    --distributed-timeout-seconds 7200
    --load-format safetensors
    --safetensors-load-strategy prefetch
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser kimi_k2
    --reasoning-parser kimi_k2
    --trust-remote-code
    --max-model-len 65536
    # 0.70 not 0.85: the fraction multiplies TOTAL board memory but only 96 of the MI300A's
    # 128 GiB is HIP-allocatable.
    --gpu-memory-utilization 0.70
    --max-num-seqs 128
)
if (( rank == 0 )); then
    ARGS+=(--host 0.0.0.0 --port "${API_PORT}")
else
    ARGS+=(--headless)
fi

SERVER_PID=""
cleanup() { [[ -n "${SERVER_PID}" ]] && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 143' TERM

vllm serve "${ARGS[@]}" >"${RUN_DIR}/vllm-rank${rank}.log" 2>&1 &
SERVER_PID=$!
echo "rank ${rank} vllm pid ${SERVER_PID}"

while [[ ! -e "${RUN_DIR}/smoke.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the smoke finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
SERVE
chmod +x "${SERVE_SCRIPT}"
rm -f "${SMOKE_DONE}"

stage "serving kimi tp=4 pp=4 with the eager-PG patch"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODE0},${NODES[1]},${NODES[2]},${NODES[3]}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!
cleanup_steps() {
    touch "${SMOKE_DONE}"
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null && { sleep 10; kill "${SERVER_STEP_PID}" 2>/dev/null || true; }
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

metric() { curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/metrics" 2>/dev/null |
    awk -v k="vllm:$1" '$0 ~ "^"k {print $2; exit}'; }

stage "waiting for the API (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 10 "http://${NODE0}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "FAIL: vLLM died before the API came up" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
    sleep 15
done

stage "gate 1: did the eager-PG patch load?"
if grep -qh "\[external-eager-pg\]" "${RUN_DIR}"/vllm-rank*.log 2>/dev/null; then
    grep -ho "\[external-eager-pg\] [a-z_]*" "${RUN_DIR}"/vllm-rank*.log | sort | uniq -c
else
    echo "FAIL: no [external-eager-pg] lines -- sitecustomize never loaded" >&2
    exit 1
fi

stage "gate 2: unbatched-P2P warnings (594541 had 24; expect 0)"
p2p=$(grep -ch "unbatched P2P op" "${RUN_DIR}"/vllm-rank*.log 2>/dev/null | awk '{s+=$1} END{print s+0}')
echo "unbatched P2P warnings: ${p2p}"

stage "gate 3: DOES IT DECODE -- the gate every previous smoke skipped"
before=$(metric generation_tokens_total)
curl -fsS --max-time 300 "http://${NODE0}:${API_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write one line of C.\",\"max_tokens\":64}" \
    -o "${RUN_DIR}/single.json" || true
after=$(metric generation_tokens_total)
echo "generation_tokens_total: ${before:-0} -> ${after:-0}"
awk -v a="${before:-0}" -v b="${after:-0}" 'BEGIN{ if (b-a <= 0) exit 1 }' || {
    echo "FAIL: zero tokens decoded -- still wedged" >&2
    exit 1
}

stage "gate 4: saturation probe, ${PROBE_STREAMS} concurrent x ${PROBE_MAX_TOKENS} tokens"
g0=$(metric generation_tokens_total); t0=$SECONDS
for i in $(seq 1 "${PROBE_STREAMS}"); do
    curl -fsS --max-time 600 "http://${NODE0}:${API_PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Explain loop tiling, variation ${i}.\",\"max_tokens\":${PROBE_MAX_TOKENS}}" \
        -o "${RUN_DIR}/probe-${i}.json" &
done
sleep 60
g1=$(metric generation_tokens_total); run=$(metric num_requests_running); t1=$SECONDS
wait
awk -v a="${g0:-0}" -v b="${g1:-0}" -v dt="$((t1-t0))" -v r="${run:-0}" 'BEGIN{
    printf "SATURATED DECODE: %.1f tok/s aggregate over %ds at %s concurrent\n", (b-a)/dt, dt, r
    if (r > 0) printf "                  %.1f tok/s per request\n", (b-a)/dt/r
}'

touch "${SMOKE_DONE}"
printf '\n===== KIMI EAGER-PG SMOKE PASSED =====\nlogs: %s\n' "${RUN_DIR}"
