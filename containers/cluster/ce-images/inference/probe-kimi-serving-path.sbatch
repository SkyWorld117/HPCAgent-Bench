#!/usr/bin/env bash
# Why is the agent path 50x slower than the plain path on the same server?
#
# 598922 ran 20 agent requests at 2.4 tok/s aggregate for 80 minutes (KV usage 1.1%, Waiting 0,
# no deadlock -- the engine was scheduling the whole time). 598954 ran 64 plain completions at
# 115-121 tok/s on the same topology. Same model, same nodes, same batch regime. The only things
# that differ are the request shape and what the API server does per token: chat vs completions,
# streaming vs not, tool schemas attached, and the kimi_k2 tool + reasoning parsers running on
# every streamed delta.
#
# So walk that difference one step at a time against ONE server, and read the SERVER-side
# generation counter each time. A parser that throttles the API server shows up in that counter
# just as clearly as a slow kernel would, and neither client-side parsing nor LiteLLM sits in
# the measurement.
#
# The same script is the topology A/B: TP_SIZE x PP_SIZE always occupies four nodes, so
# TP=4/PP=4 and TP=8/PP=2 are directly comparable at every concurrency in stage 1. Submit it
# twice, once per split, and compare the scaling slope rather than the peak.
#SBATCH --job-name=probe-kimi-path
#SBATCH --partition=mi300
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/probe-kimi-path-%j.out
#SBATCH --error=logs/probe-kimi-path-%j.err

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
export TOOL_PARSER="${TOOL_PARSER:-kimi_k2}"
export REASONING_PARSER="${REASONING_PARSER:-kimi_k2}"
# AITER is installed in the image but nothing enables it: 598954 served on TRITON_MLA with all
# sixteen VLLM_ROCM_USE_AITER* knobs unset, so both dominant costs -- MLA attention and the
# expert GEMM -- ran on Triton. USE_AITER=1 flips the master switch, which turns on the MHA/MLA/
# MOE/RMSNORM/ROPE sub-paths that default to following it. Point AITER_JIT_DIR at a prebuilt
# cache or the first request pays a multi-minute kernel build (that is what killed 598021).
export USE_AITER="${USE_AITER:-0}"
export AITER_JIT_DIR="${AITER_JIT_DIR:-${FAST_SCRATCH}/aiter-jit}"
export API_PORT="${API_PORT:-8000}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"
export READY_TIMEOUT="${READY_TIMEOUT:-5400}"
export FIRST_TOKEN_TIMEOUT="${FIRST_TOKEN_TIMEOUT:-900}"
# prefetch stages checkpoint shards through host RAM. At PP=4 each node stages a quarter of
# the 555 GB checkpoint; at PP=2 it stages half, and 599253 had every rank SIGKILLed during
# load on 513 GB nodes. Leave empty for the low-host-memory path.
export LOAD_STRATEGY="${LOAD_STRATEGY:-}"
# Each probe waits for generation to actually START, then measures the counter delta over WINDOW.
# A FIXED settle does not work here: 599252 showed every change in the request set is followed by
# roughly four minutes during which the engine emits nothing at all (prompt throughput 0.2-5.6
# tok/s, no logger lines), after which it runs at 84-94 tok/s. A 20s settle put the whole window
# inside that dead period and reported 0.0 tok/s for four of nine arms. So poll until the counter
# moves, report that ramp as a result in its own right, and only then measure.
export PROBE_SETTLE="${PROBE_SETTLE:-20}"
export PROBE_RAMP_TIMEOUT="${PROBE_RAMP_TIMEOUT:-600}"
export PROBE_WINDOW="${PROBE_WINDOW:-120}"
# Large enough that no stream finishes inside settle+window even at the single-stream rate --
# a client that completes early stops generating and would silently deflate the measured rate.
export PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-16384}"
export SWEEP_CONCURRENCY="${SWEEP_CONCURRENCY:-1 8 16 32 64}"
# 20 is what the wedged campaign actually ran, so the path arms reproduce its load exactly.
export PATH_CONCURRENCY="${PATH_CONCURRENCY:-20}"

GPUS_PER_NODE=4
NNODES=$((TP_SIZE * PP_SIZE / GPUS_PER_NODE))
[[ "${NNODES}" -eq "${SLURM_NNODES}" ]] || {
    echo "FAIL: TP=${TP_SIZE} x PP=${PP_SIZE} needs ${NNODES} nodes, allocation has ${SLURM_NNODES}" >&2
    exit 1
}

# BASH_SOURCE points at the spooled copy under /var/spool/slurmd once slurm runs the script,
# so the submit directory is the only reliable anchor back to the repo (599092).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PG_PATCH_REL="containers/cluster/ce-images/inference/external-eager-pg-patch"
for cand in "${PG_PATCH_DIR:-}" "${SUBMIT_DIR}/external-eager-pg-patch" "${SUBMIT_DIR}/${PG_PATCH_REL}"; do
    if [[ -n "${cand}" && -f "${cand}/sitecustomize.py" ]]; then
        PG_PATCH_DIR="${cand}"
        break
    fi
done
test -f "${PG_PATCH_DIR:-}/sitecustomize.py" 2>/dev/null || {
    echo "FAIL: no sitecustomize.py found from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
    exit 2
}
MOE_DIR="${PG_PATCH_DIR%/external-eager-pg-patch}/moe-configs"
[[ -d "${MOE_DIR}" ]] && export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${MOE_DIR}}"
export PG_PATCH_DIR

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}"
NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | paste -sd,)"
MASTER="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export NODELIST MASTER
SMOKE_DONE="${RUN_DIR}/probe.done"

mkdir -p "${RUN_DIR}"
stage() { printf '\n===== stage: %s =====\n' "$1"; }
printf 'job:    %s\nmodel:  %s tp=%s pp=%s on %s nodes\nmaster: %s\nlogs:   %s\n' \
    "${SLURM_JOB_ID}" "${MODEL_REPO}" "${TP_SIZE}" "${PP_SIZE}" "${NNODES}" "${MASTER}" "${RUN_DIR}"

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
if [[ "${USE_AITER}" == "1" ]]; then
    export VLLM_ROCM_USE_AITER=1
    export TORCH_BLAS_PREFER_HIPBLASLT=1
    # The master switch turns the MLA path on too, and 599260 hung there: weights loaded, then
    # the engine blocked before graph capture, emitting shm_broadcast "no block in 60 seconds"
    # every minute for over an hour. AITER_MLA=0 keeps the other aiter paths while leaving
    # attention on TRITON_MLA, which is what isolates the hang.
    export VLLM_ROCM_USE_AITER_MLA="${AITER_MLA:-1}"
    mkdir -p "${AITER_JIT_DIR}"
fi

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
echo "rank ${rank} host $(hostname) master ${MASTER}:${MASTER_PORT_BASE}"

ARGS=(
    "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${SLURM_NNODES}"
    --node-rank "${rank}"
    --master-addr "${MASTER}"
    --master-port "${MASTER_PORT_BASE}"
    --distributed-timeout-seconds 7200
    --load-format safetensors
    ${LOAD_STRATEGY:+--safetensors-load-strategy ${LOAD_STRATEGY}}
    --generation-config vllm
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_PARSER}"
    --trust-remote-code
    --max-model-len 65536
    --gpu-memory-utilization 0.70
    --max-num-seqs 128
)
[[ -n "${REASONING_PARSER}" ]] && ARGS+=(--reasoning-parser "${REASONING_PARSER}")
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

while [[ ! -e "${RUN_DIR}/probe.done" ]]; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "rank ${rank}: vLLM exited before the probes finished" >&2
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done
SERVE
chmod +x "${SERVE_SCRIPT}"
rm -f "${SMOKE_DONE}"

stage "serving tp=${TP_SIZE} pp=${PP_SIZE} across ${NNODES} nodes"
srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --exclusive --export=ALL \
    --nodelist="${NODELIST}" --environment="${INFERENCE_EDF}" \
    bash "${SERVE_SCRIPT}" >"${RUN_DIR}/vllm-step.log" 2>&1 &
SERVER_STEP_PID=$!
cleanup_steps() {
    touch "${SMOKE_DONE}"
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null && { sleep 10; kill "${SERVER_STEP_PID}" 2>/dev/null || true; }
    wait 2>/dev/null || true
}
trap cleanup_steps EXIT INT TERM

metric() { { curl -fsS --max-time 10 "http://${MASTER}:${API_PORT}/metrics" 2>/dev/null || true; } |
    awk -v k="vllm:$1" '$0 ~ "^"k {print $2; exit}'; }

stage "waiting for the API (up to ${READY_TIMEOUT}s)"
deadline=$((SECONDS + READY_TIMEOUT))
until curl -fsS --max-time 10 "http://${MASTER}:${API_PORT}/v1/models" -o "${RUN_DIR}/models.json"; do
    kill -0 "${SERVER_STEP_PID}" 2>/dev/null || { echo "FAIL: vLLM died before the API came up" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "FAIL: API not ready in ${READY_TIMEOUT}s" >&2; exit 1; }
    sleep 15
done
echo "ready: ${MASTER}"

stage "which kernels the server actually picked"
grep -ho "Using .*backend[^,]*\|moe_backend='"'"'[a-z]*'"'"'\|linear_backend='"'"'[a-z]*'"'"'" \
    "${RUN_DIR}"/vllm-rank0.log | sort -u | sed 's/^/  /' || true
echo "  USE_AITER=${USE_AITER} VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-unset}"

stage "warming the engine so no JIT or dynamo compile lands inside a measured window"
curl -fsS --max-time "${FIRST_TOKEN_TIMEOUT}" "http://${MASTER}:${API_PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"prompt\":\"Write one line of C.\",\"max_tokens\":64}" \
    -o "${RUN_DIR}/warm-completions.json" || true
curl -fsS --max-time "${FIRST_TOKEN_TIMEOUT}" "http://${MASTER}:${API_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one line of C.\"}],\"max_tokens\":64}" \
    -o "${RUN_DIR}/warm-chat.json" || true
g_warm=$(metric generation_tokens_total)
echo "generation_tokens_total after warmup: ${g_warm:-0}"
awk -v g="${g_warm:-0}" 'BEGIN{ if (g <= 0) exit 1 }' || { echo "FAIL: warmup decoded zero tokens" >&2; exit 1; }

# Request bodies live in files: four arms differ only in shape, and writing them once keeps the
# difference visible instead of buried in shell quoting.
python3 - <<'PY'
import json
import os

run_dir = os.environ["RUN_DIR"]
model = os.environ["SERVED_MODEL"]
max_tokens = int(os.environ["PROBE_MAX_TOKENS"])
prompt = "Explain loop tiling and why it helps cache reuse."

# Shaped like the optarena MCP surface the agents actually carry, so the tool-schema arm costs
# the parser what a real turn costs it.
tools = [{
    "type": "function",
    "function": {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {
                "kernel": {"type": "string", "description": "kernel name"},
                "source": {"type": "string", "description": "full source text of the file"},
                "language": {"type": "string", "enum": ["c", "cpp", "fortran"]},
                "compiler": {"type": "string", "description": "toolchain to build with"},
            },
            "required": ["kernel", "source"],
        },
    },
} for name, desc in (("syntax_check", "Compile the file without running it and return warnings."),
                     ("score", "Build, check correctness and return the measured speedup."),
                     ("submit", "Record the current version as the graded answer."),
                     ("profile", "Run the kernel under a profiler and return the report."))]


def write(name, body):
    with open(os.path.join(run_dir, f"body-{name}.json"), "w") as fh:
        json.dump(body, fh)


base_chat = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "ignore_eos": True,
}
write("completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True})
write("chat", dict(base_chat))
write("chat-stream", {**base_chat, "stream": True})
write("chat-stream-tools", {**base_chat, "stream": True, "tools": tools, "tool_choice": "auto"})
PY

# Hold `conc` clients open against one endpoint, let the engine settle, then read the counter
# delta across a fixed window. Server-side counter, so client cost never enters the number.
measure() {
    local label="$1" conc="$2" endpoint="$3" body="$4"
    local pids=() i g0 g1 t0 t1
    for ((i = 0; i < conc; i++)); do
        curl -fsS -N --max-time $((PROBE_SETTLE + PROBE_RAMP_TIMEOUT + PROBE_WINDOW + 60)) \
            "http://${MASTER}:${API_PORT}${endpoint}" \
            -H 'Content-Type: application/json' -d @"${RUN_DIR}/${body}" >/dev/null 2>&1 &
        pids+=($!)
    done
    # Wait for the engine to actually begin emitting for THIS arm. The ramp is the headline
    # number for a campaign: agents change the request set constantly, so if every change costs
    # minutes the server never reaches the steady rate measured below.
    local ramp=0 gp gc
    sleep "${PROBE_SETTLE}"
    gp=$(metric generation_tokens_total)
    while (( ramp < PROBE_RAMP_TIMEOUT )); do
        sleep 10
        ramp=$((ramp + 10))
        gc=$(metric generation_tokens_total)
        awk -v a="${gp:-0}" -v b="${gc:-0}" 'BEGIN{ exit (b - a > 20) ? 0 : 1 }' && break
        gp="${gc}"
    done
    g0=$(metric generation_tokens_total); t0=$SECONDS
    local running; running=$(metric num_requests_running)
    sleep "${PROBE_WINDOW}"
    g1=$(metric generation_tokens_total); t1=$SECONDS
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
    awk -v a="${g0:-0}" -v b="${g1:-0}" -v dt="$((t1 - t0))" -v c="${conc}" -v r="${running:-0}" \
        -v l="${label}" -v ramp="${ramp}" -v cap="${PROBE_RAMP_TIMEOUT}" '
        BEGIN {
            rate = dt > 0 ? (b - a) / dt : 0
            printf "%-28s conc=%-4d running=%-5.0f ramp=%4ds%s %8.1f tok/s   %6.2f tok/s/req\n",
                   l, c, r, ramp, (ramp >= cap ? "!" : " "), rate, c ? rate / c : 0
        }'
    # Let the queue drain so the next arm starts from the same clean state.
    sleep 15
}

stage "stage 1: concurrency sweep on plain /v1/completions (tp=${TP_SIZE} pp=${PP_SIZE})"
echo "a healthy engine multiplies aggregate tok/s as concurrency rises; flat means per-token serialization"
for c in ${SWEEP_CONCURRENCY}; do
    measure "sweep-completions" "${c}" "/v1/completions" "body-completions.json"
done

stage "stage 2: request-shape isolation at concurrency ${PATH_CONCURRENCY} (the campaign's load)"
echo "each arm adds ONE thing over the arm above it; the step that collapses is the cause"
measure "1-completions"        "${PATH_CONCURRENCY}" "/v1/completions"      "body-completions.json"
measure "2-chat"               "${PATH_CONCURRENCY}" "/v1/chat/completions" "body-chat.json"
measure "3-chat+stream"        "${PATH_CONCURRENCY}" "/v1/chat/completions" "body-chat-stream.json"
measure "4-chat+stream+tools"  "${PATH_CONCURRENCY}" "/v1/chat/completions" "body-chat-stream-tools.json"

touch "${SMOKE_DONE}"
printf '\n===== PROBE DONE (tp=%s pp=%s) =====\nlogs: %s\n' "${TP_SIZE}" "${PP_SIZE}" "${RUN_DIR}"
