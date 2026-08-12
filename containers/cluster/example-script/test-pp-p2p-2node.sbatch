#!/usr/bin/env bash
#SBATCH --job-name=pp-p2p-2node
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/pp-p2p-%j.out
#SBATCH --error=/capstor/scratch/cscs/ybudanaz/x86_64/rccl-ofi/logs/pp-p2p-%j.err
#
# Bisect the CXI env against the failure that kills kimi pipeline parallelism.
#
# test-rccl-ofi-2node.sbatch already proved inter-node COLLECTIVES ride CXI (job 590212: cxi
# provider, AWS Libfabric, 85.1 GB/s bus at 2n x 4r). Job 590270 then showed that is not enough:
# vLLM's PP handoff -- torch.distributed.irecv on a size-4 group, which makes RCCL build a fresh
# 2-rank communicator -- died with `ncclInternalError: Internal check failed` (RCCL 2.30.4,
# aws-ofi-nccl 1.20.0), taking the engine with it. Collectives healthy, P2P dead.
#
# So the question is which of the CXI/RCCL knobs added in b37cb495 the 2-rank P2P path cannot take.
# Each variant below runs the same probe under a different env, in ONE allocation: about two minutes
# each, against three hours for a vLLM probe that answers the same question.
#
#   sbatch --account=a-g200 test-pp-p2p-2node.sbatch
#
# Each variant prints `allreduce=<v> p2p_size2=<v> p2p_size4=<v>`, and the two P2P columns are the
# point. torch mints a lazy 2-rank communicator only when the group is WIDER than the pair actually
# exchanging data, so:
#
#   size2 PASS + size4 FAIL -> only the lazy sub-communicator is broken. Pipeline parallelism works
#                              at pp=2 (PP group size 2), which is what probe 589497 already showed
#                              passing where pp=4 failed. Fix is the topology, not the env.
#   both FAIL               -> inter-node P2P is broken outright; pp is unavailable at any width.
#   both PASS               -> the env under test is clean; look at the vLLM side instead.
#
# A variant with size4 PASS names a knob to drop from the EDF. If v0-current passes outright, the
# bug is NOT in the env -- do not go on to re-tune the EDF in that case.
#
# NOTE on an earlier version of this gate: the probe's chain used to WRAP AROUND, and every rank
# issued its irecv before its isend. Creating the lazy 2-rank communicator is itself blocking and
# collective over that pair, so the ring deadlocked on communicator creation and every variant
# reported a TCPStore rendezvous timeout -- a bug in the probe, not in RCCL, and nothing about the
# env could be concluded from it. The chain is a line now, which is also what vLLM's pipeline is.

set -euo pipefail

# Beverin's core_pattern is the machine-global `core_%h_%p` and dumps land in the CWD.
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
RUN_DIR="${SCRATCH}/rccl-ofi/logs/${SLURM_JOB_ID}"
# NOT dirname "${BASH_SOURCE[0]}": slurmd runs a spooled copy out of /var/spool, so that resolves to
# a directory holding nothing but the script itself. SLURM_SUBMIT_DIR is where sbatch was invoked.
PROBE_PY="${PROBE_PY:-${SLURM_SUBMIT_DIR:-.}/pp-p2p-probe.py}"
INFERENCE_EDF="${INFERENCE_EDF:-rocm714-vllm-0.23.0-cdna}"

MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
RANKS_PER_NODE="${RANKS_PER_NODE:-4}"
STAGE_TIMEOUT="${STAGE_TIMEOUT:-240}"
# Fraction of free GPU memory the probe holds before building any communicator. 0 = empty-GPU
# control. The interesting run is a high value: vLLM creates its pipeline communicator with weights
# and KV cache already resident, and registration is what failed there (cxil_map), not transport.
FILL_FRACTION="${FILL_FRACTION:-0}"

mkdir -p "${RUN_DIR}"
[[ -f "${PROBE_PY}" ]] || { echo "FAIL: missing ${PROBE_PY}" >&2; exit 2; }

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if (( ${#NODES[@]} != 2 )); then
    echo "FAIL: need exactly 2 nodes, allocation has ${#NODES[@]}" >&2
    exit 2
fi

MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
    bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1")
[[ -n "${MASTER_ADDR}" ]] || { echo "FAIL: no hsn0 address on ${NODES[0]}" >&2; exit 2; }

cat <<EOF
job:      ${SLURM_JOB_ID}
nodes:    ${NODES[*]}
edf:      ${INFERENCE_EDF}
probe:    ${PROBE_PY}
master:   ${MASTER_ADDR}:${MASTER_PORT}
run_dir:  ${RUN_DIR}
EOF

# Variants, cheapest hypothesis first. Each is shell run INSIDE the container before the probe, so
# it edits the env the EDF established. Names are what the summary prints.
#
#   v0-current   reproduce with the EDF exactly as committed -- if this passes, stop, the env is innocent
#   v1-nochan    NCCL_NCHANNELS_PER_NET_PEER=4 asks for 4 channels to every net peer; a 2-rank comm
#                created after a 16-rank one is where that runs out of CXI resources first
#   v2-nocrossnic  NCCL_CROSS_NIC=1 lets a route leave on one NIC and arrive on another
#   v3-nordzv    the four FI_CXI_RDZV_* settings force rendezvous for every message size; a 2-rank
#                P2P comm is the one place that has no collective algorithm to hide a bad choice
#   v4-minimal   drop all CXI/MR tuning, keep only what SELECTS the fabric
#   v5-min-hybrid  minimal plus the one match-mode setting that is a correctness fix, not a tuning knob
# VARIANTS overrides the sweep, e.g. VARIANTS=v0-current to run the committed env alone.
read -r -a VARIANT_NAMES <<<"${VARIANTS:-v0-current v1-nochan v2-nocrossnic v3-nordzv v4-minimal v5-min-hybrid}"

TUNING_VARS="FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_DISABLE_HOST_REGISTER \
FI_CXI_RX_MATCH_MODE FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE \
FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE \
FI_MR_CACHE_MONITOR NCCL_CROSS_NIC NCCL_NCHANNELS_PER_NET_PEER"

variant_env() {
    case "$1" in
        v0-current)    echo ":" ;;
        v1-nochan)     echo "unset NCCL_NCHANNELS_PER_NET_PEER" ;;
        v2-nocrossnic) echo "unset NCCL_CROSS_NIC" ;;
        v3-nordzv)     echo "unset FI_CXI_RDZV_PROTO FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_GET_MIN FI_CXI_RDZV_THRESHOLD" ;;
        v4-minimal)    echo "unset ${TUNING_VARS}" ;;
        v5-min-hybrid) echo "unset ${TUNING_VARS}; export FI_CXI_RX_MATCH_MODE=hybrid" ;;
        *)             echo "echo 'FAIL: unknown variant $1' >&2; exit 2" ;;
    esac
}

run_variant() {
    local name="$1" fill="$2" label="${1}@fill${2}" log="${RUN_DIR}/${1}-fill${2}.log"

    printf '\n===== variant %s (fill %s) =====\n' "${name}" "${fill}"
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
        --gpus-per-node=4 --cpus-per-task=32 \
        --environment="${INFERENCE_EDF}" \
        bash -c "
            $(variant_env "${name}")
            timeout --signal=TERM --kill-after=15s ${STAGE_TIMEOUT}s \
            torchrun --nnodes=2 --nproc-per-node=${RANKS_PER_NODE} \
                --node-rank=\${SLURM_NODEID} \
                --master-addr='${MASTER_ADDR}' --master-port='${MASTER_PORT}' \
                '${PROBE_PY}' --timeout-seconds 90 --fill-fraction ${fill}
        " 2>&1 | tee "${log}" || true

    verdict "${label}" "${log}"
}

# Report the two stages separately. A single pass/fail line would hide the one distinction that
# matters here: allreduce PASS + p2p FAIL means the fabric is fine and only the 2-rank path is broken.
verdict() {
    local name="$1" log="$2"
    local allreduce p2p pair provider

    # PASS needs rank 0's PASS *and* no rank reporting FAIL: a remote rank can get a silently wrong
    # answer while rank 0 gets the right one, which is exactly what memory pressure produced.
    stage_verdict() {
        grep -q "^stage_$1=FAIL" "${log}" && { echo FAIL; return; }
        grep -q "^stage_$1=PASS" "${log}" && echo PASS || echo FAIL
    }
    allreduce=$(stage_verdict allreduce)
    pair=$(stage_verdict p2p_size2)
    p2p=$(stage_verdict p2p_size4)
    grep -q 'Using network Socket'  "${log}" && provider=SOCKET || provider=cxi
    grep -q 'Selected provider is cxi' "${log}" || provider=UNKNOWN

    printf -- '-- %s: allreduce=%s p2p_size2=%s p2p_size4=%s net=%s\n' \
        "${name}" "${allreduce}" "${pair}" "${p2p}" "${provider}"
    # `|| true`: under `set -o pipefail` this pipeline fails when the grep matches NOTHING, which is
    # precisely the passing case -- it ended the first corrected run after one variant.
    grep -E '^(failed_stage|failure)=' "${log}" | sed 's/^/   | /' | head -4 || true

    SUMMARY+=("${name}: allreduce=${allreduce} p2p_size2=${pair} p2p_size4=${p2p} net=${provider}")
    if [[ "${p2p}" == PASS && "${provider}" == cxi ]]; then
        PASSING+=("${name}")
    fi
}

SUMMARY=()
PASSING=()
read -r -a FILL_LEVELS <<<"${FILL_FRACTION}"
for fill in "${FILL_LEVELS[@]}"; do
    for name in "${VARIANT_NAMES[@]}"; do
        run_variant "${name}" "${fill}"
    done
done

printf '\n===== summary =====\n'
printf '%s\n' "${SUMMARY[@]}"

if (( ${#PASSING[@]} == 0 )); then
    echo "PP_P2P_GATE: FAIL -- no variant carried 2-rank P2P over CXI. The env is not the cause."
    echo "Logs: ${RUN_DIR}"
    exit 1
fi
echo "PP_P2P_GATE: PASS -- first working variant: ${PASSING[0]} (all: ${PASSING[*]})"
echo "Logs: ${RUN_DIR}"
exit 0
