#!/usr/bin/env bash
# Does the int4 W4A16 expert kernel stop scaling with batch size, while bf16 keeps scaling?
#
# 598954 measured 115-121 tok/s at 64 concurrent against ~90 tok/s single stream: batching buys
# ~1.3x where a healthy engine gives 10-30x. One candidate is the expert kernel itself -- a
# weight-only int4 kernel dequantizes per token, so its cost can track batch size linearly and
# hand back no amortization at all.
#
# Serving the model in bf16 to compare end to end is NOT possible: 61 layers x 384 experts x 3
# matrices x 7168 x 2048 is 1.03T MoE parameters, 2.06 TB at bf16 against 1.5 TiB of allocatable
# HBM on four nodes. So compare the KERNEL at the serving shape instead. benchmark_moe.py
# allocates synthetic weights, so no checkpoint is read and one node answers it.
#
# Read the per-token column, not the kernel column: kernel time is expected to rise with batch.
# Healthy means us_per_token FALLS as batch grows. int4 flat-or-rising next to a falling bf16
# curve confirms the expert kernel as the throughput ceiling, and the fix is a kernel or
# checkpoint swap rather than another serving flag.
#SBATCH --job-name=bench-moe-dtype
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench-moe-dtype-%j.out
#SBATCH --error=logs/bench-moe-dtype-%j.err

set -Eeuo pipefail
ulimit -c 0

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
export RUN_DIR="${SCRATCH}/kimi-smoke/${SLURM_JOB_ID}"
FAST_SCRATCH="/iopsstor/scratch/cscs/${USER}"
[[ -d "${FAST_SCRATCH}" ]] || FAST_SCRATCH="${SCRATCH}"
export HF_HOME="${HF_HOME:-${FAST_SCRATCH}/hf}"
export MODEL_REPO="${MODEL_REPO:-moonshotai/Kimi-K2.7-Code}"
# TP the serving config uses; it sets the per-rank expert width the kernel actually runs.
export TP_SIZE="${TP_SIZE:-4}"
export BENCH_BATCH_SIZES="${BENCH_BATCH_SIZES:-1 4 16 64 256}"
# Both dtypes get the same budget so neither wins on search effort alone.
export BENCH_MAX_CONFIGS="${BENCH_MAX_CONFIGS:-24}"

INFERENCE_EDF="${INFERENCE_EDF:-rocm723-vllm-0.23.0-pytorch211-ofi}"

mkdir -p "${RUN_DIR}"
printf 'job:    %s\nmodel:  %s tp=%s\nsizes:  %s (max %s configs per dtype per size)\nlogs:   %s\n\n' \
    "${SLURM_JOB_ID}" "${MODEL_REPO}" "${TP_SIZE}" "${BENCH_BATCH_SIZES}" "${BENCH_MAX_CONFIGS}" "${RUN_DIR}"

srun --nodes=1 --ntasks=1 --exclusive --export=ALL --environment="${INFERENCE_EDF}" bash -c '
    set -Eeuo pipefail
    export TOKENIZERS_PARALLELISM=false
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
        export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES}}"
        unset ROCR_VISIBLE_DEVICES
    fi

    MODEL_PATH="$(python3 - <<PY | tail -n 1
import os
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=os.environ["MODEL_REPO"], local_files_only=True))
PY
)"
    test -d "${MODEL_PATH}"
    export MODEL_PATH
    echo "model: ${MODEL_PATH}"

    cd "${RUN_DIR}"
    python3 - <<PY
import importlib.util
import json
import os
import random

import torch
import triton

spec = importlib.util.spec_from_file_location("benchmark_moe", "/opt/vllm-src/benchmarks/kernels/benchmark_moe.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

model = os.environ["MODEL_PATH"]
tp = int(os.environ["TP_SIZE"])
sizes = [int(s) for s in os.environ["BENCH_BATCH_SIZES"].split()]
max_configs = int(os.environ["BENCH_MAX_CONFIGS"])

config = bm.get_config(model=model, trust_remote_code=True)
# Same authority the tuner uses: to_dict prunes model-specific keys in this transformers build,
# and kimi nests the MoE fields under text_config.
raw = json.load(open(os.path.join(model, "config.json")))
tc = raw.get("text_config") or raw
E = tc.get("n_routed_experts") or tc["num_experts"]
topk = tc["num_experts_per_tok"]
hidden_size = tc["hidden_size"]
shard_intermediate_size = 2 * tc["moe_intermediate_size"] // tp
dtype = bm.resolve_dtype(config)
group_size = bm.get_quantization_group_size(config)
assert group_size is not None, "checkpoint is not int4 grouped; nothing to compare against bf16"

# The only difference between the two arms. Activations stay bf16 in both -- W4A16 means the
# weights are int4 and the activations are not, so this isolates the weight format.
ARMS = [("int4_w4a16", True, [0, group_size]), ("bf16", False, None)]

print(f"E={E} topk={topk} shard_intermediate={shard_intermediate_size} hidden={hidden_size} "
      f"group_size={group_size} act_dtype={dtype}", flush=True)

torch.set_default_device("cuda")
results = {}
for bs in sizes:
    for name, use_int4, block_quant_shape in ARMS:
        space = bm.get_configs_compute_bound(False, None)
        space = bm.prune_rocm_search_space(bs, shard_intermediate_size, hidden_size, space, False, topk)
        if len(space) > max_configs:
            random.seed(0)
            space = random.sample(space, max_configs)
        best, best_time = None, float("inf")
        for cfg in space:
            try:
                kt = bm.benchmark_config(cfg, bs, E, shard_intermediate_size, hidden_size, topk, dtype,
                                         False, False, use_int4, num_iters=20,
                                         block_quant_shape=block_quant_shape, use_deep_gemm=False)
            except triton.runtime.autotuner.OutOfResources:
                continue
            if kt < best_time:
                best_time, best = kt, cfg
        bm.clear_triton_cache()
        assert best is not None, f"no config compiled for bs={bs} arm={name}"
        results[(bs, name)] = best_time
        print(f"[{name}] bs={bs} best_kernel={best_time:.1f}us per_token={best_time / bs:.2f}us "
              f"cfg={bm.sort_config(best)}", flush=True)

print("", flush=True)
print("batch |    int4 us |  int4 us/tok |    bf16 us |  bf16 us/tok | int4/bf16", flush=True)
for bs in sizes:
    i4, b16 = results[(bs, "int4_w4a16")], results[(bs, "bf16")]
    print(f"{bs:5d} | {i4:10.1f} | {i4 / bs:12.2f} | {b16:10.1f} | {b16 / bs:12.2f} | {i4 / b16:9.2f}",
          flush=True)

# Scaling is the verdict, not any single point: how much cheaper does one token get when the
# batch grows from the smallest to the largest size measured?
lo, hi = sizes[0], sizes[-1]
for name in ("int4_w4a16", "bf16"):
    gain = (results[(lo, name)] / lo) / (results[(hi, name)] / hi)
    print(f"{name}: per-token cost falls {gain:.2f}x from bs={lo} to bs={hi}", flush=True)
json.dump({f"{bs}:{n}": v for (bs, n), v in results.items()}, open("moe-dtype-bench.json", "w"), indent=2)
PY
'

echo "===== results ====="
cat "${RUN_DIR}/moe-dtype-bench.json"
