#!/usr/bin/env bash
# Is VLLM_ROCM_USE_AITER_FP4BMM reachable for an int4_w4a16 MoE like Kimi, or only for FP4
# checkpoints? MI300A is gfx942 and has no FP4 (that is gfx950), so a default-on FP4 path is a
# crash candidate -- but only if this model's quantisation actually routes through it. Kimi logs
# "Using TritonWNA16Experts", which is the WNA16 path, not FP4.
#SBATCH --job-name=probe-fp4reach
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-fp4reach-%j.out
#SBATCH --error=logs/probe-fp4reach-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0271-nomori-pytorch211-ofi}" bash -lc '
echo "=== every consumer of the FP4BMM flag / helper in vllm ==="
grep -rn "fp4bmm\|FP4BMM" /opt/vllm-src/vllm/ 2>/dev/null | grep -viE "^\S+envs.py" | head -20
echo
echo "=== is the helper gated by the master switch? ==="
grep -n -B6 -A10 "def is_fp4bmm" /opt/vllm-src/vllm/_aiter_ops.py 2>/dev/null | head -30
echo
echo "=== for comparison, how MoE gates ==="
grep -n -B4 -A10 "def is_fused_moe_enabled" /opt/vllm-src/vllm/_aiter_ops.py 2>/dev/null | head -24'
