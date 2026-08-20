#!/usr/bin/env bash
# Do the per-component aiter flags act on their own, or only when the master switch is on?
# Every component defaults True in this image while VLLM_ROCM_USE_AITER defaults False. If
# FP4BMM is independent it is LIVE on gfx942 right now (vLLM #34641: FP4 is gfx950-only, so a
# default-on FP4 path is a crash trigger on MI300A). If it is gated by the master, it is inert
# and not worth an experiment slot.
#SBATCH --job-name=probe-gating
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-gating-%j.out
#SBATCH --error=logs/probe-gating-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0271-nomori-pytorch211-ofi}" bash -lc '
echo "=== where FP4BMM / MHA / MLA / MOE are READ in vllm ==="
grep -rn "VLLM_ROCM_USE_AITER_\(FP4BMM\|MHA\|MLA\|MOE\)" /opt/vllm-src/vllm/ 2>/dev/null | grep -v "envs.py" | head -20
echo
echo "=== the envs.py definitions (note any dependence on VLLM_ROCM_USE_AITER) ==="
grep -n -A4 "VLLM_ROCM_USE_AITER_FP4BMM\|VLLM_ROCM_USE_AITER_MLA\b" /opt/vllm-src/vllm/envs.py | head -30
echo
echo "=== what _aiter_ops reports with the master OFF (current campaign state) ==="
/opt/pytorch211/bin/python - <<PY
from vllm._aiter_ops import rocm_aiter_ops
for name in dir(rocm_aiter_ops):
    if name.startswith("is_") and name.endswith("_enabled"):
        try:
            print(f"  {name}() = {getattr(rocm_aiter_ops, name)()}")
        except Exception as e:
            print(f"  {name}() raised {type(e).__name__}")
PY'
