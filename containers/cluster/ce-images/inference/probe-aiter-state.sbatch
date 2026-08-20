#!/usr/bin/env bash
# Is the aiter v3-varlen predicate patch present in the image kimi serves on, and would the
# master switch be safe? 600662 died with "invalid argument for fmha_v3_varlen_fwd" because
# can_impl_fmha_v3_fwd claims causal shapes the dispatcher cannot serve when q<k, which is EVERY
# chunked prefill against a warm prefix cache. 7ef33539 added the missing guard. Without it,
# VLLM_ROCM_USE_AITER=1 reproduces 600662.
#SBATCH --job-name=probe-aiter
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe-aiter-%j.out
#SBATCH --error=logs/probe-aiter-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0271-nomori-pytorch211-ofi}" bash -lc '
/opt/pytorch211/bin/python - <<PY
import pathlib, subprocess, sys
import aiter
print("aiter:", aiter.__file__)
print("aiter version:", getattr(aiter, "__version__", "<none>"))
mha = pathlib.Path(aiter.__file__).parent / "ops" / "mha.py"
src = mha.read_text()
guard = "ret = ret and ((not causal) or (max_seqlen_q == max_seqlen_k))"
print("PREDICATE PATCH PRESENT:", guard in src)
print("occurrences:", src.count(guard))
print("--- git describe of the aiter source, if kept ---")
print(subprocess.run(["git","-C","/opt/aiter-src","log","-1","--oneline"],
                     capture_output=True, text=True).stdout.strip() or "<no /opt/aiter-src>")
import vllm.envs as envs
print("VLLM_ROCM_USE_AITER default:", envs.VLLM_ROCM_USE_AITER)
from vllm._aiter_ops import is_aiter_found
print("vLLM finds aiter:", is_aiter_found())
PY'
