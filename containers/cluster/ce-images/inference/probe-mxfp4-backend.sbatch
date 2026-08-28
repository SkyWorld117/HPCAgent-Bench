#!/usr/bin/env bash
# Can gpt-oss-120b load on the 0.27.1 image at all? 601854-601857 all died in
# process_weights_after_loading -> fused_moe/oracle/mxfp4.py -> `from triton_kernels.matmul_ogs`,
# a module the image does not carry. oss120b last ran on 0.23 (600516), which had no such path.
# No weights, no GPUs beyond one node: this only asks what the image can import and what knobs
# vLLM offers to route around the missing module.
#SBATCH --job-name=probe-mxfp4
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe-mxfp4-%j.out
#SBATCH --error=logs/probe-mxfp4-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0.27.1-pytorch211-ofi}" bash -lc '
/opt/pytorch211/bin/python - <<PY
import importlib, importlib.util, subprocess, sys
for m in ("triton", "triton_kernels", "triton_kernels.matmul_ogs"):
    spec = importlib.util.find_spec(m) if m != "triton_kernels.matmul_ogs" else None
    try:
        importlib.import_module(m); print(f"IMPORT OK   {m}")
    except Exception as e:
        print(f"IMPORT FAIL {m}: {type(e).__name__}: {e}")
import triton
print("triton version:", triton.__version__, triton.__file__)
import vllm.envs as envs
knobs = [k for k in dir(envs) if "MXFP4" in k or "OSS" in k or "MARLIN" in k]
print("mxfp4-ish env knobs:", knobs)
for k in knobs:
    print("   ", k, "=", getattr(envs, k, None))
# Is triton_kernels shipped inside the triton source tree we built from?
print("--- pip candidates ---")
print(subprocess.run([sys.executable,"-m","pip","index","versions","triton_kernels"],
      capture_output=True, text=True).stdout or "no pip index result")
PY'
