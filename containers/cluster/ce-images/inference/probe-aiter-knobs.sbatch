#!/usr/bin/env bash
# Which aiter components can be enabled INDEPENDENTLY of the attention/prefill path? The master
# switch alone broke MLA prefill on gfx942 (600662), and the MoE kernel is ~88% of decode time,
# so "aiter everywhere except prefill" is the combination worth knowing the spelling of.
#SBATCH --job-name=probe-aiterknobs
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/probe-aiterknobs-%j.out
#SBATCH --error=logs/probe-aiterknobs-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0271-nomori-pytorch211-ofi}" bash -lc '
/opt/pytorch211/bin/python - <<PY
import vllm.envs as envs
names = sorted(n for n in dir(envs) if "AITER" in n)
print("vLLM aiter env knobs:")
for n in names:
    print(f"  {n} = {getattr(envs, n, None)!r}")
print()
import subprocess
out = subprocess.run(["grep","-rhoE","AITER_[A-Z0-9_]+",
                      "/opt/pytorch211/lib/python3.12/site-packages/aiter/"],
                     capture_output=True, text=True).stdout.split()
print("aiter-side AITER_* names seen in the package:")
for n in sorted(set(out)):
    if any(k in n for k in ("OPUS","TUNE","MHA","MLA","MOE","ASM","CK")):
        print("  ", n)
PY'
