#!/usr/bin/env bash
#SBATCH --job-name=eagle3-probe
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=results/eagle3-probe-%j.out
#SBATCH --error=results/eagle3-probe-%j.err
set -uo pipefail
# Does the EAGLE3 draft the vendor recipe names actually exist, and does this vLLM know eagle3?
srun --environment=optarena-amd-mi300-v4 --cpus-per-task=32 --hint=nomultithread bash -c '
export HF_HOME=/iopsstor/scratch/cscs/ybudanaz/hf
python3 - <<PY
import os
from huggingface_hub import model_info, snapshot_download
repo = "lightseekorg/kimi-k2.7-coder-eagle3.1-mla"
try:
    i = model_info(repo)
    n = len(i.siblings or [])
    print(f"REPO OK files={n}")
except Exception as e:
    print("REPO FAIL:", type(e).__name__, str(e)[:300]); raise SystemExit(0)
p = snapshot_download(repo_id=repo)
print("DOWNLOADED:", p)
import subprocess
sz = subprocess.run(["du","-sh",p], capture_output=True, text=True).stdout.strip()
print("SIZE:", sz)
PY
echo "--- does this vLLM support eagle3 / speculative? ---"
python3 -c "
from vllm.config import SpeculativeConfig as S
import inspect
src = inspect.getsource(S)
for m in [\"eagle3\",\"eagle\",\"ngram\",\"medusa\",\"mlp_speculator\"]:
    print(f\"  {m}: {\"yes\" if m in src else \"no\"}\")
" 2>&1 | tail -8
'
