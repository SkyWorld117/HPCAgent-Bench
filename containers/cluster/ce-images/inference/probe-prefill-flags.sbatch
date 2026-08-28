#!/usr/bin/env bash
# Enumerate the prefill-scheduling surface of the image kimi actually serves on. `vllm serve --help`
# is a SUMMARY (4.8 KB); the real list needs --help=all (109 KB) -- flagcheck/600925 and 600938 both
# reported every flag MISSING because they used the summary.
#SBATCH --job-name=probe-prefill
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/probe-prefill-%j.out
#SBATCH --error=logs/probe-prefill-%j.err
set -Eeuo pipefail
srun --environment="${INFERENCE_EDF:-rocm723-vllm-0271-nomori-pytorch211-ofi}" bash -lc '
  vllm serve --help=all > /tmp/h.txt 2>&1 || true
  echo "help bytes: $(wc -c </tmp/h.txt)"
  for f in --max-num-batched-tokens --max-num-partial-prefills --max-long-partial-prefills \
           --long-prefill-token-threshold --enable-chunked-prefill --scheduling-policy \
           --enable-prefix-caching --prefix-caching-hash-algo --max-num-seqs; do
      if grep -q -- "$f" /tmp/h.txt; then
          echo "HAVE $f"
          grep -A3 -m1 -- "^ *$f" /tmp/h.txt | sed "s/^/       /" | head -4
      else
          echo "MISS $f"
      fi
  done'
