#!/usr/bin/env bash
# Add AMD MORI to the pt211 vLLM image, for expert-parallel all-to-all.
#
# vLLM 0.27.1 already integrates it -- device_communicators/all2all.py imports mori for the
# `mori_high_throughput` / `mori_low_latency` --all2all-backend choices -- but the package is
# absent from our image, so those choices are unselectable and EP would fall back to the generic
# `naive` path over RCCL. MoE decode moves many SMALL activation slices, where per-message
# latency dominates and a generic collective is the wrong shape; MORI is AMD's answer to
# NVIDIA's DeepEP, which is CUDA-only (deep_ep and pplx_kernels both import False here).
#
# TWO RISKS, both unresolved before this build. Do not read a successful install as a working
# EP path:
#   1. AMD lists MI308X / MI300X / MI325X / MI355X as supported. MI300A -- an APU -- is NOT on
#      that list, though it is the same gfx942.
#   2. AMD lists Pollara, ConnectX-7 and Thor2 as supported NICs and wants libibverbs. Our
#      fabric is HPE Slingshot / CXI via libfabric, which is none of those, so the inter-node
#      GPU-direct path very likely does not apply. Intra-node EP may still work.
# The verification below therefore reports what it finds rather than asserting a working
# transport; a real answer needs a 2-node EP run.
#
# Patches the image IN PLACE with a .before-mori backup, exactly like add-aiter-pt211.sbatch.
#SBATCH --job-name=pt211-mori
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/add-mori-pt211-%j.out
#SBATCH --error=logs/add-mori-pt211-%j.err

set -Eeuxo pipefail
umask 0022

# Derive from SLURM_SUBMIT_DIR, not BASH_SOURCE: slurm copies the script to
# /var/spool/slurmd/job<id>/slurm_script. Submit from containers/cluster/ce-images/inference.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${VLLM_BUILD_ROOT:-}" ]]; then
    DEFAULT_IMAGE_DIR="${VLLM_BUILD_ROOT}/containers"
else
    DEFAULT_IMAGE_DIR="$(dirname -- "$(cd -- "${SUBMIT_DIR}/../../../.." && pwd)")/ce-images"
fi

CE_IMAGE_DIR="${CE_IMAGE_DIR:-$DEFAULT_IMAGE_DIR}"
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
IMAGE_STEM="${CE_IMAGE_DIR}/rocm723-vllm-${VLLM_VERSION}-pytorch211-ofi"

SOURCE_IMAGE="${IMAGE_STEM}.sqsh"
TEMP_IMAGE="${IMAGE_STEM}.mori.tmp.sqsh"
FINAL_IMAGE="${IMAGE_STEM}.sqsh"
BACKUP_IMAGE="${IMAGE_STEM}.before-mori.sqsh"

# Enroot names are machine-global, so the version keeps two concurrent patches from colliding.
BUILD_NAME="rocm723-pt211-mori-patch-${VLLM_VERSION//./}"
MORI_PACKAGE="${MORI_PACKAGE:-amd_mori}"

test -f "$SOURCE_IMAGE"

echo "Host:        $(hostname)"
echo "Source:      ${SOURCE_IMAGE}"
echo "Package:     ${MORI_PACKAGE}"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true
rm -f "$TEMP_IMAGE" "${TEMP_IMAGE}.sha256"

enroot create --name "$BUILD_NAME" "$SOURCE_IMAGE"

enroot start \
  --root \
  --rw \
  -e MORI_PACKAGE="${MORI_PACKAGE}" \
  "$BUILD_NAME" \
  bash -lc '
    set -Eeuxo pipefail

    export VIRTUAL_ENV=/opt/pytorch211
    export PATH=/opt/pytorch211/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/rocm/lib:/opt/rocm/lib64

    # MORI probes these at import for its RDMA transports. ibverbs is present on this fabric
    # only as a stub, which is exactly why the inter-node path is in doubt above.
    apt-get update -qq
    apt-get install -y -qq libpci-dev libibverbs-dev ibverbs-utils
    # mori/cpp is a compiled extension linked against libgrpc++.so.1.51. Without it EVERY MoE
    # model in the image dies at import -- vllm fused_moe/all2all_utils.py imports the mori
    # prepare/finalize unconditionally once the package is present, so 600908 shipped an image
    # that could not serve Kimi, gpt-oss or Qwen (600517 + 600926-600937, 13 jobs).
    # The soname is spelled differently across Ubuntu releases, so try the candidates and
    # verify the soname itself rather than trusting a package name.
    apt-get install -y -qq libgrpc++1.51t64 || apt-get install -y -qq libgrpc++1.51 || \
        apt-get install -y -qq libgrpc++-dev || true
    ldconfig
    ldconfig -p | grep -F libgrpc++.so.1.51

    /opt/pytorch211/bin/python -m pip install --no-cache-dir "${MORI_PACKAGE}"

    /opt/pytorch211/bin/python - <<PY
import sys

import torch

import mori

print("Python:  ", sys.executable)
print("mori:    ", mori.__file__, getattr(mori, "__version__", "<no __version__>"))
print("PyTorch: ", torch.__version__)

assert sys.executable == "/opt/pytorch211/bin/python"
assert torch.__version__.startswith("2.11.0+rocm7.2")

# mori.ops is a LAZY attribute (mori/__init__.py __getattr__), so a bare `import mori` loads
# none of the C++ extension. 600908 passed on exactly that and still shipped a broken image.
# Touch mori.ops to force mori.cpp, then import the two vLLM modules that every MoE model
# pulls in -- fused_moe, and the mori prepare/finalize all2all_utils imports from it.
print("mori.ops: ", mori.ops)

import vllm.distributed.device_communicators.all2all as a2a
import vllm.model_executor.layers.fused_moe as fused_moe
from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
    MoriPrepareAndFinalize,
)

print("vllm all2all module:", a2a.__file__)
print("vllm fused_moe:     ", fused_moe.__file__)
print("MoriPrepareAndFinalize:", MoriPrepareAndFinalize)
print("MORI IMPORT VALIDATION PASSED -- transport NOT yet proven on Slingshot")
PY

    rm -rf /root/.cache/pip /root/.cache/uv
  '

enroot export --output "$TEMP_IMAGE" "$BUILD_NAME"
test -s "$TEMP_IMAGE"

if [[ ! -e "$BACKUP_IMAGE" ]]; then
    mv "$FINAL_IMAGE" "$BACKUP_IMAGE"
else
    rm -f "$FINAL_IMAGE"
fi

mv "$TEMP_IMAGE" "$FINAL_IMAGE"

sha256sum "$FINAL_IMAGE" | tee "${FINAL_IMAGE}.sha256"

enroot remove --force "$BUILD_NAME" 2>/dev/null || true

ls -lh "$FINAL_IMAGE" "${FINAL_IMAGE}.sha256" "$BACKUP_IMAGE"

echo "========================================"
echo "MORI IMAGE PATCH PASSED"
echo "========================================"
