#!/usr/bin/env bash
#SBATCH --job-name=build-vllm-mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=build-vllm-mi300-%j.out
# Replace these with the Opus project/vCluster values:
#SBATCH --account=REPLACE_ME
#SBATCH --partition=REPLACE_ME

set -euo pipefail

# Submit from the directory containing this file and Containerfile:
#   sbatch build_once.slurm
#
# Optional overrides:
#   export VLLM_ROCM_IMAGE=docker.io/rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
#   export RAY_SPEC='ray[default]==2.48.0'
#   export SQSH="$SCRATCH/ce-images/my-vllm-mi300.sqsh"

: "${SCRATCH:?SCRATCH is not set}"

VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-docker.io/rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909}"
EXPECTED_ROCM_MAJOR="${EXPECTED_ROCM_MAJOR:-6}"
RAY_SPEC="${RAY_SPEC:-ray[default]==2.48.0}"
LOCAL_IMAGE="${LOCAL_IMAGE:-vllm-mi300-rocm${EXPECTED_ROCM_MAJOR}:local}"
SQSH="${SQSH:-$SCRATCH/ce-images/vllm-mi300-rocm${EXPECTED_ROCM_MAJOR}.sqsh}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$SLURM_SUBMIT_DIR}"

CONTAINERS_CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/containers"
mkdir -p "$CONTAINERS_CONF_DIR"
cat > "$CONTAINERS_CONF_DIR/storage.conf" <<EOF_STORAGE
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF_STORAGE

# Confirm Podman picked up the intended rootless /dev/shm store.
podman info | grep -A 4 'store:' || true

mkdir -p "$(dirname "$SQSH")"

# Recommended CSCS Lustre layout for large SquashFS images.  The command is
# intentionally non-fatal if the target filesystem or vCluster lacks `lfs`.
if command -v lfs >/dev/null 2>&1; then
  lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M \
    "$(dirname "$SQSH")" 2>/dev/null || true
fi

cd "$BUILD_CONTEXT"

podman build --pull=always \
  --build-arg "VLLM_ROCM_IMAGE=$VLLM_ROCM_IMAGE" \
  --build-arg "RAY_SPEC=$RAY_SPEC" \
  --tag "$LOCAL_IMAGE" \
  --file Containerfile \
  .

# Check the ROCm major before producing the persistent SquashFS image.  This
# does not require a GPU device inside the Podman build allocation.
ROCM_VERSION="$({ podman run --rm --entrypoint python3 "$LOCAL_IMAGE" \
  -c 'import torch; print(torch.version.hip or "")'; } | tail -n1)"

echo "Image reports torch.version.hip=$ROCM_VERSION"
case "$ROCM_VERSION" in
  "${EXPECTED_ROCM_MAJOR}."*) ;;
  *)
    echo >&2 "ERROR: image ROCm major does not match expected ${EXPECTED_ROCM_MAJOR}."
    echo >&2 "The supplied EDF uses com.hooks.aws_ofi_nccl.variant=rocm${EXPECTED_ROCM_MAJOR}."
    echo >&2 "Choose a matching image or obtain the correct Opus hook variant from CSCS."
    exit 2
    ;;
esac

podman run --rm -i --entrypoint python3 "$LOCAL_IMAGE" - <<'PY'
import torch
import vllm
import ray
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("vLLM:", vllm.__version__)
print("Ray:", ray.__version__)
PY

podman run --rm --entrypoint bash "$LOCAL_IMAGE" -lc \
  'ldconfig -p | grep -m1 libfabric; fi_info --version'

# /dev/shm is allocation-local.  Import before this same Slurm allocation ends.
rm -f "$SQSH"
enroot import -x mount -o "$SQSH" "podman://$LOCAL_IMAGE"

printf '\nCreated Container Engine image:\n  %s\n' "$SQSH"
printf 'Use it with:\n  export VLLM_MI300_SQSH=%q\n' "$SQSH"
