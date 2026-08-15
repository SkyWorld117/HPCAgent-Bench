#!/usr/bin/env bash
# Pull the vLLM-suggested AMD container for Kimi K3 and convert it to a CE sqsh.
#
# recipes.vllm.ai/moonshotai/Kimi-K3 names vllm/vllm-openai_rocm:kimi-k3 (vllm >= 0.27.1,
# aiter >= 0.1.19). That vllm generation carries the aiter/FlyDSL MoE backends our 0.23 lacks --
# proven missing in 595212: --moe-backend aiter + every FLYDSL switch still leaves
# CompressedTensorsWNA16MoEMethod on the untuned triton path at 90 tok/s.
#
# Pull-only, no Dockerfile: the point is to follow the suggested container, not to fork it.
# KNOWN RISK carried forward: a stock image has no aws-ofi-nccl/CXI plugin, so cross-node
# traffic rides TCP. PP hidden-state sends are ~1 MB/microbatch, so tp4/pp4 may be fine; the
# kimi smoke against the new EDF is what decides, not this build.
#SBATCH --job-name=pull-vllm-kimi-k3
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=logs/pull-vllm-kimi-k3-%j.out
#SBATCH --error=logs/pull-vllm-kimi-k3-%j.err

set -Eeuxo pipefail
umask 0022

SOURCE_IMAGE="${SOURCE_IMAGE:-docker.io/vllm/vllm-openai_rocm:kimi-k3}"
SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRATCH}/ce-images/vllm-openai-rocm-kimi-k3.sqsh}"

mkdir -p "$(dirname "${OUTPUT_SQSH}")"

# Diskless nodes: temp + runtime dirs on /dev/shm, stale per-node podman state wiped (only a
# cache). --mem=0 because tmpfs pages charge the job cgroup.
unset DBUS_SESSION_BUS_ADDRESS
export TMPDIR="/dev/shm/${USER}/tmp"
export XDG_RUNTIME_DIR="/dev/shm/${USER}/xdg"
rm -rf "/dev/shm/${USER}/root" "/dev/shm/${USER}/runroot" "/dev/shm/${USER}/tmp" "/dev/shm/${USER}/xdg"
mkdir -p "${TMPDIR}"
mkdir -p -m 0700 "${XDG_RUNTIME_DIR}"

# cgroupfs: with the systemd manager a dying logind session reaps podman mid-pull (silent rc=1).
podman --cgroup-manager=cgroupfs pull "${SOURCE_IMAGE}"

# The tag is a moving target; record what was actually pulled.
podman --cgroup-manager=cgroupfs inspect --format '{{.Digest}} {{.Id}}' "${SOURCE_IMAGE}"

# enroot's exit code lies when its cleanup fails after a good write; gate on the artifact
# (listing forces a read of the inode table at file END, which a truncated image fails).
rm -f "${OUTPUT_SQSH}"
enroot import -x mount -o "${OUTPUT_SQSH}" "podman://${SOURCE_IMAGE}" || true
unsquashfs -l "${OUTPUT_SQSH}" usr >/dev/null

sha256sum "${OUTPUT_SQSH}" | tee "${OUTPUT_SQSH}.sha256"
ls -lh "${OUTPUT_SQSH}"

echo "========================================"
echo "KIMI-K3 IMAGE PULL PASSED"
echo "========================================"
