#!/usr/bin/env bash
# Extract the aws-ofi-nccl plugin out of the rocm723-ofi image so another image can load it.
#
# Why this exists: the cdna image (rocm/vllm rocm7.14.0_cdna, the only one with the MLA prefill
# Kimi/DeepSeek need) ships NO net plugin. Without one RCCL dlopens librccl-net.so, misses, finds
# no verbs device either -- CXI is not verbs -- and SILENTLY falls back to TCP sockets over the
# hsn NICs. No error is logged. That fallback is what made a kimi pp=4 decode hop take 7m08s.
#
# The plugin extracted here is portable between the two images because it was built against a
# CAPTURED host libfabric SDK with its rpath STRIPPED (see pack-beverin-host-sdk.sh), so it
# resolves libfabric at runtime from whatever the CSCS CXI hook injects. Both images ship the
# same libfabric 1.20.0 anchor at the same path for that hook to override, so the hook behaves
# identically in either one.
#
# The CSCS site hook com.hooks.aws_ofi_nccl is NOT a substitute: its rocm6 plugin exports only
# ncclNetPlugin_v5 and links libamdhip64.so.6, against an image with RCCL 2.30.4 and ROCm 7.
#
#   ./extract-aws-ofi-plugin.sh [SOURCE_SQSH] [DEST_DIR]
#
# DEST_DIR is rebuilt from scratch, so do not point it at a directory a running job is loading
# the plugin from: ranks already up survive on the open inode, but every rank that starts after
# the wipe fails to find it and the job falls back to TCP or dies. Extract elsewhere and swap.
#
# Point the consuming EDF's LD_LIBRARY_PATH at <DEST_DIR>/opt/aws-ofi-nccl/lib -- PREPENDED to
# that image's own baked LD_LIBRARY_PATH, never replacing it. Validate with
# example-script/test-rccl-ofi-2node.sbatch before trusting a campaign to it.

set -euo pipefail

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/${USER}/x86_64}"
SOURCE_SQSH="${1:-${SCRATCH}/ce-images/rocm723-vllm-0.23.0-pytorch211-ofi.sqsh}"
DEST_DIR="${2:-${SCRATCH}/ce-plugins/aws-ofi-nccl-rocm723}"

[[ -f "${SOURCE_SQSH}" ]] || { echo "no such image: ${SOURCE_SQSH}" >&2; exit 2; }
command -v unsquashfs >/dev/null || { echo "unsquashfs not on PATH" >&2; exit 2; }

rm -rf "${DEST_DIR}"
mkdir -p "$(dirname "${DEST_DIR}")"

# The plugin itself, plus fi_info-host: the only way to ask, from inside a container, whether the
# cxi provider is actually reachable rather than inferring it from a collective that hangs.
unsquashfs -no-progress -d "${DEST_DIR}" "${SOURCE_SQSH}" /opt/aws-ofi-nccl >/dev/null
unsquashfs -no-progress -f -d "${DEST_DIR}" "${SOURCE_SQSH}" /usr/local/bin/fi_info-host >/dev/null
mkdir -p "${DEST_DIR}/opt/aws-ofi-nccl/bin"
mv "${DEST_DIR}/usr/local/bin/fi_info-host" "${DEST_DIR}/opt/aws-ofi-nccl/bin/"
rm -rf "${DEST_DIR}/usr"

PLUGIN="${DEST_DIR}/opt/aws-ofi-nccl/lib/librccl-net.so"
[[ -f "${PLUGIN}" ]] || { echo "extraction produced no ${PLUGIN}" >&2; exit 1; }

# Two properties decide whether a consuming image can load this at all, so assert both here
# rather than discovering them as a silent TCP fallback inside a 24 h campaign.
ABI=$(readelf -sW --dyn-syms "${PLUGIN}" | grep -oE 'ncclNetPlugin_v[0-9]+' | sort -uV | tr '\n' ' ')
[[ -n "${ABI}" ]] || { echo "plugin exports no ncclNetPlugin_v* -- wrong artifact" >&2; exit 1; }

# An RPATH would pin it to the source image's library layout and defeat the whole point.
if readelf -dW "${PLUGIN}" | grep -qE 'RPATH|RUNPATH'; then
    echo "plugin carries an RPATH/RUNPATH -- not relocatable to another image" >&2
    exit 1
fi

cat <<EOF
plugin:   ${DEST_DIR}/opt/aws-ofi-nccl
abi:      ${ABI}
needed:   $(readelf -dW "${PLUGIN}" | awk -F'[][]' '/NEEDED/ {printf "%s ", $2}')
manifest: $(grep -E 'aws_ofi_nccl_ref|compile_libfabric' "${DEST_DIR}/opt/aws-ofi-nccl/BUILD-MANIFEST.txt" | tr '\n' ' ')

Put this on the consuming EDF's LD_LIBRARY_PATH, FIRST, keeping that image's own entries:
  ${DEST_DIR}/opt/aws-ofi-nccl/lib
EOF
