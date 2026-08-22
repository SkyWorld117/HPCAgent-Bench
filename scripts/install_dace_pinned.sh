#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Install spcl/dace@extended into the ACTIVE environment, at the same commit CI grades against.
#
#   scripts/install_dace_pinned.sh [target-dir]
#
# Without it, tests/test_dace_numeric_agreement.py and tests/test_dace_frontend_validity.py do not
# skip -- they FAIL, ~40 of them, because their probe subprocesses die on ModuleNotFoundError, and
# a reader has to look inside the probe output to see the suite was never really red.
#
# The SHA is READ OUT of .github/actions/setup/action.yml rather than repeated here. That pin
# exists because the numeric test ratchets NUMERIC_BAD in both directions, so a dace that moves
# under the suite rewrites the ratchet; a second copy of the SHA is how a local run and CI end up
# grading different dace while both call it "extended".
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
action="${repo}/.github/actions/setup/action.yml"
target="${1:-${DACE_PINNED_DIR:-${repo}/../dace-pinned}}"

sha="${DACE_SHA:-$(sed -n 's/.*DACE_SHA="${DACE_SHA:-\([0-9a-f]\{40\}\)}".*/\1/p' "${action}" | head -1)}"
[[ -n "${sha}" ]] || { echo "no DACE_SHA in ${action} and none in the environment" >&2; exit 2; }

# init+fetch rather than clone: a shallow clone cannot name a bare SHA, only a branch or tag.
mkdir -p "${target}"
git -C "${target}" rev-parse --git-dir >/dev/null 2>&1 || git -C "${target}" init -q
git -C "${target}" remote get-url origin >/dev/null 2>&1 ||
    git -C "${target}" remote add origin https://github.com/spcl/dace.git
git -C "${target}" fetch -q --depth 1 origin "${sha}"
git -C "${target}" checkout -q FETCH_HEAD
# REQUIRED: dace vendors runtime headers as submodules (external/moodycamel/... is included by
# dace/runtime/include/dace/stream.h), so without them every SDFG build stops at a missing header.
git -C "${target}" submodule update --init --recursive --depth 1 -q

python -m pip install -q -e "${target}"
echo "dace @ $(git -C "${target}" rev-parse HEAD) -> $(python -c 'import dace; print(dace.__file__)')"
