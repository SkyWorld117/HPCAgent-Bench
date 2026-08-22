#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Install spcl/dace@extended into the active environment, at the commit CI grades against.
#
#   scripts/install_dace_pinned.sh [target-dir]
#
# Without dace, test_dace_numeric_agreement.py and test_dace_frontend_validity.py do not skip --
# ~40 of them fail inside probe subprocesses, reading as a red suite, not a missing dependency.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
action="${repo}/.github/actions/setup/action.yml"
target="${1:-${DACE_PINNED_DIR:-${repo}/../dace-pinned}}"

# Read out of the CI action, never repeated here: a second copy is how a local run and CI end up
# grading different dace while both call it extended.
sha="${DACE_SHA:-$(sed -n 's/.*DACE_SHA="${DACE_SHA:-\([0-9a-f]\{40\}\)}".*/\1/p' "${action}" | head -1)}"
[[ -n "${sha}" ]] || { echo "no DACE_SHA in ${action} and none in the environment" >&2; exit 2; }

# init+fetch, not clone: a shallow clone cannot name a bare SHA.
mkdir -p "${target}"
git -C "${target}" rev-parse --git-dir >/dev/null 2>&1 || git -C "${target}" init -q
git -C "${target}" remote get-url origin >/dev/null 2>&1 ||
    git -C "${target}" remote add origin https://github.com/spcl/dace.git
git -C "${target}" fetch -q --depth 1 origin "${sha}"
git -C "${target}" checkout -q FETCH_HEAD
# REQUIRED, not tidiness: dace/runtime/include/dace/stream.h includes vendored submodule headers
# (external/moodycamel/...), and without them every SDFG build stops on a missing header.
git -C "${target}" submodule update --init --recursive --depth 1 -q

python -m pip install -q -e "${target}"
echo "dace @ $(git -C "${target}" rev-parse HEAD) -> $(python -c 'import dace; print(dace.__file__)')"
