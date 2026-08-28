#!/bin/bash
# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Install the TIP of spcl/dace@extended into the active environment -- the same dace CI grades
# against, which is always the branch tip and never a pin.
#
#   scripts/install_dace_extended.sh [target-dir]
#
# Without dace, test_dace_numeric_agreement.py and test_dace_frontend_validity.py do not skip --
# ~40 of them fail inside probe subprocesses, reading as a red suite, not a missing dependency.
#
# Re-run this before trusting a local dace verdict. The tip moves, and a tree left at last week's
# extended reports failures the branch has already fixed (run 32571640992 did exactly that from
# the CI side, back when the setup action still had a pin).
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-${DACE_EXTENDED_DIR:-${repo}/../dace-extended}}"

# init+fetch, not clone: this re-runs against an existing tree to move it forward.
mkdir -p "${target}"
git -C "${target}" rev-parse --git-dir >/dev/null 2>&1 || git -C "${target}" init -q
git -C "${target}" remote get-url origin >/dev/null 2>&1 ||
    git -C "${target}" remote add origin https://github.com/spcl/dace.git
git -C "${target}" fetch -q --depth 1 origin extended
git -C "${target}" checkout -q FETCH_HEAD
# REQUIRED, not tidiness: dace/runtime/include/dace/stream.h includes vendored submodule headers
# (external/moodycamel/...), and without them every SDFG build stops on a missing header.
git -C "${target}" submodule update --init --recursive --depth 1 -q

python -m pip install -q -e "${target}"
echo "dace @ $(git -C "${target}" rev-parse HEAD) -> $(python -c 'import dace; print(dace.__file__)')"
