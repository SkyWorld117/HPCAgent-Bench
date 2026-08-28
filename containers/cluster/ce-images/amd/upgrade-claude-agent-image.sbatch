#!/usr/bin/env bash
# Upgrade the Claude Code CLI inside the agent image.
#
# The shipped binary is 2.1.197, which REJECTS --autocompact ("error: unknown option", rc 1 --
# proven against a deliberately bogus flag in the same run). Every .env.llr4-* sets
# CLAUDE_AUTOCOMPACT=100000 and agent_driver.py forwards it, so all 40 agents per arm would exit
# 1 before doing any work. Upgrading is the fix the campaign chose over dropping the flag.
#
# Validation is the point of this job, not the install: it asserts the binary accepts the EXACT
# flag set agent_driver.py passes. "Unknown option" anywhere in that output fails the build, so a
# CLI that renamed a flag between versions is caught HERE and not by 40 dead agents on 6 nodes.
#
# Patches the image IN PLACE with a .before-claude-upgrade backup, exactly like the inference
# image patches. Name stays optarena-ce-amd-mi300-v4.sqsh so every EDF and env file keeps working.
#SBATCH --job-name=upgrade-claude
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/upgrade-claude-%j.out
#SBATCH --error=logs/upgrade-claude-%j.err

set -Eeuxo pipefail
umask 0022

# "latest" is resolved ONCE here and frozen into the image, so every arm runs the same binary no
# matter when it is submitted -- the image is the pin. The resolved version is printed below and
# belongs in the campaign notes: the agent CLI is an experiment variable, and results from the
# 2.1.197 generation (jobs 594529-594538) are not harness-comparable to results from this one.
CLAUDE_VERSION="${CLAUDE_VERSION:-latest}"

SCRATCH="${SCRATCH:-/capstor/scratch/cscs/ybudanaz/x86_64}"
CE_IMAGE_DIR="${CE_IMAGE_DIR:-${SCRATCH}/ce-images}"

SOURCE_IMAGE="${CE_IMAGE_DIR}/optarena-ce-amd-mi300-v4.sqsh"
TEMP_IMAGE="${CE_IMAGE_DIR}/optarena-ce-amd-mi300-v4.claude.tmp.sqsh"
FINAL_IMAGE="${CE_IMAGE_DIR}/optarena-ce-amd-mi300-v4.sqsh"
BACKUP_IMAGE="${CE_IMAGE_DIR}/optarena-ce-amd-mi300-v4.before-claude-upgrade.sqsh"

BUILD_NAME="optarena-agent-claude-upgrade"

test -f "${SOURCE_IMAGE}"

echo "Host:   $(hostname)"
echo "Source: ${SOURCE_IMAGE}"
echo "Claude: ${CLAUDE_VERSION}"

enroot remove --force "${BUILD_NAME}" 2>/dev/null || true
rm -f "${TEMP_IMAGE}" "${TEMP_IMAGE}.sha256"

enroot create --name "${BUILD_NAME}" "${SOURCE_IMAGE}"

enroot start --root --rw --env "CLAUDE_VERSION=${CLAUDE_VERSION}" "${BUILD_NAME}" bash -lc '
    set -Eeuxo pipefail

    node --version
    npm --version
    echo "before: $(claude --version)"

    npm install -g "@anthropic-ai/claude-code@${CLAUDE_VERSION}"
    hash -r

    echo "after: $(claude --version)"

    # The gate. Every flag agent_driver.py passes, in one invocation, with a prompt that cannot
    # reach a model (no server configured) -- so this exercises ARGUMENT PARSING and nothing else.
    # A missing server makes claude fail; a missing FLAG makes it print "unknown option", and only
    # the second is a reason to reject the image.
    set +e
    out="$(claude --bare --print "ping" \
        --model optarena-llm \
        --max-turns 1 \
        --autocompact 100000 \
        --permission-mode bypassPermissions \
        --verbose \
        --output-format stream-json \
        --tools Read,Edit \
        --allowedTools Read Edit \
        --disallowedTools WebFetch WebSearch Task Agent 2>&1)"
    set -e
    echo "--- flag probe output (first 20 lines) ---"
    printf "%s\n" "${out}" | head -20

    if printf "%s" "${out}" | grep -qi "unknown option"; then
        echo "FATAL: the upgraded CLI rejects a flag agent_driver.py passes" >&2
        printf "%s" "${out}" | grep -i "unknown option" >&2
        exit 2
    fi

    # --mcp-config takes a path; probe it separately so a bad path cannot be blamed on the set above.
    printf "%s" "{\"mcpServers\":{}}" >/tmp/mcp-probe.json
    set +e
    out2="$(claude --bare --print "ping" --model optarena-llm --max-turns 1 \
        --mcp-config /tmp/mcp-probe.json --strict-mcp-config 2>&1)"
    set -e
    if printf "%s" "${out2}" | grep -qi "unknown option"; then
        echo "FATAL: the upgraded CLI rejects --mcp-config/--strict-mcp-config" >&2
        exit 2
    fi

    echo "CLAUDE FLAG VALIDATION PASSED"
    rm -rf /root/.npm /tmp/mcp-probe.json
'

enroot export --output "${TEMP_IMAGE}" "${BUILD_NAME}"
test -s "${TEMP_IMAGE}"

if [[ ! -e "${BACKUP_IMAGE}" ]]; then
    mv "${FINAL_IMAGE}" "${BACKUP_IMAGE}"
else
    rm -f "${FINAL_IMAGE}"
fi
mv "${TEMP_IMAGE}" "${FINAL_IMAGE}"

sha256sum "${FINAL_IMAGE}" | tee "${FINAL_IMAGE}.sha256"
enroot remove --force "${BUILD_NAME}" 2>/dev/null || true
ls -lh "${FINAL_IMAGE}" "${BACKUP_IMAGE}"

echo "========================================"
echo "CLAUDE UPGRADE PASSED"
echo "========================================"
