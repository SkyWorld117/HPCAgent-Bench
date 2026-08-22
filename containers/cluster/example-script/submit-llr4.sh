#!/usr/bin/env bash
# Submit the 6 llr4 arms for ONE model: {c,cpp,fortran} x {off,skills}.
# MODEL names the arm family, so it also names the env files: .env.llr4-<MODEL>-<lang>[-skills].
#   MODEL=qwen30b ./submit-llr4.sh                 # all 6
#   MODEL=kimi27code LANGS=c ARMS=off ./submit-llr4.sh
# Extra args go to sbatch verbatim (e.g. --partition). No --account: beverin schedules root,
# a-g200 and a-g34 identically, so -A only picks a billing line nobody chose.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# beverin.sbatch writes --output=results/... relative to here; slurm DROPS the file when the
# folder is missing and the job then runs with no serve log at all.
mkdir -p results
. ./arm_nodes.sh

MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || { echo "MODEL is required, e.g. MODEL=qwen30b $0" >&2; exit 2; }

LANGS="${LANGS:-c cpp fortran}"
# ARMS="off" / ARMS="skills" re-runs one side without a hand-rolled sbatch. Words, not the raw
# suffix: the off arm's suffix is empty, and an empty element cannot survive a word list.
ARMS="${ARMS:-off skills}"
# 598185 hit exactly its 16 h limit while siblings finished in 2-6 h. Kimi gets longer: at
# 6.06 tok/s per request (599301) an arm covers a fraction of what qwen/oss does in the window.
case "${MODEL}" in
    kimi*) TIME="${TIME:-24:00:00}" ;;
    *) TIME="${TIME:-20:00:00}" ;;
esac
# Skills arms read the llr4 packet; off arms reuse the unchanged llr2 task text. Refuse a stale
# or missing list rather than grading against one.
for lang in ${LANGS}; do
    for f in "problems-llr4-${lang}-skills.jsonl" "problems-llr2-${lang}.jsonl"; do
        [[ -s "$f" ]] || { echo "missing problems file: $f -- regenerate with make_problems.py" >&2; exit 2; }
    done
done

for lang in ${LANGS}; do
    for arm_kind in ${ARMS}; do
        [[ "${arm_kind}" == "off" ]] && suffix="" || suffix="-${arm_kind}"
        arm="llr4-${MODEL}-${lang}${suffix}"
        # A mistyped MODEL otherwise surfaces as an arm_nodes failure on a path nobody recognises.
        [[ -f ".env.${arm}" ]] || { echo "no env file for ${arm} -- check MODEL=${MODEL}" >&2; exit 2; }
        sbatch --nodes="$(arm_nodes ".env.${arm}")" --time="${TIME}" \
            --job-name="${arm}" "$@" \
            --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.${arm}" beverin.sbatch
        echo "submitted ${arm}"
    done
done
