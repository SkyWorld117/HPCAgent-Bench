#!/usr/bin/env bash
# Submit the 6 llr4 arms for ONE model: {c,cpp,fortran} x {off,skills}.
# MODEL names the arm family, so it also names the env files: .env.llr4-<MODEL>-<lang>[-skills].
#   MODEL=qwen30b ./submit-llr4.sh                 # all 6
#   MODEL=kimi27code LANGS=c ARMS=off ./submit-llr4.sh
# Extra args go to sbatch verbatim (e.g. --partition). No --account -- see below.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# beverin.sbatch writes --output=results/... relative to the submission cwd, which is this
# directory. Slurm drops the file when the folder is missing and the job runs blind: no vLLM
# serve log, no agent_driver output, only the per-role logs under RUN_DIR.
mkdir -p results
. ./arm_nodes.sh

MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || { echo "MODEL is required, e.g. MODEL=qwen30b $0" >&2; exit 2; }

# No --account, ever: these submit beverin.sbatch and nothing else, and on beverin root,
# a-g200 and a-g34 are scheduling-identical, so -A only ever picks a billing line nobody
# chose. The knob is absent rather than empty -- an empty default is still a knob, and it
# defaulted to a real account here once already.
LANGS="${LANGS:-c cpp fortran}"
# Which side of the skills-on/off pair to submit, so re-running ONE side does not need a
# hand-rolled sbatch:  ARMS="off" ...  or  ARMS="skills" ...  (default: both).
# Spelled with words rather than the raw suffix because the off arm's suffix is the empty
# string, and an empty element cannot survive a shell word list.
ARMS="${ARMS:-off skills}"
# 598185 hit exactly its 16 h limit while its siblings finished in 2-6 h. Kimi gets longer: it
# decodes at 6.06 tok/s per request with 8 concurrent (599301, tp4/pp4), so an arm covers a
# fraction of the 242 kernels a qwen/oss arm gets through in the same window.
case "${MODEL}" in
    kimi*) TIME="${TIME:-24:00:00}" ;;
    *) TIME="${TIME:-20:00:00}" ;;
esac
# Skills arms read the llr4 packet (lang page + openmp/openacc/stdpar/do-concurrent pages);
# off arms reuse the unchanged llr2 task text. Refuse to submit against a stale or missing list.
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
