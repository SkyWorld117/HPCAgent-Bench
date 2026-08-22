#!/usr/bin/env bash
# Submit the llr5 two-leg experiment for ONE model.
#
#   leg 1: base prompt only -- no optimization hints, no skills packet.
#   leg 2: skill-usage triggers in the main prompt ({{HINTS}} <- skill-triggers.md; the two cpp
#          arms predate that and still point at hints.md) plus the per-language packet
#          (optimization-hints + lang-<L> + loop-transformations-<L> + openmp-<L>).
#          llr6 is the successor and puts hints in the main prompt instead -- see it first.
#          C submits first; Fortran is chained with --dependency=afterany on the C job.
#
#   MODEL=oss120b ./submit-llr5.sh              # both legs, c + fortran
#   MODEL=qwen30b LEGS=2 ./submit-llr5.sh       # hints+skills leg only
#   MODEL=oss120b LANGS=c ./submit-llr5.sh
#
# Extra args go to sbatch verbatim. No --account: beverin schedules root, a-g200 and a-g34
# identically, so -A only picks a billing line nobody chose.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# beverin.sbatch writes --output=results/... relative to here; slurm DROPS the file when the
# folder is missing and the job then runs with no serve log at all.
mkdir -p results
. ./arm_nodes.sh
. ./check_problems.sh

MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || { echo "MODEL is required, e.g. MODEL=oss120b $0" >&2; exit 2; }
LANGS="${LANGS:-c fortran}"
LEGS="${LEGS:-1 2}"
case "${MODEL}" in
    kimi*) TIME="${TIME:-24:00:00}" ;;
    *) TIME="${TIME:-20:00:00}" ;;
esac

for lang in ${LANGS}; do
    for f in "problems-llr5-${lang}.jsonl" "problems-llr5-${lang}-skills.jsonl"; do
        problems_fresh "$f" || exit 2
    done
done

submit_arm() {  # submit_arm <arm> [extra sbatch args...] -> prints the job id
    local arm="$1"; shift
    [[ -f ".env.${arm}" ]] || { echo "no env file for ${arm} -- check MODEL=${MODEL}" >&2; exit 2; }
    sbatch --parsable --nodes="$(arm_nodes ".env.${arm}")" --time="${TIME}" \
        --job-name="${arm}" "$@" \
        --export=ALL,CLUSTER_ENV_FILE="$PWD/.env.${arm}" beverin.sbatch
}

for leg in ${LEGS}; do
    if [[ "${leg}" == "1" ]]; then
        for lang in ${LANGS}; do
            jid="$(submit_arm "llr5-${MODEL}-${lang}" "$@")"
            echo "submitted llr5-${MODEL}-${lang} (job ${jid})"
        done
    else
        # Leg 2 is sequenced: C first, then each remaining language after the previous finished,
        # so one inference allocation's worth of leg-2 load runs at a time.
        prev=""
        for lang in ${LANGS}; do
            if [[ -n "${prev}" ]]; then
                jid="$(submit_arm "llr5-${MODEL}-${lang}-skills" --dependency="afterany:${prev}" "$@")"
                echo "submitted llr5-${MODEL}-${lang}-skills (job ${jid}, after ${prev})"
            else
                jid="$(submit_arm "llr5-${MODEL}-${lang}-skills" "$@")"
                echo "submitted llr5-${MODEL}-${lang}-skills (job ${jid})"
            fi
            prev="${jid}"
        done
    fi
done
