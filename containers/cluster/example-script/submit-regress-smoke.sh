#!/usr/bin/env bash
# Re-run ONLY the kernels the skills arms previously got wrong, on the v3 pages.
#
# The oss120b campaign graded skills arms 2-3 points MORE incorrect than base, clustered on
# dependence-carrying loops; v3 rewrote the pages around that. This asks the narrow question
# those pages were changed to answer: on the 69 kernels that regressed, does v3 still fail?
#
# Kernel lists come from the judge DBs of 598186-598191: status='incorrect' in a skills arm
# whose base arm was clean on the same kernel (provenance in llr4-analysis).
#
# Every env is byte-identical to the arm it derives from except the packet, arm name, wave width
# and JUDGE_NODES=4, so a difference here is the pages, not the harness. One agent per kernel is
# a single wave; four judges keep out the timeouts that contaminated the campaign (21.7%).
# No --account: beverin schedules root, a-g200 and a-g34 identically, so -A only picks a
# billing line nobody chose.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# beverin.sbatch writes --output=results/... relative to here; slurm DROPS the file when the
# folder is missing and the job then runs with no serve log at all.
mkdir -p results

MODELS="${MODELS:-oss120b qwen30b}"
LANGS="${LANGS:-c cpp fortran}"

for lang in ${LANGS}; do
    f="problems-regress-${lang}-skills.jsonl"
    [[ -s "$f" ]] || {
        echo "missing $f -- regenerate with:" >&2
        echo "  PYTHONHASHSEED=0 make_problems.py --track loop_level_reasoning --language ${lang}" \
             "--skills --kernels-file regressed-kernels-${lang}.txt > $f" >&2
        exit 2
    }
done

for model in ${MODELS}; do
    for lang in ${LANGS}; do
        env_file="$PWD/.env.regress-${model}-${lang}-skills"
        [[ -s "${env_file}" ]] || { echo "missing ${env_file}" >&2; exit 2; }
        inf="$(sed -n 's/^INFERENCE_NODES=\([0-9]*\).*/\1/p' "${env_file}")"
        judge="$(sed -n 's/^JUDGE_NODES=\([0-9]*\).*/\1/p' "${env_file}")"
        nodes=$((inf + 1 + judge))
        sbatch --nodes="${nodes}" --time=04:00:00 "$@" \
            --job-name="regress-${model}-${lang}" \
            --export=ALL,CLUSTER_ENV_FILE="${env_file}" beverin.sbatch
        echo "submitted regress-${model}-${lang}-skills on ${nodes} nodes"
    done
done
