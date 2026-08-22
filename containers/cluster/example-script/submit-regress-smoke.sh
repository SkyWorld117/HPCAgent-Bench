#!/usr/bin/env bash
# Re-run ONLY the kernels the skills arms previously got wrong, on the v3 pages.
#
# The oss120b campaign graded the skills arms 2-3 points MORE incorrect than base, clustered on
# dependence-carrying loops. Skills v3 rewrote the pages around that (classify-then-thread bins,
# inscan, de-biased stdpar/do-concurrent, conflict-free scatter). This smoke asks the narrow
# question those pages were changed to answer: on the 69 kernels that regressed, does v3 still
# get them wrong?
#
# The kernel lists come from the judge DBs of 598186-598191 -- rows with status='incorrect' in a
# skills arm whose base arm was clean on the same kernel (regressed-kernels-<lang>.txt, and
# skills_regressions.json in llr4-analysis for the provenance).
#
# Every env is byte-identical to the real arm it derives from except the packet, the arm name,
# the wave width and JUDGE_NODES=4, so a difference here is the pages and not the harness.
# One agent per kernel means a single wave; four judges keep the queue clear of the timeouts
# that contaminated the campaign measurement (21.7% at 40 agents / 4 judge nodes).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# beverin.sbatch writes --output=results/... relative to the submission cwd, which is this
# directory. Slurm drops the file when the folder is missing and the job runs blind: no vLLM
# serve log, no agent_driver output, only the per-role logs under RUN_DIR.
mkdir -p results

# No --account, ever: these submit beverin.sbatch and nothing else, and on beverin root,
# a-g200 and a-g34 are scheduling-identical, so -A only ever picks a billing line nobody
# chose. The knob is absent rather than empty -- an empty default is still a knob, and it
# defaulted to a real account here once already.
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
