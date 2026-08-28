"""Build the wave-2 arms: qwen3.8 moved to SGLang, and the Kimi kernels wave 1 never submitted.

qwen3.8 on vLLM is not viable -- 610203/610204 swept seven legs on two vLLM versions and the best
decoded 8.5 tok/s, with mtp and all three aiter legs refusing to serve at all. The same model on
SGLang measured 163.0 tok/s at 9/9 accuracy (610229), so these arms carry that run's serve args
verbatim; --tp-size is NOT among them because run_cluster.sh derives it from GPUS_PER_NODE.

The Kimi retries re-run only the kernels that produced no submission, and change nothing else:
same agent budget, same judge sizing, same serve args. Wave-1 rows and retry rows have to stay
poolable, and a retry that also moves the runner config would measure two things at once.
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parents[2] / "containers/cluster/example-script"
RUN_ROOT = "RUN_ROOT=${SCRATCH:-/iopsstor/scratch/cscs/$USER}/hpcagent-bench-runs/llr8w1-20260827"

# From 610229's server_args: triton attention, one node at tp4, 0.18 static KV. aiter's master
# switch stays ON here -- on SGLang it imports a prebuilt module_aiter_core and serves; it is the
# vLLM path that JIT-builds on first request and dies (610251/610252).
SGLANG_BLOCK = """
# --- SGLang, from the 610229 config: 163.0 tok/s, 9/9 accuracy, 96 requests, 0 errors. ---
# vLLM cannot serve this hybrid Gated-DeltaNet backbone at length: 610203/610204 measured 8.5
# tok/s on its best leg of seven and DID NOT SERVE on five of them.
INFERENCE_ENGINE=sglang
SGLANG_PYTHON=/opt/venv/bin/python3
SGLANG_USE_AITER=1
SGLANG_SET_CPU_AFFINITY=0
SGLANG_EXTRA_ARGS="--trust-remote-code --attention-backend triton --language-only \
--watchdog-timeout 1800 --context-length 262144 --mem-fraction-static 0.18 \
--max-running-requests 128 --enable-metrics --reasoning-parser qwen3 --tool-call-parser qwen3_coder"
"""


def edit(path: pathlib.Path, subs: list[tuple[str, str]], append: str = "") -> None:
    text = path.read_text()
    for pattern, replacement in subs:
        new, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"{path.name}: {pattern!r} matched {n} times")
        text = new
    # Appending twice would hand the server two copies of every flag, so the block goes on only
    # once however often this runs.
    if append and "INFERENCE_ENGINE=sglang" not in text:
        text += append
    path.write_text(text)
    print(f"wrote {path.name}")


for arm in ("c", "c-skills", "fortran", "fortran-skills"):
    # 163 tok/s over 20 agents is 8.2 each, at the >=8 tok/s per-agent floor the judge sizing
    # assumes. 120 was a vLLM-era number for a server that never reached this rate.
    # The image moves WITH the engine. Naming SGLANG_PYTHON while leaving this on a vLLM image
    # points /opt/venv/bin/python3 at an interpreter with neither sglang nor huggingface_hub, and
    # the run dies resolving the model path (610646/610647).
    edit(HERE / f".env.llr8-qwen38-{arm}", [(r"^RUN_ROOT=.*$", RUN_ROOT.replace("\\", "\\\\")),
                                            (r"^AGENTS_PER_NODE=.*$", "AGENTS_PER_NODE=20"),
                                            (r"^INFERENCE_CE_ENV=.*$", "INFERENCE_CE_ENV=sglang-rocm-mi30x")],
         append=SGLANG_BLOCK)

for src, dst, problems, campaign_arm in (
    ("c-a", "c-r1", "problems-llr8kimi-c-r1.jsonl", "llr8-kimi27sglang-c-r1"),
    ("c-skills-a", "c-skills-r1", "problems-llr8kimi-c-skills-r1.jsonl", "llr8-kimi27sglang-c-skills-r1"),
):
    target = HERE / f".env.llr8-kimi27sglang-{dst}"
    target.write_text((HERE / f".env.llr8-kimi27sglang-{src}").read_text())
    edit(target, [(r"^PROBLEMS_FILE=.*$", f"PROBLEMS_FILE={problems}"),
                  (r"^CAMPAIGN_ARM=.*$", f"CAMPAIGN_ARM={campaign_arm}")])

# oss120b serves fine on vLLM once AITER's master switch is off (97ff5d4f), so these arms keep the
# engine they have always completed on and only pick up the wave root and the agent count.
for arm in ("fortran", "fortran-skills"):
    edit(HERE / f".env.llr8-oss120b-{arm}", [(r"^RUN_ROOT=.*$", RUN_ROOT.replace("\\", "\\\\")),
                                             (r"^AGENTS_PER_NODE=.*$", "AGENTS_PER_NODE=40")])
