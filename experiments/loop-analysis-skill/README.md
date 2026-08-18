# loop-analysis skill arm (NOT enabled by default)

Three per-language pages teaching loop-carried dependence PATTERNS (no compiler tooling --
agents have no shell): recurrence, carried scalar, indirect/overlapping writes, fission.
Motivation: skills arms grade 2-3 points MORE incorrect than base (08-18, oss120b: 7.8 vs
5.6 / 10.2 vs 8.5 / 15.5 vs 12.8) and the failures cluster on dependence-carrying kernels
(tsvc s221/s243, stencil-through-transient) -- aggression without a dependence check.

Experiment contract: THREE tracks per language --
  1. base            (no skill pages)              problems-llr*-<lang>.jsonl
  2. skills          (shipped lang-<lang> pages)   problems-llr*-<lang>-skills.jsonl
  3. skills+loopdeps (2. plus loop-deps-<lang>)    problems-llr*-<lang>-loopdeps.jsonl

Track 3 packets (per language, PYTHONHASHSEED=0 as always):

    python3 make_problems.py --track loop_level_reasoning --language c --skills \
        --extra-skill-root ../../../experiments/loop-analysis-skill \
        > problems-llr4-c-loopdeps.jsonl

The flag inlines only this root's page matching the packet language (suffix convention
<name>-<language>) on TOP of the normal skills set, so track 3 differs from track 2 by
exactly one page.

Keep the pages OUT of hpcagent_bench/skills/ until that experiment is scheduled -- the
default discovery would silently add them to every skills packet.
