# loop-analysis skill arm (NOT enabled by default)

Three per-language pages teaching loop-carried dependence PATTERNS (no compiler tooling --
agents have no shell): recurrence, carried scalar, indirect/overlapping writes, fission.
Motivation: skills arms grade 2-3 points MORE incorrect than base (08-18, oss120b: 7.8 vs
5.6 / 10.2 vs 8.5 / 15.5 vs 12.8) and the failures cluster on dependence-carrying kernels
(tsvc s221/s243, stencil-through-transient) -- aggression without a dependence check.

This is the THIRD experiment arm: base vs skills vs skills+loop-deps. Enable at packet
generation only, via the prompt user-root mechanism (prompts are discovered from
`<root>/skills/*/SKILL.md`, user roots shadow built-ins):

    prompt.template_dirs += ["experiments/loop-analysis-skill"]   # config.yaml, or the
    make_problems.py --skills ... with the config override            equivalent CLI/env

Keep the pages OUT of hpcagent_bench/skills/ until that experiment is scheduled -- the
default discovery would silently add them to every skills packet.
