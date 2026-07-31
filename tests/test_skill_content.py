# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the shipped skills SAY, checked against the code that has to make it true.

``tests/test_prompt_skills.py`` covers how a skill is discovered, parsed and layered -- the
mechanism. This covers the content, because a skill is documentation that gets INJECTED INTO
EVERY PROMPT and therefore drifts in the most expensive possible direction: silently, into an
agent's instructions. A skill naming a metric the code dropped, or quoting a perf event the code
no longer records, is worse than no skill -- it is a confident wrong answer with the repo's name
on it.

Every assertion here is a cross-check against a constant or a table that already exists. Nothing
pins prose for its own sake: rewording is free, contradicting the code is not.
"""
import pathlib
import re
from typing import Dict

import pytest

from hpcagent_bench import flags, paths, perf_reports
from hpcagent_bench.harness import papi
from hpcagent_bench.harness.prompts import load_skills, parse_skill

SKILLS = paths.ROOT / "hpcagent_bench" / "skills"


def skill_bodies() -> Dict[str, str]:
    """Every shipped skill's body, keyed by directory name."""
    general, others = load_skills(())
    return {s.name: s.body for s in [general] + others}


PROFILING = "profiling"


def test_every_shipped_skill_parses_and_is_indexable() -> None:
    """A skill with no description is invisible in the index the agent reads to choose one."""
    general, others = load_skills(())
    for skill in [general] + others:
        assert skill.body.strip(), f"{skill.name}: empty body"
        assert skill.description.strip(), f"{skill.name}: no description, so the index cannot list it"
        assert len(skill.description) < 200, f"{skill.name}: the index line is a line, not a paragraph"


def test_a_skill_directory_name_is_its_frontmatter_name() -> None:
    """The DIRECTORY is a skill's identity (that is what a user root overrides); frontmatter that
    disagrees makes an override silently miss."""
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        skill = parse_skill(path.read_text(), path)
        assert skill.name == path.parent.name, f"{path}: frontmatter says {skill.name!r}"


def test_skills_are_ascii_and_have_no_trailing_whitespace() -> None:
    """These go into a prompt verbatim. Smart quotes and stray trailing spaces are tokens spent on
    nothing, and the repo is ASCII everywhere else."""
    for path in sorted(SKILLS.glob("*/SKILL.md")):
        text = path.read_text()
        bad = [c for c in text if ord(c) > 127]
        assert not bad, f"{path}: non-ASCII {sorted(set(bad))}"
        offenders = [i + 1 for i, line in enumerate(text.splitlines()) if line != line.rstrip()]
        assert not offenders, f"{path}: trailing whitespace on lines {offenders}"


def test_the_profiling_skill_names_every_metric_the_wrapper_reports() -> None:
    """The skill teaches a metric table; the code owns the metric table. Neither may add or drop
    one without the other, or an agent asks for a metric that does not exist -- or never learns
    about one that does."""
    body = skill_bodies()[PROFILING]
    named = {m for m in papi.METRICS if m in body}
    assert named == set(papi.METRICS), (f"the profiling skill does not name {sorted(set(papi.METRICS) - named)}; "
                                        "a metric the code reports but the skill never mentions is one an agent "
                                        "will not know to read")


def test_the_profiling_skill_quotes_the_perf_constants_it_teaches() -> None:
    """The skill prints a `perf record` line. If PERF_EVENT / PERF_FREQUENCY / PERF_CALL_GRAPH
    change, that line becomes a command that reproduces something else."""
    body = skill_bodies()[PROFILING]
    for constant in (perf_reports.PERF_EVENT, str(perf_reports.PERF_FREQUENCY), perf_reports.PERF_CALL_GRAPH):
        assert constant in body, f"the profiling skill no longer quotes {constant!r}"


def test_the_profiling_skill_and_the_build_flags_agree_about_frame_pointers() -> None:
    """The skill tells the reader NOT to add -fno-omit-frame-pointer, on the grounds that the
    profiled build does not need it. That is only true while DEBUG_SYMBOLS really is just -g:
    add the flag to the build and the skill starts arguing against the repo's own behaviour."""
    body = skill_bodies()[PROFILING]
    assert "-fno-omit-frame-pointer" not in " ".join(flags.DEBUG_SYMBOLS)
    assert flags.DEBUG_SYMBOLS == ["-g"], f"DEBUG_SYMBOLS is now {flags.DEBUG_SYMBOLS}; the skill says only -g"
    assert "-fno-omit-frame-pointer" in body and "-g" in body


def test_the_profiling_skill_teaches_ratios_not_just_counts() -> None:
    """The complaint this rewrite answers: naming tools is not teaching them. A raw count with no
    denominator is the thing a reader most reliably misreads."""
    body = skill_bodies()[PROFILING]
    for idea in ("IPC", "per 1k instructions", "flops per cycle", "hit rate"):
        assert idea in body, f"the profiling skill no longer explains {idea!r}"


def test_the_profiling_skill_carries_the_two_counter_traps() -> None:
    """Both are measured facts about this hardware that a reader WILL hit, and both look like
    bugs in the tool rather than properties of the CPU."""
    body = skill_bodies()[PROFILING]
    # Matched across a line break: prose is free to reflow, the CLAIM is not free to disappear.
    assert "fma_instructions" in body and re.search(r"reads exactly 0|reads 0", body), (
        "the skill must warn that PAPI_FMA_INS is a derived preset that reports 0 on Zen4")
    assert re.search(r"1 instruction and 32\s+operations",
                     body), ("the skill must state that an instruction count is not an operation count")


def test_the_profiling_skill_states_the_threading_scope_of_a_count() -> None:
    """A count summed over every thread and a count taken on the master alone have the same units
    and different meanings; the payload distinguishes them with `scope`, so the skill must too."""
    body = skill_bodies()[PROFILING]
    assert "scope" in body and "calling_thread" in body
    assert "SMT" in body, "siblings share L1/L2, which is why a miss count needs the caveat"


def test_the_profiling_skill_says_counters_cost_a_run_each() -> None:
    """Opt-in is only an informed choice if the cost is stated where the choice is made."""
    body = skill_bodies()[PROFILING]
    assert "one run per metric" in body.lower()
    assert "multiplex" in body


@pytest.mark.parametrize("doc", ["docs/kernel_extraction.md", "hpcagent_bench/docs/agent_service_contract.md"])
def test_the_long_form_docs_do_not_contradict_the_perf_constants(doc: str) -> None:
    """The two prose documents quote the same event the sampler records. They are read by humans
    porting kernels, who cannot see the constant."""
    text = (paths.ROOT / pathlib.Path(doc)).read_text()
    assert perf_reports.PERF_EVENT in text, f"{doc} no longer names the sampled event"
