# Copyright 2021 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""SIZE DIMENSIONS vs CONFIG KNOBS manifest schema (see hpcagent_bench/spec.py's module docstring).

Locks the additive, backward-compatible split: a manifest still declaring the legacy flat
``parameters:`` block loads unchanged, while one declaring the new ``dimensions:`` (+ optional
``config:``/``constraints:``) keeps ``BenchSpec.parameters`` as the same merged
``{preset: {symbol: value}}`` view every existing consumer reads.
"""
from typing import Any, Dict

import pytest

from hpcagent_bench.spec import BenchSpec, ConfigKnob


def _raw(short_name: str = "dimtest", **overrides: Any) -> Dict[str, Any]:
    """A minimal, hermetic manifest dict: every field the caller doesn't override is either
    required-and-supplied or optional-and-omitted, so ``from_dict`` never touches the filesystem
    (input_args/array_args/func_name are given explicitly, not derived from a numpy reference)."""
    base: Dict[str, Any] = {
        "short_name": short_name,
        "name": short_name,
        "relative_path": short_name,
        "module_name": short_name,
        "func_name": "kernel",
        "input_args": ["x", "N"],
        "array_args": ["x"],
        "output_args": ["x"],
    }
    base.update(overrides)
    return base


def test_old_style_manifest_still_loads_and_exposes_parameters() -> None:
    """A manifest with only the legacy 'parameters:' block loads unchanged: 'dimensions' mirrors
    it verbatim and 'config' stays empty."""
    raw = _raw(parameters={"S": {"N": 16}, "M": {"N": 32}})
    spec = BenchSpec.from_dict(raw, source="<test>")
    assert spec.parameters == {"S": {"N": 16}, "M": {"N": 32}}
    assert spec.dimensions == {"S": {"N": 16}, "M": {"N": 32}}
    assert spec.config == {}
    assert spec.constraints == ()
    assert spec.coverage == {"require": "all"}


def test_new_style_separates_dimensions_and_config() -> None:
    """A manifest declaring 'dimensions:' + 'config:' keeps them apart in .dimensions/.config,
    while .parameters merges each config knob's representative value into every preset -- the
    view frameworks/benchmark.py, support/bindings/contract.py, and initialize.py already read."""
    raw = _raw(
        dimensions={
            "S": {
                "N": 16
            },
            "M": {
                "N": 32
            }
        },
        config={
            "lvn_only": {
                "domain": [0, 1],
                "selects": "branch"
            },
            "max_iter": {
                "value": 200,
                "selects": "iteration"
            },
        },
    )
    spec = BenchSpec.from_dict(raw, source="<test>")
    assert spec.dimensions == {"S": {"N": 16}, "M": {"N": 32}}
    assert set(spec.config) == {"lvn_only", "max_iter"}
    assert spec.config["lvn_only"] == ConfigKnob(domain=(0, 1), value=None, selects="branch")
    assert spec.config["max_iter"] == ConfigKnob(domain=None, value=200, selects="iteration")
    assert spec.config["lvn_only"].representative == 0
    assert spec.config["max_iter"].representative == 200
    assert spec.parameters == {
        "S": {
            "N": 16,
            "lvn_only": 0,
            "max_iter": 200
        },
        "M": {
            "N": 32,
            "lvn_only": 0,
            "max_iter": 200
        },
    }


def test_both_parameters_and_dimensions_raises() -> None:
    """Declaring both the legacy and the new size-symbol block is an ambiguity error, not a
    silent 'one wins' resolution."""
    raw = _raw(parameters={"S": {"N": 16}}, dimensions={"S": {"N": 16}})
    with pytest.raises(ValueError, match="both 'parameters'"):
        BenchSpec.from_dict(raw, source="<test>")


def test_config_domain_and_value_raises() -> None:
    """A config entry declaring BOTH 'domain' and 'value' is ambiguous: is it an axis or pinned?"""
    raw = _raw(dimensions={"S": {"N": 16}}, config={"lvn": {"domain": [32, 64], "value": 32}})
    with pytest.raises(ValueError, match="both 'domain' and 'value'"):
        BenchSpec.from_dict(raw, source="<test>")


def test_config_neither_domain_nor_value_raises() -> None:
    """A config entry declaring NEITHER 'domain' nor 'value' has no concrete or fuzzable value."""
    raw = _raw(dimensions={"S": {"N": 16}}, config={"lvn": {"selects": "tile"}})
    with pytest.raises(ValueError, match="neither 'domain'"):
        BenchSpec.from_dict(raw, source="<test>")


def test_mismatched_preset_key_sets_raises() -> None:
    """Every preset in 'dimensions:' must declare the SAME symbol set -- a symbol missing from one
    preset used to silently union away (spec.py's old 'parameters' handling) and explode at run
    time; the new schema catches it at load."""
    raw = _raw(dimensions={"S": {"N": 16}, "M": {"N": 32, "K": 2}})
    with pytest.raises(ValueError, match="same symbol set"):
        BenchSpec.from_dict(raw, source="<test>")


def test_dimension_and_config_overlap_raises() -> None:
    """A symbol cannot be declared in BOTH 'dimensions' and 'config' -- the merge in .parameters
    would let the config value silently clobber the dimension value."""
    raw = _raw(dimensions={"S": {"N": 16}}, config={"N": {"value": 16}})
    with pytest.raises(ValueError, match="declared in BOTH"):
        BenchSpec.from_dict(raw, source="<test>")


def test_violated_constraint_raises() -> None:
    """A 'constraints:' expression is evaluated at LOAD against every preset's merged dimension +
    config-representative values; a violated one raises immediately, naming the expression."""
    raw = _raw(
        dimensions={"S": {
            "nproma": 16
        }},
        config={"lvn": {
            "value": 32
        }},
        constraints=["lvn <= nproma"],
    )
    with pytest.raises(ValueError, match=r"constraint 'lvn <= nproma' is violated"):
        BenchSpec.from_dict(raw, source="<test>")


def test_satisfied_constraint_loads() -> None:
    """A constraint that holds for every preset is a no-op at load time and survives on the spec."""
    raw = _raw(
        dimensions={"S": {
            "nproma": 64
        }},
        config={"lvn": {
            "value": 32
        }},
        constraints=["lvn <= nproma"],
    )
    spec = BenchSpec.from_dict(raw, source="<test>")
    assert spec.constraints == ("lvn <= nproma", )
    assert spec.parameters == {"S": {"nproma": 64, "lvn": 32}}
