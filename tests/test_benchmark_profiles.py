"""Shipped benchmark profiles — plan item P3.5.

Authoring-time presets (`bfcl`/`seal_tools`/`rotbench`/`toolalpaca`/`custom`) that
fill MAGEConfig fields the user didn't set. Out of scope: benchmark execution.
"""

from __future__ import annotations

from mmar_mage.profiles import BENCHMARK_PROFILES, get_profile, list_profiles
from mmar_mage.schemas import MAGEConfig


# --------------------------------------------------------------------------- #
# Profile registry
# --------------------------------------------------------------------------- #
def test_four_headline_profiles_plus_custom_present():
    assert {"bfcl", "seal_tools", "rotbench", "toolalpaca", "custom"} <= set(list_profiles())


def test_every_profile_config_key_is_a_real_config_field():
    fields = set(MAGEConfig.model_fields)
    for name, profile in BENCHMARK_PROFILES.items():
        bad = set(profile.config) - fields
        assert not bad, f"profile {name} has non-config keys: {bad}"


def test_profiles_carry_metadata():
    bfcl = get_profile("bfcl")
    assert bfcl.notes  # quirks documented
    assert bfcl.default_topology  # topology preference
    assert get_profile("unknown") is None


# --------------------------------------------------------------------------- #
# Application to MAGEConfig
# --------------------------------------------------------------------------- #
def test_profile_fills_unset_fields():
    c = MAGEConfig(api_key="t", benchmark_profile="bfcl")
    assert c.allowed_step_types == ["llm", "tool"]
    assert c.simplicity_bias == 0.6
    assert c.simplicity_max_steps == 5


def test_explicit_value_overrides_profile():
    c = MAGEConfig(api_key="t", benchmark_profile="bfcl", simplicity_bias=0.1)
    assert c.simplicity_bias == 0.1  # user wins
    assert c.allowed_step_types == ["llm", "tool"]  # other profile fields still applied


def test_no_profile_keeps_defaults():
    c = MAGEConfig(api_key="t")
    assert c.allowed_step_types is None
    assert c.simplicity_bias == 0.5


def test_toolalpaca_enables_tool_discovery_types():
    c = MAGEConfig(api_key="t", benchmark_profile="toolalpaca")
    assert "tool_discovery" in c.allowed_step_types
    assert "mcp" in c.allowed_step_types


def test_profile_allowed_types_are_validated():
    # the profile's allowed_step_types must pass the after-validator (all valid types)
    for name in list_profiles():
        c = MAGEConfig(api_key="t", benchmark_profile=name)  # must not raise
        assert c.benchmark_profile == name


def test_for_benchmark_factory():
    c = MAGEConfig.for_benchmark("rotbench", api_key="t")
    assert c.benchmark_profile == "rotbench"
    assert "evaluation" in c.allowed_step_types
    # factory kwargs still win over the profile
    c2 = MAGEConfig.for_benchmark("rotbench", api_key="t", simplicity_bias=0.0)
    assert c2.simplicity_bias == 0.0


def test_custom_profile_applies_no_overrides():
    c = MAGEConfig(api_key="t", benchmark_profile="custom")
    assert c.allowed_step_types is None  # nothing forced
    assert c.simplicity_bias == 0.5


# --------------------------------------------------------------------------- #
# Real-execution scenario — a profiled config drives a real generate()
# --------------------------------------------------------------------------- #
async def test_generate_with_profile_respects_allowed_types():
    import json
    from unittest.mock import AsyncMock, MagicMock

    from mmar_mage.generator import MAGEGenerator

    # bfcl profile → allowed_step_types=["llm","tool"]; fast-mode chain with a tool step
    chain = {
        "name": "c",
        "steps": [
            {"step_type": "llm", "number": 1, "title": "A", "aim": "plan", "dependencies": []},
            {"step_type": "tool", "number": 2, "title": "B", "aim": "call", "dependencies": [1], "tool_name": "f"},
        ],
    }
    gen = MAGEGenerator(config=MAGEConfig.for_benchmark("bfcl", api_key="t", mode="fast"))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps(chain))

    result = await gen.generate("call the function")
    types = {s["step_type"] for s in result.chain_dict["steps"]}
    assert types <= {"llm", "tool"}  # within the bfcl profile's allowed types
    assert gen._config.simplicity_bias == 0.6  # profile applied
