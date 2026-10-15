"""Skill execution modes — plan item P2.4.

``refine_agent_skill_step`` fills sensible execution-mode / runtime / trust
defaults for agent_skill steps based on their signals, without overriding explicit
choices. The real-execution check confirms the refined+exported config validates
against the real ``mmar_carl.AgentSkillStepConfig``.
"""

from __future__ import annotations

from typing import Any

import pytest

from mmar_mage.carl_export import refine_agent_skill_step, to_carl_compat


def _step(**fields: Any) -> dict[str, Any]:
    base = {"number": 1, "step_type": "agent_skill", "title": "S", "aim": "do", "dependencies": []}
    base.update(fields)
    return base


# --------------------------------------------------------------------------- #
# execution_mode
# --------------------------------------------------------------------------- #
def test_output_schema_implies_llm_agent():
    out = refine_agent_skill_step(_step(skill="local://x", skill_output_schema={"k": "int"}))
    assert out["skill_execution_mode"] == "llm_agent"


def test_no_output_schema_leaves_mode_unset():
    out = refine_agent_skill_step(_step(skill="local://x"))
    assert "skill_execution_mode" not in out  # defaults to "llm" downstream


def test_explicit_mode_not_overridden():
    out = refine_agent_skill_step(
        _step(skill="local://x", skill_output_schema={"k": "int"}, skill_execution_mode="subagent")
    )
    assert out["skill_execution_mode"] == "subagent"  # explicit choice preserved


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
def test_remote_skill_with_script_gets_docker():
    out = refine_agent_skill_step(_step(skill="github://o/r/s@main", skill_script_name="run.py"))
    assert out["skill_runtime"] == "docker"


def test_local_skill_with_script_stays_local():
    out = refine_agent_skill_step(_step(skill="local://x", skill_script_name="run.py"))
    assert "skill_runtime" not in out  # local → default 'local'


def test_remote_without_script_no_docker():
    out = refine_agent_skill_step(_step(skill="https://example.com/skill"))
    assert "skill_runtime" not in out


# --------------------------------------------------------------------------- #
# trust / SHA-pinning
# --------------------------------------------------------------------------- #
def test_remote_with_sha_gets_pinned():
    out = refine_agent_skill_step(_step(skill="github://o/r/s@main", skill_sha256="deadbeef"))
    assert out["skill_trust_policy"] == "sha_pinned"


def test_remote_without_sha_not_forced_to_pinned():
    # sha_pinned REQUIRES a sha — forcing it without one would be invalid
    out = refine_agent_skill_step(_step(skill="github://o/r/s@main"))
    assert "skill_trust_policy" not in out


def test_local_skill_not_pinned():
    out = refine_agent_skill_step(_step(skill="local://x", skill_sha256="deadbeef"))
    assert "skill_trust_policy" not in out  # local is trusted; no pinning


def test_skill_source_dict_uri_extraction():
    out = refine_agent_skill_step(_step(skill={"git_url": "https://github.com/o/r"}, skill_sha256="abc"))
    assert out["skill_trust_policy"] == "sha_pinned"


# --------------------------------------------------------------------------- #
# Integration: refinement runs inside to_carl_compat
# --------------------------------------------------------------------------- #
def test_to_carl_compat_applies_refinement():
    chain = {
        "name": "c",
        "steps": [
            _step(skill="github://o/r/s@main", skill_task="x", skill_sha256="abc", skill_output_schema={"k": "int"})
        ],
    }
    cfg = to_carl_compat(chain)["steps"][0]["step_config"]
    assert cfg["execution_mode"] == "llm_agent"
    assert cfg["trust_policy"] == "sha_pinned"


# --------------------------------------------------------------------------- #
# Real-execution scenario — refined config validates against real CARL schema
# --------------------------------------------------------------------------- #
def test_refined_config_loads_into_real_carl():
    pytest.importorskip("mmar_carl")
    # AgentSkillStepConfig only exists on CARLs that support the agent_skill type
    # (agent-features); released mmar-carl translates agent_skill → llm instead.
    pytest.importorskip("mmar_carl.models.agent_skill")
    from mmar_carl.models.agent_skill import AgentSkillStepConfig

    chain = {
        "name": "c",
        "steps": [
            _step(
                skill="github://anthropics/skills/pdf@main",
                skill_task="extract tables",
                skill_script_name="extract.py",
                skill_sha256="abc123",
                skill_output_schema={"tables": "list"},
            )
        ],
    }
    cfg = to_carl_compat(chain)["steps"][0]["step_config"]
    loaded = AgentSkillStepConfig.model_validate(cfg)
    assert loaded.execution_mode.value == "llm_agent"
    assert loaded.runtime == "docker"
    assert loaded.trust_policy == "sha_pinned"
    assert loaded.skill_sha256 == "abc123"
