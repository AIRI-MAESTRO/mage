"""End-to-end round-trip tests: ``to_carl_compat`` -> ``ReasoningChain.from_json`` (plan item P0.2).

This exercises the real load boundary that the generator uses
(``mmar_mage.generator._try_load_reasoning_chain``) for **every** step type in
``CARL_VALID_STEP_TYPES``, against the real ``mmar_carl`` library.

Findings encoded here (mmar_carl @ agent-features, verified 2026-06-01):

* ``PRESERVES_CONFIG`` — the six stable non-LLM types round-trip with their
  typed ``step_config`` intact via the canonical ``from_json`` path.
* ``TRANSLATED_TO_LLM`` — ``parallel_sampling`` / ``agent_handoff`` /
  ``supervisor`` need a runtime-only object CARL cannot rebuild from JSON, so
  ``to_carl_compat`` translates them to ``llm`` (P0.1 / P0.3 interim).
* ``UPSTREAM_GAP`` — ``agent_skill`` / ``evaluation`` / ``human_input`` /
  ``debate`` / ``mcp_resource`` / ``tool_discovery`` do **not** round-trip
  through ``from_json`` today: CARL's ``_reconstruct_step_config`` builds no
  config for most of them (``step_config`` loads as ``None`` → silent drop),
  and ``agent_skill`` raises because ``AgentSkillStepConfig`` is missing from
  the generic ``StepDescription.step_config`` union. These are marked
  ``xfail(strict=True)`` so that when CARL gains JSON support the test turns
  XPASS and tells us to promote them. See IMPROVEMENT_PLAN P0.2 / P0.3.

If ``mmar_carl`` is not installed, the load tests skip; the MAGE-internal
strict-validation test always runs.
"""

from __future__ import annotations

import json

import pytest

from mmar_mage.carl_export import to_carl_compat
from mmar_mage.generator import validate_carl_json
from mmar_mage.schemas import CARL_VALID_STEP_TYPES

try:
    from mmar_carl import ReasoningChain

    _HAS_CARL = True
except ImportError:  # pragma: no cover - depends on optional dep
    ReasoningChain = None  # type: ignore[assignment,misc]
    _HAS_CARL = False

_needs_carl = pytest.mark.skipif(not _HAS_CARL, reason="mmar_carl not installed")

# Canonical from_json behavior categories (see module docstring).
PRESERVES_CONFIG = {"tool", "mcp", "memory", "transform", "conditional", "structured_output"}
TRANSLATED_TO_LLM = {"parallel_sampling", "agent_handoff", "supervisor"}
UPSTREAM_GAP = {"agent_skill", "evaluation", "human_input", "debate", "mcp_resource", "tool_discovery"}

# Minimal flat MAGE fields that satisfy each step type's required CARL config.
_EXTRAS: dict[str, dict] = {
    "llm": {},
    "tool": {"tool_name": "do_thing"},
    "mcp": {"mcp_server_name": "srv", "mcp_server_command": "run-srv", "mcp_tool_name": "query"},
    "memory": {"memory_operation": "write", "memory_key": "k"},
    "transform": {"transform_type": "extract"},
    "conditional": {"condition_branches": [{"condition": "True", "next_step": 1}]},
    "structured_output": {"structured_output_schema": {"type": "object"}},
    "agent_skill": {"skill": "github://anthropics/skills/skills/pdf@main", "skill_task": "extract"},
    "evaluation": {
        "evaluation_target_step": 1,
        "evaluation_criteria": ["min_words:5"],
        "evaluation_method": "rule",
    },
    "human_input": {"human_prompt": "Review:", "human_default": "[none]"},
    "debate": {
        "debate_agents": [{"name": "pro", "role": "Argue for."}, {"name": "con", "role": "Argue against."}],
        "debate_judge_prompt": "Synthesize a verdict. Topic: {task}",
    },
    "mcp_resource": {
        "mcp_server_name": "docs",
        "mcp_server_transport": "sse",
        "mcp_server_url": "https://example/mcp",
        "mcp_resource_uri": "docs://api/reference.md",
    },
    "tool_discovery": {"discovery_sources": [{"kind": "module", "module": "os"}]},
    "parallel_sampling": {"sampling_n": 3},
    "agent_handoff": {"handoff_chain_path": "chains/research.json"},
    "supervisor": {"supervisor_routing_prompt": "Route it.", "supervisor_routes": {"a": "a.json"}},
}


def _minimal_chain(step_type: str) -> dict:
    """A 2-step MAGE chain: an llm 'Prep' step, then the target step depending on it."""
    return {
        "name": f"rt_{step_type}",
        "steps": [
            {"number": 1, "title": "Prep", "step_type": "llm", "aim": "Produce a draft.", "dependencies": []},
            {
                "number": 2,
                "title": "Target",
                "step_type": step_type,
                "aim": "Carry out the step.",
                "dependencies": [1],
                **_EXTRAS[step_type],
            },
        ],
    }


def test_every_type_has_an_extras_fixture():
    """Guard: a new CARL step type must get a round-trip fixture here."""
    assert set(_EXTRAS) == set(CARL_VALID_STEP_TYPES)


@pytest.mark.parametrize("step_type", sorted(CARL_VALID_STEP_TYPES))
def test_strict_carl_validation_passes_for_every_type(step_type: str):
    """MAGE-internal strict validation must accept every type's exported form."""
    compat = to_carl_compat(_minimal_chain(step_type))
    validate_carl_json(compat, strict_carl=True)  # raises on failure


@_needs_carl
@pytest.mark.parametrize("step_type", sorted(PRESERVES_CONFIG | {"llm"}))
def test_round_trip_preserves_config(step_type: str):
    compat = to_carl_compat(_minimal_chain(step_type))
    rc = ReasoningChain.from_json(json.dumps(compat))
    target = rc.steps[-1]
    assert target.step_type == step_type
    if step_type != "llm":
        assert target.step_config is not None, "typed config was dropped on round-trip"


@_needs_carl
@pytest.mark.parametrize("step_type", sorted(TRANSLATED_TO_LLM))
def test_runtime_only_types_translate_to_llm(step_type: str):
    compat = to_carl_compat(_minimal_chain(step_type))
    rc = ReasoningChain.from_json(json.dumps(compat))
    target = rc.steps[-1]
    assert target.step_type == "llm"
    assert target.aim  # carries a synthesized, non-empty aim


@_needs_carl
@pytest.mark.parametrize("step_type", sorted(UPSTREAM_GAP))
@pytest.mark.xfail(
    strict=True,
    reason="CARL from_json cannot yet round-trip these agent-features configs "
    "(config dropped to None, or agent_skill crashes the StepConfig union). "
    "See IMPROVEMENT_PLAN P0.2/P0.3 + upstream carl-experiments.",
)
def test_upstream_gap_types_round_trip_config(step_type: str):
    compat = to_carl_compat(_minimal_chain(step_type))
    rc = ReasoningChain.from_json(json.dumps(compat))  # agent_skill raises here
    target = rc.steps[-1]
    assert target.step_type == step_type
    assert target.step_config is not None  # fails (None) for the drop-config types
