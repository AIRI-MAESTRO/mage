"""Field-map correctness tests for ``mmar_mage.carl_export`` (plan item P0.1).

These tests validate the *names* produced by :func:`to_carl_compat` against the
**real** ``mmar_carl`` ``*StepConfig`` Pydantic models on the ``agent-features``
branch — the authoritative schema — rather than hard-coded expectations.

When the optional ``mmar_carl`` package is installed the real-schema tests run;
otherwise they skip. The translation tests (which need no CARL import) always
run.
"""

from __future__ import annotations

import pytest

from mmar_mage.carl_export import to_carl_compat

# Validate against the real CARL config models when the (optional) ``mmar_carl``
# package is installed; otherwise skip. Importing the real classes (rather than
# stubbing ``sys.modules``) keeps these tests isolated from the rest of the suite.
try:
    from mmar_carl.models.config import (
        DebateStepConfig,
        EvaluationStepConfig,
        HumanInputStepConfig,
        MCPResourceStepConfig,
        StructuredOutputStepConfig,
        ToolDiscoveryStepConfig,
        ToolStepConfig,
    )

    _HAS_CARL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_CARL = False

_needs_carl = pytest.mark.skipif(not _HAS_CARL, reason="mmar_carl not installed")


def _export_one(step: dict) -> dict:
    """Run a single step through to_carl_compat and return the converted step."""
    chain = {"name": "t", "steps": [{"number": 1, "title": "s", "aim": "a", **step}]}
    return to_carl_compat(chain)["steps"][0]


# --------------------------------------------------------------------------- #
# Real-schema validation: produced step_config must instantiate the CARL model.
# --------------------------------------------------------------------------- #
@_needs_carl
def test_tool_config_matches_carl():
    out = _export_one(
        {
            "step_type": "tool",
            "tool_name": "search",
            "tool_input_mapping": {"q": "$history[-1]"},
            "tool_output_key": "res",
            "tool_timeout": 12.0,
        }
    )
    cfg = ToolStepConfig(**out["step_config"])
    assert cfg.tool_name == "search"
    assert cfg.input_mapping == {"q": "$history[-1]"}
    assert cfg.output_key == "res"
    assert cfg.timeout == 12.0


@_needs_carl
def test_structured_output_config_matches_carl():
    out = _export_one(
        {
            "step_type": "structured_output",
            "structured_output_schema": {"type": "object"},
            "structured_output_schema_name": "Foo",
            "structured_output_input_source": "$history[-1]",
        }
    )
    cfg = StructuredOutputStepConfig(**out["step_config"])
    assert cfg.output_schema == {"type": "object"}
    assert cfg.schema_name == "Foo"
    assert cfg.input_source == "$history[-1]"


@_needs_carl
def test_evaluation_config_matches_carl():
    out = _export_one(
        {
            "step_type": "evaluation",
            "evaluation_target_step": 1,
            "evaluation_criteria": ["at least 200 words"],
            "evaluation_fail_action": "continue",
            "evaluation_method": "llm",
            "evaluation_threshold": 0.8,  # no CARL home -> must be dropped
        }
    )
    cfg = EvaluationStepConfig(**out["step_config"])
    assert cfg.evaluates_step == 1
    assert cfg.criteria == ["at least 200 words"]
    assert cfg.on_fail.value == "continue"
    assert cfg.evaluation_method == "llm"
    # the orphan flat field must not survive anywhere
    assert "evaluation_threshold" not in out
    assert "threshold" not in out["step_config"]


@_needs_carl
def test_human_input_config_matches_carl():
    out = _export_one(
        {
            "step_type": "human_input",
            "human_prompt": "Review the draft:",
            "human_input_timeout": 30.0,
            "human_default": "[none]",
        }
    )
    cfg = HumanInputStepConfig(**out["step_config"])
    assert cfg.prompt == "Review the draft:"
    assert cfg.timeout == 30.0
    assert cfg.fallback_value == "[none]"  # FIXED: was wrongly mapped to "default"


@_needs_carl
def test_mcp_resource_config_matches_carl():
    out = _export_one(
        {
            "step_type": "mcp_resource",
            "mcp_server_name": "docs",
            "mcp_server_transport": "sse",
            "mcp_server_url": "https://example/mcp",
            "mcp_resource_uri": "docs://api/reference.md",
            "mcp_resource_memory_key": "api_docs",
        }
    )
    cfg = MCPResourceStepConfig(**out["step_config"])
    assert cfg.resource_uri == "docs://api/reference.md"  # FIXED: was "uri"
    assert cfg.output_memory_key == "api_docs"  # FIXED: was "memory_key"
    assert cfg.server.server_name == "docs"
    assert cfg.server.transport == "sse"


@_needs_carl
def test_debate_config_matches_carl():
    out = _export_one(
        {
            "step_type": "debate",
            "debate_agents": [
                {"name": "pro", "role": "Argue in favour."},
                {"name": "con", "role": "Argue against."},
            ],
            "debate_rounds": 2,
            "debate_judge_prompt": "Synthesize a verdict. Topic: {task}",
        }
    )
    cfg = DebateStepConfig(**out["step_config"])
    assert cfg.roles == ["pro", "con"]  # FIXED: reshaped from debate_agents
    assert cfg.role_prompts == {"pro": "Argue in favour.", "con": "Argue against."}
    assert cfg.rounds == 2
    assert cfg.judge_prompt.startswith("Synthesize")


@_needs_carl
def test_tool_discovery_config_matches_carl():
    out = _export_one(
        {
            "step_type": "tool_discovery",
            "discovery_sources": [{"kind": "module", "module": "myapp.tools", "name_prefix": "tool_"}],
        }
    )
    cfg = ToolDiscoveryStepConfig(**out["step_config"])
    # source FIXED: single ToolSource, reshaped from the discovery_sources list
    assert cfg.source.module == "myapp.tools"
    assert cfg.source.name_prefix == "tool_"


# --------------------------------------------------------------------------- #
# Translation: runtime-only types become llm (no CARL import needed).
# --------------------------------------------------------------------------- #
def test_parallel_sampling_translated_to_llm():
    out = _export_one({"step_type": "parallel_sampling", "sampling_n": 5, "sampling_temperature": 0.7})
    assert out["step_type"] == "llm"
    assert out.get("aim")
    for k in ("sampling_n", "sampling_temperature", "sampling_aggregation", "sampling_judge_prompt"):
        assert k not in out


def test_agent_handoff_translated_to_llm():
    out = _export_one({"step_type": "agent_handoff", "handoff_chain_path": "chains/research.json"})
    assert out["step_type"] == "llm"
    assert out.get("aim")
    for k in ("handoff_chain_path", "handoff_input_mapping", "handoff_output_key"):
        assert k not in out


def test_supervisor_translated_to_llm():
    out = _export_one(
        {
            "step_type": "supervisor",
            "supervisor_routes": {"finance": "f.json", "legal": "l.json"},
            "supervisor_routing_prompt": "Which domain?",
        }
    )
    assert out["step_type"] == "llm"
    assert out.get("aim")
    for k in ("supervisor_routes", "supervisor_routing_prompt"):
        assert k not in out


def test_unknown_step_type_falls_back_to_llm():
    out = _export_one({"step_type": "totally_made_up"})
    assert out["step_type"] == "llm"
