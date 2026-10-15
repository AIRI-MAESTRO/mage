"""Tests for the CARL-compatibility exporter."""

from __future__ import annotations

import pytest

from mmar_mage.carl_export import to_carl_compat
from mmar_mage.generator import validate_carl_json


class TestEvalTranslation:
    def test_eval_becomes_llm(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Draft",
                    "step_type": "llm",
                    "aim": "Produce draft",
                    "dependencies": [],
                },
                {
                    "number": 2,
                    "title": "Critique",
                    "step_type": "eval",
                    "eval_target_steps": [1],
                    "eval_criteria": "factual accuracy and tone",
                    "eval_on_fail": "branch",
                    "dependencies": [1],
                },
            ]
        }
        out = to_carl_compat(chain)
        s2 = out["steps"][1]
        assert s2["step_type"] == "llm"
        assert "Critique step(s) 1" in s2["aim"]
        assert "factual accuracy and tone" in s2["aim"]
        assert "On failure: branch" in s2["aim"]
        assert "eval_target_steps" not in s2
        assert "eval_criteria" not in s2

    def test_aggregator_becomes_llm(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []},
                {"number": 2, "title": "B", "step_type": "llm", "aim": "do B", "dependencies": []},
                {
                    "number": 3,
                    "title": "Merge",
                    "step_type": "aggregator",
                    "aggregate_from_steps": [1, 2],
                    "aggregation_strategy": "vote",
                    "dependencies": [1, 2],
                },
            ]
        }
        out = to_carl_compat(chain)
        s3 = out["steps"][2]
        assert s3["step_type"] == "llm"
        assert "Aggregate" in s3["aim"]
        assert "1, 2" in s3["aim"]
        assert "vote" in s3["aim"]
        assert "aggregate_from_steps" not in s3


class TestNestedStepConfig:
    def test_tool_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Fetch",
                    "step_type": "tool",
                    "tool_name": "http_get",
                    "tool_description": "GET request",
                    "tool_input_mapping": {"url": "$history[-1]"},
                    "tool_output_key": "response",
                    "tool_timeout": 10.0,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "tool_name" not in s
        assert s["step_config"]["tool_name"] == "http_get"
        assert s["step_config"]["tool_description"] == "GET request"
        assert s["step_config"]["input_mapping"] == {"url": "$history[-1]"}
        assert s["step_config"]["output_key"] == "response"
        assert s["step_config"]["timeout"] == 10.0

    def test_mcp_fields_nested_with_server(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Query DB",
                    "step_type": "mcp",
                    "mcp_server_name": "postgres-mcp",
                    "mcp_server_transport": "stdio",
                    "mcp_server_command": "mcp-postgres",
                    "mcp_server_args": ["--db", "main"],
                    "mcp_tool_name": "query",
                    "mcp_arguments": {"sql": "SELECT 1"},
                    "mcp_timeout": 30.0,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "mcp_server_name" not in s
        assert "mcp_tool_name" not in s
        cfg = s["step_config"]
        assert cfg["server"]["server_name"] == "postgres-mcp"
        assert cfg["server"]["transport"] == "stdio"
        assert cfg["server"]["command"] == "mcp-postgres"
        assert cfg["server"]["args"] == ["--db", "main"]
        assert cfg["tool_name"] == "query"
        assert cfg["arguments"] == {"sql": "SELECT 1"}
        assert cfg["timeout"] == 30.0

    def test_memory_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Save",
                    "step_type": "memory",
                    "memory_operation": "write",
                    "memory_key": "intermediate_summary",
                    "memory_value_source": "$history[-1]",
                    "memory_namespace": "session_42",
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "memory_operation" not in s
        assert s["step_config"]["operation"] == "write"
        assert s["step_config"]["memory_key"] == "intermediate_summary"
        assert s["step_config"]["value_source"] == "$history[-1]"
        assert s["step_config"]["namespace"] == "session_42"

    def test_transform_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Split",
                    "step_type": "transform",
                    "transform_type": "extract",
                    "transform_input_key": "$history[-1]",
                    "transform_expression": "$.items[*]",
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "transform_type" not in s
        assert s["step_config"]["transform_type"] == "extract"
        assert s["step_config"]["input_key"] == "$history[-1]"
        assert s["step_config"]["expression"] == "$.items[*]"

    def test_structured_output_fields_nested(self) -> None:
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Score",
                    "step_type": "structured_output",
                    "aim": "Score the input",
                    "structured_output_schema": schema,
                    "structured_output_schema_name": "ScoreOutput",
                    "structured_output_input_source": "$history[-1]",
                    "structured_output_instruction": "Return a numeric score 0-1",
                    "structured_output_strict_json": True,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "structured_output_schema" not in s
        cfg = s["step_config"]
        assert cfg["output_schema"] == schema
        assert cfg["schema_name"] == "ScoreOutput"
        assert cfg["input_source"] == "$history[-1]"
        assert cfg["instruction"] == "Return a numeric score 0-1"
        assert cfg["strict_json"] is True


class TestExporterPreservesValidity:
    def test_export_passes_strict_carl_validation(self) -> None:
        # MAGE-format chain with eval/aggregator + nested tool config
        chain = {
            "name": "demo",
            "max_workers": 2,
            "timeout": 300.0,
            "steps": [
                {
                    "number": 1,
                    "title": "Fetch",
                    "step_type": "tool",
                    "tool_name": "http_get",
                    "dependencies": [],
                },
                {
                    "number": 2,
                    "title": "Reason",
                    "step_type": "llm",
                    "aim": "Interpret result",
                    "dependencies": [1],
                },
                {
                    "number": 3,
                    "title": "Critique",
                    "step_type": "eval",
                    "eval_target_steps": [2],
                    "eval_criteria": "completeness",
                    "dependencies": [2],
                },
            ],
        }
        # Pre-export: MAGE validation passes (eval allowed).
        validate_carl_json(chain)

        out = to_carl_compat(chain)
        # Post-export: strict CARL validation passes (eval translated to llm).
        validate_carl_json(out, strict_carl=True)

    def test_unknown_step_type_falls_back_to_llm(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Mystery",
                    "step_type": "frobnicate",
                    "aim": "do mystery",
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        assert out["steps"][0]["step_type"] == "llm"

    def test_none_fields_stripped(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "A",
                    "step_type": "llm",
                    "aim": "do A",
                    "llm_config": None,
                    "tool_name": None,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "llm_config" not in s
        assert "tool_name" not in s

    def test_input_chain_not_mutated(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "A",
                    "step_type": "eval",
                    "eval_target_steps": [99],
                    "dependencies": [],
                }
            ]
        }
        original_type = chain["steps"][0]["step_type"]
        to_carl_compat(chain)
        assert chain["steps"][0]["step_type"] == original_type  # deep copy preserved


@pytest.mark.parametrize(
    "step_type",
    [
        "llm",
        "tool",
        "mcp",
        "memory",
        "transform",
        "conditional",
        "structured_output",
        # agent-features types that load from JSON (step_type preserved).
        # parallel_sampling / agent_handoff / supervisor are NOT here: they
        # require a runtime-only object CARL cannot rebuild from JSON, so they
        # translate to "llm" (see the dedicated translation tests below).
        "agent_skill",
        "evaluation",
        "human_input",
        "debate",
        "mcp_resource",
        "tool_discovery",
    ],
)
def test_carl_valid_types_passthrough(step_type: str) -> None:
    chain = {
        "steps": [
            {
                "number": 1,
                "title": "X",
                "step_type": step_type,
                "aim": "do X",
                "dependencies": [],
            }
        ]
    }
    out = to_carl_compat(chain)
    assert out["steps"][0]["step_type"] == step_type


# ---------------------------------------------------------------------------
# agent-features step types — nested step_config
# ---------------------------------------------------------------------------


class TestAgentFeaturesNesting:
    def test_agent_skill_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Extract PDF",
                    "step_type": "agent_skill",
                    "skill": "github://anthropics/skills/skills/pdf@main",
                    "skill_task": "Extract tables from {input_pdf}",
                    "skill_execution_mode": "llm_agent",
                    "skill_input_mapping": {"input_pdf": "$history[-1]"},
                    "skill_output_key": "extracted_text",
                    "skill_llm_max_iterations": 5,
                    "skill_runtime": "local",
                    "skill_timeout": 60.0,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        s = out["steps"][0]
        assert "skill" not in s
        assert "skill_task" not in s
        cfg = s["step_config"]
        assert cfg["skill"] == "github://anthropics/skills/skills/pdf@main"
        assert cfg["task"] == "Extract tables from {input_pdf}"
        assert cfg["execution_mode"] == "llm_agent"
        assert cfg["input_mapping"] == {"input_pdf": "$history[-1]"}
        assert cfg["output_key"] == "extracted_text"
        assert cfg["llm_max_iterations"] == 5
        assert cfg["runtime"] == "local"
        assert cfg["timeout"] == 60.0

    def test_evaluation_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Quality gate",
                    "step_type": "evaluation",
                    "evaluation_target_step": 0,
                    "evaluation_method": "llm",
                    "evaluation_criteria": ["Output contains structured JSON"],
                    "evaluation_fail_action": "continue",
                    "evaluation_threshold": 0.7,
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        cfg = out["steps"][0]["step_config"]
        # CARL field names (config.py::EvaluationStepConfig).
        assert cfg["evaluates_step"] == 0
        assert cfg["evaluation_method"] == "llm"
        assert cfg["criteria"] == ["Output contains structured JSON"]
        assert cfg["on_fail"] == "continue"
        # evaluation_threshold has no CARL home and must be dropped entirely.
        assert "threshold" not in cfg
        assert "evaluation_threshold" not in out["steps"][0]

    def test_evaluation_loose_vocab_normalised_to_carl_enums(self) -> None:
        # the planner often emits values CARL's EvaluationStepConfig rejects:
        # method "python" / fail_action "retry" → must be coerced to valid enums.
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Gate",
                    "step_type": "evaluation",
                    "evaluation_target_step": 0,
                    "evaluation_method": "python",          # invalid → llm
                    "evaluation_criteria": ["covers 7 days"],
                    "evaluation_fail_action": "retry",       # invalid → retry_with_feedback
                    "dependencies": [],
                }
            ]
        }
        cfg = to_carl_compat(chain)["steps"][0]["step_config"]
        assert cfg["evaluation_method"] == "llm"
        assert cfg["on_fail"] == "retry_with_feedback"

    def test_evaluation_bare_string_criteria_wrapped_in_list(self) -> None:
        # weak models sometimes return a single string; CARL needs list[str].
        chain = {
            "steps": [{
                "number": 1, "title": "Gate", "step_type": "evaluation",
                "evaluation_target_step": 0, "dependencies": [],
                "evaluation_criteria": "the forecast covers 7 days",
            }]
        }
        cfg = to_carl_compat(chain)["steps"][0]["step_config"]
        assert cfg["criteria"] == ["the forecast covers 7 days"]

    def test_evaluation_fail_action_synonyms(self) -> None:
        from mmar_mage.carl_export import refine_evaluation_step

        def on_fail(v):
            out = refine_evaluation_step(
                {"step_type": "evaluation", "evaluation_fail_action": v}
            )
            return out["evaluation_fail_action"]

        assert on_fail("fail") == "abort"
        assert on_fail("halt") == "abort"
        assert on_fail("replan") == "retry_with_feedback"
        assert on_fail("continue") == "continue"
        assert on_fail("nonsense") == "continue"  # safe default

    def test_parallel_sampling_translated_to_llm(self) -> None:
        # parallel_sampling needs a runtime-only base_step CARL cannot rebuild
        # from JSON, so it is translated to llm (P0.3 interim).
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Vote",
                    "step_type": "parallel_sampling",
                    "sampling_n": 5,
                    "sampling_temperature": 0.7,
                    "sampling_aggregation": "vote",
                    "sampling_judge_prompt": "Pick the best answer",
                    "dependencies": [],
                }
            ]
        }
        s = to_carl_compat(chain)["steps"][0]
        assert s["step_type"] == "llm"
        assert s.get("aim")
        for k in ("sampling_n", "sampling_temperature", "sampling_aggregation", "sampling_judge_prompt"):
            assert k not in s

    def test_supervisor_translated_to_llm(self) -> None:
        # supervisor needs runtime-only ``agents``; translated to llm (P0.3).
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Route",
                    "step_type": "supervisor",
                    "supervisor_routes": {"finance": "chains/finance.json", "legal": "chains/legal.json"},
                    "supervisor_routing_prompt": "Which domain?",
                    "dependencies": [],
                }
            ]
        }
        s = to_carl_compat(chain)["steps"][0]
        assert s["step_type"] == "llm"
        assert s.get("aim")
        assert "supervisor_routes" not in s
        assert "supervisor_routing_prompt" not in s

    def test_debate_fields_nested(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Debate",
                    "step_type": "debate",
                    "debate_agents": [{"name": "pro", "role": "advocate"}, {"name": "con", "role": "skeptic"}],
                    "debate_rounds": 3,
                    "debate_judge_prompt": "Synthesize the consensus",
                    "dependencies": [],
                }
            ]
        }
        out = to_carl_compat(chain)
        cfg = out["steps"][0]["step_config"]
        # MAGE's debate_agents list is reshaped into CARL roles + role_prompts.
        assert cfg["roles"] == ["pro", "con"]
        assert cfg["role_prompts"] == {"pro": "advocate", "con": "skeptic"}
        assert cfg["rounds"] == 3
        assert cfg["judge_prompt"] == "Synthesize the consensus"

    def test_replan_policy_chain_level(self) -> None:
        # chain-level RE-PLAN policy passes through to_carl_compat unchanged
        chain = {
            "name": "demo",
            "steps": [{"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []}],
            "replan_policy": {
                "checkers": [{"type": "rule_based", "name": "no-error", "trigger_on_failed_step": True}],
                "aggregation": {"strategy": "any"},
                "max_replans": 3,
            },
        }
        out = to_carl_compat(chain)
        assert out["replan_policy"]["max_replans"] == 3


class TestSkillUriNormalization:
    """A hallucinated / stale agent_skill URI is corrected to the canonical
    registry URI (the anthropics/skills repo moved to ``skills/<name>``)."""

    def test_registry_uris_use_current_skills_layout(self):
        from mmar_mage.skills import default_registry

        for name in ("pdf", "docx", "pptx", "xlsx"):
            uri = default_registry.get(name).uri
            assert "/document-skills/" not in uri  # the stale layout is gone
            assert uri == f"github://anthropics/skills/skills/{name}@main"

    def test_hallucinated_powerpoint_uri_maps_to_pptx(self):
        from mmar_mage.carl_export import refine_agent_skill_step

        step = {
            "step_type": "agent_skill",
            "title": "Format content as PowerPoint presentation (PPTX)",
            "skill": "github://anthropics/skills/presentation-skills/powerpoint@main",
        }
        out = refine_agent_skill_step(step)
        assert out["skill"] == "github://anthropics/skills/skills/pptx@main"

    def test_bare_skill_name_maps_to_registry_uri(self):
        # planner named the skill "pptx" (bare) → CARL can't resolve it locally;
        # map it to the canonical registry URI so the skill loads.
        from mmar_mage.carl_export import refine_agent_skill_step

        out = refine_agent_skill_step(
            {"step_type": "agent_skill", "skill": "pptx", "title": "Создать PPTX"},
        )
        assert out["skill"] == "github://anthropics/skills/skills/pptx@main"

    def test_bare_alias_maps_to_registry_uri(self):
        from mmar_mage.carl_export import refine_agent_skill_step

        out = refine_agent_skill_step(
            {"step_type": "agent_skill", "skill": "powerpoint", "title": "deck"},
        )
        assert out["skill"] == "github://anthropics/skills/skills/pptx@main"

    def test_canonical_uri_left_unchanged(self):
        from mmar_mage.carl_export import refine_agent_skill_step

        canonical = "github://anthropics/skills/skills/pptx@main"
        out = refine_agent_skill_step(
            {"step_type": "agent_skill", "skill": canonical, "title": "deck"},
        )
        assert out["skill"] == canonical

    def test_custom_third_party_uri_left_alone(self):
        from mmar_mage.carl_export import refine_agent_skill_step

        custom = "github://acme/my-skills/pptx-maker@main"
        out = refine_agent_skill_step(
            {"step_type": "agent_skill", "skill": custom, "title": "slides"},
        )
        assert out["skill"] == custom  # not an anthropics/skills ref → untouched
