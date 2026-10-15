"""P-1 — registry-driven planner step-type menu; no deprecated eval/aggregator."""

from __future__ import annotations

from mmar_mage.prompts import STEP_PLANNER_PROMPT, build_planning_guidance, get_step_types_constraint
from mmar_mage.schemas import CARL_VALID_STEP_TYPES
from mmar_mage.step_types import STEP_TYPE_DESCRIPTIONS, step_types_for


def test_registry_synced_with_carl_types() -> None:
    assert set(STEP_TYPE_DESCRIPTIONS) == set(CARL_VALID_STEP_TYPES)
    assert "eval" not in STEP_TYPE_DESCRIPTIONS
    assert "aggregator" not in STEP_TYPE_DESCRIPTIONS


def test_step_types_for_filters_and_falls_back() -> None:
    assert step_types_for(["llm"]) == ["llm"]
    assert step_types_for(["llm", "tool"]) == ["llm", "tool"]
    # MAGE-internal / unknown labels are dropped; an all-invalid list falls back
    assert step_types_for(["eval", "aggregator", "nope"]) == list(STEP_TYPE_DESCRIPTIONS)
    assert step_types_for(None) == list(STEP_TYPE_DESCRIPTIONS)


def test_guidance_filters_to_allowed_one_type() -> None:
    g = build_planning_guidance(["llm"])
    assert "`llm` —" in g
    # the planner sees ONE type, not the full menu + a ban
    for t in ["tool", "agent_skill", "structured_output", "debate", "evaluation"]:
        assert f"`{t}` —" not in g
    # unrestricted still lists every CARL type
    full = build_planning_guidance()
    for t in CARL_VALID_STEP_TYPES:
        assert f"`{t}` —" in full


def test_constraint_uses_filtered_registry() -> None:
    c = get_step_types_constraint(["llm"])
    assert '"llm"' in c and '"tool"' not in c
    # eval/aggregator dropped even if explicitly passed
    c2 = get_step_types_constraint(["llm", "eval", "aggregator"])
    assert '"eval"' not in c2 and '"aggregator"' not in c2
    assert get_step_types_constraint(None) == ""


def test_planner_prompt_dropped_deprecated_types() -> None:
    # the static 18-type list (with eval/aggregator) is gone from the YAML prompt
    assert '"eval"' not in STEP_PLANNER_PROMPT
    assert '"aggregator"' not in STEP_PLANNER_PROMPT
    assert '"step_type": "eval"' not in STEP_PLANNER_PROMPT  # example fixed to "evaluation"
