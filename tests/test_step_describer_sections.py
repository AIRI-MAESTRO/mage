"""P-2 — StepDescriber prompt assembled from only the DAG's step-type sections."""

from __future__ import annotations

from mmar_mage.prompts import STEP_DESCRIBER_TYPE_SECTIONS, build_step_describer_prompt
from mmar_mage.schemas import CARL_VALID_STEP_TYPES


def test_sections_synced_with_carl_types() -> None:
    assert set(STEP_DESCRIBER_TYPE_SECTIONS) == set(CARL_VALID_STEP_TYPES)
    assert "eval" not in STEP_DESCRIBER_TYPE_SECTIONS
    assert "aggregator" not in STEP_DESCRIBER_TYPE_SECTIONS


def test_only_dag_types_included() -> None:
    p = build_step_describer_prompt({"llm", "tool"})
    assert "<<STEP_TYPE_SECTIONS>>" not in p  # placeholder was filled
    assert 'step_type "llm":' in p
    assert 'step_type "tool":' in p
    # the other 14 types' field sections are NOT included
    for t in ["agent_skill", "debate", "supervisor", "parallel_sampling", "evaluation", "mcp"]:
        assert f'step_type "{t}":' not in p
    # base content is preserved
    assert "You are a CARL step description writer" in p
    assert "Quality guidelines" in p


def test_none_includes_all_types() -> None:
    p = build_step_describer_prompt(None)
    for t in CARL_VALID_STEP_TYPES:
        assert f'step_type "{t}":' in p


def test_filtered_prompt_is_shorter() -> None:
    assert len(build_step_describer_prompt({"llm"})) < len(build_step_describer_prompt(None))
