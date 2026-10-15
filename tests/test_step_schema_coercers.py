"""T22 — CARLStepSchema coercers absorb list / None / scalar from weak models."""

from __future__ import annotations

from typing import Any

from mmar_mage.schemas import CARLStepSchema


def _step(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"step_type": "llm", "number": 1, "title": "T"}
    base.update(over)
    return base


def test_text_field_list_is_joined() -> None:
    s = CARLStepSchema.model_validate(_step(reasoning_questions=["Q1?", "Q2?"]))
    assert s.reasoning_questions == "Q1?; Q2?"


def test_text_field_none_becomes_empty() -> None:
    s = CARLStepSchema.model_validate(_step(reasoning_questions=None, aim=None))
    assert s.reasoning_questions == "" and s.aim == ""


def test_text_field_scalar_stringified() -> None:
    assert CARLStepSchema.model_validate(_step(stage_action=42)).stage_action == "42"


def test_text_field_string_passthrough() -> None:
    s = CARLStepSchema.model_validate(_step(aim="Determine the dominant signals"))
    assert s.aim == "Determine the dominant signals"


def test_context_queries_bare_string_wrapped() -> None:
    assert CARLStepSchema.model_validate(_step(step_context_queries="finance")).step_context_queries == ["finance"]


def test_context_queries_none_and_nonstr_elems() -> None:
    assert CARLStepSchema.model_validate(_step(step_context_queries=None)).step_context_queries == []
    assert CARLStepSchema.model_validate(_step(step_context_queries=[1, 2])).step_context_queries == ["1", "2"]
