"""T11 — cheap verifier: step critique scoped to the risky step types."""

from __future__ import annotations

from mmar_mage.generator import _select_critique_steps
from mmar_mage.schemas import CARLStepSchema


def _steps() -> list[CARLStepSchema]:
    return [
        CARLStepSchema(number=1, title="get ip", step_type="tool", tool_name="http_request", aim="fetch"),
        CARLStepSchema(number=2, title="reason", step_type="llm", aim="think it through"),
        CARLStepSchema(number=3, title="extract", step_type="structured_output", aim="pull fields"),
    ]


def test_none_critiques_all_steps() -> None:
    assert [s.number for s in _select_critique_steps(_steps(), None)] == [1, 2, 3]


def test_scoped_to_tool_and_structured_only() -> None:
    scoped = _select_critique_steps(_steps(), ["tool", "structured_output"])
    assert [s.number for s in scoped] == [1, 3]  # the llm step is skipped — cheaper critique


def test_scoped_empty_when_no_matching_types() -> None:
    steps = [CARLStepSchema(number=1, title="reason", step_type="llm", aim="think")]
    assert _select_critique_steps(steps, ["tool", "structured_output"]) == []  # → critique skipped
