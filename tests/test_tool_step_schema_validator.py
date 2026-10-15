"""T23 — schema-level guards on tool steps (mirror the T5 lint at construction)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from mmar_mage.schemas import CARLStepSchema


def _step(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"step_type": "tool", "number": 1, "title": "Fetch", "tool_name": "http_request"}
    base.update(over)
    return base


def test_tool_step_requires_tool_name() -> None:
    with pytest.raises(ValidationError, match="non-empty tool_name"):
        CARLStepSchema.model_validate(_step(tool_name=""))
    with pytest.raises(ValidationError, match="non-empty tool_name"):
        CARLStepSchema.model_validate(_step(tool_name=None))


def test_outer_context_in_url_field_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be .outer_context"):
        CARLStepSchema.model_validate(_step(tool_input_mapping={"url": "$outer_context", "method": "GET"}))
    # nested key variants are caught too
    with pytest.raises(ValidationError, match="must not be .outer_context"):
        CARLStepSchema.model_validate(_step(tool_input_mapping={"base_url": "$outer_context"}))


def test_valid_tool_step_passes() -> None:
    s = CARLStepSchema.model_validate(_step(tool_input_mapping={"url": "https://api.ipify.org", "method": "GET"}))
    assert s.tool_name == "http_request"
    # $outer_context is fine in a NON-url field (a text query)
    CARLStepSchema.model_validate(_step(tool_name="web_search", tool_input_mapping={"query": "$outer_context"}))
    # $history reference in a url field is fine
    CARLStepSchema.model_validate(_step(tool_input_mapping={"url": "$history[-1]"}))


def test_non_tool_step_unaffected() -> None:
    # an llm step with empty aim/tool_name is still valid (only tool steps are gated)
    CARLStepSchema.model_validate({"step_type": "llm", "number": 1, "title": "Think"})
