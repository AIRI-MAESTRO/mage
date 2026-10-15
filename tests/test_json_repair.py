"""T21 — JSON-repair recovers malformed/truncated LLM output instead of restarting."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.llm import LLMClient, _balance_json, _repair_json
from mmar_mage.schemas import MAGEConfig


def test_repair_strips_fences_and_prose() -> None:
    raw = 'Sure! Here is the JSON:\n```json\n{"a": 1, "b": [2, 3]}\n```\nHope that helps.'
    assert _repair_json(raw) == {"a": 1, "b": [2, 3]}


def test_repair_truncated_object() -> None:
    # model cut off mid-output — the run-773 failure class
    raw = '{"steps": [{"number": 1, "title": "Fetch", "aim": "get it"'
    assert _repair_json(raw) == {"steps": [{"number": 1, "title": "Fetch", "aim": "get it"}]}


def test_balance_json_closes_brackets_and_string() -> None:
    assert json.loads(_balance_json('{"x": [1, 2') or "") == {"x": [1, 2]}
    assert json.loads(_balance_json('{"x": "unclosed string') or "") == {"x": "unclosed string"}


def test_repair_returns_none_on_garbage() -> None:
    assert _repair_json("not json at all") is None
    assert _balance_json("no brace here") is None


async def test_generate_json_recovers_via_repair(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))  # default provider: no json_mode_fallback
    # the model fenced its JSON, so json.loads(raw) fails — repair must recover it
    mocker.patch.object(llm, "generate", AsyncMock(return_value='```json\n{"ok": true}\n```'))
    assert await llm.generate_json("sys", "user", stage="s") == {"ok": True}
