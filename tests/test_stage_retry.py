"""T8 — generate_json re-requests the failed stage instead of restarting the pipeline."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mmar_mage.exceptions import MAGELLMError
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig


async def test_generate_json_rerequests_then_succeeds(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test", json_parse_retries=1))
    gen = AsyncMock(side_effect=["totally not json", '{"ok": true}'])
    mocker.patch.object(llm, "generate", gen)
    assert await llm.generate_json("sys", "user", stage="describe") == {"ok": True}
    assert gen.await_count == 2  # re-requested the failed stage exactly once


async def test_generate_json_exhausts_retries_then_raises(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test", json_parse_retries=2))
    gen = AsyncMock(return_value="not json")
    mocker.patch.object(llm, "generate", gen)
    with pytest.raises(MAGELLMError):
        await llm.generate_json("sys", "user", stage="describe")
    assert gen.await_count == 3  # 1 initial + json_parse_retries


async def test_generate_json_no_retry_when_disabled(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test", json_parse_retries=0))
    gen = AsyncMock(return_value="not json")
    mocker.patch.object(llm, "generate", gen)
    with pytest.raises(MAGELLMError):
        await llm.generate_json("sys", "user", stage="describe")
    assert gen.await_count == 1  # no extra attempts


async def test_repair_short_circuits_rerequest(mocker: Any) -> None:
    # a recoverable (fenced) reply is repaired on attempt 1 — no re-request
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test", json_parse_retries=2))
    gen = AsyncMock(side_effect=['```json\n{"ok": 1}\n```', "should not be used"])
    mocker.patch.object(llm, "generate", gen)
    assert await llm.generate_json("sys", "user", stage="describe") == {"ok": 1}
    assert gen.await_count == 1
