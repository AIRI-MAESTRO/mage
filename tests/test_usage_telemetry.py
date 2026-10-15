"""WP-5 — LLMClient accumulates real token usage; it lands in MAGEResult."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

_CHAIN = {
    "name": "T",
    "max_workers": 2,
    "timeout": 300.0,
    "search_config": {
        "strategy": "substring",
        "substring_config": {"case_sensitive": False, "min_word_length": 3, "max_matches_per_query": 5},
    },
    "steps": [
        {
            "step_type": "llm", "number": 1, "title": "A", "aim": "do a",
            "reasoning_questions": "q", "step_context_queries": ["x"], "stage_action": "do",
            "example_reasoning": "e", "dependencies": [], "retry_max": 3, "llm_config": None,
        },
    ],
}


def _resp(content: str, *, prompt: int | None = 10, completion: int | None = 20) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    r = MagicMock()
    r.choices = [choice]
    if prompt is None and completion is None:
        r.usage = None
    else:
        usage = MagicMock()
        usage.prompt_tokens = prompt
        usage.completion_tokens = completion
        usage.completion_tokens_details = None
        r.usage = usage
    return r


def _client(mocker: Any, response: MagicMock) -> MagicMock:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    inst = mock_cls.return_value
    inst.chat.completions.create = AsyncMock(return_value=response)
    inst.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return inst


async def test_usage_accumulates_and_lands_in_metadata(mocker: Any) -> None:
    _client(mocker, _resp(json.dumps(_CHAIN), prompt=10, completion=20))
    gen = MAGEGenerator(config=MAGEConfig(mode="fast", api_key="test", enable_memory=False))

    result = await gen.generate("analyze X")

    total = gen._llm.total_usage()  # noqa: SLF001
    assert total.calls >= 1
    assert gen._llm.usage_by_stage  # noqa: SLF001 — accumulated per stage
    # each call reported prompt=10 / completion=20
    assert total.prompt_tokens == 10 * total.calls
    assert total.completion_tokens == 20 * total.calls
    assert total.total_tokens == total.prompt_tokens + total.completion_tokens
    assert total.estimated is False

    # MAGEResult carries the actual tokens, consistent with the client total
    assert result.metadata.prompt_tokens == total.prompt_tokens
    assert result.metadata.completion_tokens == total.completion_tokens
    assert result.metadata.total_tokens == total.total_tokens
    assert result.metadata.usage_estimated is False
    assert result.metadata.usage_by_stage  # per-stage breakdown present

    # snapshot is an independent deep copy
    snap = gen._llm.usage_snapshot()  # noqa: SLF001
    snap_total = sum(u.total_tokens for u in snap.values())
    assert snap_total == total.total_tokens


async def test_usage_estimated_when_provider_omits_usage(mocker: Any) -> None:
    _client(mocker, _resp(json.dumps(_CHAIN), prompt=None, completion=None))
    gen = MAGEGenerator(config=MAGEConfig(mode="fast", api_key="test", enable_memory=False))

    result = await gen.generate("analyze X")

    assert gen._llm.total_usage().estimated is True  # noqa: SLF001
    assert result.metadata.usage_estimated is True
    assert result.metadata.total_tokens == 0
