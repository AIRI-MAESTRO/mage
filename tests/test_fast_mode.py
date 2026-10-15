"""Tests for fast-mode chain generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.exceptions import MAGELLMError, MAGEValidationError
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


@pytest.mark.asyncio
async def test_fast_mode_basic(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode generates a valid CARL chain."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Analyze a document")

    assert result.mode == "fast"
    assert result.metadata.num_steps == 3
    assert result.metadata.model == "gpt-4o-test"

    chain = json.loads(result.chain_json)
    assert "steps" in chain
    assert len(chain["steps"]) == 3


@pytest.mark.asyncio
async def test_fast_mode_chain_structure(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode output has correct CARL step fields."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test query")

    chain = json.loads(result.chain_json)
    step = chain["steps"][0]

    assert step["step_type"] == "llm"
    assert step["number"] == 1
    assert "aim" in step
    assert "reasoning_questions" in step
    assert "step_context_queries" in step
    assert isinstance(step["dependencies"], list)


@pytest.mark.asyncio
async def test_fast_mode_no_memory(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode with memory disabled returns None memory_key."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")

    assert result.memory_key is None


@pytest.mark.asyncio
async def test_fast_mode_invalid_json(config: MAGEConfig, mocker: MagicMock) -> None:
    """Fast mode raises MAGEValidationError on invalid JSON."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    choice = MagicMock()
    choice.message.content = "not valid json {{"
    response = MagicMock()
    response.choices = [choice]
    instance.chat.completions.create = AsyncMock(return_value=response)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=config)
    with pytest.raises(MAGEValidationError):
        await gen.generate("Bad query")


@pytest.mark.asyncio
async def test_fast_mode_empty_response(config: MAGEConfig, mocker: MagicMock) -> None:
    """Fast mode raises MAGELLMError on empty LLM content."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    choice = MagicMock()
    choice.message.content = None
    response = MagicMock()
    response.choices = [choice]
    instance.chat.completions.create = AsyncMock(return_value=response)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=config)
    with pytest.raises(MAGELLMError):
        await gen.generate("Empty query")


@pytest.mark.asyncio
async def test_fast_mode_metadata(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode populates metadata correctly."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Analyze report")

    assert result.metadata.generation_time_seconds >= 0
    assert result.metadata.deep_stages_completed is None  # fast mode


@pytest.mark.asyncio
async def test_fast_mode_chain_dict(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode result includes chain_dict."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")

    assert isinstance(result.chain_dict, dict)
    assert "steps" in result.chain_dict
