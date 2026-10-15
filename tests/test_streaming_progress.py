"""Tests for §3 P0 — streaming + intermediate artifact progress.

Covers:
- `_extract_chunk_delta()` helper handles both SDK-object and dict shapes
- `LLMClient.generate(stream_callback=...)` fires the callback per chunk
  and returns the full content
- Streaming gracefully falls back to non-streaming when the provider
  rejects ``stream=True``
- A failing user callback does NOT break generation
- `StepDescriber.describe(on_step=...)` fires once per parsed step
- Generator wires `progress.on_stage_progress` to StepDescriber, and
  fires `on_llm_chunk` only when `enable_streaming=True`
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.llm import LLMClient, _extract_chunk_delta
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# _extract_chunk_delta
# ---------------------------------------------------------------------------


class TestExtractChunkDelta:
    def test_sdk_object_shape(self) -> None:
        delta_obj = MagicMock()
        delta_obj.content = "hello"
        choice = MagicMock()
        choice.delta = delta_obj
        chunk = MagicMock()
        chunk.choices = [choice]
        assert _extract_chunk_delta(chunk) == "hello"

    def test_dict_shape(self) -> None:
        chunk = {"choices": [{"delta": {"content": "world"}}]}
        assert _extract_chunk_delta(chunk) == "world"

    def test_empty_delta_returns_empty_string(self) -> None:
        chunk = {"choices": [{"delta": {}}]}
        assert _extract_chunk_delta(chunk) == ""

    def test_no_choices_returns_empty_string(self) -> None:
        assert _extract_chunk_delta({"choices": []}) == ""
        assert _extract_chunk_delta({}) == ""

    def test_none_content_returns_empty_string(self) -> None:
        chunk = {"choices": [{"delta": {"content": None}}]}
        assert _extract_chunk_delta(chunk) == ""


# ---------------------------------------------------------------------------
# LLMClient.generate(stream_callback=...)
# ---------------------------------------------------------------------------


def _make_chunk(content: str | None) -> Any:
    """Build an OpenAI SDK-style streaming chunk."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


class _AsyncChunkIterator:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self._i = 0

    def __aiter__(self) -> _AsyncChunkIterator:
        return self

    async def __anext__(self) -> Any:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


@pytest.mark.asyncio
async def test_generate_streaming_invokes_callback_per_chunk(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    chunks = [_make_chunk("hel"), _make_chunk("lo"), _make_chunk(" "), _make_chunk("wor"), _make_chunk("ld")]

    async def create_side_effect(**kwargs: Any) -> Any:
        # Streaming branch returns an async iterator; non-streaming branch
        # would return a normal response. Assert the streaming flag was
        # actually set.
        assert kwargs.get("stream") is True
        return _AsyncChunkIterator(chunks)

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=0)
    client = LLMClient(cfg)

    received: list[tuple[str, str]] = []

    def cb(stage: str, delta: str) -> None:
        received.append((stage, delta))

    out = await client.generate("sys", "usr", stage="my_stage", stream_callback=cb)

    assert out == "hello world"
    assert [d for _, d in received] == ["hel", "lo", " ", "wor", "ld"]
    assert {s for s, _ in received} == {"my_stage"}


@pytest.mark.asyncio
async def test_generate_streaming_falls_back_when_provider_rejects(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    call_count = {"n": 0}

    async def create_side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if kwargs.get("stream"):
            raise RuntimeError("provider rejects stream=true")
        return _make_openai_response("fallback content")

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=0)
    client = LLMClient(cfg)

    out = await client.generate("sys", "usr", stream_callback=lambda s, d: None)
    assert out == "fallback content"
    # Both attempts happened — stream=True then plain fallback
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_generate_streaming_skips_callback_failures(mocker: Any) -> None:
    """A buggy user callback must not break generation."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]

    async def create_side_effect(**kwargs: Any) -> Any:
        return _AsyncChunkIterator(chunks)

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=0)
    client = LLMClient(cfg)

    call_log: list[str] = []

    def bad_cb(stage: str, delta: str) -> None:
        call_log.append(delta)
        raise RuntimeError("buggy callback")

    out = await client.generate("sys", "usr", stream_callback=bad_cb)
    # Full content is preserved even though the callback blew up.
    assert out == "abc"
    # Callback was invoked at least once before being swapped out for the
    # no-op (we shouldn't keep invoking a broken callback).
    assert call_log == ["a"]


# Note: "provider returns non-streaming response despite stream=True" is
# the real-world degenerate case, but `MagicMock` auto-implements
# `__aiter__` so it can't be modelled cleanly in a unit test. The
# exception-path fallback (above) covers the same code branch.


# ---------------------------------------------------------------------------
# StepDescriber on_step hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_describer_fires_on_step_per_parsed_step(mocker: Any) -> None:
    from mmar_mage.agents import StepDescriber
    from mmar_mage.schemas import DAGStructure

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=0)
    llm = LLMClient(cfg)
    describer = StepDescriber(llm)

    fired: list[tuple[str, int]] = []

    def on_step(stage: str, step: Any) -> None:
        fired.append((stage, step.number))

    steps = await describer.describe(
        "Some query",
        DAGStructure.model_validate(SAMPLE_DAG),
        on_step=on_step,
    )

    # 3 steps in SAMPLE_DESCRIBED_STEPS
    assert len(steps) == 3
    assert fired == [("step_describing", 1), ("step_describing", 2), ("step_describing", 3)]


@pytest.mark.asyncio
async def test_step_describer_swallows_on_step_exceptions(mocker: Any) -> None:
    """on_step callback raising must NOT abort step description parsing."""
    from mmar_mage.agents import StepDescriber
    from mmar_mage.schemas import DAGStructure

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=0)
    llm = LLMClient(cfg)
    describer = StepDescriber(llm)

    def buggy(stage: str, step: Any) -> None:
        raise RuntimeError("boom")

    steps = await describer.describe(
        "Some query",
        DAGStructure.model_validate(SAMPLE_DAG),
        on_step=buggy,
    )
    assert len(steps) == 3


# ---------------------------------------------------------------------------
# Generator wires the callbacks correctly
# ---------------------------------------------------------------------------


class _RecordingCallback:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.completes: list[str] = []
        self.errors: list[tuple[str, Exception]] = []
        self.chunks: list[tuple[str, str]] = []
        self.progress: list[tuple[str, Any]] = []
        self.previews: list[Any] = []
        self.preview_completes_snapshot: list[str] = []

    def on_stage_start(self, stage: str) -> None:
        self.starts.append(stage)

    def on_stage_complete(self, stage: str, result: Any) -> None:
        self.completes.append(stage)

    def on_error(self, stage: str, error: Exception) -> None:
        self.errors.append((stage, error))

    def on_llm_chunk(self, stage: str, delta: str) -> None:
        self.chunks.append((stage, delta))

    def on_stage_progress(self, stage: str, artifact: Any) -> None:
        self.progress.append((stage, artifact))

    def on_chain_preview(self, preview: Any) -> None:
        self.previews.append(preview)
        # snapshot completes-so-far → proves the preview preceded later stages
        self.preview_completes_snapshot = list(self.completes)


def _mock_deep_responses(mocker: Any) -> MagicMock:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance


@pytest.mark.asyncio
async def test_generator_fires_on_stage_progress_for_step_descriptions(deep_config: MAGEConfig, mocker: Any) -> None:
    _mock_deep_responses(mocker)
    # Disable naming LLM so we don't have to mock a 5th response
    deep_config.enable_suggested_naming_llm = False

    cb = _RecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)
    await gen.generate("Analyze a quarterly report")

    # on_stage_progress fired once per step in SAMPLE_DESCRIBED_STEPS
    assert [s for s, _ in cb.progress] == ["step_describing"] * 3
    # step_describing went through start + complete in the legacy hooks too
    assert "step_describing" in cb.starts
    assert "step_describing" in cb.completes


@pytest.mark.asyncio
async def test_generator_fires_on_chain_preview_before_serialization(
    deep_config: MAGEConfig, mocker: Any,
) -> None:
    """P5.8 — the assembled skeleton/DAG is streamed via on_chain_preview
    (a draft chain: topology + per-step title/deps) BEFORE the steps are
    described + serialized."""
    _mock_deep_responses(mocker)
    deep_config.enable_suggested_naming_llm = False

    cb = _RecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)
    await gen.generate("Analyze a quarterly report")

    assert len(cb.previews) == 1  # fired exactly once
    preview = cb.previews[0]
    assert len(preview["steps"]) > 0  # draft chain (step count > 0)
    assert preview.get("topology")  # the chosen pattern
    first = preview["steps"][0]
    assert {"number", "title", "step_type", "dependencies"} <= set(first)
    # the preview preceded step descriptions / serialization
    assert "step_describing" not in cb.preview_completes_snapshot


@pytest.mark.asyncio
async def test_generator_does_not_stream_when_enable_streaming_false(deep_config: MAGEConfig, mocker: Any) -> None:
    """Streaming is opt-in via MAGEConfig.enable_streaming."""
    instance = _mock_deep_responses(mocker)
    deep_config.enable_suggested_naming_llm = False
    assert deep_config.enable_streaming is False  # default

    cb = _RecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)
    await gen.generate("Analyze")

    # All 4 LLM calls were non-streaming
    for call in instance.chat.completions.create.call_args_list:
        assert call.kwargs.get("stream") is not True
    # No chunks fired
    assert cb.chunks == []


@pytest.mark.asyncio
async def test_generator_streams_when_enable_streaming_true(deep_config: MAGEConfig, mocker: Any) -> None:
    """With enable_streaming=True, the StepDescriber call goes through
    the streaming path and `on_llm_chunk` fires."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    # First 3 stages (domain, plan, dag) are non-streaming;
    # 4th stage (step_describing) is streaming and yields chunks.
    step_chunks = [
        _make_chunk(json.dumps(SAMPLE_DESCRIBED_STEPS)[:30]),
        _make_chunk(json.dumps(SAMPLE_DESCRIBED_STEPS)[30:]),
    ]

    call_log: list[bool] = []

    async def create_side_effect(**kwargs: Any) -> Any:
        is_stream = bool(kwargs.get("stream"))
        call_log.append(is_stream)
        if is_stream:
            return _AsyncChunkIterator(step_chunks)
        # Return next non-streaming response in sequence
        idx = sum(1 for x in call_log if not x) - 1
        responses = [
            json.dumps(SAMPLE_DOMAIN_ANALYSIS),
            json.dumps(SAMPLE_STEP_PLAN),
            json.dumps(SAMPLE_DAG),
        ]
        return _make_openai_response(responses[idx])

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_streaming = True
    deep_config.enable_suggested_naming_llm = False

    cb = _RecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)
    await gen.generate("Analyze")

    # Exactly one streaming call happened (the StepDescriber stage)
    assert call_log.count(True) == 1
    # And both chunks were forwarded to on_llm_chunk under stage "step_describing"
    assert len(cb.chunks) == 2
    assert {s for s, _ in cb.chunks} == {"step_describing"}


@pytest.mark.asyncio
async def test_generator_works_with_legacy_callback_missing_optional_methods(
    deep_config: MAGEConfig, mocker: Any
) -> None:
    """Callbacks predating §3 only implement the three legacy methods."""
    _mock_deep_responses(mocker)
    deep_config.enable_suggested_naming_llm = False

    class LegacyCallback:
        def __init__(self) -> None:
            self.starts: list[str] = []

        def on_stage_start(self, stage: str) -> None:
            self.starts.append(stage)

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

    cb = LegacyCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)  # type: ignore[arg-type]
    result = await gen.generate("Analyze")
    # Generation succeeded — the legacy callback raised no AttributeError.
    assert result.metadata.num_steps == 3
    assert "step_describing" in cb.starts
