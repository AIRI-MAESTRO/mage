"""Tests for §3 P1 — cancellation + retry visibility.

Covers:
- `_check_cancel()` helper raises `MAGECancelledError` only when the
  event is set, no-ops when ``cancel`` is ``None``
- `MAGEGenerator.generate(cancel=...)` aborts before the first LLM call
  when event is pre-set
- Cancellation observed mid-pipeline (between stages) aborts cleanly
- `LLMClient.generate(retry_callback=...)` invokes the callback once
  per failed attempt that has a retry queued, NOT on the final attempt
- `MAGEGenerator` wires `progress.on_retry` to `LLMClient` via
  ``set_retry_callback`` for the lifetime of the call
- Legacy callbacks without `on_retry` continue to work
- `MAGECancelledError` is exported from the package
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import mmar_mage as mage
from mmar_mage.exceptions import MAGECancelledError
from mmar_mage.generator import MAGEGenerator, _check_cancel
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_CARL_CHAIN,
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# _check_cancel helper
# ---------------------------------------------------------------------------


class TestCheckCancel:
    def test_none_event_is_noop(self) -> None:
        # Should not raise
        _check_cancel(None, "anywhere")

    def test_unset_event_is_noop(self) -> None:
        ev = asyncio.Event()
        _check_cancel(ev, "anywhere")

    def test_set_event_raises_with_stage(self) -> None:
        ev = asyncio.Event()
        ev.set()
        with pytest.raises(MAGECancelledError) as exc_info:
            _check_cancel(ev, "step_planning")
        assert exc_info.value.stage == "step_planning"
        assert "step_planning" in str(exc_info.value)


def test_mage_cancelled_error_exported_from_package() -> None:
    assert mage.MAGECancelledError is MAGECancelledError


# ---------------------------------------------------------------------------
# Cancellation via MAGEGenerator.generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_set_cancel_event_aborts_before_any_llm_call(config: MAGEConfig, mocker: Any) -> None:
    """If the cancel event is already set, no LLM call should happen."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cancel = asyncio.Event()
    cancel.set()

    gen = MAGEGenerator(config=config)
    with pytest.raises(MAGECancelledError):
        await gen.generate("Anything", cancel=cancel)

    instance.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_event_set_mid_deep_pipeline_aborts(deep_config: MAGEConfig, mocker: Any) -> None:
    """Set the event after the domain-analysis call; the next stage check fires."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    cancel = asyncio.Event()

    call_count = {"n": 0}

    async def create_side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        # After the first LLM call (domain_analysis) finishes, set the
        # event so the next _check_cancel raises.
        cancel.set()
        return _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS))

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    with pytest.raises(MAGECancelledError) as exc_info:
        await gen.generate("Analyze a quarterly report", cancel=cancel)

    # Exactly ONE LLM call happened — the next stage's pre-check aborted.
    assert call_count["n"] == 1
    # Stage attribute identifies where cancellation was observed.
    assert exc_info.value.stage == "step_planning"


@pytest.mark.asyncio
async def test_unset_cancel_event_does_not_interfere(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Passing a non-set event must not change behavior."""
    cancel = asyncio.Event()  # never set
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test query", cancel=cancel)
    assert result.metadata.num_steps == 3


@pytest.mark.asyncio
async def test_generate_without_cancel_arg_still_works(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Backwards-compat: omitting cancel keyword keeps the legacy behavior."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test query")
    assert result.metadata.num_steps == 3


# ---------------------------------------------------------------------------
# Retry visibility via LLMClient.generate(retry_callback=...)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_callback_fires_for_each_retryable_attempt(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    call_count = {"n": 0}

    async def create_side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise RuntimeError(f"transient {call_count['n']}")
        return _make_openai_response("ok")

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=3)
    client = LLMClient(cfg)

    fired: list[tuple[str, int, str]] = []

    def cb(stage: str, attempt: int, error: Exception) -> None:
        fired.append((stage, attempt, type(error).__name__))

    out = await client.generate("s", "u", stage="my_stage", retry_callback=cb)
    assert out == "ok"
    # Two failed attempts → two on_retry events (third succeeded).
    assert fired == [
        ("my_stage", 1, "RuntimeError"),
        ("my_stage", 2, "RuntimeError"),
    ]


@pytest.mark.asyncio
async def test_retry_callback_not_fired_on_final_attempt(mocker: Any) -> None:
    """When the call exhausts all retries, the FINAL failure doesn't fire on_retry."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("nope"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=2)
    client = LLMClient(cfg)

    fired: list[int] = []
    with pytest.raises(mage.MAGELLMError):
        await client.generate(
            "s",
            "u",
            stage="my_stage",
            retry_callback=lambda stage, attempt, error: fired.append(attempt),
        )
    # max_retries=2 → 3 attempts total, first 2 fire on_retry, last one
    # raises through to the caller without firing on_retry.
    assert fired == [1, 2]


@pytest.mark.asyncio
async def test_retry_callback_failure_is_isolated(mocker: Any) -> None:
    """A buggy on_retry callback must not break LLM generation."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    call_count = {"n": 0}

    async def create_side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("transient")
        return _make_openai_response("recovered")

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=2)
    client = LLMClient(cfg)

    def buggy(stage: str, attempt: int, error: Exception) -> None:
        raise ValueError("callback bug")

    out = await client.generate("s", "u", retry_callback=buggy)
    assert out == "recovered"


# ---------------------------------------------------------------------------
# LLMClient.set_retry_callback installs a default for subsequent calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_retry_callback_used_when_call_omits_one(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    call_count = {"n": 0}

    async def create_side_effect(**kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("transient")
        return _make_openai_response("ok")

    instance.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(model="m", api_key="k", max_retries=2)
    client = LLMClient(cfg)

    fired: list[int] = []
    client.set_retry_callback(lambda stage, attempt, error: fired.append(attempt))

    out = await client.generate("s", "u")
    assert out == "ok"
    assert fired == [1]


# ---------------------------------------------------------------------------
# Generator wires on_retry via set_retry_callback for the lifetime of generate()
# ---------------------------------------------------------------------------


class _RecordingCallback:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.completes: list[str] = []
        self.errors: list[tuple[str, Exception]] = []
        self.retries: list[tuple[str, int, str]] = []

    def on_stage_start(self, stage: str) -> None:
        self.starts.append(stage)

    def on_stage_complete(self, stage: str, result: Any) -> None:
        self.completes.append(stage)

    def on_error(self, stage: str, error: Exception) -> None:
        self.errors.append((stage, error))

    def on_retry(self, stage: str, attempt: int, error: Exception) -> None:
        self.retries.append((stage, attempt, type(error).__name__))


@pytest.mark.asyncio
async def test_generator_wires_on_retry_into_llm_client(deep_config: MAGEConfig, mocker: Any) -> None:
    """A retry inside a deep-mode stage fires the callback."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    domain_resp = _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS))
    plan_resp = _make_openai_response(json.dumps(SAMPLE_STEP_PLAN))
    dag_resp = _make_openai_response(json.dumps(SAMPLE_DAG))
    described_resp = _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS))

    # First domain call fails, the retry succeeds, then plan/dag/described.
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            RuntimeError("flaky once"),
            domain_resp,
            plan_resp,
            dag_resp,
            described_resp,
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.max_retries = 2
    deep_config.enable_suggested_naming_llm = False

    cb = _RecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)
    result = await gen.generate("Analyze quarterly")

    assert result.metadata.num_steps == 3
    # Exactly one retry observed, on the domain_analysis stage.
    assert cb.retries == [("domain_analysis", 1, "RuntimeError")]


@pytest.mark.asyncio
async def test_generator_works_with_legacy_callback_missing_on_retry(config: MAGEConfig, mocker: Any) -> None:
    """Old callbacks without on_retry must not break retries."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    # First call fails, second succeeds.
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            RuntimeError("flaky"),
            _make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    class LegacyCallback:
        def __init__(self) -> None:
            self.starts: list[str] = []

        def on_stage_start(self, stage: str) -> None:
            self.starts.append(stage)

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

    config.max_retries = 2
    cb = LegacyCallback()
    gen = MAGEGenerator(config=config, progress=cb)  # type: ignore[arg-type]
    result = await gen.generate("Test")
    assert result.metadata.num_steps == 3
