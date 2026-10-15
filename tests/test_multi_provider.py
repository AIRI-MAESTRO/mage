"""Tests for §6 P2 — multi-provider robustness.

Covers:
- `StageLLMOverride` rejects unknown keys (typo-safety)
- `MAGEConfig.stage_llm_overrides` accepts bare dicts and promotes to
  the typed model
- `LLMClient.get_stage_config` reads both attribute and dict shapes
- `MAGEConfig.json_mode_fallback` defaults: True for provider="local",
  False for the rest, explicit value always wins
- `LLMClient.generate(json_mode=True)` with fallback ON appends the
  JSON-only suffix and skips `response_format=json_object`
- `LLMClient.generate_json` recovers parseable JSON from a
  markdown-fenced / prose-wrapped response when fallback is on
- `_extract_json_object` handles fenced code, prose-wrapped, nested
  objects, and missing braces
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.exceptions import MAGELLMError
from mmar_mage.llm import LLMClient, _extract_json_object
from mmar_mage.schemas import MAGEConfig, StageLLMOverride

from .conftest import _make_openai_response

# ---------------------------------------------------------------------------
# StageLLMOverride
# ---------------------------------------------------------------------------


class TestStageLLMOverride:
    def test_accepts_model_and_temperature(self) -> None:
        ov = StageLLMOverride(model="gpt-4o-mini", temperature=0.1)
        assert ov.model == "gpt-4o-mini"
        assert ov.temperature == 0.1

    def test_both_optional(self) -> None:
        ov = StageLLMOverride()
        assert ov.model is None
        assert ov.temperature is None

    def test_rejects_unknown_keys(self) -> None:
        """The whole point: typos must raise."""
        with pytest.raises(ValueError) as exc_info:
            StageLLMOverride(temprature=0.1)  # type: ignore[call-arg]
        assert "temprature" in str(exc_info.value).lower() or "extra" in str(exc_info.value).lower()


class TestMAGEConfigStageOverrides:
    def test_promotes_bare_dict_to_typed_override(self) -> None:
        cfg = MAGEConfig(stage_llm_overrides={"domain_analysis": {"model": "gpt-4o-mini", "temperature": 0.2}})
        ov = cfg.stage_llm_overrides["domain_analysis"]
        assert isinstance(ov, StageLLMOverride)
        assert ov.model == "gpt-4o-mini"

    def test_accepts_typed_override_directly(self) -> None:
        cfg = MAGEConfig(
            stage_llm_overrides={
                "step_planning": StageLLMOverride(model="x", temperature=0.0),
            }
        )
        ov = cfg.stage_llm_overrides["step_planning"]
        assert ov.model == "x"

    def test_rejects_typo_in_nested_dict(self) -> None:
        with pytest.raises(ValueError):
            MAGEConfig(stage_llm_overrides={"x": {"temprature": 0.5}})


# ---------------------------------------------------------------------------
# LLMClient.get_stage_config — reads typed + dict-shaped overrides
# ---------------------------------------------------------------------------


def test_get_stage_config_uses_typed_override(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    cfg = MAGEConfig(
        model="default-model",
        temperature=0.5,
        api_key="x",
        stage_llm_overrides={
            "step_planning": StageLLMOverride(model="override-model", temperature=0.1),
        },
    )
    client = LLMClient(cfg)
    out = client.get_stage_config("step_planning")
    assert out["model"] == "override-model"
    assert out["temperature"] == 0.1


def test_get_stage_config_falls_back_to_global_when_no_override(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    cfg = MAGEConfig(model="default-model", temperature=0.5, api_key="x")
    client = LLMClient(cfg)
    out = client.get_stage_config("any_stage")
    assert out["model"] == "default-model"
    assert out["temperature"] == 0.5


def test_get_stage_config_uses_global_when_override_field_is_none(mocker: Any) -> None:
    """Override with only model set: temperature falls back to global."""
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    cfg = MAGEConfig(
        model="default-model",
        temperature=0.5,
        api_key="x",
        stage_llm_overrides={"x": StageLLMOverride(model="override-model")},
    )
    client = LLMClient(cfg)
    out = client.get_stage_config("x")
    assert out["model"] == "override-model"
    assert out["temperature"] == 0.5  # fell back


# ---------------------------------------------------------------------------
# json_mode_fallback defaults
# ---------------------------------------------------------------------------


class TestJsonModeFallbackDefaults:
    def test_local_provider_auto_enables_fallback(self) -> None:
        cfg = MAGEConfig(provider="local")
        assert cfg.json_mode_fallback is True

    def test_openai_provider_disables_fallback(self) -> None:
        cfg = MAGEConfig(provider="openai")
        assert cfg.json_mode_fallback is False

    def test_openrouter_provider_disables_fallback(self) -> None:
        cfg = MAGEConfig(provider="openrouter")
        assert cfg.json_mode_fallback is False

    def test_explicit_true_wins_over_provider(self) -> None:
        cfg = MAGEConfig(provider="openai", json_mode_fallback=True)
        assert cfg.json_mode_fallback is True

    def test_explicit_false_wins_over_local_default(self) -> None:
        cfg = MAGEConfig(provider="local", json_mode_fallback=False)
        assert cfg.json_mode_fallback is False


# ---------------------------------------------------------------------------
# _extract_json_object — brace-balanced extraction
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_plain_object(self) -> None:
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_with_prose_prefix(self) -> None:
        out = _extract_json_object('Here is the answer: {"a": 1}.')
        assert out == '{"a": 1}'

    def test_markdown_fenced_with_lang(self) -> None:
        out = _extract_json_object('```json\n{"a": 1}\n```')
        assert out is not None
        assert json.loads(out) == {"a": 1}

    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": {"deep": 1}}, "list": [1,2,3]}'
        assert _extract_json_object(text) == text

    def test_string_with_braces_inside(self) -> None:
        """Braces inside a string literal must NOT confuse the depth counter."""
        text = '{"description": "use {placeholder}"}'
        assert _extract_json_object(text) == text

    def test_escaped_quote_inside_string(self) -> None:
        text = '{"q": "he said \\"hi\\""}'
        assert _extract_json_object(text) == text

    def test_no_braces_returns_none(self) -> None:
        assert _extract_json_object("just prose") is None

    def test_empty_returns_none(self) -> None:
        assert _extract_json_object("") is None

    def test_unbalanced_returns_none(self) -> None:
        assert _extract_json_object('{"a": 1') is None


# ---------------------------------------------------------------------------
# LLMClient.generate — json_mode_fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_fallback_skips_response_format(mocker: Any) -> None:
    """With fallback ON, the call must NOT include response_format=json_object."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_openai_response('{"ok": true}')

    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="local", model="m", api_key="x", max_retries=0)
    assert cfg.json_mode_fallback is True
    client = LLMClient(cfg)

    out = await client.generate("System prompt.", "User prompt.", json_mode=True)
    assert out == '{"ok": true}'
    # response_format must NOT be in the API call when fallback is on
    assert "response_format" not in captured
    # System prompt got the strict-JSON suffix appended
    sys_msg = captured["messages"][0]["content"]
    assert "Return ONLY a single JSON object" in sys_msg


@pytest.mark.asyncio
async def test_generate_without_fallback_uses_response_format(mocker: Any) -> None:
    """With fallback OFF (default for openai), response_format IS set."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_openai_response('{"ok": true}')

    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="openai", model="m", api_key="x", max_retries=0)
    assert cfg.json_mode_fallback is False
    client = LLMClient(cfg)

    await client.generate("S", "U", json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}
    # And the system prompt did NOT get the suffix.
    assert "Return ONLY a single JSON object" not in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_generate_no_json_mode_never_appends_suffix(mocker: Any) -> None:
    """json_mode=False bypasses fallback machinery entirely."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_openai_response("free text")

    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="local", model="m", api_key="x", max_retries=0)
    client = LLMClient(cfg)

    await client.generate("S", "U", json_mode=False)
    assert "response_format" not in captured
    assert "Return ONLY a single JSON object" not in captured["messages"][0]["content"]


# ---------------------------------------------------------------------------
# LLMClient.generate_json — regex extraction on fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_json_recovers_from_markdown_fenced_response(mocker: Any) -> None:
    """Local model wrapped its JSON in markdown fences — fallback recovers it."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    raw = 'Sure, here you go:\n```json\n{"k": "v"}\n```'
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(raw))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="local", model="m", api_key="x", max_retries=0)
    client = LLMClient(cfg)
    out = await client.generate_json("S", "U")
    assert out == {"k": "v"}


@pytest.mark.asyncio
async def test_generate_json_recovers_from_prose_wrapped_response(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    raw = 'The plan is: {"steps": 3, "ok": true}. Hope this helps!'
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(raw))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="local", model="m", api_key="x", max_retries=0)
    client = LLMClient(cfg)
    out = await client.generate_json("S", "U")
    assert out == {"steps": 3, "ok": True}


@pytest.mark.asyncio
async def test_generate_json_repairs_or_raises_clearly(mocker: Any) -> None:
    """T21: prose-wrapped JSON is now repaired even with fallback off (the run-773
    failure class was on OpenRouter, not the local fallback path), but TRULY
    unparseable output still raises a clear MAGELLMError."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    cfg = MAGEConfig(provider="openai", model="m", api_key="x", max_retries=0)
    client = LLMClient(cfg)

    # recoverable: prose-wrapped object → repaired
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response('Sure: {"k": "v"} OK'))
    assert await client.generate_json("S", "U") == {"k": "v"}

    # unrecoverable: no JSON at all → still raises
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("totally not json"))
    with pytest.raises(MAGELLMError):
        await client.generate_json("S", "U")


@pytest.mark.asyncio
async def test_generate_json_clean_response_works_under_fallback(mocker: Any) -> None:
    """When the model DOES return a clean object, fallback still works."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response('{"clean": "json"}'))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(provider="local", model="m", api_key="x", max_retries=0)
    client = LLMClient(cfg)
    out = await client.generate_json("S", "U")
    assert out == {"clean": "json"}
