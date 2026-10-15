"""Tests for §5 P1 — memory writes for ecosystem reuse.

Covers:
- `is_reusable_step()` heuristic accepts clean LLM steps and rejects
  non-llm / dynamic-ref / sparse-aim steps
- `MemoryManager.save_agent_skill()` prefers `gigaevo_client.MemoryClient.
  ingest_skill_from_carl` and falls back to legacy `gigaevo_memory`
- `MemoryManager.save_reusable_step()` writes a `step` entity tagged
  `template` via `gigaevo_memory.MemoryClient.save_chain`
- `MAGEGenerator` fires both helpers after chain finalize, gated on the
  new `enable_skill_provenance_save` / `enable_reusable_step_save` flags
- `_resolve_skill_payload()` reads both flat and nested CARL shapes
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator, _resolve_skill_payload
from mmar_mage.memory import MemoryManager, is_reusable_step
from mmar_mage.schemas import MAGEConfig

from .conftest import _make_openai_response

# ---------------------------------------------------------------------------
# is_reusable_step
# ---------------------------------------------------------------------------


class TestIsReusableStep:
    def _llm_step(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "step_type": "llm",
            "number": 1,
            "title": "Synthesize",
            "aim": "Synthesize a structured summary of the analysis",
            "reasoning_questions": "What conclusions follow?",
            "stage_action": "Produce a coherent summary",
            "example_reasoning": "Summary answers the question",
            "dependencies": [],
            "retry_max": 3,
        }
        base.update(overrides)
        return base

    def test_clean_llm_step_is_reusable(self) -> None:
        assert is_reusable_step(self._llm_step()) is True

    def test_non_llm_step_is_not_reusable(self) -> None:
        step = self._llm_step(step_type="tool")
        assert is_reusable_step(step) is False

    def test_dynamic_template_var_disqualifies(self) -> None:
        step = self._llm_step(aim="Process {{user_input}} carefully")
        assert is_reusable_step(step) is False

    def test_step_n_output_ref_disqualifies(self) -> None:
        step = self._llm_step(stage_action="Use step_1_output as the seed")
        assert is_reusable_step(step) is False

    def test_short_aim_disqualifies(self) -> None:
        step = self._llm_step(aim="Do it")
        assert is_reusable_step(step) is False

    def test_missing_aim_disqualifies(self) -> None:
        step = self._llm_step(aim="")
        assert is_reusable_step(step) is False

    def test_not_a_dict_returns_false(self) -> None:
        assert is_reusable_step("not a step") is False  # type: ignore[arg-type]
        assert is_reusable_step(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_skill_payload
# ---------------------------------------------------------------------------


class TestResolveSkillPayload:
    def test_flat_shape(self) -> None:
        step = {
            "step_type": "agent_skill",
            "skill_name": "pdf",
            "skill": "github://anthropics/skills/pdf@main",
            "aim": "Extract tables",
        }
        out = _resolve_skill_payload(step)
        assert out["name"] == "pdf"
        assert out["uri"] == "github://anthropics/skills/pdf@main"
        assert "Extract tables" in out["description"]

    def test_nested_step_config_shape(self) -> None:
        step = {
            "step_type": "agent_skill",
            "title": "Extract",
            "step_config": {
                "skill_name": "docx",
                "skill": "github://anthropics/skills/docx@main",
            },
        }
        out = _resolve_skill_payload(step)
        assert out["name"] == "docx"
        assert out["uri"] == "github://anthropics/skills/docx@main"

    def test_uri_only_derives_name(self) -> None:
        step = {"step_type": "agent_skill", "skill": "github://acme/x-skill@v1"}
        out = _resolve_skill_payload(step)
        assert out["name"] == "x-skill"
        assert out["uri"] == "github://acme/x-skill@v1"


# ---------------------------------------------------------------------------
# MemoryManager.save_agent_skill — gigaevo_client preferred path
# ---------------------------------------------------------------------------


class _FakeClientV2:
    last: dict[str, Any] = {}

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _FakeClientV2:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def ingest_skill_from_carl(self, resolved: dict[str, Any], entity_id: str | None = None) -> Any:
        type(self).last = {"resolved": resolved, "entity_id": entity_id}
        ref = MagicMock()
        ref.entity_id = "v2-skill-id"
        return ref


def _install_fake_v2_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_client")
    pkg.MemoryClient = _FakeClientV2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_client", pkg)


@pytest.mark.asyncio
async def test_save_agent_skill_uses_gigaevo_client_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_v2_client(monkeypatch)
    _FakeClientV2.last = {}

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = False  # forces fallback path to be unavailable

    eid = await mgr.save_agent_skill(
        {"name": "pdf", "uri": "github://anthropics/skills/pdf", "sha256": "abc"},
        entity_id="my-id",
    )
    assert eid == "v2-skill-id"
    assert _FakeClientV2.last["entity_id"] == "my-id"
    assert _FakeClientV2.last["resolved"]["name"] == "pdf"


@pytest.mark.asyncio
async def test_save_agent_skill_returns_none_when_memory_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_v2_client(monkeypatch)
    cfg = MAGEConfig(enable_memory=False)
    mgr = MemoryManager(cfg)

    eid = await mgr.save_agent_skill({"name": "x", "uri": "u"})
    assert eid is None


@pytest.mark.asyncio
async def test_save_agent_skill_falls_back_to_legacy_when_no_v2_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without gigaevo_client, falls back to gigaevo_memory.save_chain shape."""
    # Make sure gigaevo_client is NOT in sys.modules
    monkeypatch.delitem(sys.modules, "gigaevo_client", raising=False)

    # Install fake gigaevo_memory
    class _FakeLegacyClient:
        last: dict[str, Any] = {}

        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def __enter__(self) -> _FakeLegacyClient:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def save_chain(self, **kwargs: Any) -> Any:
            type(self).last = kwargs
            ref = MagicMock()
            ref.entity_id = "legacy-skill-id"
            return ref

    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _FakeLegacyClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    eid = await mgr.save_agent_skill({"name": "pdf", "uri": "github://x/y", "tags": ["documents"]})
    assert eid == "legacy-skill-id"
    chain = _FakeLegacyClient.last["chain"]
    assert chain["kind"] == "agent_skill"
    assert chain["name"] == "pdf"


# ---------------------------------------------------------------------------
# MemoryManager.save_reusable_step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_reusable_step_persists_with_template_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLegacyClient:
        last: dict[str, Any] = {}

        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def __enter__(self) -> _FakeLegacyClient:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def save_chain(self, **kwargs: Any) -> Any:
            type(self).last = kwargs
            ref = MagicMock()
            ref.entity_id = "step-id"
            return ref

    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _FakeLegacyClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    eid = await mgr.save_reusable_step(
        {"step_type": "llm", "number": 1, "title": "Synthesize", "aim": "..."},
        domain="finance",
        extra_tags=["uses-pdf"],
    )
    assert eid == "step-id"
    saved = _FakeLegacyClient.last
    assert "template" in saved["tags"]
    assert "finance" in saved["tags"]
    assert "uses-pdf" in saved["tags"]
    assert saved["chain"]["kind"] == "reusable_step"


@pytest.mark.asyncio
async def test_save_reusable_step_noop_when_memory_unavailable() -> None:
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = False
    eid = await mgr.save_reusable_step({"step_type": "llm", "title": "x", "aim": "y"}, domain="d")
    assert eid is None


# ---------------------------------------------------------------------------
# Generator integration — both saves fire when flags are on
# ---------------------------------------------------------------------------


SAMPLE_CHAIN_WITH_SKILL: dict[str, Any] = {
    "name": "Test",
    "steps": [
        {
            "step_type": "agent_skill",
            "number": 1,
            "title": "Read PDF",
            "aim": "Use the pdf skill",
            "reasoning_questions": "?",
            "step_context_queries": ["x"],
            "stage_action": "do",
            "example_reasoning": "y",
            "skill_name": "pdf",
            "skill": "github://anthropics/skills/pdf",
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Synthesize",
            "aim": "Synthesize a structured summary of the analysis",
            "reasoning_questions": "What conclusions follow?",
            "step_context_queries": ["summary"],
            "stage_action": "Produce a coherent summary",
            "example_reasoning": "Answers the question",
            "dependencies": [1],
            "retry_max": 3,
        },
    ],
}


@pytest.mark.asyncio
async def test_generator_persists_skill_provenance_when_enabled(config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(json.dumps(SAMPLE_CHAIN_WITH_SKILL))
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config.enable_memory = False  # short-circuit save_chain
    config.enable_skill_provenance_save = True

    gen = MAGEGenerator(config=config)
    save_skill_spy = mocker.patch.object(
        gen._memory, "save_agent_skill", new_callable=AsyncMock, return_value="skill-id"
    )
    save_step_spy = mocker.patch.object(
        gen._memory, "save_reusable_step", new_callable=AsyncMock, return_value="step-id"
    )

    await gen.generate("Test query")

    # save_agent_skill fired exactly once — for the agent_skill step
    assert save_skill_spy.call_count == 1
    resolved = save_skill_spy.call_args.args[0]
    assert resolved["name"] == "pdf"
    # the stale sample URI is normalised to the canonical registry URI
    # (the anthropics/skills repo moved to ``skills/<name>``)
    assert resolved["uri"] == "github://anthropics/skills/skills/pdf@main"

    # save_reusable_step did NOT fire (flag off)
    assert save_step_spy.call_count == 0


@pytest.mark.asyncio
async def test_generator_persists_reusable_steps_when_enabled(config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(json.dumps(SAMPLE_CHAIN_WITH_SKILL))
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config.enable_memory = False
    config.enable_reusable_step_save = True

    gen = MAGEGenerator(config=config)
    save_skill_spy = mocker.patch.object(gen._memory, "save_agent_skill", new_callable=AsyncMock, return_value=None)
    save_step_spy = mocker.patch.object(
        gen._memory, "save_reusable_step", new_callable=AsyncMock, return_value="step-id"
    )

    await gen.generate("Test query")

    # save_reusable_step fired exactly once — for the LLM step (number 2)
    assert save_step_spy.call_count == 1
    saved_step = save_step_spy.call_args.args[0]
    assert saved_step["number"] == 2
    assert saved_step["step_type"] == "llm"
    # save_agent_skill NOT called (flag off)
    assert save_skill_spy.call_count == 0


@pytest.mark.asyncio
async def test_generator_noop_when_both_flags_off(config: MAGEConfig, mocker: Any) -> None:
    """Default behavior: no ecosystem writes happen."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(json.dumps(SAMPLE_CHAIN_WITH_SKILL))
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config.enable_memory = False
    # both ecosystem-save flags default to False

    gen = MAGEGenerator(config=config)
    save_skill_spy = mocker.patch.object(gen._memory, "save_agent_skill", new_callable=AsyncMock)
    save_step_spy = mocker.patch.object(gen._memory, "save_reusable_step", new_callable=AsyncMock)

    await gen.generate("Test query")
    assert save_skill_spy.call_count == 0
    assert save_step_spy.call_count == 0


@pytest.mark.asyncio
async def test_generator_save_skill_failure_is_isolated(config: MAGEConfig, mocker: Any) -> None:
    """A failing save_agent_skill must not break generation."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(json.dumps(SAMPLE_CHAIN_WITH_SKILL))
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config.enable_memory = False
    config.enable_skill_provenance_save = True

    gen = MAGEGenerator(config=config)
    mocker.patch.object(
        gen._memory,
        "save_agent_skill",
        new_callable=AsyncMock,
        side_effect=RuntimeError("memory down"),
    )

    # Generation still completes successfully.
    result = await gen.generate("Test query")
    assert result.metadata.num_steps == 2
