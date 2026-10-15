"""Tests for §4 P2 — persist intermediate artifacts as memory_cards.

Covers:
- `MemoryManager.save_intermediate_artifacts` writes one entity per
  stage with the right tags (`mage_intermediate` +
  `mage_intermediate_{stage}` + `{domain}`)
- Empty artifacts dict → no saves, returns empty list
- Memory disabled / unavailable → no-op
- Per-stage failure is isolated (one bad save doesn't break the others)
- The chain memory_key is embedded so audit cards link back to their chain
- Generator fires the save when `enable_intermediate_artifact_save=True`,
  populating `MAGEResult.memory_keys["intermediate_artifacts"]`
- Default-off: no audit cards persisted
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# Fake gigaevo_memory used across this module
# ---------------------------------------------------------------------------


class _FakeLegacyClient:
    saved: list[dict[str, Any]] = []
    next_fail_for_stage: str | None = None
    counter: int = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _FakeLegacyClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def save_chain(self, **kwargs: Any) -> Any:
        cls = type(self)
        chain = kwargs.get("chain", {})
        stage = chain.get("stage") if isinstance(chain, dict) else None
        if cls.next_fail_for_stage is not None and stage == cls.next_fail_for_stage:
            raise RuntimeError(f"forced failure for stage {stage}")
        cls.counter += 1
        cls.saved.append(kwargs)
        ref = MagicMock()
        ref.entity_id = f"art-{cls.counter}"
        return ref


def _install_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _FakeLegacyClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)
    _FakeLegacyClient.saved = []
    _FakeLegacyClient.counter = 0
    _FakeLegacyClient.next_fail_for_stage = None


# ---------------------------------------------------------------------------
# MemoryManager.save_intermediate_artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_persists_each_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    artifacts = {
        "domain_analysis": {"domain": "finance"},
        "step_plan": {"steps": [], "rationale": "ok"},
        "dag": {"steps": [], "max_workers": 2},
    }

    ids = await mgr.save_intermediate_artifacts(
        artifacts,
        domain="finance",
        query="Analyze quarterly revenue",
        chain_memory_key="chain-XYZ",
    )
    assert len(ids) == 3
    assert all(eid.startswith("art-") for eid in ids)
    # Each saved record carries kind, stage, domain, query, chain_memory_key
    by_stage = {s["chain"]["stage"]: s for s in _FakeLegacyClient.saved}
    assert set(by_stage.keys()) == {"domain_analysis", "step_plan", "dag"}
    for stage_name, save in by_stage.items():
        assert save["chain"]["kind"] == "intermediate_artifact"
        assert save["chain"]["chain_memory_key"] == "chain-XYZ"
        assert save["chain"]["query"] == "Analyze quarterly revenue"
        tags = save["tags"]
        assert "finance" in tags
        assert "mage_intermediate" in tags
        assert f"mage_intermediate_{stage_name}" in tags


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_empty_input_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    out = await mgr.save_intermediate_artifacts({}, domain="x")
    assert out == []
    assert _FakeLegacyClient.saved == []


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_noop_when_memory_disabled() -> None:
    cfg = MAGEConfig(enable_memory=False, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True
    out = await mgr.save_intermediate_artifacts({"x": {}}, domain="d")
    assert out == []


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_noop_when_unavailable() -> None:
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = False
    out = await mgr.save_intermediate_artifacts({"x": {}}, domain="d")
    assert out == []


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_isolates_per_stage_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One bad stage save must not break the others."""
    _install_legacy(monkeypatch)
    _FakeLegacyClient.next_fail_for_stage = "step_plan"

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    artifacts = {
        "domain_analysis": {"domain": "x"},
        "step_plan": {"steps": []},  # this one fails
        "dag": {"max_workers": 1},
    }
    ids = await mgr.save_intermediate_artifacts(artifacts, domain="x")
    # 2 of 3 succeeded
    assert len(ids) == 2
    saved_stages = {s["chain"]["stage"] for s in _FakeLegacyClient.saved}
    assert saved_stages == {"domain_analysis", "dag"}


@pytest.mark.asyncio
async def test_save_intermediate_artifacts_omits_query_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    await mgr.save_intermediate_artifacts({"domain_analysis": {"x": 1}}, domain="d", query=None)
    assert "query" not in _FakeLegacyClient.saved[0]["chain"]


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------


def _mock_deep_4_calls(mocker: Any) -> MagicMock:
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
async def test_generator_persists_intermediate_artifacts_when_enabled(deep_config: MAGEConfig, mocker: Any) -> None:
    _mock_deep_4_calls(mocker)
    deep_config.enable_memory = False  # short-circuit save_chain
    deep_config.enable_intermediate_artifact_save = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    save_spy = mocker.patch.object(
        gen._memory,
        "save_intermediate_artifacts",
        new_callable=AsyncMock,
        return_value=["int-1", "int-2", "int-3", "int-4"],
    )
    result = await gen.generate("Analyze quarterly")

    save_spy.assert_called_once()
    # The call's first positional arg is the intermediate_artifacts dict
    artifacts_passed = save_spy.call_args.args[0]
    assert "domain_analysis" in artifacts_passed
    assert "step_plan" in artifacts_passed
    assert "dag" in artifacts_passed
    assert "described_steps" in artifacts_passed
    # Returned IDs land on memory_keys
    assert result.memory_keys["intermediate_artifacts"] == ["int-1", "int-2", "int-3", "int-4"]


@pytest.mark.asyncio
async def test_generator_noop_intermediate_save_when_disabled(deep_config: MAGEConfig, mocker: Any) -> None:
    """Default: no intermediate audit cards persisted."""
    _mock_deep_4_calls(mocker)
    deep_config.enable_memory = False
    deep_config.enable_suggested_naming_llm = False
    # enable_intermediate_artifact_save defaults to False

    gen = MAGEGenerator(config=deep_config)
    save_spy = mocker.patch.object(gen._memory, "save_intermediate_artifacts", new_callable=AsyncMock)
    result = await gen.generate("Analyze")
    save_spy.assert_not_called()
    assert "intermediate_artifacts" not in result.memory_keys or not result.memory_keys.get("intermediate_artifacts")


@pytest.mark.asyncio
async def test_generator_passes_chain_memory_key_to_save(deep_config: MAGEConfig, mocker: Any) -> None:
    """The chain's memory_key (when known) reaches save_intermediate_artifacts."""
    _mock_deep_4_calls(mocker)
    deep_config.enable_memory = False
    deep_config.enable_intermediate_artifact_save = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    # Pretend save_chain returned a known id
    mocker.patch.object(gen._memory, "save_chain", new_callable=AsyncMock, return_value="MAIN-ID")
    save_spy = mocker.patch.object(
        gen._memory,
        "save_intermediate_artifacts",
        new_callable=AsyncMock,
        return_value=[],
    )
    await gen.generate("X")
    kwargs = save_spy.call_args.kwargs
    assert kwargs["chain_memory_key"] == "MAIN-ID"
    assert kwargs["domain"] == "finance"
    assert kwargs["query"] == "X"


@pytest.mark.asyncio
async def test_generator_save_intermediate_failure_is_isolated(deep_config: MAGEConfig, mocker: Any) -> None:
    """Failure in save_intermediate_artifacts must not break generation."""
    _mock_deep_4_calls(mocker)
    deep_config.enable_memory = False
    deep_config.enable_intermediate_artifact_save = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    mocker.patch.object(
        gen._memory,
        "save_intermediate_artifacts",
        new_callable=AsyncMock,
        side_effect=RuntimeError("memory down"),
    )
    # Generation still succeeds.
    result = await gen.generate("X")
    assert result.metadata.num_steps == 3
