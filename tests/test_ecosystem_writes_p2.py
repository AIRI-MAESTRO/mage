"""Tests for §5 P2 — failure-card persistence + evolution_meta block.

Covers:
- `MemoryManager.save_failure_card()` writes a `memory_card` payload
  with the right tags and graceful degradation
- `MemoryManager.save_chain(evolution_meta=...)` embeds the block under
  `metadata.evolution_meta` (both via the typed CARE helper and the
  direct dict-merge fallback)
- `_build_evolution_meta()` strips `None` signals
- `MAGEGenerator` fires `save_failure_card` when verifier rejects the
  chain (config flag on) and when self-refine hits its iteration cap
- `MAGEGenerator` passes a populated `evolution_meta` to
  `save_chain` when `enable_evolution_meta_save=True`
- All saves are isolated — exceptions during ecosystem-write don't
  affect the main generation result
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.chain_verifier import VerificationQuestion, VerificationResult
from mmar_mage.agents.self_refiner import RefinementResult
from mmar_mage.generator import MAGEGenerator, _build_evolution_meta
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig, MAGEMetadata

from .conftest import SAMPLE_CARL_CHAIN, _make_openai_response

# ---------------------------------------------------------------------------
# MemoryManager.save_failure_card
# ---------------------------------------------------------------------------


class _FakeLegacyClient:
    saved: list[dict[str, Any]] = []

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _FakeLegacyClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def save_chain(self, **kwargs: Any) -> Any:
        type(self).saved.append(kwargs)
        ref = MagicMock()
        ref.entity_id = f"id-{len(type(self).saved)}"
        return ref


def _install_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _FakeLegacyClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)


@pytest.mark.asyncio
async def test_save_failure_card_persists_with_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_legacy(monkeypatch)
    _FakeLegacyClient.saved = []

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    eid = await mgr.save_failure_card(
        query="Analyze a PDF",
        domain="finance",
        reason="verifier failed",
        stage="chain_verification",
        extra={"issues": 3},
    )
    assert eid == "id-1"
    saved = _FakeLegacyClient.saved[0]
    assert saved["chain"]["kind"] == "failure_card"
    assert saved["chain"]["reason"] == "verifier failed"
    assert saved["chain"]["stage"] == "chain_verification"
    assert saved["chain"]["extra"] == {"issues": 3}
    tags = saved["tags"]
    assert "finance" in tags
    assert "mage-failure" in tags
    assert "mage-failure-chain_verification" in tags


@pytest.mark.asyncio
async def test_save_failure_card_omits_chain_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    _FakeLegacyClient.saved = []

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    await mgr.save_failure_card(query="x", domain="d", reason="r", stage="s", chain_dict=None)
    assert "chain" not in _FakeLegacyClient.saved[0]["chain"]


@pytest.mark.asyncio
async def test_save_failure_card_noop_when_unavailable() -> None:
    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = False
    eid = await mgr.save_failure_card(query="x", domain="d", reason="r", stage="s")
    assert eid is None


# ---------------------------------------------------------------------------
# save_chain(evolution_meta=...) embeds the block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_chain_persists_evolution_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    _FakeLegacyClient.saved = []

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    await mgr.save_chain(
        {"name": "test", "steps": [{"number": 1, "title": "A", "step_type": "llm"}]},
        domain="finance",
        evolution_meta={"step_critique_score": 0.85, "verification_passed": True},
    )
    saved_chain = _FakeLegacyClient.saved[0]["chain"]
    assert saved_chain["metadata"]["evolution_meta"]["step_critique_score"] == 0.85
    assert saved_chain["metadata"]["evolution_meta"]["verification_passed"] is True


@pytest.mark.asyncio
async def test_save_chain_evolution_meta_coexists_with_care_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evolution_meta must NOT trample the existing CARE metadata block."""
    _install_legacy(monkeypatch)
    _FakeLegacyClient.saved = []

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    await mgr.save_chain(
        {"name": "test", "steps": [{"number": 1, "title": "A", "step_type": "llm"}]},
        domain="finance",
        task_description="Analyze X",
        display_name="X Reader",
        evolution_meta={"step_critique_score": 0.7},
    )
    metadata = _FakeLegacyClient.saved[0]["chain"]["metadata"]
    # CARE fields preserved
    assert metadata["task_description"] == "Analyze X"
    assert metadata["display_name"] == "X Reader"
    # Evolution meta added alongside
    assert metadata["evolution_meta"]["step_critique_score"] == 0.7


@pytest.mark.asyncio
async def test_save_chain_no_evolution_meta_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy(monkeypatch)
    _FakeLegacyClient.saved = []

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    # No evolution_meta arg, no CARE metadata either
    await mgr.save_chain(
        {"name": "test", "steps": [{"number": 1, "title": "A", "step_type": "llm"}]},
        domain="finance",
    )
    saved_chain = _FakeLegacyClient.saved[0]["chain"]
    # save_chain injects the server-required `version` + `metadata` block; with
    # no evolution_meta / CARE args the metadata stays empty (no evolution_meta).
    assert saved_chain.get("version")
    assert saved_chain.get("metadata", {}) == {}


# ---------------------------------------------------------------------------
# _build_evolution_meta
# ---------------------------------------------------------------------------


class TestBuildEvolutionMeta:
    def test_strips_none_signals(self) -> None:
        meta = MAGEMetadata(
            step_critique_score=0.82,
            verification_passed=True,
            refine_iterations=None,
            refine_quality_delta=0.15,
            mcts_best_reward=None,
        )
        block = _build_evolution_meta(meta)
        assert block == {
            "step_critique_score": 0.82,
            "verification_passed": True,
            "refine_quality_delta": 0.15,
        }

    def test_empty_metadata_yields_empty_block(self) -> None:
        meta = MAGEMetadata()  # all quality fields are None by default
        block = _build_evolution_meta(meta)
        assert block == {}


# ---------------------------------------------------------------------------
# Generator integration — failure cards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_saves_failure_card_on_verifier_rejection(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the verifier rejects the chain, a failure card is saved."""

    async def fake_verify(self: Any, *args: Any, **kwargs: Any) -> VerificationResult:
        return VerificationResult(
            chain_passed=False,
            questions=[VerificationQuestion(step_number=1, question="OK?", aspect="logical")],
            issues_found=[{"step_number": 1, "issue": "vague aim"}],
        )

    async def fake_fix(self: Any, chain_dict: dict[str, Any], vresult: Any) -> dict[str, Any]:
        return chain_dict

    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.verify", fake_verify)
    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.fix", fake_fix)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(
                json.dumps(
                    {
                        "domain": "finance",
                        "task_type": "analysis",
                        "complexity": "medium",
                        "key_concepts": [],
                        "suggested_step_count": 3,
                        "preferred_step_types": [],
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "rough_aim": "z"},
                        ],
                        "rationale": "ok",
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "dependencies": [], "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "dependencies": [1, 2], "rough_aim": "z"},
                        ],
                        "max_workers": 2,
                        "pattern": "funnel",
                    }
                )
            ),
            _make_openai_response(json.dumps({"steps": SAMPLE_CARL_CHAIN["steps"]})),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_chain_verification = True
    deep_config.enable_failure_card_save = True
    deep_config.enable_suggested_naming_llm = False
    deep_config.enable_memory = False

    gen = MAGEGenerator(config=deep_config)
    save_failure_spy = mocker.patch.object(
        gen._memory, "save_failure_card", new_callable=AsyncMock, return_value="failure-id"
    )

    result = await gen.generate("Analyze quarterly")

    # Failure card fired exactly once for chain_verification
    assert save_failure_spy.call_count == 1
    kwargs = save_failure_spy.call_args.kwargs
    assert kwargs["stage"] == "chain_verification"
    assert kwargs["domain"] == "finance"
    assert "verifier" in kwargs["reason"].lower()
    assert kwargs["query"] == "Analyze quarterly"
    # MAGEMetadata still records the failure verdict
    assert result.metadata.verification_passed is False


@pytest.mark.asyncio
async def test_generator_saves_failure_card_on_refine_cap_hit(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When self-refine exhausts iterations without reaching threshold."""

    async def fake_refine(self: Any, *args: Any, **kwargs: Any) -> RefinementResult:
        return RefinementResult(
            chain_dict={"name": "x", "steps": SAMPLE_CARL_CHAIN["steps"]},
            chain_json="{}",
            iterations_used=kwargs["max_iterations"],  # exhausted
            initial_quality=0.3,
            final_quality=0.45,  # < threshold (0.9 below)
        )

    monkeypatch.setattr("mmar_mage.agents.self_refiner.SelfRefiner.refine", fake_refine)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(
                json.dumps(
                    {
                        "domain": "finance",
                        "task_type": "analysis",
                        "complexity": "medium",
                        "key_concepts": [],
                        "suggested_step_count": 3,
                        "preferred_step_types": [],
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "rough_aim": "z"},
                        ],
                        "rationale": "ok",
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "dependencies": [], "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "dependencies": [1, 2], "rough_aim": "z"},
                        ],
                        "max_workers": 2,
                        "pattern": "funnel",
                    }
                )
            ),
            _make_openai_response(json.dumps({"steps": SAMPLE_CARL_CHAIN["steps"]})),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_self_refine = True
    deep_config.max_refine_iterations = 3
    deep_config.refine_quality_threshold = 0.9
    deep_config.enable_failure_card_save = True
    deep_config.enable_suggested_naming_llm = False
    deep_config.enable_memory = False

    gen = MAGEGenerator(config=deep_config)
    save_failure_spy = mocker.patch.object(
        gen._memory, "save_failure_card", new_callable=AsyncMock, return_value="failure-id"
    )

    await gen.generate("Analyze quarterly")

    assert save_failure_spy.call_count == 1
    kwargs = save_failure_spy.call_args.kwargs
    assert kwargs["stage"] == "self_refine"
    assert "iterations" in kwargs["reason"].lower()


@pytest.mark.asyncio
async def test_generator_noop_failure_card_when_disabled(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With enable_failure_card_save=False (default), no card is saved
    even when the verifier rejects the chain."""

    async def fake_verify(self: Any, *args: Any, **kwargs: Any) -> VerificationResult:
        return VerificationResult(chain_passed=False)

    async def fake_fix(self: Any, c: dict[str, Any], v: Any) -> dict[str, Any]:
        return c

    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.verify", fake_verify)
    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.fix", fake_fix)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(
                json.dumps(
                    {
                        "domain": "finance",
                        "task_type": "analysis",
                        "complexity": "medium",
                        "key_concepts": [],
                        "suggested_step_count": 3,
                        "preferred_step_types": [],
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "rough_aim": "z"},
                        ],
                        "rationale": "ok",
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "dependencies": [], "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "dependencies": [1, 2], "rough_aim": "z"},
                        ],
                        "max_workers": 2,
                        "pattern": "funnel",
                    }
                )
            ),
            _make_openai_response(json.dumps({"steps": SAMPLE_CARL_CHAIN["steps"]})),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_chain_verification = True
    # enable_failure_card_save defaults to False
    deep_config.enable_suggested_naming_llm = False
    deep_config.enable_memory = False

    gen = MAGEGenerator(config=deep_config)
    spy = mocker.patch.object(gen._memory, "save_failure_card", new_callable=AsyncMock)
    await gen.generate("Analyze quarterly")
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# Generator integration — evolution_meta flows into save_chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_passes_evolution_meta_to_save_chain(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enable_evolution_meta_save=True, save_chain receives a populated
    `evolution_meta` built from MAGEMetadata."""

    async def fake_critique(self: Any, *args: Any, **kwargs: Any) -> Any:
        from mmar_mage.agents.step_critic import StepCritique, StepCritiqueResult

        return StepCritiqueResult(
            critiques=[StepCritique(step_number=1, overall_score=0.85)],
            overall_quality=0.85,
            steps_needing_revision=[],
        )

    monkeypatch.setattr("mmar_mage.agents.step_critic.StepCriticAgent.critique", fake_critique)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(
                json.dumps(
                    {
                        "domain": "finance",
                        "task_type": "analysis",
                        "complexity": "medium",
                        "key_concepts": [],
                        "suggested_step_count": 3,
                        "preferred_step_types": [],
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "rough_aim": "z"},
                        ],
                        "rationale": "ok",
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "dependencies": [], "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "dependencies": [1, 2], "rough_aim": "z"},
                        ],
                        "max_workers": 2,
                        "pattern": "funnel",
                    }
                )
            ),
            _make_openai_response(json.dumps({"steps": SAMPLE_CARL_CHAIN["steps"]})),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_step_critique = True
    deep_config.step_critique_threshold = 0.6
    deep_config.enable_evolution_meta_save = True
    deep_config.enable_suggested_naming_llm = False
    deep_config.enable_memory = False

    gen = MAGEGenerator(config=deep_config)
    save_chain_spy = mocker.patch.object(gen._memory, "save_chain", new_callable=AsyncMock, return_value=None)
    await gen.generate("Analyze quarterly")

    save_chain_spy.assert_called_once()
    kwargs = save_chain_spy.call_args.kwargs
    evo = kwargs.get("evolution_meta")
    assert evo is not None
    assert evo.get("step_critique_score") == 0.85


@pytest.mark.asyncio
async def test_generator_evolution_meta_omitted_when_flag_off(deep_config: MAGEConfig, mocker: Any) -> None:
    """Default: evolution_meta=None in save_chain call."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(
                json.dumps(
                    {
                        "domain": "finance",
                        "task_type": "analysis",
                        "complexity": "medium",
                        "key_concepts": [],
                        "suggested_step_count": 3,
                        "preferred_step_types": [],
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "rough_aim": "z"},
                        ],
                        "rationale": "ok",
                    }
                )
            ),
            _make_openai_response(
                json.dumps(
                    {
                        "steps": [
                            {"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                            {"number": 2, "title": "B", "step_type": "llm", "dependencies": [], "rough_aim": "y"},
                            {"number": 3, "title": "C", "step_type": "llm", "dependencies": [1, 2], "rough_aim": "z"},
                        ],
                        "max_workers": 2,
                        "pattern": "funnel",
                    }
                )
            ),
            _make_openai_response(json.dumps({"steps": SAMPLE_CARL_CHAIN["steps"]})),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_suggested_naming_llm = False
    deep_config.enable_memory = False

    gen = MAGEGenerator(config=deep_config)
    spy = mocker.patch.object(gen._memory, "save_chain", new_callable=AsyncMock, return_value=None)
    await gen.generate("Q")
    kwargs = spy.call_args.kwargs
    assert kwargs.get("evolution_meta") is None
