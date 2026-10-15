"""T33 — session run-facts: save + recall + memory-research prioritization."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.agents.memory_research_agent import MemoryResearchAgent
from mmar_mage.llm import LLMClient
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import DomainAnalysis, MAGEConfig


class _FakeClient:
    def __init__(self, **kw: Any) -> None: ...
    def __enter__(self) -> _FakeClient:
        return self
    def __exit__(self, *a: Any) -> bool:
        return False


# --- MemoryManager.save_run_facts / recall_run_facts ---


async def test_save_run_facts_noop_without_session() -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_memory=True))  # no memory_session_id
    mgr._available = True
    assert await mgr.save_run_facts("q", {"ip": "1.2.3.4"}) is None


async def test_save_run_facts_noop_when_no_facts() -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_memory=True, memory_session_id="s1"))
    mgr._available = True
    assert await mgr.save_run_facts("q", {}) is None


async def test_save_run_facts_builds_session_card(mocker: Any) -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_memory=True, memory_session_id="s1"))
    mgr._available = True
    mocker.patch("mmar_mage.memory._memory_sdk", return_value=SimpleNamespace(MemoryClient=_FakeClient))
    captured: dict[str, Any] = {}

    def _capture(client: Any, content: dict[str, Any], **kw: Any) -> str:
        captured.update(content)
        captured["_tags"] = kw.get("tags")
        return "id-1"

    mocker.patch.object(mgr, "_save_card", side_effect=_capture)
    eid = await mgr.save_run_facts("weather where my ip", {"ip": "185.x", "country": "Sweden"}, ttl_seconds=600)
    assert eid == "id-1"
    assert captured["kind"] == "run_facts"
    assert captured["session_id"] == "s1"
    assert captured["facts"] == {"ip": "185.x", "country": "Sweden"}
    assert "s1" in captured["_tags"] and "run_facts" in captured["_tags"]
    assert captured["expires_at"] > 0


async def test_recall_run_facts_filters_and_formats(mocker: Any) -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_memory=True, memory_session_id="s1"))
    mgr._available = True
    mocker.patch("mmar_mage.memory._memory_sdk", return_value=SimpleNamespace(MemoryClient=_FakeClient))
    hits = [
        SimpleNamespace(content={"kind": "run_facts", "session_id": "s1", "question": "my ip?",
                                 "facts": {"ip": "185.x"}, "expires_at": 9_999_999_999.0}),
        SimpleNamespace(content={"kind": "run_facts", "session_id": "OTHER", "facts": {"x": 1},
                                 "expires_at": 9_999_999_999.0}),  # other session → dropped
        SimpleNamespace(content={"kind": "run_facts", "session_id": "s1", "facts": {"y": 2},
                                 "expires_at": 1.0}),  # expired → dropped
        SimpleNamespace(content={"kind": "mage_template", "session_id": "s1"}),  # wrong kind → dropped
    ]
    mocker.patch.object(mgr, "_search_cards", return_value=hits)
    out = await mgr.recall_run_facts()
    assert "185.x" in out and "my ip?" in out
    assert '"x": 1' not in out and '"y": 2' not in out


async def test_recall_run_facts_noop_without_session() -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_memory=True))
    mgr._available = True
    assert await mgr.recall_run_facts() == ""


# --- memory_research prioritization (the T33 criterion) ---


def _agent(mocker: Any) -> MemoryResearchAgent:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    return MemoryResearchAgent(LLMClient(MAGEConfig(api_key="test")))


async def test_session_facts_prevent_cold_start(mocker: Any) -> None:
    agent = _agent(mocker)
    agent._generate_sub_queries = AsyncMock(return_value=["q"])  # type: ignore[method-assign]
    memory = mocker.MagicMock()
    memory.iterative_search = AsyncMock(return_value=[])  # empty recall — would be a cold start
    memory.recall_run_facts = AsyncMock(return_value="Facts: my ip → 185.x")
    cfg = MAGEConfig(api_key="test", memory_session_id="s1")

    result = await agent.research("what's my ip again", DomainAnalysis(domain="general"), memory, cfg)
    assert "185.x" in result.digest
    assert result.was_cold_start is False  # session facts → not a cold start


async def test_session_facts_lead_the_digest(mocker: Any) -> None:
    agent = _agent(mocker)
    hit = SimpleNamespace(entity_id="e1", name="X", abstract_structure={})
    agent._generate_sub_queries = AsyncMock(return_value=["q"])  # type: ignore[method-assign]
    agent._filter_applicable = AsyncMock(return_value=[hit])  # type: ignore[method-assign]
    agent._purify_hits = AsyncMock(return_value="recalled memory digest")  # type: ignore[method-assign]
    memory = mocker.MagicMock()
    memory.iterative_search = AsyncMock(return_value=[hit])
    memory.recall_run_facts = AsyncMock(return_value="SESSION_FACTS_BLOCK")
    cfg = MAGEConfig(
        api_key="test", memory_session_id="s1", enable_memory_applicability_gate=True, memory_min_applicable_hits=0
    )

    result = await agent.research("q", DomainAnalysis(domain="general"), memory, cfg)
    assert result.digest.startswith("SESSION_FACTS_BLOCK")
    assert "recalled memory digest" in result.digest
