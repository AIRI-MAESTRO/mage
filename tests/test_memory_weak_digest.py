"""T32 — memory research surfaces weak context instead of an empty digest."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.agents.memory_research_agent import MemoryResearchAgent
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import DomainAnalysis, MAGEConfig


def _hit(eid: str, name: str) -> SimpleNamespace:
    return SimpleNamespace(entity_id=eid, name=name, abstract_structure={})


def _agent(mocker: Any) -> MemoryResearchAgent:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    return MemoryResearchAgent(LLMClient(MAGEConfig(api_key="test")))


def _cfg() -> MAGEConfig:
    return MAGEConfig(api_key="test", enable_memory_applicability_gate=True, memory_min_applicable_hits=0)


def test_weak_digest_lists_titles() -> None:
    d = MemoryResearchAgent._weak_digest([_hit("e1", "Weather API tool"), _hit("e2", "IP geolocation")])
    assert "Weather API tool" in d and "IP geolocation" in d


def test_weak_digest_empty_when_no_titles() -> None:
    assert MemoryResearchAgent._weak_digest([_hit("e1", ""), _hit("e2", "  ")]) == ""
    assert MemoryResearchAgent._weak_digest([]) == ""


async def test_gate_drops_all_returns_weak_digest(mocker: Any) -> None:
    agent = _agent(mocker)
    agent._generate_sub_queries = AsyncMock(return_value=["q"])  # type: ignore[method-assign]
    agent._filter_applicable = AsyncMock(return_value=[])  # type: ignore[method-assign] — gate rejects all
    memory = mocker.MagicMock()
    memory.iterative_search = AsyncMock(return_value=[_hit("e1", "Weather API tool"), _hit("e2", "IP geolocation")])

    result = await agent.research("weather where my ip is", DomainAnalysis(domain="general"), memory, _cfg())

    assert result.digest  # NOT empty — the run-096 bug was digest=""
    assert "Weather API tool" in result.digest
    assert result.hits_count == 2
    assert result.was_cold_start is False  # memory wasn't empty, so not a cold start


async def test_purify_empty_falls_back_to_titles(mocker: Any) -> None:
    agent = _agent(mocker)
    agent._generate_sub_queries = AsyncMock(return_value=["q"])  # type: ignore[method-assign]
    hits = [_hit("e1", "Weather API tool"), _hit("e2", "IP geolocation")]
    agent._filter_applicable = AsyncMock(return_value=hits)  # type: ignore[method-assign] — gate keeps them
    agent._purify_hits = AsyncMock(return_value="   ")  # type: ignore[method-assign] — purifier yields nothing
    memory = mocker.MagicMock()
    memory.iterative_search = AsyncMock(return_value=hits)

    result = await agent.research("q", DomainAnalysis(domain="general"), memory, _cfg())

    assert "Weather API tool" in result.digest  # fell back to titles, not ""
    assert result.hits_count == 2


async def test_true_cold_start_still_empty(mocker: Any) -> None:
    agent = _agent(mocker)
    agent._generate_sub_queries = AsyncMock(return_value=["q"])  # type: ignore[method-assign]
    memory = mocker.MagicMock()
    memory.iterative_search = AsyncMock(return_value=[])  # memory genuinely empty

    result = await agent.research("q", DomainAnalysis(domain="general"), memory, _cfg())
    assert result.digest == ""
    assert result.hits_count == 0
    assert result.was_cold_start is True
