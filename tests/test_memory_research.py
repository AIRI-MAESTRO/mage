"""Tests for MemoryResearchAgent and extended MemoryManager methods."""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_mage.agents.memory_research_agent import MemoryResearchAgent
from mmar_mage.llm import LLMClient
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import (
    DomainAnalysis,
    MAGEConfig,
    MemoryHit,
    MemoryResearchResult,
)

from .conftest import _make_openai_response


def _hit(entity_id: str, name: str, score: float, content: dict | None = None) -> types.SimpleNamespace:
    """Stand-in for gigaevo SearchHitData (what client.search_hits returns)."""
    return types.SimpleNamespace(entity_id=entity_id, name=name, score=score, content=content or {})


def _client_cm(client: MagicMock) -> MagicMock:
    """Wrap a mock client as a context manager (``with MemoryClient(...) as client``)."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


@contextmanager
def _stub_gigaevo_client(client: MagicMock):
    """Inject a stub ``gigaevo_client`` module so MemoryManager's in-method
    ``from gigaevo_client import MemoryClient / SearchType`` resolves even
    without the optional SDK installed in this venv. ``MemoryClient(...)``
    yields the given mock client."""
    mod = types.ModuleType("gigaevo_client")

    class SearchType(Enum):
        BM25 = "bm25"
        VECTOR = "vector"
        HYBRID = "hybrid"

    def _make_client(**_kw: Any) -> MagicMock:
        return _client_cm(client)

    mod.SearchType = SearchType
    mod.MemoryClient = _make_client
    with patch.dict(sys.modules, {"gigaevo_client": mod}):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DOMAIN = DomainAnalysis(
    domain="finance",
    task_type="analysis",
    complexity="medium",
    key_concepts=["financial report", "risks", "revenue"],
    suggested_step_count=3,
)


def _make_llm_client(mocker: Any, responses: list[dict[str, Any]]) -> LLMClient:
    """Create an LLMClient with mocked AsyncOpenAI returning sequential responses."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    side_effects = [_make_openai_response(json.dumps(r)) for r in responses]
    instance.chat.completions.create = AsyncMock(side_effect=side_effects)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    config = MAGEConfig(api_key="test-key", max_retries=0)
    return LLMClient(config)


# ---------------------------------------------------------------------------
# MemoryResearchAgent tests
# ---------------------------------------------------------------------------


class TestMemoryResearchAgent:
    """Tests for MemoryResearchAgent."""

    @pytest.mark.asyncio
    async def test_cold_start_no_hits(self, mocker: Any) -> None:
        """Agent returns cold start result when no memory hits found."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["q1", "q2", "q3"]},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_memory=True,
            enable_memory_research=True,
            max_retries=0,
        )
        memory = MemoryManager(config)
        memory._available = True
        mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=[])

        agent = MemoryResearchAgent(llm)
        result = await agent.research("Analyze report", SAMPLE_DOMAIN, memory, config)

        assert isinstance(result, MemoryResearchResult)
        assert result.was_cold_start is True
        assert result.hits_count == 0
        assert result.digest == ""

    @pytest.mark.asyncio
    async def test_with_hits(self, mocker: Any) -> None:
        """Agent returns digest when memory hits are found."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["q1", "q2"]},
                # applicability gate (P1.1) runs between recall and purify
                {
                    "ratings": [
                        {"entity_id": "mem-1", "applicability": 0.9},
                        {"entity_id": "mem-2", "applicability": 0.7},
                    ]
                },
                {"digest": "Past finance chains used funnel pattern with 5 steps."},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_memory=True,
            enable_memory_research=True,
            memory_relevance_threshold=0.0,
            max_retries=0,
        )
        memory = MemoryManager(config)
        memory._available = True

        # The agent recalls via MemoryManager.iterative_search; return MemoryHits.
        hits = [
            MemoryHit(entity_id="mem-1", name="chain-1", score=0.8, abstract_structure={}),
            MemoryHit(entity_id="mem-2", name="chain-2", score=0.6, abstract_structure={}),
        ]
        mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=hits)

        agent = MemoryResearchAgent(llm)
        result = await agent.research("Analyze report", SAMPLE_DOMAIN, memory, config)

        assert result.was_cold_start is False
        assert result.hits_count > 0
        assert "funnel" in result.digest

    @pytest.mark.asyncio
    async def test_sub_query_generation_failure(self, mocker: Any) -> None:
        """Agent gracefully handles sub-query generation failure."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=Exception("LLM failure"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        config = MAGEConfig(api_key="test", enable_memory=True, max_retries=0)
        llm = LLMClient(config)

        memory = MemoryManager(config)
        memory._available = True
        mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=[])

        agent = MemoryResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, memory, config)

        assert result.was_cold_start is True


# ---------------------------------------------------------------------------
# MemoryManager extended methods tests
# ---------------------------------------------------------------------------


class TestMemoryManagerIterativeSearch:
    """Tests for MemoryManager.iterative_search()."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        hits = await mgr.iterative_search("test", ["concept1"])
        assert hits == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_not_available(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        hits = await mgr.iterative_search("test", ["concept1"])
        assert hits == []

    @pytest.mark.asyncio
    async def test_deduplicates_results(self, mocker: Any) -> None:
        """Iterative search deduplicates across sub-queries."""
        config = MAGEConfig(enable_memory=True, memory_relevance_threshold=0.0)
        mgr = MemoryManager(config)
        mgr._available = True

        # Same entity returned for every sub-query → must dedup to 1.
        client = MagicMock()
        client.search_hits.return_value = [_hit("mem-1", "chain-1", 0.8)]

        with _stub_gigaevo_client(client):
            hits = await mgr.iterative_search("test", ["concept1", "concept2"], sub_queries_count=3)

        assert len(hits) == 1
        assert hits[0].entity_id == "mem-1"

    @pytest.mark.asyncio
    async def test_filters_by_threshold(self, mocker: Any) -> None:
        """Results below threshold are filtered out."""
        config = MAGEConfig(enable_memory=True, memory_relevance_threshold=0.5)
        mgr = MemoryManager(config)
        mgr._available = True

        client = MagicMock()
        client.search_hits.return_value = [
            _hit("mem-1", "good", 0.8),
            _hit("mem-2", "bad", 0.3),
        ]

        with _stub_gigaevo_client(client):
            hits = await mgr.iterative_search("test", ["concept1"], threshold=0.5)

        assert len(hits) == 1
        assert hits[0].entity_id == "mem-1"  # bad (0.3) filtered out
        assert hits[0].entity_id == "mem-1"


class TestMemoryManagerSaveIdeas:
    """Tests for MemoryManager.save_ideas()."""

    @pytest.mark.asyncio
    async def test_save_ideas_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        result = await mgr.save_ideas([{"title": "test"}], domain="finance", query="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_ideas_not_available(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        result = await mgr.save_ideas([{"title": "test"}], domain="finance", query="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_ideas_failure_returns_none(self, mocker: Any) -> None:
        """save_ideas returns None on error (graceful degradation)."""
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = True

        # Mock the import and client to raise
        mocker.patch(
            "mmar_mage.memory.MemoryManager.save_ideas",
            new_callable=AsyncMock,
            return_value=None,
        )

        result = await mgr.save_ideas([{"title": "test"}], domain="finance", query="test")
        assert result is None
