"""Tests for WebResearchAgent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from mmar_mage.agents.web_research_agent import WebResearchAgent
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import (
    DomainAnalysis,
    MAGEConfig,
    WebResearchResult,
)

from .conftest import _make_openai_response

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
# WebResearchAgent tests
# ---------------------------------------------------------------------------


class TestWebResearchAgent:
    """Tests for WebResearchAgent."""

    @pytest.mark.asyncio
    async def test_no_api_key(self, mocker: Any) -> None:
        """Agent returns empty result when no API key configured."""
        llm = _make_llm_client(mocker, [])

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key=None,
            max_retries=0,
        )

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        assert isinstance(result, WebResearchResult)
        assert result.digest == ""
        assert result.was_used is False

    @pytest.mark.asyncio
    async def test_tavily_search(self, mocker: Any) -> None:
        """Agent calls Tavily API and produces a digest."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["finance risk analysis"]},
                {"digest": "Financial risk analysis involves VaR calculations."},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="tavily-key",
            web_search_provider="tavily",
            web_research_sub_queries=1,
            web_min_sourced_claims=0,  # this test checks search plumbing, not the P1.2 second wave
            max_retries=0,
        )

        # Mock httpx
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Risk Analysis", "url": "https://example.com/risk", "content": "VaR is..."},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("mmar_mage.agents.web_research_agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebResearchAgent(llm)
        result = await agent.research("Analyze risks", SAMPLE_DOMAIN, config)

        assert result.was_used is True
        assert "VaR" in result.digest
        assert len(result.sources) >= 1

    @pytest.mark.asyncio
    async def test_serpapi_search(self, mocker: Any) -> None:
        """Agent calls SerpAPI and produces a digest."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["financial risk serpapi"]},
                {"digest": "Financial analysis best practices."},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="serp-key",
            web_search_provider="serpapi",
            web_research_sub_queries=1,
            web_min_sourced_claims=0,  # this test checks search plumbing, not the P1.2 second wave
            max_retries=0,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [
                {"title": "Analysis", "link": "https://example.com/analysis", "snippet": "Best practices..."},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        mocker.patch("mmar_mage.agents.web_research_agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        assert result.was_used is True

    @pytest.mark.asyncio
    async def test_brave_search(self, mocker: Any) -> None:
        """Agent calls Brave Search API and produces a digest."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["brave search test"]},
                {"digest": "Brave results digest."},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="brave-key",
            web_search_provider="brave",
            web_research_sub_queries=1,
            web_min_sourced_claims=0,  # this test checks search plumbing, not the P1.2 second wave
            max_retries=0,
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Result", "url": "https://brave.com/result", "description": "Description..."},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        mocker.patch("mmar_mage.agents.web_research_agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        assert result.was_used is True

    @pytest.mark.asyncio
    async def test_unknown_provider(self, mocker: Any) -> None:
        """Agent handles unknown search provider gracefully."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["test query"]},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="key",
            web_search_provider="unknown_provider",
            web_research_sub_queries=1,
            max_retries=0,
        )

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        assert result.digest == ""
        assert result.was_used is False

    @pytest.mark.asyncio
    async def test_search_failure_graceful(self, mocker: Any) -> None:
        """Agent handles search API failure gracefully."""
        llm = _make_llm_client(
            mocker,
            [
                {"sub_queries": ["test query"]},
            ],
        )

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="key",
            web_search_provider="tavily",
            web_research_sub_queries=1,
            max_retries=0,
        )

        # Mock httpx to raise
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

        mocker.patch("mmar_mage.agents.web_research_agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        assert result.digest == ""
        assert result.was_used is False

    @pytest.mark.asyncio
    async def test_sub_query_generation_failure(self, mocker: Any) -> None:
        """Agent falls back to original query on sub-query generation failure."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=Exception("LLM failure"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        config = MAGEConfig(
            api_key="test",
            enable_web_research=True,
            web_search_api_key="key",
            web_search_provider="tavily",
            web_research_sub_queries=1,
            max_retries=0,
        )
        llm = LLMClient(config)

        # Mock tavily to also fail
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("fail"))

        mocker.patch("mmar_mage.agents.web_research_agent.httpx.AsyncClient", return_value=mock_client)

        agent = WebResearchAgent(llm)
        result = await agent.research("test", SAMPLE_DOMAIN, config)

        # Should still return a result (not crash)
        assert isinstance(result, WebResearchResult)
