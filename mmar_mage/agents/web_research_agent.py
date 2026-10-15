"""
WebResearchAgent — stage 0b of the deep-mode pipeline (optional).

Searches the internet for domain context:
1. Generates web search sub-queries from domain analysis
2. Calls configured search provider (Tavily / SerpAPI / Brave)
3. Runs a summarizer LLM call → returns compressed "web digest" (max 300 words)
4. Returns WebResearchResult
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ..llm import LLMClient
from ..prompts import WEB_DIGEST_PROMPT, WEB_GAP_SUBQUERY_PROMPT, WEB_SUB_QUERY_GENERATOR_PROMPT
from ..schemas import DomainAnalysis, MAGEConfig, WebResearchResult

logger = logging.getLogger(__name__)

STAGE = "web_research"

# Source-citation markers the digest prompt emits, e.g. "...metric [1][3]".
_SOURCE_MARKER_RE = re.compile(r"\[(\d+)\]")


def _count_sourced_claims(digest: str) -> int:
    """Count distinct numbered-source citations in a digest (P1.2 grounding proxy)."""
    return len(set(_SOURCE_MARKER_RE.findall(digest or "")))


class WebResearchAgent:
    """Search the internet for domain context.

    Parameters
    ----------
    llm : LLMClient
        LLM client for sub-query generation and digest summarization.

    Examples
    --------
    >>> agent = WebResearchAgent(llm_client)
    >>> result = await agent.research(query, domain_analysis, config)
    >>> result.digest
    'Financial risk analysis typically uses...'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def research(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        config: MAGEConfig,
    ) -> WebResearchResult:
        """Run web research.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain_analysis : DomainAnalysis
            Output from the DomainAnalyzer.
        config : MAGEConfig
            MAGE configuration.

        Returns
        -------
        WebResearchResult
            Digest, sources, and metadata from web research.
        """
        logger.info("WebResearchAgent: starting web research")

        if not config.web_search_api_key:
            logger.warning("WebResearchAgent: no web search API key configured")
            return WebResearchResult()

        # Generate sub-queries
        try:
            sub_queries = await self._generate_sub_queries(query, domain_analysis, config.web_research_sub_queries)
        except Exception as exc:
            logger.warning("WebResearchAgent: sub-query generation failed: %s", exc)
            sub_queries = [query]

        # Search web
        all_results: list[dict[str, Any]] = []
        sources: list[str] = []

        for sq in sub_queries:
            try:
                results = await self._search_web(
                    sq,
                    provider=config.web_search_provider,
                    api_key=config.web_search_api_key,
                    max_results=config.web_research_max_results,
                )
                all_results.extend(results)
                for r in results:
                    url = r.get("url", "")
                    if url and url not in sources:
                        sources.append(url)
            except Exception as exc:
                logger.warning("WebResearchAgent: search failed for '%s': %s", sq, exc)
                continue

        if not all_results:
            logger.info("WebResearchAgent: no web results found")
            return WebResearchResult(sub_queries_used=sub_queries)

        # Summarize into digest
        try:
            digest = await self._create_digest(query, all_results)
        except Exception as exc:
            logger.warning("WebResearchAgent: digest creation failed: %s", exc)
            digest = ""

        # Source-grounded second wave (P1.2): if the digest cites too few
        # distinct sources, search gap-targeting queries and re-digest the
        # combined results. Non-fatal.
        n_min = config.web_min_sourced_claims
        if digest and n_min > 0 and _count_sourced_claims(digest) < n_min:
            try:
                digest = await self._second_wave(
                    query, domain_analysis, config, digest, sub_queries, all_results, sources
                )
            except Exception as exc:
                logger.warning("WebResearchAgent: second wave failed: %s", exc)

        logger.info("WebResearchAgent: found %d results from %d sources", len(all_results), len(sources))

        return WebResearchResult(
            digest=digest,
            sources=sources,
            sub_queries_used=sub_queries,
            was_used=bool(digest),
        )

    async def _second_wave(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        config: MAGEConfig,
        digest: str,
        sub_queries: list[str],
        all_results: list[dict[str, Any]],
        sources: list[str],
    ) -> str:
        """Search gap-targeting queries and re-digest the combined results (P1.2).

        Mutates ``all_results`` / ``sources`` in place (so the caller's source
        list reflects the second wave) and returns the regenerated digest, or the
        original digest if no new results were found.
        """
        gap_queries = await self._generate_gap_queries(query, domain_analysis, digest, config.web_research_sub_queries)
        gap_queries = [q for q in gap_queries if q not in sub_queries]
        if not gap_queries:
            return digest

        before = len(all_results)
        for gq in gap_queries:
            try:
                results = await self._search_web(
                    gq,
                    provider=config.web_search_provider,
                    api_key=config.web_search_api_key or "",
                    max_results=config.web_research_max_results,
                )
            except Exception as exc:
                logger.warning("WebResearchAgent: second-wave search failed for '%s': %s", gq, exc)
                continue
            all_results.extend(results)
            for r in results:
                url = r.get("url", "")
                if url and url not in sources:
                    sources.append(url)

        if len(all_results) == before:
            return digest  # no new evidence — keep the first-wave digest
        return await self._create_digest(query, all_results)

    async def _generate_gap_queries(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        digest: str,
        count: int,
    ) -> list[str]:
        """Generate gap-targeting web queries for the second wave (P1.2)."""
        context = (
            f"User query: {query}\n\n"
            f"Domain: {domain_analysis.domain}; task_type: {domain_analysis.task_type}\n\n"
            f"Partial digest so far (aim at its gaps):\n{digest}\n\n"
            f"Generate up to {count} gap-targeting web search queries."
        )
        data = await self._llm.generate_json(
            system_prompt=WEB_GAP_SUBQUERY_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        subs = data.get("sub_queries", [])
        return subs[:count] if isinstance(subs, list) else []

    async def _generate_sub_queries(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        count: int,
    ) -> list[str]:
        """Generate web search sub-queries from domain analysis."""
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Generate exactly {count} web search queries."
        )

        data = await self._llm.generate_json(
            system_prompt=WEB_SUB_QUERY_GENERATOR_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        return data.get("sub_queries", [query])[:count]

    async def _search_web(
        self,
        query: str,
        *,
        provider: str,
        api_key: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Call the configured web search provider."""
        if provider == "tavily":
            return await self._search_tavily(query, api_key, max_results)
        elif provider == "serpapi":
            return await self._search_serpapi(query, api_key, max_results)
        elif provider == "brave":
            return await self._search_brave(query, api_key, max_results)
        else:
            logger.warning("WebResearchAgent: unknown provider '%s'", provider)
            return []

    async def _search_tavily(self, query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
        """Search via Tavily API."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in data.get("results", [])
            ]

    async def _search_serpapi(self, query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
        """Search via SerpAPI."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.get(
                "https://serpapi.com/search",
                params={
                    "api_key": api_key,
                    "q": query,
                    "num": max_results,
                    "engine": "google",
                },
            )
            response.raise_for_status()
            data = response.json()
            return [
                {"title": r.get("title", ""), "url": r.get("link", ""), "content": r.get("snippet", "")}
                for r in data.get("organic_results", [])
            ]

    async def _search_brave(self, query: str, api_key: str, max_results: int) -> list[dict[str, Any]]:
        """Search via Brave Search API."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("description", ""),
                }
                for r in data.get("web", {}).get("results", [])
            ]

    async def _create_digest(self, query: str, results: list[dict[str, Any]]) -> str:
        """Compress raw web results into a source-grounded digest (P1.2).

        Results are presented as a numbered list so the digest prompt can cite
        each claim with a ``[N]`` marker (and omit anything it can't attribute).
        """
        numbered = "\n".join(
            f"[{i + 1}] {r.get('title', '')} ({r.get('url', '')}): {str(r.get('content', ''))[:500]}"
            for i, r in enumerate(results[:20])
        )

        context = f"User query: {query}\n\nNumbered web sources:\n{numbered}"

        data = await self._llm.generate_json(
            system_prompt=WEB_DIGEST_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        return data.get("digest", "")
