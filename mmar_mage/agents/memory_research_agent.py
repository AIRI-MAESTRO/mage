"""
MemoryResearchAgent — stage 0a of the deep-mode pipeline.

Performs iterative multi-query recall from gigaevo-memory (CoRAG-style):
1. Uses domain concepts to generate N targeted sub-queries
2. Retrieves hits for each sub-query
3. Runs PurifierLLM call: takes all raw hits, returns a compressed
   "memory digest" (max 300 words) relevant to the current query
4. Returns MemoryResearchResult with digest + was_cold_start flag
"""

from __future__ import annotations

import json
import logging

from ..llm import LLMClient
from ..memory import MemoryManager
from ..prompts import (
    MEMORY_APPLICABILITY_PROMPT,
    MEMORY_GAP_SUBQUERY_PROMPT,
    MEMORY_PURIFIER_PROMPT,
    MEMORY_SUB_QUERY_GENERATOR_PROMPT,
)
from ..schemas import DomainAnalysis, MAGEConfig, MemoryResearchResult

logger = logging.getLogger(__name__)

STAGE = "memory_research"


def _summarize_structure(structure: dict | None) -> str:
    """Compact a recalled chain's structure into a short hint for the judge.

    Prefers the step-type sequence; falls back to step types pulled from a
    ``steps`` list, then to top-level keys. Keeps the judge prompt small.
    """
    if not isinstance(structure, dict) or not structure:
        return ""
    seq = structure.get("step_type_sequence")
    if isinstance(seq, list) and seq:
        return " -> ".join(str(s) for s in seq[:12])
    steps = structure.get("steps")
    if isinstance(steps, list) and steps:
        types = [str(s.get("step_type", "llm")) for s in steps if isinstance(s, dict)]
        if types:
            return " -> ".join(types[:12])
    return ", ".join(sorted(structure.keys())[:8])


class MemoryResearchAgent:
    """Perform iterative multi-query recall from memory.

    Parameters
    ----------
    llm : LLMClient
        LLM client for sub-query generation and digest purification.

    Examples
    --------
    >>> agent = MemoryResearchAgent(llm_client)
    >>> result = await agent.research(query, domain_analysis, memory, config)
    >>> result.digest
    'Past chains in finance domain used funnel patterns...'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def research(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        memory: MemoryManager,
        config: MAGEConfig,
    ) -> MemoryResearchResult:
        """Run memory research.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain_analysis : DomainAnalysis
            Output from the DomainAnalyzer.
        memory : MemoryManager
            Memory manager instance.
        config : MAGEConfig
            MAGE configuration.

        Returns
        -------
        MemoryResearchResult
            Digest and metadata from memory research.
        """
        logger.info("MemoryResearchAgent: starting memory research")

        # Generate sub-queries
        try:
            sub_queries = await self._generate_sub_queries(query, domain_analysis, config.memory_research_sub_queries)
        except Exception as exc:
            logger.warning("MemoryResearchAgent: sub-query generation failed: %s", exc)
            sub_queries = [query]

        # Iterative search
        hits = await memory.iterative_search(
            query=query,
            domain_concepts=domain_analysis.key_concepts,
            sub_queries_count=config.memory_research_sub_queries,
            top_k=config.memory_recall_top_k,
            threshold=config.memory_relevance_threshold,
        )

        raw_count = len(hits)
        raw_hits = list(hits)  # T32: keep the pre-gate hits for the weak-context fallback

        # T33: facts established earlier THIS session — recalled first, prepended
        # to the digest, and they alone keep a follow-up question from cold-starting.
        session_facts = ""
        if config.memory_session_id:
            try:
                session_facts = await memory.recall_run_facts()
            except Exception as exc:
                logger.warning("MemoryResearchAgent: session-fact recall failed: %s", exc)

        # Applicability gate (P1.1): drop hits the LLM judge deems inapplicable
        # to the current query, so superficial keyword matches don't pollute the
        # digest. Non-fatal — on failure we keep the unfiltered hits.
        if hits and config.enable_memory_applicability_gate:
            try:
                hits = await self._filter_applicable(query, hits, config.memory_applicability_threshold)
            except Exception as exc:
                logger.warning("MemoryResearchAgent: applicability gate failed (keeping hits): %s", exc)

        # Second wave (P1.1): wave 1 found content (raw_count > 0) but too few
        # applicable hits survived — recall again with gap-targeting sub-queries,
        # re-gate, and merge. Skipped on a true cold start (raw_count == 0, likely
        # empty memory) to avoid a wasted pass. Non-fatal.
        n_min = config.memory_min_applicable_hits
        if n_min > 0 and raw_count > 0 and len(hits) < n_min:
            try:
                hits = await self._second_wave(query, domain_analysis, memory, config, hits)
            except Exception as exc:
                logger.warning("MemoryResearchAgent: second-wave recall failed: %s", exc)

        if not hits:
            if session_facts or raw_hits:
                # T32/T33: no applicable recall, but memory wasn't empty (or we have
                # session facts) — surface session facts + hit titles as weak context
                # instead of an empty digest + false cold-start (run-096 "hits=4, ''").
                weak = self._weak_digest(raw_hits)
                combined = "\n\n".join(p for p in (session_facts, weak) if p)
                logger.info(
                    "MemoryResearchAgent: no applicable hits; session_facts=%s, weak=%d chars",
                    bool(session_facts), len(weak),
                )
                return MemoryResearchResult(
                    digest=combined,
                    hits_count=len(raw_hits),
                    was_cold_start=False,
                    sub_queries_used=sub_queries,
                )
            logger.info("MemoryResearchAgent: no relevant hits (cold start)")
            return MemoryResearchResult(
                digest="",
                hits_count=0,
                was_cold_start=True,
                sub_queries_used=sub_queries,
            )

        # Purify hits into digest
        try:
            digest = await self._purify_hits(query, hits)
        except Exception as exc:
            logger.warning("MemoryResearchAgent: purification failed: %s", exc)
            digest = ""

        # T32: a non-empty hit set that purifies to nothing must still yield SOME
        # context — fall back to the hit titles rather than reporting hits with an
        # empty digest (the run-096 symptom).
        if not digest.strip():
            digest = self._weak_digest(hits)
            if digest:
                logger.info("MemoryResearchAgent: purifier empty; fell back to %d hit titles", len(hits))

        # T33: session facts lead the digest (highest-priority context).
        if session_facts:
            digest = (session_facts + "\n\n" + digest).strip() if digest.strip() else session_facts

        logger.info("MemoryResearchAgent: found %d hits, cold_start=%s", len(hits), len(hits) == 0)

        return MemoryResearchResult(
            digest=digest,
            hits_count=len(hits),
            was_cold_start=False,
            sub_queries_used=sub_queries,
        )

    @staticmethod
    def _weak_digest(hits: list) -> str:
        """Low-confidence fallback context (T32): the recalled hit titles as a
        labelled list, so a non-empty recall never collapses to an empty digest
        even when the applicability gate or the purifier yields nothing."""
        titles: list[str] = []
        for h in hits:
            name = str(getattr(h, "name", "") or "").strip()
            if name and name not in titles:
                titles.append(name)
            if len(titles) >= 8:
                break
        if not titles:
            return ""
        return "Possibly-relevant memory items (low confidence, unverified): " + "; ".join(titles)

    async def _generate_sub_queries(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        count: int,
    ) -> list[str]:
        """Generate targeted sub-queries from domain analysis."""
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Generate exactly {count} sub-queries."
        )

        data = await self._llm.generate_json(
            system_prompt=MEMORY_SUB_QUERY_GENERATOR_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        return data.get("sub_queries", [query])[:count]

    async def _filter_applicable(self, query: str, hits: list, threshold: float) -> list:
        """Re-rank/gate recalled hits by LLM-judged applicability to the query (P1.1).

        Runs a single judge call that scores each hit 0-1; keeps only hits at or
        above ``threshold``, ordered by applicability (best first). Hits the judge
        omits are dropped (treated as score 0). Returns the original list
        unchanged if the judge returns nothing usable.
        """
        items = [
            {
                "entity_id": h.entity_id,
                "name": h.name,
                "structure": _summarize_structure(getattr(h, "abstract_structure", {})),
            }
            for h in hits
        ]
        context = f"User query: {query}\n\nRecalled items:\n{json.dumps(items, ensure_ascii=False, indent=2)}"

        data = await self._llm.generate_json(
            system_prompt=MEMORY_APPLICABILITY_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        ratings = data.get("ratings")
        if not isinstance(ratings, list) or not ratings:
            # Judge produced nothing usable — don't silently drop everything.
            return hits

        scores = {
            r.get("entity_id"): float(r.get("applicability", 0.0))
            for r in ratings
            if isinstance(r, dict) and r.get("entity_id")
        }
        scored = [(h, scores.get(h.entity_id, 0.0)) for h in hits]
        for h, s in scored:  # T32: log why each hit was kept / dropped
            logger.debug(
                "MemoryResearchAgent: hit %r applicability=%.2f (threshold %.2f) → %s",
                getattr(h, "name", h.entity_id), s, threshold, "kept" if s >= threshold else "dropped",
            )
        kept = [(h, s) for h, s in scored if s >= threshold]
        kept.sort(key=lambda hs: hs[1], reverse=True)
        logger.info("MemoryResearchAgent: applicability gate kept %d/%d hits", len(kept), len(hits))
        return [h for h, _ in kept]

    async def _second_wave(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        memory: MemoryManager,
        config: MAGEConfig,
        current_hits: list,
    ) -> list:
        """Recall again with gap-targeting sub-queries, re-gate, and merge (P1.1)."""
        gap_queries = await self._generate_gap_subqueries(
            query, domain_analysis, current_hits, config.memory_research_sub_queries
        )
        if not gap_queries:
            return current_hits

        more = await memory.iterative_search(
            query=query,
            domain_concepts=domain_analysis.key_concepts,
            top_k=config.memory_recall_top_k,
            threshold=config.memory_relevance_threshold,
            extra_queries=gap_queries,
        )
        if more and config.enable_memory_applicability_gate:
            more = await self._filter_applicable(query, more, config.memory_applicability_threshold)

        seen = {h.entity_id for h in current_hits}
        merged = current_hits + [h for h in more if h.entity_id not in seen]
        logger.info("MemoryResearchAgent: second wave %d -> %d hits", len(current_hits), len(merged))
        return merged

    async def _generate_gap_subqueries(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        found_hits: list,
        count: int,
    ) -> list[str]:
        """Generate gap-targeting sub-queries for the second recall wave (P1.1)."""
        found = [
            {"name": h.name, "structure": _summarize_structure(getattr(h, "abstract_structure", {}))}
            for h in found_hits
        ]
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Already-found items (aim elsewhere):\n{json.dumps(found, ensure_ascii=False, indent=2)}\n\n"
            f"Generate up to {count} gap-targeting sub-queries."
        )
        data = await self._llm.generate_json(
            system_prompt=MEMORY_GAP_SUBQUERY_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        subs = data.get("sub_queries", [])
        return subs[:count] if isinstance(subs, list) else []

    async def _purify_hits(self, query: str, hits: list) -> str:
        """Compress raw memory hits into a digest."""
        hits_text = json.dumps(
            [{"entity_id": h.entity_id, "name": h.name, "score": h.score} for h in hits],
            ensure_ascii=False,
            indent=2,
        )

        context = f"User query: {query}\n\nMemory search results:\n{hits_text}"

        data = await self._llm.generate_json(
            system_prompt=MEMORY_PURIFIER_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        return data.get("digest", "")
