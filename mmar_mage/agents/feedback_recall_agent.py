"""
FeedbackRecallAgent — recalls past execution feedback from memory.

Searches for chain_feedback entities matching the current domain/task,
filters by minimum quality, and compiles a digest that helps the
planner avoid past mistakes and repeat successful patterns.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..memory import MemoryManager
from ..prompts import FEEDBACK_APPLICABILITY_PROMPT, FEEDBACK_DIGEST_PROMPT
from ..schemas import DomainAnalysis, MAGEConfig

logger = logging.getLogger(__name__)

STAGE = "feedback_recall"


class RecalledFeedback(BaseModel):
    """A single recalled feedback record.

    Parameters
    ----------
    chain_entity_id : str
        Memory entity ID of the original chain.
    domain : str
        Domain of the chain.
    quality_score : float
        Execution quality score.
    success : bool
        Whether execution succeeded.
    failed_at_step : int | None
        Step number where failure occurred.
    failure_reason : str | None
        Reason for failure.
    pattern_used : str
        DAG pattern that was used.
    step_types_used : list[str]
        Step types in the chain.
    """

    chain_entity_id: str = ""
    domain: str = ""
    quality_score: float = 0.0
    success: bool = True
    failed_at_step: int | None = None
    failure_reason: str | None = None
    pattern_used: str = ""
    step_types_used: list[str] = Field(default_factory=list)


class FeedbackDigest(BaseModel):
    """Compiled feedback digest for the planner.

    Parameters
    ----------
    digest : str
        Compressed text summary for the planner.
    recalled_count : int
        Number of feedback records recalled.
    avg_quality : float
        Average quality score of recalled feedback.
    common_failures : list[str]
        Patterns that commonly failed.
    successful_patterns : list[str]
        Patterns that commonly succeeded.
    """

    digest: str = ""
    recalled_count: int = 0
    avg_quality: float = 0.0
    common_failures: list[str] = Field(default_factory=list)
    successful_patterns: list[str] = Field(default_factory=list)


class FeedbackRecallAgent:
    """Recall past execution feedback for similar tasks.

    Parameters
    ----------
    llm : LLMClient
        LLM client for digest generation.
    memory : MemoryManager
        Memory manager for feedback search.

    Examples
    --------
    >>> agent = FeedbackRecallAgent(llm_client, memory_manager)
    >>> digest = await agent.recall(query, domain_analysis, config)
    >>> digest.digest
    'Diamond pattern worked well for comparison tasks...'
    """

    def __init__(self, llm: LLMClient, memory: MemoryManager) -> None:
        self._llm = llm
        self._memory = memory

    async def recall(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        config: MAGEConfig,
    ) -> FeedbackDigest:
        """Recall past execution feedback for similar tasks.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        config : MAGEConfig
            Configuration with feedback parameters.

        Returns
        -------
        FeedbackDigest
            Compiled feedback digest.
        """
        logger.info("FeedbackRecallAgent: searching for feedback in domain '%s'", domain_analysis.domain)

        # Search memory for chain_feedback entities
        raw_feedback = await self._memory.search_feedback(
            domain=domain_analysis.domain,
            task_type=domain_analysis.task_type,
            top_k=config.feedback_recall_top_k,
        )

        if not raw_feedback:
            logger.info("FeedbackRecallAgent: no feedback found")
            return FeedbackDigest()

        # Filter by minimum quality
        filtered = [f for f in raw_feedback if f.get("quality_score", 0.0) >= config.feedback_min_quality]

        if not filtered:
            logger.info("FeedbackRecallAgent: no feedback above quality threshold %.2f", config.feedback_min_quality)
            return FeedbackDigest()

        # Applicability gate (P1.3): drop feedback whose lesson isn't relevant to
        # the *current* query (domain/task-type match alone is not enough).
        # Non-fatal — on failure we keep the quality-filtered set.
        if config.enable_feedback_applicability_gate:
            try:
                filtered = await self._filter_applicable(query, filtered, config.feedback_applicability_threshold)
            except Exception as exc:
                logger.warning("FeedbackRecallAgent: applicability gate failed (keeping feedback): %s", exc)
            if not filtered:
                logger.info("FeedbackRecallAgent: no feedback applicable to the query after gate")
                return FeedbackDigest()

        # Compile digest via LLM
        feedback_text = json.dumps(filtered, ensure_ascii=False, indent=2)
        context = (
            f"Domain: {domain_analysis.domain}\n"
            f"Task type: {domain_analysis.task_type}\n"
            f"Query: {query}\n\n"
            f"Raw feedback records ({len(filtered)}):\n{feedback_text}"
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=FEEDBACK_DIGEST_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

        digest_text = data.get("digest", "")

        # Compute summary stats
        avg_quality = sum(f.get("quality_score", 0.0) for f in filtered) / len(filtered)
        common_failures = [
            f.get("failure_reason", "") for f in filtered if not f.get("success", True) and f.get("failure_reason")
        ]
        successful_patterns = [
            f.get("pattern_used", "") for f in filtered if f.get("success", True) and f.get("pattern_used")
        ]

        result = FeedbackDigest(
            digest=digest_text,
            recalled_count=len(filtered),
            avg_quality=avg_quality,
            common_failures=common_failures,
            successful_patterns=successful_patterns,
        )

        logger.info(
            "FeedbackRecallAgent: recalled %d feedback records, avg_quality=%.2f",
            len(filtered),
            avg_quality,
        )
        return result

    async def _filter_applicable(
        self,
        query: str,
        records: list[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Gate recalled feedback by LLM-judged applicability to the query (P1.3).

        Records are addressed by list index (robust to missing
        ``chain_entity_id``); only those scoring at or above ``threshold`` are
        kept, ordered best-first. Returns the input unchanged if the judge
        produces nothing usable.
        """
        items = [
            {
                "id": str(i),
                "pattern": f.get("pattern_used", ""),
                "step_types": f.get("step_types_used", []),
                "success": f.get("success", True),
                "failure_reason": f.get("failure_reason"),
            }
            for i, f in enumerate(records)
        ]
        context = f"User query: {query}\n\nPast feedback records:\n{json.dumps(items, ensure_ascii=False, indent=2)}"

        data = await self._llm.generate_json(
            system_prompt=FEEDBACK_APPLICABILITY_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )
        ratings = data.get("ratings")
        if not isinstance(ratings, list) or not ratings:
            return records  # judge produced nothing usable — don't drop everything

        scores = {
            str(r.get("id")): float(r.get("applicability", 0.0))
            for r in ratings
            if isinstance(r, dict) and r.get("id") is not None
        }
        kept = [(f, scores.get(str(i), 0.0)) for i, f in enumerate(records)]
        kept = [(f, s) for f, s in kept if s >= threshold]
        kept.sort(key=lambda fs: fs[1], reverse=True)
        logger.info("FeedbackRecallAgent: applicability gate kept %d/%d records", len(kept), len(records))
        return [f for f, _ in kept]
