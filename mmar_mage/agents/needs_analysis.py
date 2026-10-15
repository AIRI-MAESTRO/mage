"""
NeedsAnalysisAgent — P6.2 "pull" stage.

Given the user query (+ its domain analysis), the model declares what the task
REQUIRES — packaged skills, built-in tools, external services, data only the
user can provide, and credentials — as a structured :class:`NeedsAnalysis`. The
pipeline then resolves what it can and requests the rest, so the model is not
locked to the always-on surfaced catalog; it can ask for more.
"""

from __future__ import annotations

import logging

from ..llm import LLMClient
from ..prompts import NEEDS_ANALYSIS_PROMPT
from ..schemas import DomainAnalysis, NeedsAnalysis

logger = logging.getLogger(__name__)

STAGE = "needs_analysis"


class NeedsAnalysisAgent:
    """Declare a task's capability/input requirements (the pull side)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(
        self,
        query: str,
        domain_analysis: DomainAnalysis | None = None,
    ) -> NeedsAnalysis:
        """Return the structured requirements for ``query``.

        Mirrors :class:`DomainAnalyzer` — a single ``generate_structured`` call
        validated against :class:`NeedsAnalysis`.
        """
        dom = ""
        if domain_analysis is not None:
            concepts = ", ".join(getattr(domain_analysis, "key_concepts", None) or [])
            dom = (
                f"\nDomain: {getattr(domain_analysis, 'domain', '')}; "
                f"concepts: {concepts}"
            )
        logger.info("NeedsAnalysisAgent: analysing requirements")
        result = await self._llm.generate_structured(
            system_prompt=NEEDS_ANALYSIS_PROMPT,
            user_prompt=f"User query: {query}{dom}",
            response_model=NeedsAnalysis,
            stage=STAGE,
        )
        return result
