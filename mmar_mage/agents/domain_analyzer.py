"""
DomainAnalyzer agent — first stage of the deep-mode pipeline.

Analyzes the user query to determine knowledge domain, task type,
complexity, key concepts, and suggested step count.
"""

from __future__ import annotations

import logging

from ..llm import LLMClient
from ..prompts import DOMAIN_ANALYZER_PROMPT
from ..schemas import DomainAnalysis

logger = logging.getLogger(__name__)

STAGE = "domain_analysis"


class DomainAnalyzer:
    """Analyze a user query to extract domain metadata.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> analyzer = DomainAnalyzer(llm_client)
    >>> analysis = await analyzer.analyze("Analyze a financial report")
    >>> analysis.domain
    'finance'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def analyze(self, query: str) -> DomainAnalysis:
        """Run domain analysis on the user query.

        Parameters
        ----------
        query : str
            The user's original text query.

        Returns
        -------
        DomainAnalysis
            Structured domain analysis result.
        """
        logger.info("DomainAnalyzer: analyzing query")
        result = await self._llm.generate_structured(
            system_prompt=DOMAIN_ANALYZER_PROMPT,
            user_prompt=f"User query: {query}",
            response_model=DomainAnalysis,
            stage=STAGE,
        )
        logger.info(
            "DomainAnalyzer: domain=%s, complexity=%s, steps=%d",
            result.domain,
            result.complexity,
            result.suggested_step_count,
        )
        return result
