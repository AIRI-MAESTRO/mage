"""
SelfRefiner — iterative critique → revise loop for chain quality improvement.

Runs a configurable number of iterations, stopping early when the chain
reaches a satisfactory quality threshold.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..prompts import SELF_REFINE_CRITIQUE_PROMPT, SELF_REFINE_REVISE_PROMPT
from ..schemas import DomainAnalysis

logger = logging.getLogger(__name__)

STAGE = "self_refine"


class RefinementCritique(BaseModel):
    """Critique from one refinement iteration.

    Parameters
    ----------
    overall_quality : float
        Quality score 0-1.
    is_satisfactory : bool
        Whether the chain meets the quality bar.
    issues : list[str]
        Identified quality issues.
    suggestions : list[str]
        Improvement suggestions.
    focus_areas : list[str]
        Areas to improve in next iteration.
    """

    overall_quality: float = 0.0
    is_satisfactory: bool = False
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(default_factory=list)


class RefinementResult(BaseModel):
    """Result of the iterative refinement process.

    Parameters
    ----------
    iterations_used : int
        Number of iterations actually performed.
    initial_quality : float
        Quality score of the initial chain.
    final_quality : float
        Quality score after refinement.
    chain_dict : dict[str, Any]
        Refined chain dictionary.
    chain_json : str
        Refined chain as JSON string.
    history : list[dict]
        Per-iteration history.
    """

    iterations_used: int = 0
    initial_quality: float = 0.0
    final_quality: float = 0.0
    chain_dict: dict[str, Any] = Field(default_factory=dict)
    chain_json: str = ""
    history: list[dict] = Field(default_factory=list)


class SelfRefiner:
    """Iteratively critique and refine a CARL chain.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> refiner = SelfRefiner(llm_client)
    >>> result = await refiner.refine(query, chain_dict, domain_analysis)
    >>> result.final_quality
    0.85
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def refine(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain_analysis: DomainAnalysis,
        max_iterations: int = 3,
        quality_threshold: float = 0.8,
    ) -> RefinementResult:
        """Iteratively critique and refine the chain.

        Parameters
        ----------
        query : str
            The user's original text query.
        chain_dict : dict[str, Any]
            Initial chain dictionary.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        max_iterations : int
            Maximum refinement iterations.
        quality_threshold : float
            Stop when quality reaches this threshold.

        Returns
        -------
        RefinementResult
            Refinement result with improved chain.
        """
        logger.info(
            "SelfRefiner: starting refinement (max %d iterations, threshold %.2f)", max_iterations, quality_threshold
        )

        current_chain = chain_dict
        history: list[dict] = []
        initial_quality = 0.0
        # Track the best chain seen — a revision can regress quality, and we must
        # not return a chain worse than one we already had (P1.13).
        best_chain = chain_dict
        best_quality = -1.0
        prev_quality = -1.0

        for iteration in range(max_iterations):
            # Whole-chain critique (one pass over all steps — already batched).
            critique = await self._critique(query, current_chain, domain_analysis)
            q = critique.overall_quality

            if iteration == 0:
                initial_quality = q
            if q > best_quality:
                best_quality = q
                best_chain = current_chain

            history.append({"iteration": iteration + 1, "quality": q, "issues_fixed": len(critique.issues)})
            logger.info(
                "SelfRefiner: iteration %d, quality=%.2f, satisfactory=%s",
                iteration + 1,
                q,
                critique.is_satisfactory,
            )

            if critique.is_satisfactory or q >= quality_threshold:
                break

            # Plateau / regression guard: if the previous revision did not improve
            # quality, further revisions are unlikely to help — stop early and keep
            # the best chain (P1.13: efficiency at large max_iterations).
            if iteration > 0 and q <= prev_quality:
                logger.info("SelfRefiner: no improvement (%.2f <= %.2f), stopping early", q, prev_quality)
                break

            prev_quality = q
            current_chain = await self._revise(query, current_chain, critique)

        chain_json = json.dumps(best_chain, ensure_ascii=False, indent=2)

        result = RefinementResult(
            iterations_used=len(history),
            initial_quality=initial_quality,
            final_quality=max(best_quality, 0.0),
            chain_dict=best_chain,
            chain_json=chain_json,
            history=history,
        )

        logger.info(
            "SelfRefiner: done in %d iterations, quality %.2f → %.2f",
            result.iterations_used,
            result.initial_quality,
            result.final_quality,
        )
        return result

    async def _critique(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain_analysis: DomainAnalysis,
    ) -> RefinementCritique:
        """Critique the current chain.

        Parameters
        ----------
        query : str
            User query.
        chain_dict : dict[str, Any]
            Chain to critique.
        domain_analysis : DomainAnalysis
            Domain analysis.

        Returns
        -------
        RefinementCritique
            Quality assessment.
        """
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Chain:\n{json.dumps(chain_dict, ensure_ascii=False, indent=2)}"
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=SELF_REFINE_CRITIQUE_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

        return RefinementCritique.model_validate(data)

    async def _revise(
        self,
        query: str,
        chain_dict: dict[str, Any],
        critique: RefinementCritique,
    ) -> dict[str, Any]:
        """Revise the chain based on critique.

        Parameters
        ----------
        query : str
            User query.
        chain_dict : dict[str, Any]
            Chain to revise.
        critique : RefinementCritique
            Critique with issues and suggestions.

        Returns
        -------
        dict[str, Any]
            Revised chain dictionary.
        """
        critique_text = json.dumps(critique.model_dump(), ensure_ascii=False, indent=2)
        prompt = SELF_REFINE_REVISE_PROMPT.format(critique=critique_text)

        context = f"User query: {query}\n\nChain to revise:\n{json.dumps(chain_dict, ensure_ascii=False, indent=2)}"

        revised: dict[str, Any] = await self._llm.generate_json(
            system_prompt=prompt,
            user_prompt=context,
            stage=STAGE,
        )

        return revised
