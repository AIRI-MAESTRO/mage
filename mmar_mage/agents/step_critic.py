"""
StepCriticAgent — critiques each CARL step on 5 quality dimensions.

Identifies weak steps that fall below a configurable quality threshold
and provides actionable suggestions for improvement.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..prompts import STEP_CRITIC_PROMPT
from ..schemas import CARLStepSchema, DomainAnalysis

logger = logging.getLogger(__name__)

STAGE = "step_critique"

# Dimensions that most determine whether a step actually works at runtime. A
# failure here forces revision even if the averaged overall_score passes (P1.11).
_CRITICAL_DIMENSIONS = ("aim_clarity", "action_specificity")


class StepCritique(BaseModel):
    """Critique result for a single step.

    Parameters
    ----------
    step_number : int
        Step number being critiqued.
    scores : dict[str, float]
        Per-dimension scores (aim_clarity, reasoning_quality,
        context_relevance, dependency_logic, action_specificity).
    overall_score : float
        Average of dimension scores.
    issues : list[str]
        Identified quality issues.
    suggestions : list[str]
        Actionable improvement suggestions.
    """

    step_number: int
    scores: dict[str, float] = Field(default_factory=dict)
    overall_score: float = 0.0
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class StepCritiqueResult(BaseModel):
    """Result of critiquing all steps.

    Parameters
    ----------
    critiques : list[StepCritique]
        Per-step critique results.
    overall_quality : float
        Average quality score across all steps.
    steps_needing_revision : list[int]
        Step numbers that fall below the quality threshold.
    """

    critiques: list[StepCritique] = Field(default_factory=list)
    overall_quality: float = 0.0
    steps_needing_revision: list[int] = Field(default_factory=list)


class StepCriticAgent:
    """Critique CARL steps on 5 quality dimensions.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> critic = StepCriticAgent(llm_client)
    >>> result = await critic.critique(query, carl_steps, domain_analysis)
    >>> result.steps_needing_revision
    [2, 4]
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def critique(
        self,
        query: str,
        carl_steps: list[CARLStepSchema],
        domain_analysis: DomainAnalysis,
        threshold: float = 0.6,
        revision_threshold: float = 0.6,
    ) -> StepCritiqueResult:
        """Critique each step and identify quality issues.

        Parameters
        ----------
        query : str
            The user's original text query.
        carl_steps : list[CARLStepSchema]
            Steps to critique.
        domain_analysis : DomainAnalysis
            Domain analysis result for context.
        threshold : float
            Minimum averaged ``overall_score`` to pass (0-1).
        revision_threshold : float
            Minimum score on a *critical* dimension (``aim_clarity`` /
            ``action_specificity``); a step below this on either dimension is
            flagged for revision even if its ``overall_score`` passes (P1.11).

        Returns
        -------
        StepCritiqueResult
            Critique results with per-step scores and issues.
        """
        logger.info("StepCriticAgent: critiquing %d steps", len(carl_steps))

        steps_data = [s.model_dump(exclude_none=True) for s in carl_steps]
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Steps to critique:\n{json.dumps(steps_data, ensure_ascii=False, indent=2)}"
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=STEP_CRITIC_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

        raw_critiques = data.get("critiques", [])
        critiques: list[StepCritique] = []
        for raw in raw_critiques:
            critiques.append(StepCritique.model_validate(raw))

        overall_quality = sum(c.overall_score for c in critiques) / len(critiques) if critiques else 0.0
        # Flag for revision when the average fails OR a critical dimension does
        # (a load-bearing weakness the average could otherwise hide — P1.11).
        steps_needing_revision = [
            c.step_number
            for c in critiques
            if c.overall_score < threshold
            or any(c.scores.get(dim, 1.0) < revision_threshold for dim in _CRITICAL_DIMENSIONS)
        ]

        result = StepCritiqueResult(
            critiques=critiques,
            overall_quality=overall_quality,
            steps_needing_revision=steps_needing_revision,
        )

        logger.info(
            "StepCriticAgent: overall quality %.2f, %d steps need revision",
            overall_quality,
            len(steps_needing_revision),
        )
        return result
