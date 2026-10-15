"""
ToTPlanner — Tree of Thoughts DAG planning.

Explores multiple DAG structures for the same step plan using a
breadth-first approach, scores each with LLM-as-judge, refines
the best branches, and selects the winner.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..prompts import DAG_BUILDER_PROMPT, TOT_BRANCH_SCORER_PROMPT
from ..schemas import DAGStructure, DomainAnalysis, StepPlan

logger = logging.getLogger(__name__)

STAGE = "tot_planning"

# Patterns to explore across branches
DAG_PATTERNS = ["pipeline", "diamond", "funnel", "critique_revise", "map_reduce", "multi_hop"]


class ToTBranch(BaseModel):
    """A single DAG candidate branch.

    Parameters
    ----------
    dag : DAGStructure
        Generated DAG structure.
    score : float
        Quality score from LLM-as-judge.
    rationale : str
        Why this score was given.
    pattern : str
        The pattern used for this branch.
    """

    dag: DAGStructure = Field(default_factory=DAGStructure)
    score: float = 0.0
    rationale: str = ""
    pattern: str = "pipeline"


class ToTResult(BaseModel):
    """Result of Tree of Thoughts DAG planning.

    Parameters
    ----------
    best_dag : DAGStructure
        Best DAG structure selected.
    branches_explored : int
        Total number of branches explored.
    best_score : float
        Score of the best branch.
    all_branches : list[ToTBranch]
        All explored branches with scores.
    """

    best_dag: DAGStructure = Field(default_factory=DAGStructure)
    branches_explored: int = 0
    best_score: float = 0.0
    all_branches: list[ToTBranch] = Field(default_factory=list)


class ToTPlanner:
    """Explore multiple DAG structures using Tree of Thoughts.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> tot = ToTPlanner(llm_client)
    >>> result = await tot.plan(query, step_plan, domain_analysis)
    >>> result.best_dag.pattern
    'diamond'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self,
        query: str,
        step_plan: StepPlan,
        domain_analysis: DomainAnalysis,
        num_branches: int = 3,
        depth: int = 2,
    ) -> ToTResult:
        """Explore multiple DAG structures using Tree of Thoughts.

        Parameters
        ----------
        query : str
            The user's original text query.
        step_plan : StepPlan
            Step plan from the StepPlanner.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        num_branches : int
            Number of alternative DAG structures to generate.
        depth : int
            Depth of exploration (level 2 refines top branches).

        Returns
        -------
        ToTResult
            Best DAG structure and exploration metadata.
        """
        logger.info("ToTPlanner: exploring %d branches at depth %d", num_branches, depth)

        # Level 1: Generate diverse DAG structures
        branches: list[ToTBranch] = []
        patterns_to_try = DAG_PATTERNS[:num_branches]

        for pattern in patterns_to_try:
            dag = await self._generate_dag_with_pattern(query, step_plan, pattern)
            score_data = await self._score_dag(query, step_plan, dag)
            branches.append(
                ToTBranch(
                    dag=dag,
                    score=score_data.get("overall_score", 0.0),
                    rationale=score_data.get("rationale", ""),
                    pattern=dag.pattern,
                )
            )

        # Level 2: Refine top branches
        if depth >= 2 and len(branches) >= 2:
            branches.sort(key=lambda b: b.score, reverse=True)
            top_branches = branches[:2]

            for branch in top_branches:
                refined_dag = await self._refine_dag(query, step_plan, branch.dag)
                score_data = await self._score_dag(query, step_plan, refined_dag)
                branches.append(
                    ToTBranch(
                        dag=refined_dag,
                        score=score_data.get("overall_score", 0.0),
                        rationale=score_data.get("rationale", ""),
                        pattern=refined_dag.pattern,
                    )
                )

        # Select best
        branches.sort(key=lambda b: b.score, reverse=True)
        best = branches[0] if branches else ToTBranch()

        result = ToTResult(
            best_dag=best.dag,
            branches_explored=len(branches),
            best_score=best.score,
            all_branches=branches,
        )

        logger.info(
            "ToTPlanner: explored %d branches, best score=%.2f, pattern=%s",
            len(branches),
            best.score,
            best.pattern,
        )
        return result

    async def _generate_dag_with_pattern(
        self,
        query: str,
        step_plan: StepPlan,
        suggested_pattern: str,
    ) -> DAGStructure:
        """Generate a DAG structure suggesting a specific pattern.

        Parameters
        ----------
        query : str
            User query.
        step_plan : StepPlan
            Step plan.
        suggested_pattern : str
            Pattern to suggest.

        Returns
        -------
        DAGStructure
            Generated DAG.
        """
        context = (
            f"Step plan:\n{json.dumps(step_plan.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Suggested pattern: {suggested_pattern}\n"
            f"Try to structure the DAG as a '{suggested_pattern}' pattern if it fits the task."
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=DAG_BUILDER_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

        return DAGStructure.model_validate(data)

    async def _score_dag(
        self,
        query: str,
        step_plan: StepPlan,
        dag: DAGStructure,
    ) -> dict[str, Any]:
        """Score a DAG structure using LLM-as-judge.

        Parameters
        ----------
        query : str
            User query.
        step_plan : StepPlan
            Step plan.
        dag : DAGStructure
            DAG to score.

        Returns
        -------
        dict[str, Any]
            Score data with overall_score and rationale.
        """
        context = (
            f"User query: {query}\n\n"
            f"Step plan:\n{json.dumps(step_plan.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"DAG to evaluate:\n{json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)}"
        )

        return await self._llm.generate_json(
            system_prompt=TOT_BRANCH_SCORER_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

    async def _refine_dag(
        self,
        query: str,
        step_plan: StepPlan,
        dag: DAGStructure,
    ) -> DAGStructure:
        """Refine an existing DAG structure.

        Parameters
        ----------
        query : str
            User query.
        step_plan : StepPlan
            Step plan.
        dag : DAGStructure
            DAG to refine.

        Returns
        -------
        DAGStructure
            Refined DAG.
        """
        context = (
            f"Step plan:\n{json.dumps(step_plan.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Current DAG (refine and improve it):\n{json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Improve parallelism and dependency correctness while keeping the same steps."
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=DAG_BUILDER_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

        return DAGStructure.model_validate(data)
