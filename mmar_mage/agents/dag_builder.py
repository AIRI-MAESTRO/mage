"""
DAGBuilder agent — third stage of the deep-mode pipeline.

Determines dependency edges between planned steps to form a valid
directed acyclic graph (DAG).
"""

from __future__ import annotations

import json
import logging

from ..llm import LLMClient
from ..prompts import build_dag_builder_prompt
from ..schemas import DAGStructure, StepPlan

logger = logging.getLogger(__name__)

STAGE = "dag_building"


class DAGBuilder:
    """Build a dependency DAG from a step plan.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> builder = DAGBuilder(llm_client)
    >>> dag = await builder.build(step_plan)
    >>> dag.pattern
    'funnel'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def build(self, plan: StepPlan, topology_hint: str = "") -> DAGStructure:
        """Produce a DAG from a step plan.

        Parameters
        ----------
        plan : StepPlan
            Output from the StepPlanner.
        topology_hint : str
            Optional structural constraint (plan item P1.15) describing the target
            topology, so the builder wires dependencies toward that pattern instead
            of inferring one from scratch. Empty string preserves prior behavior.

        Returns
        -------
        DAGStructure
            Steps annotated with dependencies and parallelism info.
        """
        logger.info("DAGBuilder: building DAG for %d steps", len(plan.steps))

        context = f"Step plan:\n{json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)}"
        if topology_hint:
            context += f"\n\nTarget topology (wire dependencies to match this):\n{topology_hint}"

        result = await self._llm.generate_structured(
            system_prompt=build_dag_builder_prompt(),
            user_prompt=context,
            response_model=DAGStructure,
            stage=STAGE,
        )
        logger.info("DAGBuilder: pattern=%s, max_workers=%d", result.pattern, result.max_workers)
        return result
