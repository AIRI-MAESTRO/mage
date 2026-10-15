"""
StepDescriber agent — fourth stage of the deep-mode pipeline.

Generates detailed CARL step descriptions (aim, reasoning_questions,
step_context_queries, etc.) from the DAG structure and user query.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

# Avoid circular import — StepCritiqueResult used only for type hints
from typing import TYPE_CHECKING, Any

from ..llm import LLMClient, StreamCallback
from ..prompts import (
    build_planning_guidance,
    build_step_describer_prompt,
    get_step_types_constraint,
    render_tool_signatures,
)
from ..schemas import CARLStepSchema, DAGStructure

if TYPE_CHECKING:
    from .step_critic import StepCritiqueResult

logger = logging.getLogger(__name__)

STAGE = "step_describing"


class StepDescriber:
    """Generate detailed CARL step descriptions.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> describer = StepDescriber(llm_client)
    >>> steps = await describer.describe(query, dag)
    >>> steps[0].aim
    'Determine the document type'
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def describe(
        self,
        query: str,
        dag: DAGStructure,
        allowed_step_types: list[str] | None = None,
        *,
        stream_callback: StreamCallback | None = None,
        on_step: Callable[[str, CARLStepSchema], None] | None = None,
        iterative: bool = False,
        mode: str = "batch",
        tools: list[dict[str, Any]] | None = None,
    ) -> list[CARLStepSchema]:
        """Produce detailed step descriptions.

        Parameters
        ----------
        query : str
            The user's original text query.
        dag : DAGStructure
            Output from the DAGBuilder.
        allowed_step_types : list[str] | None
            If set, restricts step types in the output.
        stream_callback : StreamCallback | None
            Optional per-chunk LLM streaming callback. CARE wires its
            TUI here so the StepDescriber renders as it arrives.
        on_step : Callable[[stage, CARLStepSchema], None] | None
            Optional callback invoked once per parsed step after the LLM
            response is decoded. Lets CARE show "step 2/5 ready" before
            the full stage completes.
        iterative : bool
            When ``True`` describe one step at a time, feeding the
            fully-described prior steps as context (P1.5) so step N can
            reference step N-1's real output / keys / format. Costs N LLM
            calls instead of 1. Default ``False`` (single batch call).

        Returns
        -------
        list[CARLStepSchema]
            Fully described CARL steps ready for serialization.
        """
        if iterative or mode == "iterative":
            return await self._describe_iterative(
                query, dag, allowed_step_types,
                stream_callback=stream_callback, on_step=on_step, tools=tools,
            )
        if mode == "parallel":
            return await self._describe_parallel(
                query, dag, allowed_step_types, on_step=on_step, tools=tools,
            )

        logger.info("StepDescriber: describing %d steps", len(dag.steps))

        system_prompt = (
            build_planning_guidance()
            + "\n"
            + build_step_describer_prompt({s.step_type for s in dag.steps})
            + get_step_types_constraint(allowed_step_types)
            + render_tool_signatures(tools)
        )

        context = f"User query: {query}\n\nDAG structure:\n{json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)}"

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=context,
            stage=STAGE,
            stream_callback=stream_callback,
        )

        raw_steps = data.get("steps", [])
        steps: list[CARLStepSchema] = []
        for raw in raw_steps:
            step = CARLStepSchema.model_validate(raw)
            steps.append(step)
            if on_step is not None:
                try:
                    on_step(STAGE, step)
                except Exception as exc:  # noqa: BLE001 — callback bugs must not break generation
                    logger.warning("on_step callback raised %s; ignoring", exc)

        logger.info("StepDescriber: generated %d detailed step descriptions", len(steps))
        return steps

    async def _describe_iterative(
        self,
        query: str,
        dag: DAGStructure,
        allowed_step_types: list[str] | None = None,
        *,
        stream_callback: StreamCallback | None = None,
        on_step: Callable[[str, CARLStepSchema], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[CARLStepSchema]:
        """Describe steps ONE AT A TIME (P1.5).

        Each step is generated in its own LLM call that sees the user query,
        the full DAG, AND the fully-described prior steps — so step N can
        reference step N-1's actual aim / output_key / context_queries
        (e.g. consume ``$history[-1]``) instead of guessing. Costs N calls;
        gated by ``MAGEConfig.enable_iterative_step_describing``.
        """
        logger.info("StepDescriber: describing %d steps iteratively", len(dag.steps))
        system_prompt = (
            build_planning_guidance()
            + "\n"
            + build_step_describer_prompt({s.step_type for s in dag.steps})
            + get_step_types_constraint(allowed_step_types)
            + render_tool_signatures(tools)
        )
        dag_json = json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)

        steps: list[CARLStepSchema] = []
        for dag_step in dag.steps:
            described = json.dumps([s.model_dump() for s in steps], ensure_ascii=False, indent=2)
            target = json.dumps(dag_step.model_dump(), ensure_ascii=False, indent=2)
            context = (
                f"User query: {query}\n\n"
                f"DAG structure:\n{dag_json}\n\n"
                "Steps already fully described — when this step depends on any of "
                "them, reference their EXACT aim / output / context_queries (e.g. "
                f"read a prior step's result via $history[-1]):\n{described}\n\n"
                "Now describe ONLY the next step below. Return JSON "
                '{"steps": [<the one described step>]}:\n'
                f"{target}"
            )
            data: dict[str, Any] = await self._llm.generate_json(
                system_prompt=system_prompt,
                user_prompt=context,
                stage=STAGE,
                stream_callback=stream_callback,
            )
            raw_steps = data.get("steps", []) if isinstance(data, dict) else []
            if not raw_steps:
                logger.warning(
                    "iterative describe: no step returned for step %s; skipping",
                    getattr(dag_step, "number", "?"),
                )
                continue
            step = CARLStepSchema.model_validate(raw_steps[0])
            steps.append(step)
            if on_step is not None:
                try:
                    on_step(STAGE, step)
                except Exception as exc:  # noqa: BLE001 — callback bugs must not break generation
                    logger.warning("on_step callback raised %s; ignoring", exc)

        logger.info("StepDescriber: generated %d step descriptions iteratively", len(steps))
        return steps

    @staticmethod
    def _dag_levels(dag: DAGStructure) -> list[list[Any]]:
        """Group DAG steps into dependency levels (level = longest path from a root).

        Steps in the same level are mutually independent → safe to describe
        concurrently. Relies on the DAG invariant that a step only depends on
        lower-numbered steps.
        """
        by_number = {s.number: s for s in dag.steps}
        level_of: dict[int, int] = {}

        def level(n: int) -> int:
            if n in level_of:
                return level_of[n]
            step = by_number.get(n)
            deps = [d for d in (getattr(step, "dependencies", []) or []) if d in by_number and d != n]
            level_of[n] = 0 if not deps else 1 + max(level(d) for d in deps)
            return level_of[n]

        for s in dag.steps:
            level(s.number)
        max_level = max(level_of.values(), default=0)
        levels: list[list[Any]] = [[] for _ in range(max_level + 1)]
        for s in dag.steps:
            levels[level_of[s.number]].append(s)
        return levels

    async def _describe_parallel(
        self,
        query: str,
        dag: DAGStructure,
        allowed_step_types: list[str] | None = None,
        *,
        on_step: Callable[[str, CARLStepSchema], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[CARLStepSchema]:
        """Describe steps level-by-level, the steps WITHIN each level concurrently (WP-7).

        Each step gets its own small LLM call whose prompt carries ONLY that
        step's type section (P-2) plus the already-described ancestor steps as
        context. Independent steps (same DAG level) run under one
        ``asyncio.gather``, so wall-time scales with DAG depth, not step count.
        The per-stage model override applies via the shared ``step_describing``
        stage.
        """
        logger.info("StepDescriber: describing %d steps in parallel by level", len(dag.steps))
        dag_json = json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)
        described: dict[int, CARLStepSchema] = {}

        async def _describe_one(dag_step: Any) -> CARLStepSchema | None:
            system_prompt = (
                build_planning_guidance([dag_step.step_type])
                + "\n"
                + build_step_describer_prompt({dag_step.step_type})
                + get_step_types_constraint(allowed_step_types)
                + render_tool_signatures(tools)
            )
            ancestors = json.dumps(
                [described[n].model_dump() for n in sorted(described)], ensure_ascii=False, indent=2
            )
            target = json.dumps(dag_step.model_dump(), ensure_ascii=False, indent=2)
            context = (
                f"User query: {query}\n\n"
                f"DAG structure:\n{dag_json}\n\n"
                "Steps already fully described — when this step depends on any of them, "
                "reference their EXACT aim / output / context_queries by dependency step "
                f"number (e.g. $history[<number>]):\n{ancestors}\n\n"
                'Now describe ONLY the step below. Return JSON {"steps": [<the one described step>]}:\n'
                f"{target}"
            )
            data: dict[str, Any] = await self._llm.generate_json(
                system_prompt=system_prompt, user_prompt=context, stage=STAGE
            )
            raw = data.get("steps", []) if isinstance(data, dict) else []
            if not raw:
                logger.warning("parallel describe: no step returned for step %s", getattr(dag_step, "number", "?"))
                return None
            return CARLStepSchema.model_validate(raw[0])

        for level_steps in self._dag_levels(dag):
            results = await asyncio.gather(*[_describe_one(s) for s in level_steps])
            for dag_step, step in zip(level_steps, results, strict=True):
                if step is None:
                    continue
                described[dag_step.number] = step
                if on_step is not None:
                    try:
                        on_step(STAGE, step)
                    except Exception as exc:  # noqa: BLE001 — callback bugs must not break generation
                        logger.warning("on_step callback raised %s; ignoring", exc)

        logger.info("StepDescriber: generated %d step descriptions in parallel", len(described))
        return [described[s.number] for s in dag.steps if s.number in described]

    async def revise(
        self,
        query: str,
        dag: DAGStructure,
        carl_steps: list[CARLStepSchema],
        critique_result: StepCritiqueResult,
    ) -> list[CARLStepSchema]:
        """Re-describe only the steps flagged by StepCriticAgent.

        Parameters
        ----------
        query : str
            The user's original text query.
        dag : DAGStructure
            DAG structure for context.
        carl_steps : list[CARLStepSchema]
            Current step descriptions.
        critique_result : StepCritiqueResult
            Critique results identifying weak steps.

        Returns
        -------
        list[CARLStepSchema]
            Updated step list with revised weak steps.
        """
        steps_to_revise = set(critique_result.steps_needing_revision)
        if not steps_to_revise:
            return carl_steps

        logger.info("StepDescriber: revising steps %s", steps_to_revise)

        # Build critique context for the weak steps
        critique_context = ""
        for c in critique_result.critiques:
            if c.step_number in steps_to_revise:
                critique_context += (
                    f"\nStep {c.step_number} issues: {c.issues}\nStep {c.step_number} suggestions: {c.suggestions}\n"
                )

        context = (
            f"User query: {query}\n\n"
            f"DAG structure:\n{json.dumps(dag.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"ONLY re-describe these steps: {sorted(steps_to_revise)}\n\n"
            f"Quality critique feedback:\n{critique_context}\n\n"
            f"Fix the issues mentioned above. Return ONLY the revised steps "
            f"(the ones listed above), not all steps."
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=build_planning_guidance() + "\n" + build_step_describer_prompt({s.step_type for s in dag.steps}),
            user_prompt=context,
            stage=STAGE,
        )

        revised_map: dict[int, CARLStepSchema] = {}
        for raw in data.get("steps", []):
            step = CARLStepSchema.model_validate(raw)
            revised_map[step.number] = step

        # Merge: replace only the revised steps
        result: list[CARLStepSchema] = []
        for s in carl_steps:
            if s.number in revised_map:
                result.append(revised_map[s.number])
            else:
                result.append(s)

        logger.info("StepDescriber: revised %d steps", len(revised_map))
        return result
