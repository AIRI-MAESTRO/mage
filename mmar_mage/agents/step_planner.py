"""
StepPlanner agent — second stage of the deep-mode pipeline.

Plans the number, types, and rough aims of reasoning steps based on
the user query and domain analysis output.
"""

from __future__ import annotations

import json
import logging

from ..llm import LLMClient
from ..prompts import (
    STEP_PLANNER_ANTI_PATTERNS_PROMPT,
    STEP_PLANNER_PROMPT,
    build_planning_guidance,
    get_step_types_constraint,
)
from ..schemas import DomainAnalysis, StepPlan

logger = logging.getLogger(__name__)

STAGE = "step_planning"

# Default cap for each research digest fed to the planner (~4 chars/token,
# so ~300 tokens). Mirrors ``MAGEConfig.max_digest_chars``.
DEFAULT_MAX_DIGEST_CHARS = 1200


def _truncate_digest(text: str, max_chars: int) -> str:
    """Cap a research digest to ``max_chars`` on a word boundary.

    Unbounded memory/web/skill digests otherwise bloat the planner prompt and
    dilute the query (plan item P1.6). Truncation is on whitespace so words are
    never split, with a visible marker so the model knows content was elided.

    ``max_chars <= 0`` disables capping. Relevance-aware truncation (dropping
    low-scoring fragments rather than the tail) depends on structured digests
    from P1.1/P1.2 and is layered on later.
    """
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars].rsplit(None, 1)[0] or text[:max_chars]
    return f"{head} …[truncated]"


class StepPlanner:
    """Plan reasoning steps for a CARL chain.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> planner = StepPlanner(llm_client)
    >>> plan = await planner.plan(query, domain_analysis)
    >>> len(plan.steps)
    5
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self,
        query: str,
        domain: DomainAnalysis,
        memory_digest: str = "",
        web_digest: str = "",
        capability_context: str = "",
        template_skeleton: str = "",
        allowed_step_types: list[str] | None = None,
        skill_digest: str = "",
        max_digest_chars: int = DEFAULT_MAX_DIGEST_CHARS,
        bootstrapped_demos: str = "",
        topology_hint: str = "",
        proposed_tools: str = "",
        curated_demos: str = "",
        max_type_retries: int = 2,
    ) -> StepPlan:
        """Generate a step plan.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain : DomainAnalysis
            Output from the DomainAnalyzer.
        memory_digest : str
            Compressed memory digest from MemoryResearchAgent.
        web_digest : str
            Compressed web research digest from WebResearchAgent.
        capability_context : str
            Formatted capability context (available tools/MCPs).
        template_skeleton : str
            Formatted template skeleton (suggested DAG structure).
        max_digest_chars : int
            Per-digest character cap (plan item P1.6). Each digest below is
            truncated on a word boundary before being assembled into the
            prompt; ``<= 0`` disables capping.
        bootstrapped_demos : str
            Pre-formatted few-shot demonstrations (plan item P1.7) — compact step
            plans that scored well on a synthetic benchmark for this query.
            Appended verbatim as worked examples; empty string is a no-op.
        topology_hint : str
            Pre-formatted structural constraint (plan item P1.15) from the
            TopologySelectorAgent — the target topology, its characteristic step
            types, and an example layout. Empty string is a no-op (free-form plan).
        proposed_tools : str
            Pre-formatted proposed-tool spec (plan item P1.17 Phase A) — a tool
            MAGE would synthesize because none exists. Marked as not-yet-existing;
            empty string is a no-op.

        Returns
        -------
        StepPlan
            Planned steps with rationale.
        """
        logger.info("StepPlanner: planning steps for domain=%s", domain.domain)

        # P1.6: cap each digest so unbounded research can't dominate the prompt.
        memory_digest = _truncate_digest(memory_digest, max_digest_chars)
        web_digest = _truncate_digest(web_digest, max_digest_chars)
        capability_context = _truncate_digest(capability_context, max_digest_chars)
        template_skeleton = _truncate_digest(template_skeleton, max_digest_chars)
        skill_digest = _truncate_digest(skill_digest, max_digest_chars)

        # Format the system prompt with capability and template context.
        # P6.6: lead with the planning guidance (think-first + every step type +
        # the "skills produce files, llm only writes text" rule) so the planner
        # picks `agent_skill` for file artifacts instead of defaulting to `llm`.
        # P2.6: append the anti-pattern catalogue to steer away from over-engineering.
        system_prompt = (
            build_planning_guidance(allowed_step_types)
            + "\n"
            + STEP_PLANNER_PROMPT.format(
                capability_context=capability_context,
                template_skeleton=template_skeleton,
            )
            + get_step_types_constraint(allowed_step_types)
            + STEP_PLANNER_ANTI_PATTERNS_PROMPT
        )

        context = (
            f"User query: {query}\n\nDomain analysis:\n{json.dumps(domain.model_dump(), ensure_ascii=False, indent=2)}"
        )

        # P1.15: topology constraint frames the whole plan, so place it first.
        if topology_hint:
            context += f"\n\n## Structural constraint:\n{topology_hint}"

        # P1.17 Phase A: a proposed (synthesized-not-existing) tool for this task.
        if proposed_tools:
            context += f"\n\n## Proposed synthesized tool:\n{proposed_tools}"

        if memory_digest:
            context += (
                f"\n\n## Relevant past experience (from memory):\n{memory_digest}\n\n"
                "Use this as structural guidance, not as a template to copy."
            )

        if web_digest:
            context += f"\n\n## Domain context (web research):\n{web_digest}"

        if skill_digest:
            context += (
                f"\n\n## Recommended packaged skills:\n{skill_digest}\n\n"
                "If a recommended skill can DO what a step needs — especially "
                "producing a file/document/presentation/spreadsheet — you MUST "
                "use an `agent_skill` step with that skill's EXACT URI instead "
                "of a hand-rolled `llm` step. Ignore only skills that don't fit."
            )

        # P1.7: curated per-benchmark reference plans (from the selected profile).
        # Cheaper than the runtime bootstrap and benchmark-specific.
        if curated_demos:
            context += f"\n\n{curated_demos}"

        # P1.7: bootstrapped few-shot demos — plans that verifiably scored well on
        # a synthetic benchmark for this query. Strong structural signal, so place
        # them last (closest to the generation point).
        if bootstrapped_demos:
            context += (
                f"\n\n## Worked examples (verified on a synthetic benchmark):\n{bootstrapped_demos}\n\n"
                "Match the granularity and dependency structure of these plans "
                "where the query is similar; adapt rather than copy them."
            )

        result = await self._llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=context,
            response_model=StepPlan,
            stage=STAGE,
        )
        # P2.8: enforce allowed_step_types on the OUTPUT, not just via the prompt.
        result = await self._enforce_allowed_step_types(
            result, allowed_step_types, system_prompt, context, max_retries=max_type_retries
        )
        logger.info("StepPlanner: planned %d steps", len(result.steps))
        return result

    async def _enforce_allowed_step_types(
        self,
        plan: StepPlan,
        allowed: list[str] | None,
        system_prompt: str,
        context: str,
        max_retries: int = 2,
    ) -> StepPlan:
        """Reject + re-prompt plans that use disallowed step types (plan item P2.8).

        The ``allowed_step_types`` constraint was prompt-only; the LLM could still
        emit a disallowed type and ``carl_export`` would silently downgrade it to
        ``llm`` (lossy). Here we validate the OUTPUT: on a violation, re-prompt with
        the offending types (up to ``max_retries``); if any remain, coerce them to a
        safe allowed type (graceful — never fail generation over this).
        """
        if not allowed:
            return plan
        allowed_set = set(allowed)

        for attempt in range(max_retries):
            violations = sorted({s.step_type for s in plan.steps if s.step_type not in allowed_set})
            if not violations:
                return plan
            logger.info(
                "StepPlanner: plan used disallowed step types %s; re-prompting (%d/%d)",
                violations,
                attempt + 1,
                max_retries,
            )
            feedback = (
                f"\n\nYOUR PREVIOUS PLAN WAS INVALID: it used step types {violations}, which are NOT permitted. "
                f"Use ONLY these step types: {sorted(allowed_set)}. Regenerate the full plan accordingly."
            )
            plan = await self._llm.generate_structured(
                system_prompt=system_prompt + feedback,
                user_prompt=context,
                response_model=StepPlan,
                stage=STAGE,
            )

        # Still violating after retries → coerce to a safe allowed type.
        leftover = [s for s in plan.steps if s.step_type not in allowed_set]
        if leftover:
            fallback = "llm" if "llm" in allowed_set else sorted(allowed_set)[0]
            logger.warning(
                "StepPlanner: coercing %d disallowed step(s) to %r after %d retries",
                len(leftover),
                fallback,
                max_retries,
            )
            for step in leftover:
                step.step_type = fallback
        return plan
