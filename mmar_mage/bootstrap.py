"""Bootstrapped few-shot demo synthesis (plan item P1.7, MIPRO-style).

The single biggest known quality lever for prompt-based agent authoring
(MIPROv2, §4.2) is *bootstrapped demonstrations*: run the current program on a
held-out slice, keep the traces that score well, and feed them back as few-shot
exemplars for the prompt module being optimized.

In MAGE the optimizable prompt module is the :class:`StepPlanner` (the evolve
loop's structural operators don't re-invoke a prompt, so demos belong here, not
in the mutation operators). The flow:

1. Synthesize a mini-benchmark for the query (reusing P1.16's
   :class:`~mmar_mage.benchmark.BenchmarkSynthesizer`).
2. Generate ``N`` candidate chains cheaply (the caller supplies them — typically
   via fast-mode generation, which never recurses into the deep pipeline).
3. Score each candidate on the benchmark with the P1.16 design-judge
   (:class:`~mmar_mage.benchmark.ChainRunner` + ``QualityReporter``).
4. Keep the top-``K`` candidates whose pass-rate clears a floor; their compact
   step plans become few-shot demos for the *authoritative* StepPlanner call.

This module is deliberately decoupled from generation: it *receives* candidate
chains rather than producing them, so it has no dependency on
:class:`~mmar_mage.generator.MAGEGenerator` (no import cycle) and is trivially
testable.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .benchmark import ChainRunner, QualityReporter, SyntheticBenchmark
from .llm import LLMClient
from .schemas import DomainAnalysis

logger = logging.getLogger(__name__)

# Step fields worth showing in a demo — enough to convey structure/granularity
# without bloating the planner prompt with full reasoning_questions/etc.
_DEMO_STEP_FIELDS = ("number", "title", "step_type", "aim", "dependencies")


class BootstrapDemo(BaseModel):
    """One bootstrapped demonstration: a query and a step plan that scored well."""

    query: str
    plan: list[dict[str, Any]] = Field(default_factory=list)
    pass_rate: float = 0.0


def _compact_plan(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce a chain's steps to the compact fields shown in a demo."""
    compact: list[dict[str, Any]] = []
    for step in chain.get("steps", []):
        if not isinstance(step, dict):
            continue
        compact.append({k: step.get(k) for k in _DEMO_STEP_FIELDS if k in step})
    return compact


def format_demos_for_planner(demos: list[BootstrapDemo]) -> str:
    """Render demos as a few-shot block to append to the StepPlanner context.

    Returns an empty string when there are no demos, so callers can unconditionally
    pass the result through.
    """
    if not demos:
        return ""
    blocks: list[str] = []
    for i, demo in enumerate(demos, 1):
        lines = []
        for s in demo.plan:
            deps = s.get("dependencies", []) or []
            lines.append(
                f"  {s.get('number')}. [{s.get('step_type', 'llm')}] "
                f"{s.get('title', '')} — {s.get('aim', '')} (deps={deps})"
            )
        steps_txt = "\n".join(lines) if lines else "  (no steps)"
        blocks.append(
            f"### Demo {i} — scored {demo.pass_rate:.0%} on a synthetic benchmark\n"
            f"Query: {demo.query}\nPlan:\n{steps_txt}"
        )
    return "\n\n".join(blocks)


class DemoBootstrapper:
    """Score candidate chains on a synthetic benchmark and keep the best as demos.

    Parameters
    ----------
    llm : LLMClient
        Client used to design-judge each candidate (via :class:`ChainRunner`).
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def bootstrap(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        candidate_chains: list[dict[str, Any]],
        benchmark: SyntheticBenchmark,
        top_k: int = 2,
        min_pass_rate: float = 0.5,
        run_mode: str = "mock",
    ) -> list[BootstrapDemo]:
        """Return up to ``top_k`` demos whose pass-rate clears ``min_pass_rate``.

        Parameters
        ----------
        candidate_chains : list[dict]
            Candidate chain dicts to score (the caller generates these — keeping
            generation out of this module avoids any generator import cycle).
        benchmark : SyntheticBenchmark
            The held-out slice to score against.
        top_k : int
            Maximum number of demos to keep, highest pass-rate first.
        min_pass_rate : float
            Floor a candidate must clear to qualify as a demo.

        Returns
        -------
        list[BootstrapDemo]
            Best-first, possibly empty (no candidates, no tasks, or none cleared
            the floor). Never raises — scoring failures drop the candidate.
        """
        if not candidate_chains or not benchmark.tasks or top_k <= 0:
            return []

        runner = ChainRunner(self._llm)
        scored: list[tuple[float, dict[str, Any]]] = []
        for chain in candidate_chains:
            try:
                results = await runner.run(chain, benchmark, mode=run_mode)
                report = QualityReporter.report(benchmark, results)
            except Exception as exc:  # non-fatal: a bad candidate just doesn't qualify
                logger.debug("DemoBootstrapper: scoring a candidate failed: %s", exc)
                continue
            scored.append((report.pass_rate, chain))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        demos: list[BootstrapDemo] = []
        for pass_rate, chain in scored:
            if pass_rate < min_pass_rate:
                break  # sorted desc — nothing below clears the floor either
            demos.append(BootstrapDemo(query=query, plan=_compact_plan(chain), pass_rate=pass_rate))
            if len(demos) >= top_k:
                break

        logger.info(
            "DemoBootstrapper: %d/%d candidate(s) cleared %.0f%% → %d demo(s)",
            len(demos),
            len(candidate_chains),
            min_pass_rate * 100,
            len(demos),
        )
        return demos
