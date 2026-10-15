"""Focused benchmark synthesizer for self-verification (plan item P1.16).

Synthesizes a small ``SyntheticBenchmark`` (a handful of concrete test cases)
for ONE query via a single LLM call. This is the *verification* use case — it
deliberately does NOT fold in memory / web / user-provided tasks the way
``evolve.BenchmarkSynthesizer`` does (those are evolutionary-loop concerns). Both
share the same ``BENCHMARK_SYNTHESIZER_PROMPT`` so the test-case style is
consistent. Non-fatal: returns an empty benchmark if synthesis fails.
"""

from __future__ import annotations

import json
import logging
from typing import Literal, cast

from ..llm import LLMClient
from ..prompts import BENCHMARK_SYNTHESIZER_PROMPT
from ..schemas import DomainAnalysis
from .schemas import SynthesizedTask, SyntheticBenchmark

logger = logging.getLogger(__name__)

STAGE = "benchmark_synthesis"


class BenchmarkSynthesizer:
    """Synthesize a per-query mini-benchmark via one LLM call (P1.16)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        n_cases: int = 5,
        metric_kind: str = "hybrid",
    ) -> SyntheticBenchmark:
        """Synthesize a :class:`SyntheticBenchmark` for ``query``.

        Parameters
        ----------
        query : str
            The user's original query — what the chain is meant to solve.
        domain_analysis : DomainAnalysis
            Domain context to sharpen the test cases.
        n_cases : int
            Number of test cases to request.
        metric_kind : str
            How the chain output should be scored (recorded on the benchmark).

        Returns
        -------
        SyntheticBenchmark
            The synthesized benchmark; ``tasks`` is empty on failure (non-fatal).
        """
        logger.info("BenchmarkSynthesizer: synthesizing %d test cases for query", n_cases)
        try:
            tasks = await self._synthesize_via_llm(query, domain_analysis, n_cases)
        except Exception as exc:
            logger.warning("BenchmarkSynthesizer: synthesis failed (non-fatal): %s", exc)
            tasks = []

        return SyntheticBenchmark(
            query=query,
            domain=domain_analysis.domain,
            tasks=tasks,
            metric_kind=cast(
                'Literal["rule", "llm_judge", "structural_rubric", "hybrid"]',
                metric_kind if metric_kind in ("rule", "llm_judge", "structural_rubric", "hybrid") else "hybrid",
            ),
            provenance={"synthesizer": "llm", "n_requested": n_cases},
        )

    async def _synthesize_via_llm(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        count: int,
    ) -> list[SynthesizedTask]:
        """One LLM call → a list of concrete, rubric-bearing test cases."""
        user_prompt = (
            f"Original query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Generate exactly {count} test cases."
        )
        data = await self._llm.generate_json(
            system_prompt=BENCHMARK_SYNTHESIZER_PROMPT,
            user_prompt=user_prompt,
            stage=STAGE,
        )

        raw_tasks = data.get("tasks", []) or []
        tasks: list[SynthesizedTask] = []
        for i, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict):
                continue
            try:
                tasks.append(
                    SynthesizedTask(
                        task_id=raw.get("task_id") or f"synth_{i + 1:03d}",
                        input=str(raw.get("input", "")).strip(),
                        expected_output=raw.get("expected_output"),
                        rubric=list(raw.get("rubric", []) or []),
                        difficulty=raw.get("difficulty", "medium"),
                        source="llm",
                        metadata=dict(raw.get("metadata", {}) or {}),
                    )
                )
            except Exception as exc:
                logger.debug("BenchmarkSynthesizer: skipping malformed task %d: %s", i, exc)
                continue

        # Drop tasks with empty input — useless for grading. Re-number stably.
        tasks = [t for t in tasks if t.input]
        for i, t in enumerate(tasks):
            t.task_id = f"synth_{i + 1:03d}"
        return tasks
