"""Run a generated chain against a synthetic benchmark (plan item P1.16).

``ChainRunner`` scores how well a chain would satisfy each synthesized test case.
Mode ``"mock"`` (default) uses one LLM call to judge the chain's *design* against
all cases — no executor or tools required, so it works for any chain. Mode
``"real"`` (a future slice) would load the chain via ``mmar_carl`` and execute it
before judging; until then it falls back to ``"mock"``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMClient
from ..prompts import BENCHMARK_RUN_PROMPT
from .schemas import SyntheticBenchmark

logger = logging.getLogger(__name__)

STAGE = "benchmark_run"


def chain_summary(chain_dict: dict[str, Any]) -> str:
    """Compact one-line-per-step summary of a chain (for the judge prompt)."""
    lines = [
        f"#{s.get('number')} [{s.get('step_type', 'llm')}] {s.get('title', '')} — aim: {str(s.get('aim', ''))[:80]}"
        for s in chain_dict.get("steps", [])
    ]
    return "\n".join(lines)


class ChainRunner:
    """Score a chain against a :class:`SyntheticBenchmark` (P1.16)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(
        self,
        chain_dict: dict[str, Any],
        benchmark: SyntheticBenchmark,
        mode: str = "mock",
    ) -> list[dict[str, Any]]:
        """Return a per-case result list ``[{task_id, passed, score, failure}]``.

        Non-fatal: on judge failure every case is recorded as a non-pass with the
        error as its ``failure`` (so the report still reflects "we couldn't
        verify"), rather than raising.
        """
        if not benchmark.tasks:
            return []
        if mode == "real":  # real-executor mode is a later slice; mock is the fallback
            logger.info("ChainRunner: 'real' mode not yet implemented, using 'mock'")

        cases = [{"task_id": t.task_id, "input": t.input, "rubric": t.rubric} for t in benchmark.tasks]
        context = (
            f"User query: {benchmark.query}\n\n"
            f"Chain under test (step summary):\n{chain_summary(chain_dict)}\n\n"
            f"Test cases:\n{json.dumps(cases, ensure_ascii=False, indent=2)}"
        )

        try:
            data = await self._llm.generate_json(
                system_prompt=BENCHMARK_RUN_PROMPT,
                user_prompt=context,
                stage=STAGE,
            )
            raw = data.get("results", []) if isinstance(data, dict) else []
            verdicts = {str(r.get("task_id")): r for r in raw if isinstance(r, dict) and r.get("task_id") is not None}
        except Exception as exc:
            logger.warning("ChainRunner: judging failed (non-fatal): %s", exc)
            verdicts = {}

        results: list[dict[str, Any]] = []
        for t in benchmark.tasks:
            v = verdicts.get(t.task_id, {})
            score = float(v.get("score", 0.0) or 0.0)
            passed = bool(v.get("passed", score >= 0.5))
            results.append(
                {
                    "task_id": t.task_id,
                    "score": score,
                    "passed": passed,
                    "failure": None if passed else (v.get("reason") or "not verified"),
                }
            )
        return results
