"""Aggregate per-case run results into a QualityReport (plan item P1.16).

Deterministic (no LLM): turns the ``ChainRunner`` per-case results into a
``QualityReport`` — pass rate, distinct failure modes, and targeted fix
suggestions derived from those failures.
"""

from __future__ import annotations

from typing import Any

from .schemas import QualityReport, SyntheticBenchmark


class QualityReporter:
    """Summarize per-case results into a :class:`QualityReport` (P1.16)."""

    @staticmethod
    def report(benchmark: SyntheticBenchmark, results: list[dict[str, Any]]) -> QualityReport:
        """Aggregate ``results`` (from :meth:`ChainRunner.run`) into a report."""
        n = len(results)
        if n == 0:
            return QualityReport(n_cases=0, pass_rate=0.0)

        passed = sum(1 for r in results if r.get("passed"))
        pass_rate = passed / n

        # Distinct failure reasons, stable order (deduped, first-seen wins).
        failure_modes: list[str] = []
        seen: set[str] = set()
        for r in results:
            if r.get("passed"):
                continue
            reason = (r.get("failure") or "").strip()
            if reason and reason not in seen:
                seen.add(reason)
                failure_modes.append(reason)

        suggested_fixes = [f"Revise the chain to address: {m}" for m in failure_modes[:5]]

        return QualityReport(
            n_cases=n,
            pass_rate=pass_rate,
            per_case_results=results,
            failure_modes=failure_modes,
            suggested_fixes=suggested_fixes,
        )
