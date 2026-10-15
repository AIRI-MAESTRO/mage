"""Benchmark module owns the shared schemas; evolve re-exports them (P1.16)."""

from __future__ import annotations

import mmar_mage.evolve.schemas as evolve_schemas
from mmar_mage.benchmark import (
    BenchmarkSynthesisResult,
    ChainScore,
    QualityReport,
    SynthesizedTask,
    SyntheticBenchmark,
)


def test_evolve_reexports_the_same_schema_classes():
    # not copies — evolve consumes the benchmark module's owned classes
    assert evolve_schemas.SynthesizedTask is SynthesizedTask
    assert evolve_schemas.ChainScore is ChainScore
    assert evolve_schemas.BenchmarkSynthesisResult is BenchmarkSynthesisResult


def test_synthetic_benchmark_construction():
    b = SyntheticBenchmark(
        query="summarize a PDF",
        domain="documents",
        tasks=[SynthesizedTask(task_id="t1", input="report.pdf", rubric=["mentions revenue"])],
    )
    assert b.query == "summarize a PDF"
    assert b.metric_kind == "hybrid"  # default
    assert b.tasks[0].rubric == ["mentions revenue"]


def test_quality_report_defaults():
    r = QualityReport(n_cases=3, pass_rate=2 / 3)
    assert r.n_cases == 3
    assert r.per_case_results == []
    assert r.failure_modes == []
    assert r.suggested_fixes == []


def test_synthesized_task_still_usable_from_evolve_path():
    # evolution code paths that import from evolve.schemas keep working
    t = evolve_schemas.SynthesizedTask(task_id="x", input="i", difficulty="hard", source="memory")
    assert t.difficulty == "hard"
    assert t.source == "memory"
