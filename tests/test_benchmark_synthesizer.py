"""Focused benchmark synthesizer (plan item P1.16). Uses a directly-mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.benchmark import BenchmarkSynthesizer, SyntheticBenchmark
from mmar_mage.schemas import DomainAnalysis

DOMAIN = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")


def _synth(return_value=None, side_effect=None) -> BenchmarkSynthesizer:
    llm = MagicMock()
    if side_effect is not None:
        llm.generate_json = AsyncMock(side_effect=side_effect)
    else:
        llm.generate_json = AsyncMock(return_value=return_value)
    return BenchmarkSynthesizer(llm)


async def test_synthesizes_tasks_into_benchmark():
    synth = _synth(
        {
            "tasks": [
                {"input": "Analyze Q3 revenue", "rubric": ["mentions YoY growth"], "difficulty": "medium"},
                {"input": "Flag liquidity risks", "rubric": ["lists at least 2 risks"], "difficulty": "hard"},
            ]
        }
    )

    bench = await synth.synthesize("Analyze the report", DOMAIN, n_cases=2)

    assert isinstance(bench, SyntheticBenchmark)
    assert bench.query == "Analyze the report"
    assert bench.domain == "finance"
    assert [t.task_id for t in bench.tasks] == ["synth_001", "synth_002"]  # renumbered stably
    assert bench.tasks[1].difficulty == "hard"
    assert bench.tasks[0].rubric == ["mentions YoY growth"]
    assert bench.provenance["n_requested"] == 2


async def test_drops_tasks_with_empty_input():
    synth = _synth({"tasks": [{"input": "real task"}, {"input": "   "}, {"rubric": ["x"]}]})
    bench = await synth.synthesize("q", DOMAIN, n_cases=3)
    assert len(bench.tasks) == 1
    assert bench.tasks[0].input == "real task"


async def test_synthesis_failure_is_nonfatal():
    synth = _synth(side_effect=RuntimeError("llm down"))
    bench = await synth.synthesize("q", DOMAIN, n_cases=3)
    assert bench.tasks == []  # empty benchmark, no raise
    assert bench.query == "q"


async def test_unknown_metric_kind_falls_back_to_hybrid():
    synth = _synth({"tasks": [{"input": "t"}]})
    bench = await synth.synthesize("q", DOMAIN, n_cases=1, metric_kind="bogus")
    assert bench.metric_kind == "hybrid"


async def test_explicit_metric_kind_respected():
    synth = _synth({"tasks": [{"input": "t"}]})
    bench = await synth.synthesize("q", DOMAIN, n_cases=1, metric_kind="rule")
    assert bench.metric_kind == "rule"
