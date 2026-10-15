"""ChainRunner (mock mode) + QualityReporter (plan item P1.16). Directly-mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.benchmark import ChainRunner, QualityReporter, SynthesizedTask, SyntheticBenchmark
from mmar_mage.benchmark.runner import chain_summary

CHAIN = {"steps": [{"number": 1, "step_type": "llm", "title": "Analyze", "aim": "do x"}]}


def _bench(*ids: str) -> SyntheticBenchmark:
    return SyntheticBenchmark(query="q", tasks=[SynthesizedTask(task_id=i, input=f"in-{i}", rubric=["r"]) for i in ids])


def _runner(return_value=None, side_effect=None) -> ChainRunner:
    llm = MagicMock()
    if side_effect is not None:
        llm.generate_json = AsyncMock(side_effect=side_effect)
    else:
        llm.generate_json = AsyncMock(return_value=return_value)
    return ChainRunner(llm)


# --------------------------------------------------------------------------- #
# ChainRunner
# --------------------------------------------------------------------------- #
async def test_run_aligns_verdicts_by_task_id():
    runner = _runner(
        {
            "results": [
                {"task_id": "a", "passed": True, "score": 0.9},
                {"task_id": "b", "passed": False, "score": 0.2, "reason": "missing tool"},
            ]
        }
    )
    results = await runner.run(CHAIN, _bench("a", "b"))

    assert {r["task_id"]: r["passed"] for r in results} == {"a": True, "b": False}
    b = next(r for r in results if r["task_id"] == "b")
    assert b["failure"] == "missing tool"
    assert b["score"] == 0.2


async def test_run_missing_verdict_defaults_to_failed():
    runner = _runner({"results": [{"task_id": "a", "passed": True, "score": 0.9}]})  # 'b' omitted
    results = await runner.run(CHAIN, _bench("a", "b"))

    b = next(r for r in results if r["task_id"] == "b")
    assert b["passed"] is False
    assert b["score"] == 0.0
    assert b["failure"] == "not verified"


async def test_run_empty_benchmark_skips_llm():
    runner = _runner({"results": []})
    assert await runner.run(CHAIN, SyntheticBenchmark(query="q")) == []
    runner._llm.generate_json.assert_not_called()


async def test_run_judge_failure_is_nonfatal():
    runner = _runner(side_effect=RuntimeError("judge down"))
    results = await runner.run(CHAIN, _bench("a"))

    assert results[0]["passed"] is False  # recorded as unverified, not raised
    assert results[0]["failure"]


def test_chain_summary_is_compact():
    s = chain_summary(CHAIN)
    assert "#1" in s and "llm" in s and "Analyze" in s


# --------------------------------------------------------------------------- #
# QualityReporter
# --------------------------------------------------------------------------- #
def test_reporter_aggregates_pass_rate_and_dedups_failures():
    results = [
        {"task_id": "a", "passed": True, "score": 0.9, "failure": None},
        {"task_id": "b", "passed": False, "score": 0.2, "failure": "missing tool"},
        {"task_id": "c", "passed": False, "score": 0.1, "failure": "missing tool"},  # dup reason
    ]
    rep = QualityReporter.report(_bench("a", "b", "c"), results)

    assert rep.n_cases == 3
    assert rep.pass_rate == pytest.approx(1 / 3)
    assert rep.failure_modes == ["missing tool"]  # deduped
    assert rep.suggested_fixes == ["Revise the chain to address: missing tool"]
    assert rep.per_case_results == results


def test_reporter_empty_results():
    rep = QualityReporter.report(_bench(), [])
    assert rep.n_cases == 0
    assert rep.pass_rate == 0.0
    assert rep.failure_modes == []
