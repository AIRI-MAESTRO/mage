"""Execution-feedback loop — verify → repair → re-verify (plan item P1.9).

Drives ``MAGEGenerator._run_execution_feedback`` / ``_repair_chain`` with a
directly-mocked LLM. The loop reuses the P1.16 synth→run→report pipeline and adds
a repair step with best-chain tracking + early-stop.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.benchmark import QualityReport, SynthesizedTask, SyntheticBenchmark
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

# minimal strict-CARL-valid chains
CHAIN = {"name": "c", "steps": [{"number": 1, "title": "S", "step_type": "llm", "aim": "do", "dependencies": []}]}
REVISED = {
    "name": "c2",
    "steps": [{"number": 1, "title": "Better", "step_type": "llm", "aim": "do better", "dependencies": []}],
}

SYNTH = {"tasks": [{"input": "a", "rubric": ["r"]}, {"input": "b", "rubric": ["r"]}]}
RUN_PASS = {
    "results": [
        {"task_id": "synth_001", "passed": True, "score": 0.9},
        {"task_id": "synth_002", "passed": True, "score": 0.9},
    ]
}
RUN_HALF = {
    "results": [
        {"task_id": "synth_001", "passed": True, "score": 0.9},
        {"task_id": "synth_002", "passed": False, "score": 0.2, "reason": "gap"},
    ]
}


def _gen(side_effect, **cfg) -> MAGEGenerator:
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", enable_execution_feedback=True, benchmark_n_cases=2, **cfg))
    gen._llm = MagicMock()
    gen._llm.generate_json = AsyncMock(side_effect=side_effect)
    return gen


# --------------------------------------------------------------------------- #
# _run_execution_feedback
# --------------------------------------------------------------------------- #
async def test_threshold_already_met_skips_repair():
    gen = _gen([SYNTH, RUN_PASS], execution_feedback_threshold=0.7)
    chain, report = await gen._run_execution_feedback("q", CHAIN, "finance")

    assert chain == CHAIN  # unchanged
    assert report["pass_rate"] == 1.0
    assert gen._llm.generate_json.await_count == 2  # synth + run only, no repair


async def test_repair_improves_and_is_adopted():
    gen = _gen([SYNTH, RUN_HALF, REVISED, RUN_PASS], execution_feedback_threshold=0.7)
    chain, report = await gen._run_execution_feedback("q", CHAIN, "finance")

    assert chain["name"] == "c2"  # repaired chain adopted
    assert report["pass_rate"] == 1.0
    assert gen._llm.generate_json.await_count == 4  # synth, run, repair, re-run


async def test_repair_without_improvement_keeps_best():
    # repaired chain scores the same (0.5) → not adopted, loop stops
    gen = _gen([SYNTH, RUN_HALF, REVISED, RUN_HALF], execution_feedback_threshold=0.7)
    chain, report = await gen._run_execution_feedback("q", CHAIN, "finance")

    assert chain == CHAIN  # original kept (repair didn't help)
    assert report["pass_rate"] == 0.5


async def test_invalid_repair_keeps_original():
    gen = _gen([SYNTH, RUN_HALF, {"steps": []}], execution_feedback_threshold=0.7)  # empty repair → discarded
    chain, report = await gen._run_execution_feedback("q", CHAIN, "finance")

    assert chain == CHAIN
    assert report["pass_rate"] == 0.5


async def test_no_tasks_returns_none_report():
    gen = _gen([{"tasks": []}])
    chain, report = await gen._run_execution_feedback("q", CHAIN, "finance")
    assert chain == CHAIN
    assert report is None


# --------------------------------------------------------------------------- #
# _repair_chain
# --------------------------------------------------------------------------- #
def _bench() -> SyntheticBenchmark:
    return SyntheticBenchmark(query="q", tasks=[SynthesizedTask(task_id="synth_001", input="a", rubric=["r"])])


async def test_repair_returns_validated_chain():
    gen = _gen([REVISED])
    report = QualityReport(per_case_results=[{"task_id": "synth_001", "passed": False, "failure": "gap"}])
    out = await gen._repair_chain("q", CHAIN, _bench(), report)
    assert out is not None
    assert out["name"] == "c2"


async def test_repair_no_failing_cases_returns_none():
    gen = _gen([REVISED])
    report = QualityReport(per_case_results=[{"task_id": "synth_001", "passed": True}])
    out = await gen._repair_chain("q", CHAIN, _bench(), report)
    assert out is None
    gen._llm.generate_json.assert_not_called()  # nothing to repair → no LLM call


async def test_repair_invalid_revision_returns_none():
    # revised chain references a non-existent dependency → strict validation fails
    bad = {"name": "x", "steps": [{"number": 1, "title": "S", "step_type": "llm", "aim": "do", "dependencies": [9]}]}
    gen = _gen([bad])
    report = QualityReport(per_case_results=[{"task_id": "synth_001", "passed": False, "failure": "gap"}])
    out = await gen._repair_chain("q", CHAIN, _bench(), report)
    assert out is None
