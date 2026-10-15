"""T26 — benchmark grading runs tasks concurrently, bounded by a semaphore."""

from __future__ import annotations

import asyncio
from typing import Any

from mmar_mage.benchmark.schemas import BenchmarkSynthesisResult, SynthesizedTask
from mmar_mage.evolve.grader import ChainGrader
from mmar_mage.evolve.schemas import EvolveConfig
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig


def _benchmark(n: int) -> BenchmarkSynthesisResult:
    return BenchmarkSynthesisResult(tasks=[SynthesizedTask(task_id=f"t{i}", input=f"in {i}") for i in range(n)])


def _grader(mocker: Any) -> ChainGrader:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    return ChainGrader(LLMClient(MAGEConfig(api_key="test")))


def _cfg(concurrency: int) -> EvolveConfig:
    return EvolveConfig(
        grading_concurrency=concurrency, use_llm_judge=True, use_structural_rubric=False, execution_model=None
    )


async def test_grading_is_concurrent_and_capped(mocker: Any) -> None:
    grader = _grader(mocker)
    live = peak = 0

    async def fake_judge(chain_dict: Any, task: Any, temp: float) -> tuple[float, str]:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)  # hold the slot so others pile up
        live -= 1
        return 1.0, f"note-{task.task_id}"

    grader._llm_judge = fake_judge  # type: ignore[method-assign]
    score = await grader.grade({"steps": []}, _benchmark(7), _cfg(3))

    assert len(score.per_task_scores) == 7
    assert all(v == 1.0 for v in score.per_task_scores.values())
    assert score.overall == 1.0
    assert 1 < peak <= 3  # ran in parallel, but the semaphore capped it at 3


async def test_concurrency_one_is_sequential(mocker: Any) -> None:
    grader = _grader(mocker)
    live = peak = 0

    async def fake_judge(chain_dict: Any, task: Any, temp: float) -> tuple[float, str]:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.005)
        live -= 1
        return 0.8, ""

    grader._llm_judge = fake_judge  # type: ignore[method-assign]
    score = await grader.grade({"steps": []}, _benchmark(4), _cfg(1))
    assert peak == 1  # grading_concurrency=1 → never two at once
    assert len(score.per_task_scores) == 4


async def test_per_task_results_not_scrambled_by_gather(mocker: Any) -> None:
    grader = _grader(mocker)

    async def fake_judge(chain_dict: Any, task: Any, temp: float) -> tuple[float, str]:
        idx = int(task.task_id[1:])  # distinct score per task id
        await asyncio.sleep(0)
        return idx / 10.0, f"n{idx}"

    grader._llm_judge = fake_judge  # type: ignore[method-assign]
    score = await grader.grade({"steps": []}, _benchmark(5), _cfg(4))
    assert score.per_task_scores == {"t0": 0.0, "t1": 0.1, "t2": 0.2, "t3": 0.3, "t4": 0.4}
