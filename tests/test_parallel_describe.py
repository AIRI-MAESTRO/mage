"""WP-7 — parallel per-step describe by DAG level."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mmar_mage.agents.step_describer import StepDescriber
from mmar_mage.llm import LLMClient
from mmar_mage.prompts import STEP_DESCRIBER_TYPE_SECTIONS
from mmar_mage.schemas import DAGStep, DAGStructure, MAGEConfig

_STEP = {
    "step_type": "llm",
    "number": 1,
    "title": "t",
    "aim": "a",
    "reasoning_questions": "q",
    "step_context_queries": ["x"],
    "stage_action": "s",
    "example_reasoning": "e",
    "dependencies": [],
    "retry_max": 3,
    "llm_config": None,
}


def _dag() -> DAGStructure:
    return DAGStructure(
        pattern="diamond",
        max_workers=2,
        steps=[
            DAGStep(number=1, title="A", step_type="llm", dependencies=[], rough_aim="a"),
            DAGStep(number=2, title="B", step_type="tool", dependencies=[], rough_aim="b"),
            DAGStep(number=3, title="C", step_type="llm", dependencies=[1], rough_aim="c"),
            DAGStep(number=4, title="D", step_type="structured_output", dependencies=[2], rough_aim="d"),
            DAGStep(number=5, title="E", step_type="llm", dependencies=[3], rough_aim="e"),
            DAGStep(number=6, title="F", step_type="llm", dependencies=[4], rough_aim="f"),
        ],
    )


def test_dag_levels_group_by_depth() -> None:
    levels = StepDescriber._dag_levels(_dag())
    assert [sorted(s.number for s in lv) for lv in levels] == [[1, 2], [3, 4], [5, 6]]


async def test_parallel_describe_runs_in_waves(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    calls: list[str] = []

    async def _fake(*, system_prompt: str, user_prompt: str, stage: str) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        calls.append(system_prompt)
        return {"steps": [_STEP]}

    llm.generate_json = _fake  # type: ignore[method-assign]
    describer = StepDescriber(llm)

    t0 = time.perf_counter()
    out = await describer.describe("q", _dag(), mode="parallel")
    elapsed = time.perf_counter() - t0

    assert len(out) == 6
    assert len(calls) == 6  # one LLM call per step
    # 3 DAG levels run as 3 concurrent waves (≈0.3s); 6 sequential would be ≈0.6s
    assert elapsed < 0.45, f"describe was not parallel by level: {elapsed:.3f}s"
    # each per-step prompt carries exactly ONE type section (P-2 reuse → smaller input)
    for sp in calls:
        n = sum(1 for sec in STEP_DESCRIBER_TYPE_SECTIONS.values() if sec in sp)
        assert n == 1, f"expected exactly one type section in the prompt, got {n}"
