"""Bootstrapped few-shot demo synthesis — plan item P1.7 (MIPRO-style).

Covers the three pieces of the feature with a directly-mocked LLM:
- ``DemoBootstrapper.bootstrap`` — score candidate chains on a synthetic
  benchmark, keep top-K above a pass-rate floor.
- ``format_demos_for_planner`` / ``_compact_plan`` — rendering.
- ``StepPlanner.plan`` — demos are injected into the planner context.
- ``MAGEGenerator._bootstrap_demos`` — the real-execution path (synth →
  fast-gen candidates → score → format), end to end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.benchmark import SynthesizedTask, SyntheticBenchmark
from mmar_mage.bootstrap import (
    BootstrapDemo,
    DemoBootstrapper,
    _compact_plan,
    format_demos_for_planner,
)
from mmar_mage.schemas import DomainAnalysis


def _chain(name: str, n: int = 2) -> dict[str, Any]:
    return {
        "name": name,
        "steps": [
            {
                "step_type": "llm",
                "number": i + 1,
                "title": f"{name} step {i + 1}",
                "aim": f"do {i + 1}",
                "dependencies": [i] if i > 0 else [],
            }
            for i in range(n)
        ],
    }


def _bench(n: int = 2) -> SyntheticBenchmark:
    return SyntheticBenchmark(
        query="q",
        tasks=[SynthesizedTask(task_id=f"synth_{i + 1:03d}", input=f"in{i}", rubric=["r"]) for i in range(n)],
    )


def _verdicts(pass_flags: list[bool]) -> dict[str, Any]:
    return {
        "results": [
            {"task_id": f"synth_{i + 1:03d}", "passed": p, "score": 0.9 if p else 0.1} for i, p in enumerate(pass_flags)
        ]
    }


def _mock_llm(json_side_effect: list[Any]) -> MagicMock:
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=json_side_effect)
    return llm


DOMAIN = DomainAnalysis(domain="finance")


# --------------------------------------------------------------------------- #
# DemoBootstrapper.bootstrap
# --------------------------------------------------------------------------- #
async def test_keeps_only_candidates_clearing_floor():
    # candidate A passes both tasks (1.0), candidate B fails both (0.0)
    llm = _mock_llm([_verdicts([True, True]), _verdicts([False, False])])
    demos = await DemoBootstrapper(llm).bootstrap(
        "q", DOMAIN, [_chain("A"), _chain("B")], _bench(2), top_k=2, min_pass_rate=0.5
    )
    assert len(demos) == 1
    assert demos[0].pass_rate == 1.0
    assert demos[0].query == "q"
    assert [s["title"] for s in demos[0].plan] == ["A step 1", "A step 2"]


async def test_respects_top_k_and_sorts_by_pass_rate():
    # three passing candidates with differing pass rates; top_k=2 keeps the best 2
    llm = _mock_llm([_verdicts([True, True]), _verdicts([True, False]), _verdicts([True, True])])
    demos = await DemoBootstrapper(llm).bootstrap(
        "q", DOMAIN, [_chain("A"), _chain("B"), _chain("C")], _bench(2), top_k=2, min_pass_rate=0.5
    )
    assert len(demos) == 2
    assert demos[0].pass_rate >= demos[1].pass_rate  # sorted desc
    assert demos[0].pass_rate == 1.0


async def test_no_candidates_returns_empty():
    llm = _mock_llm([])
    demos = await DemoBootstrapper(llm).bootstrap("q", DOMAIN, [], _bench(2))
    assert demos == []
    llm.generate_json.assert_not_called()


async def test_no_tasks_returns_empty():
    llm = _mock_llm([])
    demos = await DemoBootstrapper(llm).bootstrap("q", DOMAIN, [_chain("A")], _bench(0))
    assert demos == []
    llm.generate_json.assert_not_called()


async def test_none_clear_floor_returns_empty():
    llm = _mock_llm([_verdicts([False, False])])
    demos = await DemoBootstrapper(llm).bootstrap("q", DOMAIN, [_chain("A")], _bench(2), min_pass_rate=0.5)
    assert demos == []


async def test_scoring_failure_drops_candidate():
    # first candidate's judge call raises → dropped; second passes → kept
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=[RuntimeError("judge down"), _verdicts([True, True])])
    demos = await DemoBootstrapper(llm).bootstrap(
        "q", DOMAIN, [_chain("A"), _chain("B")], _bench(2), top_k=2, min_pass_rate=0.5
    )
    assert len(demos) == 1
    assert demos[0].plan[0]["title"] == "B step 1"


async def test_top_k_zero_returns_empty():
    llm = _mock_llm([])
    demos = await DemoBootstrapper(llm).bootstrap("q", DOMAIN, [_chain("A")], _bench(2), top_k=0)
    assert demos == []
    llm.generate_json.assert_not_called()


# --------------------------------------------------------------------------- #
# format_demos_for_planner / _compact_plan
# --------------------------------------------------------------------------- #
def test_compact_plan_keeps_only_demo_fields():
    chain = _chain("A", 1)
    chain["steps"][0]["reasoning_questions"] = "huge blob"  # should be dropped
    compact = _compact_plan(chain)
    assert compact == [{"number": 1, "title": "A step 1", "step_type": "llm", "aim": "do 1", "dependencies": []}]


def test_format_demos_contains_structure_and_score():
    demos = [BootstrapDemo(query="how much tax", plan=_compact_plan(_chain("A", 2)), pass_rate=1.0)]
    text = format_demos_for_planner(demos)
    assert "Demo 1" in text
    assert "100%" in text
    assert "how much tax" in text
    assert "[llm] A step 1" in text


def test_format_empty_returns_empty_string():
    assert format_demos_for_planner([]) == ""


# --------------------------------------------------------------------------- #
# StepPlanner injection
# --------------------------------------------------------------------------- #
async def test_planner_injects_demos_into_context():
    from mmar_mage.agents.step_planner import StepPlanner
    from mmar_mage.schemas import StepPlan

    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value=StepPlan(steps=[], rationale="ok"))
    planner = StepPlanner(llm)
    await planner.plan("q", DOMAIN, bootstrapped_demos="### Demo 1 — scored 100%\nPlan:\n  1. [llm] X")

    user_prompt = llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Worked examples (verified on a synthetic benchmark)" in user_prompt
    assert "### Demo 1 — scored 100%" in user_prompt


async def test_planner_no_demos_no_injection():
    from mmar_mage.agents.step_planner import StepPlanner
    from mmar_mage.schemas import StepPlan

    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value=StepPlan(steps=[], rationale="ok"))
    await StepPlanner(llm).plan("q", DOMAIN)
    user_prompt = llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Worked examples" not in user_prompt


# --------------------------------------------------------------------------- #
# Real-execution scenario — MAGEGenerator._bootstrap_demos end to end
# --------------------------------------------------------------------------- #
def _gen_with_bootstrap(json_side: list[Any], gen_side: list[str], **cfg):
    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    g = MAGEGenerator(config=MAGEConfig(api_key="t", enable_demo_bootstrap=True, benchmark_n_cases=2, **cfg))
    g._llm = MagicMock()
    g._llm.generate_json = AsyncMock(side_effect=json_side)
    g._llm.generate = AsyncMock(side_effect=gen_side)  # fast-mode candidate generation
    return g


async def test_bootstrap_demos_end_to_end():
    import json as _json

    # synth returns 2 tasks; 2 fast-gen candidates; runner scores A=pass, B=fail
    synth = {"tasks": [{"input": "a", "rubric": ["r"]}, {"input": "b", "rubric": ["r"]}]}
    gen = _gen_with_bootstrap(
        json_side=[synth, _verdicts([True, True]), _verdicts([False, False])],
        gen_side=[_json.dumps(_chain("Good")), _json.dumps(_chain("Bad"))],
        demo_bootstrap_candidates=2,
        demo_bootstrap_top_k=2,
    )
    text = await gen._bootstrap_demos("q", DOMAIN)
    assert "Demo 1" in text
    assert "Good step 1" in text  # the passing candidate's plan was kept
    assert "Bad step 1" not in text  # the failing candidate was dropped
    assert gen._llm.generate.await_count == 2  # one fast-gen per candidate


async def test_bootstrap_demos_no_tasks_returns_empty():
    gen = _gen_with_bootstrap(json_side=[{"tasks": []}], gen_side=[], demo_bootstrap_candidates=2)
    text = await gen._bootstrap_demos("q", DOMAIN)
    assert text == ""
    gen._llm.generate.assert_not_called()  # short-circuits before generating candidates


async def test_bootstrap_demos_nonfatal_on_synth_error():
    gen = _gen_with_bootstrap(json_side=[RuntimeError("synth down")], gen_side=[], demo_bootstrap_candidates=2)
    text = await gen._bootstrap_demos("q", DOMAIN)
    assert text == ""  # swallowed, deep pipeline continues
