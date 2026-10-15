"""Tests for evolve mode (synthesizer, grader, engine).

LLM is fully mocked through the AsyncOpenAI patch — every call returns a
canned JSON response. Operators are exercised on real chain dicts.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage import (
    BenchmarkSynthesisResult,
    ChainEvolver,
    EvolveConfig,
    MAGEConfig,
    MAGEGenerator,
    SynthesizedTask,
)
from mmar_mage.evolve.benchmark_synthesizer import BenchmarkSynthesizer
from mmar_mage.evolve.grader import ChainGrader
from mmar_mage.schemas import DomainAnalysis


def _resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Canned JSON payloads
# ---------------------------------------------------------------------------


def _benchmark_payload(n: int = 3) -> dict[str, Any]:
    return {
        "rubric_summary": "Output must include the verdict and supporting evidence.",
        "tasks": [
            {
                "task_id": f"synth_{i + 1:03d}",
                "input": f"Test instance #{i + 1}: Verify that <Fact {i + 1}> holds.",
                "expected_output": f"Verdict: TRUE for fact {i + 1}.",
                "rubric": [
                    f"answer mentions fact {i + 1}",
                    "verdict is TRUE or FALSE",
                ],
                "difficulty": ["easy", "medium", "hard"][i % 3],
                "metadata": {},
            }
            for i in range(n)
        ],
    }


def _judge_payload(score: float, notes: str = "ok") -> dict[str, Any]:
    return {
        "score": score,
        "rubric_results": [{"item": "x", "passed": True, "evidence": "..."}],
        "notes": notes,
    }


SAMPLE_CHAIN: dict[str, Any] = {
    "name": "test chain",
    "max_workers": 2,
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Read input",
            "aim": "Identify the core claim and supporting evidence",
            "reasoning_questions": "What is being asserted?",
            "step_context_queries": ["fact", "verdict"],
            "stage_action": "Parse and identify",
            "example_reasoning": "",
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Verify",
            "aim": "Verify the claim against given knowledge",
            "reasoning_questions": "Does the claim match known facts?",
            "step_context_queries": ["fact", "verification"],
            "stage_action": "Cross-check",
            "example_reasoning": "",
            "dependencies": [1],
            "retry_max": 3,
        },
    ],
}


# ---------------------------------------------------------------------------
# BenchmarkSynthesizer
# ---------------------------------------------------------------------------


class TestBenchmarkSynthesizer:
    async def test_llm_only_synthesis(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_resp(json.dumps(_benchmark_payload(4))))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        synth = BenchmarkSynthesizer(llm=llm, memory=None, web_research_agent=None)

        result = await synth.synthesize(
            "Verify a claim.",
            DomainAnalysis(domain="general", task_type="verification"),
            EvolveConfig(num_benchmark_tasks=4, enable_web_synthesis=False),
            cfg,
        )
        assert len(result.tasks) == 4
        assert all(t.input for t in result.tasks)
        # ids re-numbered deterministically
        assert [t.task_id for t in result.tasks] == [
            "synth_001",
            "synth_002",
            "synth_003",
            "synth_004",
        ]
        assert "llm" in result.sources_used

    async def test_user_provided_tasks_survive(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_resp(json.dumps({"tasks": []})))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        synth = BenchmarkSynthesizer(llm=llm, memory=None, web_research_agent=None)

        user_task = SynthesizedTask(
            task_id="user_1",
            input="user-supplied input",
            rubric=["must include 'sentinel'"],
            source="user",
        )
        result = await synth.synthesize(
            "q",
            DomainAnalysis(),
            EvolveConfig(num_benchmark_tasks=3, user_provided_tasks=[user_task]),
            cfg,
        )
        # user task survives even when LLM returns nothing
        assert any(t.input == "user-supplied input" for t in result.tasks)


# ---------------------------------------------------------------------------
# ChainGrader
# ---------------------------------------------------------------------------


class TestChainGrader:
    async def test_grade_combines_structural_and_judge(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_resp(json.dumps(_judge_payload(0.8, "looks fine"))))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        grader = ChainGrader(llm)

        benchmark = BenchmarkSynthesisResult(
            tasks=[
                SynthesizedTask(
                    task_id="t1",
                    input="x",
                    expected_output="y",
                    rubric=["fact verdict"],
                )
            ]
        )
        score = await grader.grade(SAMPLE_CHAIN, benchmark, EvolveConfig())
        assert 0.0 <= score.overall <= 1.0
        assert "t1" in score.per_task_scores

    async def test_real_execution_grading(self, mocker: Any) -> None:
        """When execution_model is set, grader runs the chain and judges the actual output."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # First call = chain execution (returns plain text), second = judge JSON.
        execution_response = _resp("The verdict is TRUE for fact X.")
        judge_response = _resp(json.dumps(_judge_payload(0.9, "matches expected")))
        instance.chat.completions.create = AsyncMock(side_effect=[execution_response, judge_response])
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        grader = ChainGrader(llm)

        benchmark = BenchmarkSynthesisResult(
            tasks=[
                SynthesizedTask(
                    task_id="t1",
                    input="Is fact X true?",
                    expected_output="TRUE",
                    rubric=["mentions TRUE", "references fact X"],
                )
            ]
        )
        evolve_cfg = EvolveConfig(
            use_structural_rubric=False,
            execution_model="gpt-4o-mini",
            execution_temperature=0.0,
        )
        score = await grader.grade(SAMPLE_CHAIN, benchmark, evolve_cfg)

        # Both calls were made: one for execution, one for judging.
        assert instance.chat.completions.create.await_count == 2
        # First call used the configured execution model.
        execution_call = instance.chat.completions.create.await_args_list[0]
        assert execution_call.kwargs["model"] == "gpt-4o-mini"
        # And the judge saw a high-confidence matching response.
        assert score.overall == pytest.approx(0.9, abs=0.01)

    async def test_real_execution_empty_output_scores_zero(self, mocker: Any) -> None:
        """Empty execution output → judge returns 0.0 without calling the LLM."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # Execution returns empty string; judge would error if called.
        empty_exec = _resp("")
        instance.chat.completions.create = AsyncMock(return_value=empty_exec)
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        grader = ChainGrader(llm)

        benchmark = BenchmarkSynthesisResult(tasks=[SynthesizedTask(task_id="t1", input="x", rubric=["x"])])
        evolve_cfg = EvolveConfig(
            use_structural_rubric=False,
            execution_model="gpt-4o-mini",
        )
        score = await grader.grade(SAMPLE_CHAIN, benchmark, evolve_cfg)
        assert score.per_task_scores["t1"] == 0.0

    async def test_judge_failure_does_not_crash(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # malformed JSON breaks the judge
        instance.chat.completions.create = AsyncMock(return_value=_resp("not-json"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="gpt-4o-test", api_key="k", enable_memory=False)
        llm = LLMClient(cfg)
        grader = ChainGrader(llm)

        benchmark = BenchmarkSynthesisResult(tasks=[SynthesizedTask(task_id="t1", input="x", rubric=["x"])])
        score = await grader.grade(SAMPLE_CHAIN, benchmark, EvolveConfig())
        # judge failed → judge contribution 0, but structural still scores
        assert score.overall >= 0.0


# ---------------------------------------------------------------------------
# ChainEvolver — end-to-end smoke test
# ---------------------------------------------------------------------------


class TestChainEvolverSmoke:
    async def test_evolve_runs_to_completion(self, mocker: Any) -> None:
        """Smoke test: evolve runs through Stage 1-5 with mocked LLM."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value

        # Build a queue of canned responses.
        # Order roughly: domain analysis → benchmark synthesis →
        # initial population (deep-mode chain generation × population_size) →
        # judge calls × tasks × population →
        # next-gen operator calls (mutate/insert/etc; some are LLM-driven repair) →
        # judge calls again.
        # Easiest: cycle a pool of "valid" responses for every kind of call.
        domain = json.dumps({"domain": "general", "task_type": "verification", "complexity": "simple"})
        benchmark = json.dumps(_benchmark_payload(2))
        judge = json.dumps(_judge_payload(0.6))
        repair = json.dumps(
            {"edit_type": "rewrite_step", "target_step_number": 1, "new_step": SAMPLE_CHAIN["steps"][0]}
        )
        plan = json.dumps(
            {
                "steps": [
                    {"number": 1, "title": "Read", "step_type": "llm", "rough_aim": "Read"},
                    {"number": 2, "title": "Verify", "step_type": "llm", "rough_aim": "Verify"},
                ],
                "rationale": "linear",
            }
        )
        dag = json.dumps(
            {
                "steps": [
                    {"number": 1, "title": "Read", "step_type": "llm", "dependencies": [], "rough_aim": "Read"},
                    {"number": 2, "title": "Verify", "step_type": "llm", "dependencies": [1], "rough_aim": "Verify"},
                ],
                "max_workers": 2,
                "pattern": "pipeline",
            }
        )
        described = json.dumps({"steps": SAMPLE_CHAIN["steps"]})

        # The mock side-effect returns the next response each call. We cycle
        # through a representative pool — order doesn't matter as long as
        # every call gets *some* parseable JSON.
        pool = [
            _resp(domain),  # domain analyzer
            _resp(benchmark),  # benchmark synthesis
            *[_resp(domain), _resp(plan), _resp(dag), _resp(described)] * 3,  # 3 initial chains × 4 deep-mode calls
            *[_resp(judge)] * 30,  # plenty of judge calls
            *[_resp(repair)] * 5,
            *[_resp(judge)] * 30,
        ]
        # Loop forever once exhausted, just in case.
        idx = {"i": 0}

        def _next(*args: Any, **kwargs: Any) -> Any:
            i = idx["i"]
            idx["i"] = i + 1
            return pool[i % len(pool)]

        instance.chat.completions.create = AsyncMock(side_effect=_next)
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        cfg = MAGEConfig(
            mode="deep",
            model="gpt-4o-test",
            api_key="k",
            enable_memory=False,
            enable_memory_research=False,
            enable_web_research=False,
            enable_topology_selection=False,  # keep the canned-response pool aligned (P1.15)
        )
        gen = MAGEGenerator(config=cfg)
        evolver = ChainEvolver(
            generator=gen,
            evolve_config=EvolveConfig(
                num_benchmark_tasks=2,
                population_size=3,
                num_generations=2,
                plateau_patience=10,  # don't early-stop
                save_benchmark_to_memory=False,
                enable_memory_synthesis=False,
                enable_web_synthesis=False,
            ),
            rng_seed=42,
        )
        result = await evolver.evolve("Verify a claim.")

        assert result.best_individual is not None
        assert result.benchmark.tasks
        assert len(result.history) >= 1
        assert result.total_individuals_evaluated > 0
        # generations actually ran
        assert all(g.population_size > 0 for g in result.history)
