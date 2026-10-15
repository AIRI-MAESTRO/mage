"""Tests for §4 P1 — per-stage public entrypoints on MAGEGenerator.

Covers the seven new public methods:
- ``analyze_domain``
- ``plan_steps``
- ``build_dag``
- ``describe_steps``
- ``critique_steps``
- ``verify_chain``
- ``refine``

Each test asserts:
1. The method calls the underlying agent exactly once with the expected
   arguments.
2. The method returns the agent's result unchanged (no wrapping).
3. Per-stage methods stand alone — no other pipeline stages fire.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.chain_verifier import VerificationResult
from mmar_mage.agents.self_refiner import RefinementResult
from mmar_mage.agents.step_critic import StepCritique, StepCritiqueResult
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import CARLStepSchema, DAGStructure, DomainAnalysis, MAGEConfig, StepPlan

from .conftest import (
    SAMPLE_CARL_CHAIN,
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# analyze_domain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_domain_calls_domain_analyzer_once(deep_config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    result = await gen.analyze_domain("Analyze a quarterly report")

    assert isinstance(result, DomainAnalysis)
    assert result.domain == "finance"
    assert result.task_type == "analysis"
    assert result.complexity == "medium"
    # Exactly ONE LLM call — no other pipeline stages ran.
    assert instance.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_analyze_domain_does_not_persist_anything(deep_config: MAGEConfig, mocker: Any) -> None:
    """Per-stage entrypoints are stateless — no chain save to memory."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_memory = False
    gen = MAGEGenerator(config=deep_config)
    # Spy on save_chain — should NOT be invoked by a per-stage entrypoint
    save_spy = mocker.spy(gen._memory, "save_chain")
    await gen.analyze_domain("anything")
    save_spy.assert_not_called()
    assert instance.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# plan_steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_steps_returns_step_plan(deep_config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_STEP_PLAN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    plan = await gen.plan_steps(
        "Analyze a quarterly report",
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
    )

    assert isinstance(plan, StepPlan)
    assert len(plan.steps) == 3
    assert instance.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_plan_steps_forwards_kwargs_to_planner(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_plan(self: Any, query: str, domain: Any, **kwargs: Any) -> Any:
        captured["query"] = query
        captured.update(kwargs)
        return StepPlan(steps=[], rationale="ok")

    monkeypatch.setattr("mmar_mage.agents.StepPlanner.plan", fake_plan)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    await gen.plan_steps(
        "Test",
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
        memory_digest="MEM",
        web_digest="WEB",
        capability_context="CAP",
        template_skeleton="TPL",
        skill_digest="SKILL",
        allowed_step_types=["llm", "tool"],
    )
    assert captured["memory_digest"] == "MEM"
    assert captured["web_digest"] == "WEB"
    assert captured["capability_context"] == "CAP"
    assert captured["template_skeleton"] == "TPL"
    assert captured["skill_digest"] == "SKILL"
    assert captured["allowed_step_types"] == ["llm", "tool"]


# ---------------------------------------------------------------------------
# build_dag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_dag_returns_dag_structure(deep_config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DAG)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    plan = StepPlan(**SAMPLE_STEP_PLAN)
    dag = await gen.build_dag(plan)
    assert isinstance(dag, DAGStructure)
    assert len(dag.steps) == 3
    assert instance.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# describe_steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_steps_returns_carl_steps(deep_config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    steps = await gen.describe_steps(
        "Analyze a quarterly report",
        DAGStructure(**SAMPLE_DAG),
    )
    assert isinstance(steps, list)
    assert len(steps) == 3
    assert all(isinstance(s, CARLStepSchema) for s in steps)


@pytest.mark.asyncio
async def test_describe_steps_fires_on_stage_progress(deep_config: MAGEConfig, mocker: Any) -> None:
    """The per-step progress callback fires when invoked standalone."""

    class _Cb:
        def __init__(self) -> None:
            self.progress: list[tuple[str, Any]] = []

        def on_stage_start(self, stage: str) -> None:
            pass

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

        def on_stage_progress(self, stage: str, artifact: Any) -> None:
            self.progress.append((stage, artifact.number))

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cb = _Cb()
    gen = MAGEGenerator(config=deep_config, progress=cb)  # type: ignore[arg-type]
    await gen.describe_steps("Q", DAGStructure(**SAMPLE_DAG))
    assert [p[0] for p in cb.progress] == ["step_describing"] * 3


# ---------------------------------------------------------------------------
# critique_steps
# ---------------------------------------------------------------------------


SAMPLE_CRITIQUE: dict[str, Any] = {
    "critiques": [
        {
            "step_number": 1,
            "scores": {
                "aim_clarity": 0.8,
                "reasoning_quality": 0.7,
                "context_relevance": 0.9,
                "dependency_logic": 0.8,
                "action_specificity": 0.7,
            },
            "overall_score": 0.78,
            "issues": [],
            "suggestions": [],
        },
        {
            "step_number": 2,
            "scores": {
                "aim_clarity": 0.4,
                "reasoning_quality": 0.5,
                "context_relevance": 0.5,
                "dependency_logic": 0.5,
                "action_specificity": 0.4,
            },
            "overall_score": 0.46,
            "issues": ["vague aim"],
            "suggestions": ["be specific"],
        },
        {
            "step_number": 3,
            "scores": {
                "aim_clarity": 0.9,
                "reasoning_quality": 0.9,
                "context_relevance": 0.9,
                "dependency_logic": 0.9,
                "action_specificity": 0.9,
            },
            "overall_score": 0.9,
            "issues": [],
            "suggestions": [],
        },
    ],
}


@pytest.mark.asyncio
async def test_critique_steps_returns_critique_result(deep_config: MAGEConfig, mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CRITIQUE)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    steps = [CARLStepSchema.model_validate(s) for s in SAMPLE_CARL_CHAIN["steps"]]
    result = await gen.critique_steps(
        "Q",
        steps,
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
        threshold=0.6,
    )
    assert isinstance(result, StepCritiqueResult)
    assert len(result.critiques) == 3
    # Step 2 (overall_score=0.46) falls below threshold 0.6
    assert 2 in result.steps_needing_revision
    assert 1 not in result.steps_needing_revision


@pytest.mark.asyncio
async def test_critique_steps_uses_config_threshold_when_omitted(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_critique(self: Any, *args: Any, **kwargs: Any) -> StepCritiqueResult:
        captured["threshold"] = kwargs.get("threshold")
        return StepCritiqueResult(critiques=[StepCritique(step_number=1)])

    monkeypatch.setattr("mmar_mage.agents.step_critic.StepCriticAgent.critique", fake_critique)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.step_critique_threshold = 0.42
    gen = MAGEGenerator(config=deep_config)
    await gen.critique_steps(
        "Q",
        [CARLStepSchema.model_validate(SAMPLE_CARL_CHAIN["steps"][0])],
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
    )
    assert captured["threshold"] == 0.42


# ---------------------------------------------------------------------------
# verify_chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_chain_returns_verification_result(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_verify(self: Any, *args: Any, **kwargs: Any) -> VerificationResult:
        return VerificationResult(chain_passed=True)

    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.verify", fake_verify)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    result = await gen.verify_chain(
        "Q",
        SAMPLE_CARL_CHAIN,
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
    )
    assert isinstance(result, VerificationResult)
    assert result.chain_passed is True


@pytest.mark.asyncio
async def test_verify_chain_uses_config_max_questions_when_omitted(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_verify(self: Any, *args: Any, **kwargs: Any) -> VerificationResult:
        captured["max_questions"] = kwargs.get("max_questions")
        return VerificationResult(chain_passed=True)

    monkeypatch.setattr("mmar_mage.agents.chain_verifier.ChainVerifier.verify", fake_verify)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.max_verification_questions = 7
    gen = MAGEGenerator(config=deep_config)
    await gen.verify_chain("Q", SAMPLE_CARL_CHAIN, DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS))
    assert captured["max_questions"] == 7


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refine_returns_refinement_result(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_refine(self: Any, *args: Any, **kwargs: Any) -> RefinementResult:
        return RefinementResult(
            chain_dict=SAMPLE_CARL_CHAIN,
            chain_json=json.dumps(SAMPLE_CARL_CHAIN),
            iterations_used=2,
            initial_quality=0.5,
            final_quality=0.85,
        )

    monkeypatch.setattr("mmar_mage.agents.self_refiner.SelfRefiner.refine", fake_refine)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    result = await gen.refine(
        "Q",
        SAMPLE_CARL_CHAIN,
        DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS),
    )
    assert isinstance(result, RefinementResult)
    assert result.iterations_used == 2
    assert result.final_quality > result.initial_quality


@pytest.mark.asyncio
async def test_refine_uses_config_defaults_when_omitted(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    async def fake_refine(self: Any, *args: Any, **kwargs: Any) -> RefinementResult:
        captured["max_iterations"] = kwargs.get("max_iterations")
        captured["quality_threshold"] = kwargs.get("quality_threshold")
        return RefinementResult(
            chain_dict=SAMPLE_CARL_CHAIN,
            chain_json="{}",
            iterations_used=0,
            initial_quality=0.0,
            final_quality=0.0,
        )

    monkeypatch.setattr("mmar_mage.agents.self_refiner.SelfRefiner.refine", fake_refine)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response("{}"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.max_refine_iterations = 5
    deep_config.refine_quality_threshold = 0.95
    gen = MAGEGenerator(config=deep_config)
    await gen.refine("Q", SAMPLE_CARL_CHAIN, DomainAnalysis(**SAMPLE_DOMAIN_ANALYSIS))
    assert captured["max_iterations"] == 5
    assert captured["quality_threshold"] == 0.95


# ---------------------------------------------------------------------------
# Per-stage chaining — equivalent to a full deep run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stages_compose_into_a_full_pipeline(deep_config: MAGEConfig, mocker: Any) -> None:
    """Calling the public methods in sequence reproduces the deep flow.

    CARE's regenerate-from-X feature relies on this composability:
    the caller can re-run from any stage by passing the prior stage's
    output into the next public method.
    """
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)

    domain = await gen.analyze_domain("Analyze a quarterly report")
    plan = await gen.plan_steps("Analyze a quarterly report", domain)
    dag = await gen.build_dag(plan)
    steps = await gen.describe_steps("Analyze a quarterly report", dag)

    assert isinstance(domain, DomainAnalysis)
    assert isinstance(plan, StepPlan)
    assert isinstance(dag, DAGStructure)
    assert len(steps) == 3
    # 4 LLM calls — one per stage.
    assert instance.chat.completions.create.call_count == 4
