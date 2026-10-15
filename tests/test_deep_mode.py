"""Tests for deep-mode chain generation and individual agents."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents import DAGBuilder, DomainAnalyzer, StepDescriber, StepPlanner
from mmar_mage.generator import MAGEGenerator
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import (
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    StepPlan,
)

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# Individual agent tests
# ---------------------------------------------------------------------------


def _make_llm_client(mocker: Any, response_json: dict[str, Any]) -> LLMClient:
    """Create an LLMClient with a mocked AsyncOpenAI backend."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(response_json)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    config = MAGEConfig(api_key="test-key", max_retries=0)
    return LLMClient(config)


@pytest.mark.asyncio
async def test_domain_analyzer(mocker: Any) -> None:
    """DomainAnalyzer parses LLM output into DomainAnalysis."""
    llm = _make_llm_client(mocker, SAMPLE_DOMAIN_ANALYSIS)
    analyzer = DomainAnalyzer(llm)
    result = await analyzer.analyze("Analyze a financial report")

    assert isinstance(result, DomainAnalysis)
    assert result.domain == "finance"
    assert result.complexity == "medium"
    assert result.suggested_step_count == 3


@pytest.mark.asyncio
async def test_step_planner(mocker: Any) -> None:
    """StepPlanner produces a StepPlan from domain analysis."""
    llm = _make_llm_client(mocker, SAMPLE_STEP_PLAN)
    planner = StepPlanner(llm)
    domain = DomainAnalysis.model_validate(SAMPLE_DOMAIN_ANALYSIS)
    result = await planner.plan("Analyze report", domain)

    assert isinstance(result, StepPlan)
    assert len(result.steps) == 3
    assert result.steps[0].title == "Analyze structure"


@pytest.mark.asyncio
async def test_step_planner_with_digests(mocker: Any) -> None:
    """StepPlanner accepts memory and web digests."""
    llm = _make_llm_client(mocker, SAMPLE_STEP_PLAN)
    planner = StepPlanner(llm)
    domain = DomainAnalysis.model_validate(SAMPLE_DOMAIN_ANALYSIS)
    result = await planner.plan(
        "Analyze report",
        domain,
        memory_digest="Past chains used funnel patterns for finance",
        web_digest="Financial analysis best practices suggest...",
    )

    assert isinstance(result, StepPlan)
    assert len(result.steps) == 3


@pytest.mark.asyncio
async def test_dag_builder(mocker: Any) -> None:
    """DAGBuilder produces a DAGStructure from a step plan."""
    llm = _make_llm_client(mocker, SAMPLE_DAG)
    builder = DAGBuilder(llm)
    plan = StepPlan.model_validate(SAMPLE_STEP_PLAN)
    result = await builder.build(plan)

    assert isinstance(result, DAGStructure)
    assert result.pattern == "funnel"
    assert result.max_workers == 2
    assert result.steps[2].dependencies == [1, 2]


@pytest.mark.asyncio
async def test_step_describer(mocker: Any) -> None:
    """StepDescriber generates CARLStepSchema list."""
    llm = _make_llm_client(mocker, SAMPLE_DESCRIBED_STEPS)
    describer = StepDescriber(llm)
    dag = DAGStructure.model_validate(SAMPLE_DAG)
    result = await describer.describe("Analyze report", dag)

    assert len(result) == 3
    assert result[0].aim == "Determine the type of document"
    assert result[2].dependencies == [1, 2]


# ---------------------------------------------------------------------------
# Full deep pipeline test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_mode_full_pipeline(
    deep_config: MAGEConfig,
    mock_openai_deep: MagicMock,
) -> None:
    """Deep mode runs all 4 agents and produces a valid chain."""
    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a financial report")

    assert result.mode == "deep"
    assert result.metadata.num_steps == 3
    assert result.metadata.domain == "finance"
    assert result.metadata.deep_stages_completed is not None
    assert "domain_analysis" in result.metadata.deep_stages_completed
    assert "step_planning" in result.metadata.deep_stages_completed
    assert "dag_building" in result.metadata.deep_stages_completed
    assert "step_describing" in result.metadata.deep_stages_completed

    chain = json.loads(result.chain_json)
    assert len(chain["steps"]) == 3


@pytest.mark.asyncio
async def test_deep_mode_with_disabled_stages(mocker: Any) -> None:
    """Deep mode with some agents disabled uses fallbacks."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    # Only domain_analysis and step_describing need LLM calls
    responses = [
        _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
        _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
    ]
    instance.chat.completions.create = AsyncMock(side_effect=responses)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config = MAGEConfig(
        mode="deep",
        api_key="test-key",
        enable_memory=False,
        enable_domain_analysis=True,
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=True,
        enable_topology_selection=False,  # P1.15 stage off — this test targets planning/dag fallbacks
        max_retries=0,
    )

    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test query")

    assert result.mode == "deep"
    assert "domain_analysis" in result.metadata.deep_stages_completed
    assert "step_planning" not in result.metadata.deep_stages_completed
    assert "dag_building" not in result.metadata.deep_stages_completed


@pytest.mark.asyncio
async def test_deep_mode_all_disabled(mocker: Any) -> None:
    """Deep mode with ALL agents disabled uses only fallbacks (no LLM calls)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    # No LLM calls expected
    instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("Should not be called"))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config = MAGEConfig(
        mode="deep",
        api_key="test-key",
        enable_memory=False,
        enable_domain_analysis=False,
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,  # P1.15 stage off — "all disabled" includes topology
        max_retries=0,
    )

    gen = MAGEGenerator(config=config)
    result = await gen.generate("Fallback test")

    assert result.mode == "deep"
    assert result.metadata.deep_stages_completed == []
    # DomainAnalysis default suggested_step_count is 3, biased to 2 by the default
    # simplicity_bias=0.5 (P1.10): round(3 * 0.5)=2, floored at the 2-step minimum.
    assert result.metadata.num_steps == 2


@pytest.mark.asyncio
async def test_deep_mode_progress_callback(
    deep_config: MAGEConfig,
    mock_openai_deep: MagicMock,
) -> None:
    """Progress callback is invoked for each stage."""
    events: list[tuple[str, str]] = []

    class TestCallback:
        def on_stage_start(self, stage: str) -> None:
            events.append(("start", stage))

        def on_stage_complete(self, stage: str, result: Any) -> None:
            events.append(("complete", stage))

        def on_error(self, stage: str, error: Exception) -> None:
            events.append(("error", stage))

    gen = MAGEGenerator(config=deep_config, progress=TestCallback())
    await gen.generate("Test")

    stage_names = [e[1] for e in events if e[0] == "start"]
    assert "domain_analysis" in stage_names
    assert "step_planning" in stage_names
    assert "dag_building" in stage_names
    assert "step_describing" in stage_names
