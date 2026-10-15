"""Tests for §4 P1 — `MAGEResult.replay_from`.

Covers:
- Re-running from each of the 4 named stages (`domain_analysis`,
  `step_planning`, `dag_building`, `step_describing`) with cached
  artifacts in `intermediate_artifacts`
- `partial_state` override at the replay point (e.g. edited plan)
- Helper reconstruction functions accept dump dicts AND typed instances
- Query recovery via `chain_dict.metadata.task_description` /
  `intermediate_artifacts.domain_analysis.task_type` / explicit kwarg
- Error cases: unknown stage, missing _config, missing prior artifacts
- New MAGEResult preserves `suggested_*` metadata + memory_key
- `deep_stages_completed` records the replay stage label
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.schemas import (
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    MAGEMetadata,
    MAGEResult,
    StepPlan,
    _reconstruct_dag,
    _reconstruct_domain_analysis,
    _reconstruct_step_plan,
    _recover_query_from_result,
)

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


class TestReconstructionHelpers:
    def test_domain_analysis_from_dict(self) -> None:
        da = _reconstruct_domain_analysis(SAMPLE_DOMAIN_ANALYSIS)
        assert isinstance(da, DomainAnalysis)
        assert da.domain == "finance"

    def test_domain_analysis_from_instance(self) -> None:
        instance = DomainAnalysis(domain="legal")
        out = _reconstruct_domain_analysis(instance)
        assert out is instance

    def test_domain_analysis_none(self) -> None:
        assert _reconstruct_domain_analysis(None) is None

    def test_step_plan_from_dict(self) -> None:
        plan = _reconstruct_step_plan(SAMPLE_STEP_PLAN)
        assert isinstance(plan, StepPlan)
        assert len(plan.steps) == 3

    def test_dag_from_dict(self) -> None:
        dag = _reconstruct_dag(SAMPLE_DAG)
        assert isinstance(dag, DAGStructure)
        assert dag.max_workers == 2


class TestRecoverQuery:
    def test_recovers_from_metadata_task_description(self) -> None:
        result = MAGEResult(
            chain_json="{}",
            chain_dict={"steps": [], "metadata": {"task_description": "Analyze X"}},
        )
        assert _recover_query_from_result(result) == "Analyze X"

    def test_falls_back_to_task_type(self) -> None:
        result = MAGEResult(
            chain_json="{}",
            chain_dict={"steps": []},
            intermediate_artifacts={"domain_analysis": {"task_type": "analysis"}},
        )
        assert _recover_query_from_result(result) == "analysis"

    def test_empty_when_nothing_available(self) -> None:
        result = MAGEResult(chain_json="{}", chain_dict={"steps": []})
        assert _recover_query_from_result(result) == ""


# ---------------------------------------------------------------------------
# replay_from — error paths
# ---------------------------------------------------------------------------


class TestReplayFromErrors:
    @pytest.mark.asyncio
    async def test_unknown_stage_raises(self) -> None:
        result = MAGEResult(chain_json="{}")
        result._config = MAGEConfig(model="m", api_key="x")
        with pytest.raises(ValueError, match="Unknown replay stage"):
            await result.replay_from("not_a_stage", query="x")

    @pytest.mark.asyncio
    async def test_missing_config_raises(self) -> None:
        result = MAGEResult(chain_json="{}")
        # _config defaults to None
        with pytest.raises(ValueError, match="no _config"):
            await result.replay_from("domain_analysis", query="x")

    @pytest.mark.asyncio
    async def test_missing_cached_artifact_raises(self) -> None:
        """replay_from('dag_building') without a cached step_plan raises."""
        result = MAGEResult(
            chain_json="{}",
            intermediate_artifacts={"domain_analysis": SAMPLE_DOMAIN_ANALYSIS},
        )
        result._config = MAGEConfig(model="m", api_key="x")
        with pytest.raises(ValueError, match="step_plan"):
            await result.replay_from("dag_building", query="x")

    @pytest.mark.asyncio
    async def test_missing_query_raises(self) -> None:
        result = MAGEResult(chain_json="{}", chain_dict={"steps": []})
        result._config = MAGEConfig(model="m", api_key="x")
        with pytest.raises(ValueError, match="no query"):
            await result.replay_from("domain_analysis")


# ---------------------------------------------------------------------------
# replay_from — happy path per stage
# ---------------------------------------------------------------------------


def _prior_result_with_artifacts(*, include: tuple[str, ...] = ("domain_analysis", "step_plan", "dag")) -> MAGEResult:
    """Build a `MAGEResult` simulating a prior deep-mode run."""
    artifacts: dict[str, Any] = {}
    if "domain_analysis" in include:
        artifacts["domain_analysis"] = dict(SAMPLE_DOMAIN_ANALYSIS)
    if "step_plan" in include:
        artifacts["step_plan"] = dict(SAMPLE_STEP_PLAN)
    if "dag" in include:
        artifacts["dag"] = dict(SAMPLE_DAG)

    result = MAGEResult(
        chain_json="{}",
        chain_dict={
            "steps": [],
            "metadata": {"task_description": "Analyze a quarterly report"},
        },
        memory_key="prior-chain-id",
        mode="deep",
        metadata=MAGEMetadata(
            suggested_display_name="Quarterly Analyst",
            suggested_description="Reads quarterly reports.",
            suggested_tags=["finance"],
        ),
        intermediate_artifacts=artifacts,
    )
    result._config = MAGEConfig(
        mode="deep",
        model="gpt-4o-test",
        api_key="test",
        enable_memory=False,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    return result


@pytest.mark.asyncio
async def test_replay_from_step_describing_only_calls_one_agent(mocker: Any) -> None:
    """Replay from step_describing → only describe_steps fires (1 LLM call)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts()
    new = await prior.replay_from("step_describing")

    assert instance.chat.completions.create.call_count == 1
    assert new.metadata.num_steps == 3
    assert new.metadata.deep_stages_completed == ["step_describing_replay"]
    # suggested_* fields preserved from prior run
    assert new.metadata.suggested_display_name == "Quarterly Analyst"
    # memory_key carried over so submit_feedback lands on the right entity
    assert new.memory_key == "prior-chain-id"


@pytest.mark.asyncio
async def test_replay_from_dag_building_runs_dag_and_describe(mocker: Any) -> None:
    """Replay from dag_building → 2 LLM calls (dag + describe)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts(include=("domain_analysis", "step_plan"))
    new = await prior.replay_from("dag_building")
    assert instance.chat.completions.create.call_count == 2
    assert new.metadata.deep_stages_completed == ["dag_building_replay"]


@pytest.mark.asyncio
async def test_replay_from_step_planning_runs_plan_dag_describe(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts(include=("domain_analysis",))
    new = await prior.replay_from("step_planning")
    assert instance.chat.completions.create.call_count == 3
    assert new.metadata.deep_stages_completed == ["step_planning_replay"]


@pytest.mark.asyncio
async def test_replay_from_domain_analysis_runs_full_pipeline(mocker: Any) -> None:
    """Replay from the very start → 4 LLM calls."""
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

    # No cached artifacts at all — must rerun every stage
    result = MAGEResult(chain_json="{}")
    result._config = MAGEConfig(
        mode="deep",
        model="m",
        api_key="x",
        enable_memory=False,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    new = await result.replay_from("domain_analysis", query="Analyze quarterly")
    assert instance.chat.completions.create.call_count == 4
    assert new.metadata.deep_stages_completed == ["domain_analysis_replay"]


# ---------------------------------------------------------------------------
# replay_from — partial_state override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_state_override_at_dag_building(mocker: Any) -> None:
    """Caller can edit the plan via partial_state and replay dag onward."""
    captured: dict[str, Any] = {}

    async def fake_build(self: Any, plan: StepPlan) -> DAGStructure:
        captured["received_plan_rationale"] = plan.rationale
        return DAGStructure.model_validate(SAMPLE_DAG)

    mocker.patch("mmar_mage.agents.DAGBuilder.build", fake_build)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts(include=("domain_analysis", "step_plan"))

    # User edited the plan in CARE — rationale tweaked
    edited_plan = dict(SAMPLE_STEP_PLAN)
    edited_plan["rationale"] = "Edited in CARE TUI"

    new = await prior.replay_from("dag_building", partial_state={"step_plan": edited_plan})

    # The edited plan reached the DAG builder
    assert captured["received_plan_rationale"] == "Edited in CARE TUI"
    assert new.metadata.num_steps == 3


# ---------------------------------------------------------------------------
# replay_from — query recovery from chain_dict.metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_recovers_query_from_chain_metadata(mocker: Any) -> None:
    """Query is recovered from `chain_dict.metadata.task_description`
    when not passed explicitly."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts()
    # No explicit query → recovers from chain_dict.metadata.task_description
    new = await prior.replay_from("step_describing")
    # Chain name encodes the recovered query as a slice
    assert "Analyze a quarterly report" in new.chain_dict["name"]


@pytest.mark.asyncio
async def test_replay_explicit_query_wins_over_recovery(mocker: Any) -> None:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    prior = _prior_result_with_artifacts()
    new = await prior.replay_from("step_describing", query="OVERRIDDEN QUERY")
    assert "OVERRIDDEN QUERY" in new.chain_dict["name"]
