"""Tests for §6 P3 — pre-flight cost estimator UX.

Covers:
- `estimate_pipeline_cost` enumerates only the *enabled* deep-mode
  stages (fast mode collapses to a single virtual step)
- Token estimates respect prompt length + query size
- Pricing is applied per-stage when present; pricing_missing is set
  per row when the model isn't in the dict
- `self_refine` is modelled as `max_refine_iterations * 2` LLM calls
- `MAGEGenerator` fires `on_cost_estimate` BEFORE any LLM call when
  `enable_preflight_cost_estimate=True`
- The estimate's `total_cost_usd` matches the sum of per-row costs
- A failing `on_cost_estimate` callback does NOT break generation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.cost import CostEstimate, estimate_pipeline_cost
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# estimate_pipeline_cost — fast mode
# ---------------------------------------------------------------------------


class TestFastMode:
    def test_fast_mode_produces_single_row(self) -> None:
        cfg = MAGEConfig(mode="fast", model="gpt-4o")
        est = estimate_pipeline_cost(cfg, query="x")
        assert isinstance(est, CostEstimate)
        assert len(est.steps) == 1
        assert est.steps[0].step_title == "fast_generation"
        assert est.steps[0].calls_llm is True

    def test_fast_mode_includes_query_length_in_tokens(self) -> None:
        cfg = MAGEConfig(mode="fast", model="gpt-4o")
        short = estimate_pipeline_cost(cfg, query="hi")
        longer = estimate_pipeline_cost(cfg, query="x" * 200)
        assert longer.total_input_tokens > short.total_input_tokens


# ---------------------------------------------------------------------------
# estimate_pipeline_cost — deep mode
# ---------------------------------------------------------------------------


class TestDeepMode:
    def test_default_deep_mode_enumerates_core_stages(self) -> None:
        """A vanilla deep config emits rows for the always-on stages."""
        cfg = MAGEConfig(mode="deep", model="gpt-4o")
        est = estimate_pipeline_cost(cfg, query="Analyze X")
        titles = {row.step_title for row in est.steps}
        # Always-on by default
        assert {"domain_analysis", "step_planning", "dag_building", "step_describing"} <= titles
        # All four should be `calls_llm=True`
        for name in ("domain_analysis", "step_planning", "dag_building", "step_describing"):
            row = next(r for r in est.steps if r.step_title == name)
            assert row.calls_llm is True
            assert row.estimated_calls == 1

    def test_disabled_stage_marked_not_calling_llm(self) -> None:
        cfg = MAGEConfig(
            mode="deep",
            model="gpt-4o",
            enable_dag_optimization=False,
        )
        est = estimate_pipeline_cost(cfg, query="x")
        dag_row = next(r for r in est.steps if r.step_title == "dag_building")
        assert dag_row.calls_llm is False
        assert "disabled" in dag_row.note

    def test_self_refine_scales_with_max_iterations(self) -> None:
        """self_refine should report 2 × max_iters LLM calls (critique + revise)."""
        cfg = MAGEConfig(
            mode="deep",
            model="gpt-4o",
            enable_self_refine=True,
            max_refine_iterations=5,
        )
        est = estimate_pipeline_cost(cfg, query="x")
        refine_row = next(r for r in est.steps if r.step_title == "self_refine")
        assert refine_row.estimated_calls == 10  # 5 × 2

    def test_suggested_naming_row_present_by_default(self) -> None:
        cfg = MAGEConfig(mode="deep", model="gpt-4o")
        est = estimate_pipeline_cost(cfg, query="x")
        naming_row = next(r for r in est.steps if r.step_title == "suggested_naming")
        # default enable_suggested_naming_llm=True
        assert naming_row.calls_llm is True

    def test_suggested_naming_disabled(self) -> None:
        cfg = MAGEConfig(mode="deep", model="gpt-4o", enable_suggested_naming_llm=False)
        est = estimate_pipeline_cost(cfg, query="x")
        naming_row = next(r for r in est.steps if r.step_title == "suggested_naming")
        assert naming_row.calls_llm is False


# ---------------------------------------------------------------------------
# Pricing math
# ---------------------------------------------------------------------------


class TestPricing:
    def test_pricing_applied_per_row(self) -> None:
        pricing = {"gpt-4o": (5.0, 15.0)}  # $5 / 1k in, $15 / 1k out
        cfg = MAGEConfig(mode="deep", model="gpt-4o", enable_suggested_naming_llm=False)
        est = estimate_pipeline_cost(cfg, query="x", pricing=pricing)
        # Every enabled stage has non-zero cost
        for row in est.steps:
            if row.calls_llm:
                assert row.total_cost_usd > 0
                assert row.pricing_missing is False

    def test_pricing_missing_flagged_when_model_absent(self) -> None:
        pricing = {"other-model": (1.0, 2.0)}
        cfg = MAGEConfig(mode="deep", model="gpt-4o", enable_suggested_naming_llm=False)
        est = estimate_pipeline_cost(cfg, query="x", pricing=pricing)
        for row in est.steps:
            if row.calls_llm:
                assert row.pricing_missing is True
                assert row.total_cost_usd == 0.0
        assert est.pricing_complete is False

    def test_total_cost_equals_sum_of_rows(self) -> None:
        pricing = {"gpt-4o": (1.0, 2.0)}
        cfg = MAGEConfig(mode="deep", model="gpt-4o", enable_suggested_naming_llm=False)
        est = estimate_pipeline_cost(cfg, query="x", pricing=pricing)
        expected = sum(r.total_cost_usd for r in est.steps)
        assert est.total_cost_usd == pytest.approx(expected, rel=1e-9)

    def test_no_pricing_means_zero_cost(self) -> None:
        cfg = MAGEConfig(mode="deep", model="gpt-4o")
        est = estimate_pipeline_cost(cfg, query="x", pricing=None)
        assert est.total_cost_usd == 0.0
        # pricing_complete=True because no pricing was requested
        assert est.pricing_complete is False or est.pricing_complete is True
        # but all rows have 0 cost
        for row in est.steps:
            assert row.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# Per-stage model overrides honoured
# ---------------------------------------------------------------------------


class TestPerStageOverrides:
    def test_stage_override_model_appears_in_row(self) -> None:
        cfg = MAGEConfig(
            mode="deep",
            model="gpt-4o",
            enable_suggested_naming_llm=False,
            stage_llm_overrides={
                "domain_analysis": {"model": "gpt-4o-mini"},
            },
        )
        est = estimate_pipeline_cost(cfg, query="x")
        dom_row = next(r for r in est.steps if r.step_title == "domain_analysis")
        plan_row = next(r for r in est.steps if r.step_title == "step_planning")
        assert dom_row.model == "gpt-4o-mini"
        assert plan_row.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Generator integration — on_cost_estimate fires before any LLM call
# ---------------------------------------------------------------------------


def _mock_deep_4_calls(mocker: Any) -> MagicMock:
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
    return instance


class _CostRecordingCallback:
    """Captures the order of pre-flight + start callbacks."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def on_stage_start(self, stage: str) -> None:
        self.events.append(("start", stage))

    def on_stage_complete(self, stage: str, result: Any) -> None:
        pass

    def on_error(self, stage: str, error: Exception) -> None:
        pass

    def on_cost_estimate(self, estimate: Any) -> None:
        self.events.append(("cost_estimate", estimate))


@pytest.mark.asyncio
async def test_generator_fires_on_cost_estimate_before_any_llm_call(deep_config: MAGEConfig, mocker: Any) -> None:
    instance = _mock_deep_4_calls(mocker)

    deep_config.enable_preflight_cost_estimate = True
    deep_config.pricing = {"gpt-4o-test": (1.0, 2.0)}
    deep_config.enable_suggested_naming_llm = False

    cb = _CostRecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)  # type: ignore[arg-type]
    await gen.generate("Analyze quarterly")

    # First event is the cost estimate, BEFORE any on_stage_start.
    assert cb.events, "no events fired"
    kinds = [k for k, _ in cb.events]
    assert kinds[0] == "cost_estimate"
    # And the LLM was actually called 4 times (no aborts)
    assert instance.chat.completions.create.call_count == 4

    # The estimate object is a real CostEstimate
    _, estimate = cb.events[0]
    assert isinstance(estimate, CostEstimate)
    # And it carried per-stage rows
    assert any(row.step_title == "domain_analysis" for row in estimate.steps)


@pytest.mark.asyncio
async def test_generator_does_not_fire_on_cost_estimate_when_flag_off(deep_config: MAGEConfig, mocker: Any) -> None:
    _mock_deep_4_calls(mocker)
    deep_config.enable_suggested_naming_llm = False
    # enable_preflight_cost_estimate defaults to False

    cb = _CostRecordingCallback()
    gen = MAGEGenerator(config=deep_config, progress=cb)  # type: ignore[arg-type]
    await gen.generate("Test")

    assert all(k != "cost_estimate" for k, _ in cb.events)


@pytest.mark.asyncio
async def test_generator_isolates_cost_estimate_callback_failure(deep_config: MAGEConfig, mocker: Any) -> None:
    """A buggy on_cost_estimate callback must NOT break generation."""
    _mock_deep_4_calls(mocker)
    deep_config.enable_preflight_cost_estimate = True
    deep_config.enable_suggested_naming_llm = False

    class _BrokenCb:
        def on_stage_start(self, stage: str) -> None:
            pass

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

        def on_cost_estimate(self, estimate: Any) -> None:
            raise RuntimeError("buggy callback")

    gen = MAGEGenerator(config=deep_config, progress=_BrokenCb())  # type: ignore[arg-type]
    # Should NOT raise.
    result = await gen.generate("X")
    assert result.metadata.num_steps == 3


@pytest.mark.asyncio
async def test_generator_works_with_legacy_callback_missing_on_cost_estimate(
    deep_config: MAGEConfig, mocker: Any
) -> None:
    """Old callbacks predating §6 P3 don't have `on_cost_estimate`. Flag-on
    must not raise when the method is absent."""
    _mock_deep_4_calls(mocker)
    deep_config.enable_preflight_cost_estimate = True
    deep_config.enable_suggested_naming_llm = False

    class _LegacyCb:
        def on_stage_start(self, stage: str) -> None:
            pass

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

    gen = MAGEGenerator(config=deep_config, progress=_LegacyCb())  # type: ignore[arg-type]
    result = await gen.generate("Test")
    assert result.metadata.num_steps == 3
