"""Simplicity bias + bounded operator vocabulary — plan item P1.10 (AFlow-style).

Two levers, both tested here:
- ``_apply_simplicity_bias`` scales the suggested step count by ``(1 - bias)``,
  floored so a chain never collapses below a 2-step minimum.
- ``_enforce_step_budget`` truncates an unjustified plan back to the cap, but
  leaves a plan with a filled rationale intact.

The real-execution check drives ``generate()`` on the fallback path (which uses
``suggested_step_count`` directly) and shows the config lever changing the
produced chain's step count.
"""

from __future__ import annotations

from mmar_mage.generator import (
    MAGEGenerator,
    _apply_simplicity_bias,
    _enforce_step_budget,
)
from mmar_mage.schemas import MAGEConfig, PlannedStep, StepPlan


def _plan(n: int, rationale: str = "") -> StepPlan:
    return StepPlan(
        steps=[PlannedStep(number=i + 1, title=f"S{i + 1}", step_type="llm", rough_aim="x") for i in range(n)],
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# _apply_simplicity_bias
# --------------------------------------------------------------------------- #
def test_bias_halves_large_count():
    assert _apply_simplicity_bias(8, 0.5) == 4


def test_bias_rounds_and_floors_small_count():
    # round(3 * 0.5) = 2, which is also the floor
    assert _apply_simplicity_bias(3, 0.5) == 2


def test_bias_zero_is_noop():
    assert _apply_simplicity_bias(6, 0.0) == 6


def test_bias_never_collapses_below_floor():
    # full bias would give 0, but the 2-step floor holds
    assert _apply_simplicity_bias(10, 1.0) == 2


def test_count_at_or_below_floor_unchanged():
    assert _apply_simplicity_bias(2, 0.5) == 2
    assert _apply_simplicity_bias(1, 0.9) == 1


def test_bias_above_one_is_clamped():
    assert _apply_simplicity_bias(9, 5.0) == 2  # clamped to 1.0 → floored


# --------------------------------------------------------------------------- #
# _enforce_step_budget
# --------------------------------------------------------------------------- #
def test_over_budget_without_rationale_is_truncated():
    out = _enforce_step_budget(_plan(9), max_steps=7)
    assert len(out.steps) == 7
    assert [s.number for s in out.steps] == list(range(1, 8))  # renumbered


def test_over_budget_with_rationale_is_kept():
    out = _enforce_step_budget(_plan(9, rationale="multi-hop: 3 sources + verify + synth"), max_steps=7)
    assert len(out.steps) == 9  # justified → intact


def test_under_budget_is_kept():
    out = _enforce_step_budget(_plan(4), max_steps=7)
    assert len(out.steps) == 4


def test_cap_zero_disables_enforcement():
    out = _enforce_step_budget(_plan(12), max_steps=0)
    assert len(out.steps) == 12


# --------------------------------------------------------------------------- #
# Real-execution scenario — generate() fallback path honours the config lever
# --------------------------------------------------------------------------- #
def _fallback_config(**cfg) -> MAGEConfig:
    return MAGEConfig(
        mode="deep",
        api_key="test-key",
        enable_memory=False,
        enable_domain_analysis=False,
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,  # keep the fallback path hermetic (no LLM calls)
        max_retries=0,
        **cfg,
    )


async def test_generate_applies_default_simplicity_bias():
    # DomainAnalysis default suggested_step_count=3 → biased to 2 by default 0.5
    gen = MAGEGenerator(config=_fallback_config())
    result = await gen.generate("simplicity default")
    assert result.metadata.num_steps == 2


async def test_generate_with_bias_disabled_keeps_full_count():
    gen = MAGEGenerator(config=_fallback_config(simplicity_bias=0.0))
    result = await gen.generate("simplicity off")
    assert result.metadata.num_steps == 3  # unbiased default suggested_step_count
