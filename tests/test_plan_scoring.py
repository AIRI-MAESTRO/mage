"""Pairwise cold-start plan scoring + research-aware candidate default (plan item P1.5).

``MAGEGenerator._score_plans`` selects the best cold-start plan candidate via a
single-elimination *pairwise* tournament (more reliable than absolute multi-way
scoring). ``MAGEConfig`` bumps ``cold_start_candidates`` 1→3 when memory research
is enabled. These tests use a directly-mocked LLM (no real calls).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig, PlannedStep, StepPlan


def _gen(generate_json_return=None, generate_json_side_effect=None) -> MAGEGenerator:
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", mode="deep"))
    llm = MagicMock()
    if generate_json_side_effect is not None:
        llm.generate_json = AsyncMock(side_effect=generate_json_side_effect)
    else:
        llm.generate_json = AsyncMock(return_value=generate_json_return)
    gen._llm = llm
    return gen


def _plan(tag: str) -> StepPlan:
    return StepPlan(steps=[PlannedStep(number=1, title=tag)], rationale=tag)


def _domain():
    from mmar_mage.schemas import DomainAnalysis

    return DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")


# --------------------------------------------------------------------------- #
# _pick_better_plan
# --------------------------------------------------------------------------- #
async def test_pick_returns_b_when_winner_b():
    gen = _gen({"winner": "B"})
    out = await gen._pick_better_plan("q", _domain(), _plan("A"), _plan("B"))
    assert out.rationale == "B"


async def test_pick_returns_a_when_winner_a():
    gen = _gen({"winner": "A"})
    out = await gen._pick_better_plan("q", _domain(), _plan("A"), _plan("B"))
    assert out.rationale == "A"


async def test_pick_defaults_to_incumbent_on_missing_winner():
    gen = _gen({})  # no "winner" key
    out = await gen._pick_better_plan("q", _domain(), _plan("A"), _plan("B"))
    assert out.rationale == "A"


async def test_pick_defaults_to_incumbent_on_llm_error():
    gen = _gen(generate_json_side_effect=RuntimeError("judge down"))
    out = await gen._pick_better_plan("q", _domain(), _plan("A"), _plan("B"))
    assert out.rationale == "A"  # non-fatal: keep incumbent


# --------------------------------------------------------------------------- #
# _score_plans tournament
# --------------------------------------------------------------------------- #
async def test_score_single_candidate_returns_it_without_llm():
    gen = _gen({"winner": "B"})
    out = await gen._score_plans("q", _domain(), [_plan("only")])
    assert out.rationale == "only"
    gen._llm.generate_json.assert_not_called()  # no comparison needed


async def test_score_runs_pairwise_tournament():
    # 3 candidates → 2 comparisons; "B" wins each → last candidate advances
    gen = _gen(generate_json_side_effect=[{"winner": "B"}, {"winner": "B"}])
    out = await gen._score_plans("q", _domain(), [_plan("c0"), _plan("c1"), _plan("c2")])
    assert out.rationale == "c2"
    assert gen._llm.generate_json.await_count == 2  # N-1 head-to-head comparisons


async def test_score_incumbent_can_win_whole_tournament():
    gen = _gen(generate_json_side_effect=[{"winner": "A"}, {"winner": "A"}])
    out = await gen._score_plans("q", _domain(), [_plan("c0"), _plan("c1"), _plan("c2")])
    assert out.rationale == "c0"  # incumbent beat both challengers


# --------------------------------------------------------------------------- #
# cold_start_candidates research-aware default
# --------------------------------------------------------------------------- #
def test_cold_start_default_is_one_without_research():
    assert MAGEConfig().cold_start_candidates == 1


def test_cold_start_bumped_to_three_with_research():
    assert MAGEConfig(enable_memory_research=True).cold_start_candidates == 3


def test_explicit_cold_start_value_respected_with_research():
    cfg = MAGEConfig(enable_memory_research=True, cold_start_candidates=5)
    assert cfg.cold_start_candidates == 5
