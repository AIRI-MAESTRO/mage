"""Parallel topology sampling — plan item P1.15 part 4 (deferred follow-up).

When the TopologySelectorAgent returns >1 distinct topology and
``enable_parallel_topology_sampling`` is on, the generator fans out one planning
pass per topology and keeps the best via the P1.5 pairwise judge — exploring
diverse structures rather than only the selector's top pick. Off by default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig, PlannedStep, StepPlan


def _plan(title: str) -> StepPlan:
    return StepPlan(steps=[PlannedStep(number=1, title=title, step_type="llm", rough_aim="x")], rationale=title)


def _gen(*, parallel: bool, json_side: list[Any], structured_side: list[Any]) -> MAGEGenerator:
    cfg = MAGEConfig(
        api_key="t",
        mode="deep",
        enable_memory=False,
        enable_domain_analysis=False,  # default domain, no LLM call
        enable_topology_selection=True,
        enable_parallel_topology_sampling=parallel,
        enable_step_planning=True,
        enable_dag_optimization=False,  # fallback dag → no extra generate_structured
        enable_step_descriptions=False,  # fallback descriptions
        simplicity_bias=0.0,
    )
    gen = MAGEGenerator(config=cfg)
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate_json = AsyncMock(side_effect=json_side)
    gen._llm.generate_structured = AsyncMock(side_effect=structured_side)
    return gen


# two distinct topologies from the selector
_TWO_TOPOLOGIES = {"candidates": [{"name": "diamond"}, {"name": "debate"}]}
_ONE_TOPOLOGY = {"candidates": [{"name": "diamond"}]}


async def test_parallel_sampling_fans_out_per_topology_and_picks_winner():
    gen = _gen(
        parallel=True,
        json_side=[_TWO_TOPOLOGIES, {"winner": "B"}],  # topology selection, then pairwise judge picks B
        structured_side=[_plan("DiamondPlan"), _plan("DebatePlan")],
    )
    result = await gen.generate("compare options then synthesize")

    # one planning pass per topology (2), and each carried its own topology hint
    assert gen._llm.generate_structured.await_count == 2
    prompts = [c.kwargs["user_prompt"] for c in gen._llm.generate_structured.await_args_list]
    assert any("Target topology: diamond" in p for p in prompts)
    assert any("Target topology: debate" in p for p in prompts)
    # the pairwise judge picked B (DebatePlan) → it drives the final chain
    assert result.chain_dict["steps"][0]["title"] == "DebatePlan"


async def test_off_by_default_single_planning_pass():
    gen = _gen(
        parallel=False,  # default
        json_side=[_TWO_TOPOLOGIES],  # only topology selection; no pairwise judge call
        structured_side=[_plan("OnlyPlan")],
    )
    result = await gen.generate("q")
    assert gen._llm.generate_structured.await_count == 1  # no fan-out
    assert result.chain_dict["steps"][0]["title"] == "OnlyPlan"


async def test_single_topology_no_fan_out_even_when_enabled():
    gen = _gen(
        parallel=True,
        json_side=[_ONE_TOPOLOGY],  # only 1 distinct topology → nothing to sample over
        structured_side=[_plan("SoloPlan")],
    )
    result = await gen.generate("q")
    assert gen._llm.generate_structured.await_count == 1
    assert result.chain_dict["steps"][0]["title"] == "SoloPlan"


async def test_pipeline_plus_one_topology_samples_both():
    # selector returns pipeline (→ empty hint) + diamond (→ real hint): 2 distinct hints
    gen = _gen(
        parallel=True,
        json_side=[{"candidates": [{"name": "pipeline"}, {"name": "diamond"}]}, {"winner": "A"}],
        structured_side=[_plan("PipelinePlan"), _plan("DiamondPlan")],
    )
    result = await gen.generate("q")
    assert gen._llm.generate_structured.await_count == 2
    # winner A → the first (free-form / pipeline) plan
    assert result.chain_dict["steps"][0]["title"] == "PipelinePlan"


async def test_config_flag_plumbs_through_env(monkeypatch: Any):
    monkeypatch.setenv("MAGE_ENABLE_PARALLEL_TOPOLOGY_SAMPLING", "true")
    monkeypatch.setenv("MAGE_API_KEY", "t")
    cfg = MAGEConfig.from_env()
    assert cfg.enable_parallel_topology_sampling is True
