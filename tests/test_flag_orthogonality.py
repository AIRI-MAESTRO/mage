"""Config flag orthogonality — plan item P3.2.

Research / bootstrap / tool-synthesis stages feed ONLY the StepPlanner, so they're
skipped when ``enable_step_planning=False`` (otherwise their digests are computed
at LLM cost then ignored by the fallback). Topology also feeds the DAGBuilder, so
it's skipped only when both planning and dag-optimization are off. These tests
drive real ``generate()`` calls and assert the skip/run behaviour via
``deep_stages_completed`` + LLM-call counts.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


def _gen(**cfg: Any) -> MAGEGenerator:
    base = dict(
        mode="deep",
        api_key="t",
        enable_memory=False,
        enable_domain_analysis=False,  # default domain, no LLM call
        enable_suggested_naming_llm=False,  # CARE naming stage is orthogonal: keep LLM-call counts clean
        max_retries=0,
    )
    base.update(cfg)
    gen = MAGEGenerator(config=MAGEConfig(**base))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps({"steps": []}))
    gen._llm.generate_json = AsyncMock(return_value={"candidates": [{"name": "pipeline"}]})
    return gen


# --------------------------------------------------------------------------- #
# Planner-input stages are skipped when the planner won't run
# --------------------------------------------------------------------------- #
async def test_research_skipped_when_planning_off():
    gen = _gen(
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,
        enable_memory_research=True,  # would normally run + call the LLM
        enable_capability_lookup=True,
        enable_skill_discovery=True,
    )
    result = await gen.generate("anything")
    stages = result.metadata.deep_stages_completed or []
    assert "memory_research" not in stages
    assert "capability_lookup" not in stages
    assert "skill_discovery" not in stages
    # the research agents never touched the LLM (only nothing or the fallback path)
    gen._llm.generate_json.assert_not_called()


async def test_tool_synthesis_and_bootstrap_skipped_when_planning_off():
    gen = _gen(
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,
        enable_tool_synthesis=True,
        enable_demo_bootstrap=True,
    )
    result = await gen.generate("anything")
    stages = result.metadata.deep_stages_completed or []
    assert "tool_synthesis" not in stages
    assert "demo_bootstrap" not in stages


# --------------------------------------------------------------------------- #
# Topology: skipped only when BOTH planning and dag are off
# --------------------------------------------------------------------------- #
async def test_topology_skipped_when_planning_and_dag_off():
    gen = _gen(
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=True,
    )
    result = await gen.generate("anything")
    assert "topology_selection" not in (result.metadata.deep_stages_completed or [])
    gen._llm.generate_json.assert_not_called()  # topology call never made


async def test_topology_runs_for_dag_even_when_planning_off():
    gen = _gen(
        enable_step_planning=False,
        enable_dag_optimization=True,  # DAGBuilder will consume the topology hint
        enable_step_descriptions=False,
        enable_topology_selection=True,
    )
    # DAGBuilder uses generate_structured; topology uses generate_json
    from mmar_mage.schemas import DAGStep, DAGStructure

    gen._llm.generate_structured = AsyncMock(
        return_value=DAGStructure(
            steps=[DAGStep(number=1, title="S", step_type="llm", dependencies=[], rough_aim="do")],
            max_workers=1,
            pattern="pipeline",
        )
    )
    result = await gen.generate("anything")
    assert "topology_selection" in (result.metadata.deep_stages_completed or [])
    gen._llm.generate_json.assert_awaited()  # topology selection ran


# --------------------------------------------------------------------------- #
# When planning IS on, research still runs (no regression)
# --------------------------------------------------------------------------- #
async def test_research_runs_when_planning_on():
    gen = _gen(
        enable_step_planning=True,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,
        enable_capability_lookup=True,
    )
    # step planning uses the LLM; return a non-empty plan so the chain validates
    from mmar_mage.schemas import PlannedStep, StepPlan

    gen._llm.generate_structured = AsyncMock(
        return_value=StepPlan(
            steps=[PlannedStep(number=1, title="Do", step_type="llm", rough_aim="do the task")], rationale="r"
        )
    )
    result = await gen.generate("anything")
    assert "capability_lookup" in (result.metadata.deep_stages_completed or [])
