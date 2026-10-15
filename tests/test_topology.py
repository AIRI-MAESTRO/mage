"""Diverse topology generation — plan item P1.15.

Covers the three layers with a directly-mocked LLM:
- the built-in topology catalog + helpers (`library.py`),
- ``TopologySelectorAgent.select`` (parse/validate/rank, graceful fallback),
- the real-execution path: ``generate()`` runs stage 0.5 and the chosen
  topology's hint reaches the StepPlanner prompt.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.topology_selector import TopologySelectorAgent
from mmar_mage.schemas import DomainAnalysis, MAGEConfig, PlannedStep, StepPlan
from mmar_mage.topology import (
    DEFAULT_TOPOLOGY,
    TopologyCandidates,
    TopologyChoice,
    default_choice,
    describe_for_prompt,
    format_topology_hint,
    get_topology,
    list_topologies,
)

DOMAIN = DomainAnalysis(domain="finance")


# --------------------------------------------------------------------------- #
# library
# --------------------------------------------------------------------------- #
def test_get_topology_case_insensitive_and_unknown():
    assert get_topology("Diamond").name == "diamond"
    assert get_topology("  PIPELINE ").name == "pipeline"
    assert get_topology("nonsense") is None
    assert get_topology(None) is None


def test_list_topologies_sorted_simplest_first():
    topos = list_topologies()
    assert topos[0].name == "pipeline"  # lowest complexity
    scores = [t.complexity_score for t in topos]
    assert scores == sorted(scores)


def test_describe_for_prompt_lists_every_topology():
    text = describe_for_prompt()
    for name in ("pipeline", "diamond", "debate", "map_reduce", "mixture_of_experts"):
        assert name in text


def test_format_hint_pipeline_is_empty():
    # pipeline is the default — no constraint needed
    assert format_topology_hint(TopologyChoice(name="pipeline")) == ""


def test_format_hint_non_pipeline_describes_pattern():
    hint = format_topology_hint(TopologyChoice(name="diamond"))
    assert "Target topology: diamond" in hint
    assert "aggregator" in hint  # a characteristic step type


def test_format_hint_unknown_is_empty():
    assert format_topology_hint(TopologyChoice(name="bogus")) == ""


def test_default_choice_is_pipeline():
    assert default_choice().name == DEFAULT_TOPOLOGY


def test_candidates_best():
    cands = TopologyCandidates(candidates=[TopologyChoice(name="debate"), TopologyChoice(name="pipeline")])
    assert cands.best.name == "debate"
    assert TopologyCandidates().best is None


# --------------------------------------------------------------------------- #
# TopologySelectorAgent.select
# --------------------------------------------------------------------------- #
def _llm(return_value=None, side_effect=None) -> MagicMock:
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value=return_value, side_effect=side_effect)
    return llm


async def test_select_parses_and_validates_candidates():
    llm = _llm(
        {
            "candidates": [
                {"name": "diamond", "rationale": "3 independent analyses", "confidence": 0.8},
                {"name": "pipeline", "confidence": 0.4},
            ]
        }
    )
    out = await TopologySelectorAgent(llm).select("compare 3 options", DOMAIN)
    assert [c.name for c in out.candidates] == ["diamond", "pipeline"]
    assert out.best.name == "diamond"


async def test_select_drops_unknown_and_duplicate_names():
    llm = _llm(
        {
            "candidates": [
                {"name": "made_up", "confidence": 0.9},
                {"name": "debate", "confidence": 0.7},
                {"name": "debate", "confidence": 0.6},  # duplicate
            ]
        }
    )
    out = await TopologySelectorAgent(llm).select("q", DOMAIN)
    assert [c.name for c in out.candidates] == ["debate"]


async def test_select_caps_at_max_candidates():
    llm = _llm({"candidates": [{"name": n} for n in ("diamond", "funnel", "debate", "map_reduce")]})
    out = await TopologySelectorAgent(llm).select("q", DOMAIN, max_candidates=2)
    assert len(out.candidates) == 2


async def test_select_clamps_confidence():
    llm = _llm({"candidates": [{"name": "diamond", "confidence": 5.0}]})
    out = await TopologySelectorAgent(llm).select("q", DOMAIN)
    assert out.candidates[0].confidence == 1.0


async def test_select_llm_failure_falls_back_to_pipeline():
    llm = _llm(side_effect=RuntimeError("selector down"))
    out = await TopologySelectorAgent(llm).select("q", DOMAIN)
    assert out.best.name == "pipeline"


async def test_select_no_valid_candidate_falls_back_to_pipeline():
    llm = _llm({"candidates": [{"name": "unknown_pattern"}]})
    out = await TopologySelectorAgent(llm).select("q", DOMAIN)
    assert out.best.name == "pipeline"


# --------------------------------------------------------------------------- #
# Real-execution scenario — generate() stage 0.5 + hint reaches the planner
# --------------------------------------------------------------------------- #
def _planning_only_gen(topology_name: str) -> tuple:
    """A generator whose only live LLM stages are topology selection + planning."""
    from mmar_mage.generator import MAGEGenerator

    cfg = MAGEConfig(
        api_key="t",
        mode="deep",
        enable_memory=False,
        enable_domain_analysis=False,  # default DomainAnalysis, no LLM call
        enable_topology_selection=True,
        enable_step_planning=True,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        simplicity_bias=0.0,  # keep planning unconstrained for the assertion
        max_retries=0,
    )
    gen = MAGEGenerator(config=cfg)
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate_json = AsyncMock(return_value={"candidates": [{"name": topology_name, "confidence": 0.9}]})
    plan = StepPlan(
        steps=[
            PlannedStep(number=1, title="A", step_type="llm", rough_aim="x"),
            PlannedStep(number=2, title="B", step_type="llm", rough_aim="y"),
        ],
        rationale="r",
    )
    gen._llm.generate_structured = AsyncMock(return_value=plan)
    return gen


async def test_generate_runs_topology_stage_and_passes_hint():
    gen = _planning_only_gen("diamond")
    result = await gen.generate("compare three independent options then synthesize")

    assert "topology_selection" in result.metadata.deep_stages_completed
    # the diamond hint was threaded into the StepPlanner prompt
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Target topology: diamond" in planner_prompt


async def test_generate_pipeline_choice_injects_no_constraint():
    gen = _planning_only_gen("pipeline")
    result = await gen.generate("just answer the question")

    assert "topology_selection" in result.metadata.deep_stages_completed
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Target topology" not in planner_prompt  # pipeline → empty hint


# --------------------------------------------------------------------------- #
# P3.4 — new patterns + enriched selector prompt
# --------------------------------------------------------------------------- #
_NEW_PATTERNS = (
    "router",
    "self_consistency",
    "react",
    "chain_of_verification",
    "plan_and_execute",
    "human_in_the_loop",
)


def test_new_patterns_registered_and_resolvable():
    catalog = describe_for_prompt()
    for name in _NEW_PATTERNS:
        topo = get_topology(name)
        assert topo is not None and topo.name == name
        assert name in catalog  # surfaced in the selector prompt table


def test_all_topologies_well_formed():
    from mmar_mage.topology.library import BUILTIN_TOPOLOGIES

    for name, topo in BUILTIN_TOPOLOGIES.items():
        assert topo.name == name
        assert topo.when_to_use.strip()
        assert topo.required_step_types  # non-empty
        assert topo.example_chain.strip()
        assert 0.0 <= topo.complexity_score <= 1.0
    # pipeline stays the simplest (the deterministic default).
    assert list_topologies()[0].name == "pipeline"


async def test_select_keeps_a_new_pattern():
    llm = _llm({"candidates": [{"name": "react", "rationale": "tool-loop", "confidence": 0.8}]})
    out = await TopologySelectorAgent(llm).select("research online then compute", DOMAIN)
    assert out.best.name == "react"


def test_selector_prompt_has_worked_examples_and_rationale_rule():
    from mmar_mage.prompts import TOPOLOGY_SELECTOR_PROMPT

    assert "Worked examples" in TOPOLOGY_SELECTOR_PROMPT
    # a per-pattern few-shot for a new pattern + the "why this shape" rule
    assert "self_consistency" in TOPOLOGY_SELECTOR_PROMPT
    assert "STRUCTURAL feature" in TOPOLOGY_SELECTOR_PROMPT


async def test_generate_new_pattern_hint_reaches_planner():
    gen = _planning_only_gen("react")
    result = await gen.generate("reason and call tools iteratively")
    assert "topology_selection" in result.metadata.deep_stages_completed
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Target topology: react" in planner_prompt


# --------------------------------------------------------------------------- #
# P5.4 — big / multi-file requests route to map_reduce (parallel + aggregate)
# --------------------------------------------------------------------------- #
def test_map_reduce_catalog_covers_files():
    # The selector catalog (when_to_use) now tells the LLM that large /
    # multiple files are a map_reduce shape, not one giant llm step.
    row = next(
        line for line in describe_for_prompt().splitlines()
        if line.startswith("map_reduce |")
    )
    assert "file" in row.lower()
    assert "chunk" in row.lower()


def test_selector_prompt_routes_files_to_map_reduce():
    from mmar_mage.prompts import TOPOLOGY_SELECTOR_PROMPT

    # A worked example routes large/multiple files to map_reduce.
    assert "map_reduce" in TOPOLOGY_SELECTOR_PROMPT
    assert "multiple @-referenced files" in TOPOLOGY_SELECTOR_PROMPT


def test_map_reduce_hint_has_parallel_and_aggregate_stages():
    # Choosing map_reduce constrains the chain to a parallel analyse fan-out
    # (parallel_sampling) + an aggregate/synthesis stage (aggregator).
    hint = format_topology_hint(TopologyChoice(name="map_reduce"))
    assert "Target topology: map_reduce" in hint
    assert "parallel_sampling" in hint
    assert "aggregator" in hint


async def test_big_file_query_yields_map_reduce_parallel_aggregate():
    # End-to-end (mocked planner): a big-file query whose selector picks
    # map_reduce reaches the StepPlanner with a parallel-analyse + aggregate
    # constraint, rather than a single llm step.
    gen = _planning_only_gen("map_reduce")
    result = await gen.generate(
        "review this large source file @big.py for bugs and summarise the findings"
    )
    assert "topology_selection" in result.metadata.deep_stages_completed
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Target topology: map_reduce" in planner_prompt
    assert "parallel_sampling" in planner_prompt  # the parallel analyse fan-out
    assert "aggregator" in planner_prompt  # the aggregate / synthesis stage
