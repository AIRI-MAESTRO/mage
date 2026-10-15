"""WP-6 — ``generate(query, return_candidates=True)`` carries each
topology-sampled plan candidate through DAG → describe → validation
*independently* (returning ``list[MAGEResult]``) instead of collapsing them via
the pairwise ``_score_plans`` judge, while the research / domain / topology
stages run exactly **once**.

The LLM is mocked at ``mmar_mage.llm.AsyncOpenAI`` (per the project convention)
with a content-routing ``side_effect`` so the parallel per-candidate finalize
calls resolve regardless of completion order. Stage call-counts are asserted by
spying on the agent classes / ``_score_plans``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig, MAGEResult

# --- canned stage responses ------------------------------------------------

# Two distinct topologies → two distinct planning passes → two candidates.
_TWO_TOPOLOGIES = {"candidates": [{"name": "diamond"}, {"name": "debate"}]}

# Structurally distinct plans keyed off the topology hint that the StepPlanner
# embeds in its user prompt ("Target topology: <name>"). Distinct step counts +
# titles guarantee the finalized chains differ.
_DIAMOND_PLAN = {
    "steps": [
        {"number": 1, "title": "Diamond-Split", "step_type": "llm", "rough_aim": "split"},
        {"number": 2, "title": "Diamond-Merge", "step_type": "llm", "rough_aim": "merge"},
    ],
    "rationale": "diamond",
}
_DEBATE_PLAN = {
    "steps": [
        {"number": 1, "title": "Debate-Pro", "step_type": "llm", "rough_aim": "argue for"},
        {"number": 2, "title": "Debate-Con", "step_type": "llm", "rough_aim": "argue against"},
        {"number": 3, "title": "Debate-Judge", "step_type": "llm", "rough_aim": "decide"},
    ],
    "rationale": "debate",
}

_DOMAIN = {
    "domain": "general",
    "task_type": "analysis",
    "complexity": "medium",
    "key_concepts": ["x"],
    "suggested_step_count": 3,
    "preferred_step_types": ["llm"],
}


def _resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    r = MagicMock()
    r.choices = [choice]
    r.usage = None
    return r


def _route(**kwargs: Any) -> MagicMock:
    """Return canned JSON keyed off the (system, user) prompt of each call."""
    messages = kwargs.get("messages") or []
    system = messages[0]["content"] if messages else ""
    user = messages[1]["content"] if len(messages) > 1 else ""

    if "domain analysis expert" in system:
        return _resp(json.dumps(_DOMAIN))
    if "STRUCTURAL TOPOLOGY" in system:
        return _resp(json.dumps(_TWO_TOPOLOGIES))
    if "reasoning step planner" in system:
        # Distinct plan per topology hint embedded in the user prompt.
        if "debate" in user:
            return _resp(json.dumps(_DEBATE_PLAN))
        return _resp(json.dumps(_DIAMOND_PLAN))
    if "reasoning-plan judge" in system:  # pairwise collapse — must NOT fire here
        raise AssertionError("pairwise plan judge was called under return_candidates=True")
    # Any other (unexpected) stage: echo an empty object so JSON parsing succeeds.
    return _resp("{}")


def _candidate_config() -> MAGEConfig:
    """Deep config: topology sampling ON, DAG/describe OFF so the fallback DAG +
    descriptions carry each plan's distinct steps straight into the chain (same
    shape as ``test_parallel_topology_sampling``). Keeps the LLM call set to
    domain + topology + one planning pass per candidate."""
    return MAGEConfig(
        api_key="t",
        mode="deep",
        enable_memory=False,
        enable_domain_analysis=True,
        enable_topology_selection=True,
        enable_parallel_topology_sampling=True,
        enable_step_planning=True,
        enable_dag_optimization=False,  # fallback DAG from the plan
        enable_step_descriptions=False,  # fallback descriptions from the DAG
        enable_suggested_naming_llm=False,  # avoid per-candidate naming LLM calls
        simplicity_bias=0.0,
        max_retries=0,
    )


def _make_gen(mocker: Any, config: MAGEConfig) -> MAGEGenerator:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=_route)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return MAGEGenerator(config=config)


# --- tests -----------------------------------------------------------------


async def test_return_candidates_yields_list_of_distinct_results(mocker: Any) -> None:
    from mmar_mage.agents.domain_analyzer import DomainAnalyzer
    from mmar_mage.agents.topology_selector import TopologySelectorAgent

    analyze_spy = mocker.spy(DomainAnalyzer, "analyze")
    select_spy = mocker.spy(TopologySelectorAgent, "select")

    gen = _make_gen(mocker, _candidate_config())
    score_spy = mocker.spy(gen, "_score_plans")

    results = await gen.generate("compare options then synthesize", return_candidates=True)

    # A list of MAGEResults, one per sampled topology candidate (≥2).
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, MAGEResult) for r in results)

    # The candidates are structurally distinct (different plans carried through).
    titles = {tuple(s["title"] for s in r.chain_dict["steps"]) for r in results}
    assert len(titles) == 2, f"expected 2 distinct candidate chains, got {titles}"
    step_counts = sorted(len(r.chain_dict["steps"]) for r in results)
    assert step_counts == [2, 3]

    # Research-once: domain analysis + topology selection each ran exactly once,
    # despite two candidates being finalized.
    assert analyze_spy.call_count == 1
    assert select_spy.call_count == 1
    # The pairwise collapse judge was bypassed entirely.
    assert score_spy.call_count == 0


async def test_default_generate_returns_single_result(mocker: Any) -> None:
    """The default (no ``return_candidates``) call is unchanged: one MAGEResult,
    not a list — even with topology sampling enabled it collapses as before."""
    gen = _make_gen(mocker, _candidate_config())
    score_spy = mocker.spy(gen, "_score_plans")

    result = await gen.generate("compare options then synthesize")

    assert isinstance(result, MAGEResult)
    assert not isinstance(result, list)
    # Default path still collapses the two sampled plans via the tournament.
    assert score_spy.call_count == 1


async def test_return_candidates_single_when_no_sampling(mocker: Any) -> None:
    """With parallel topology sampling OFF there is nothing to fan out, so
    ``return_candidates=True`` returns a one-element list (the single chain)."""
    cfg = _candidate_config()
    cfg = cfg.model_copy(update={"enable_parallel_topology_sampling": False})
    gen = _make_gen(mocker, cfg)
    score_spy = mocker.spy(gen, "_score_plans")

    results = await gen.generate("q", return_candidates=True)

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MAGEResult)
    # No candidates sampled → no collapse either.
    assert score_spy.call_count == 0


async def test_return_candidates_fast_mode_single_element_list(mocker: Any) -> None:
    """Fast mode has no topology stage at all: a one-element list is returned for
    a consistent caller contract."""
    from mmar_mage import generator as gen_mod

    cfg = MAGEConfig(mode="fast", api_key="t", enable_memory=False, max_retries=0)
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    # Fast mode emits a single full CARL chain in one shot.
    fast_chain = json.dumps(
        {
            "name": "Fast",
            "max_workers": 1,
            "timeout": 300.0,
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Only",
                    "aim": "do it",
                    "reasoning_questions": "q",
                    "step_context_queries": ["x"],
                    "stage_action": "do",
                    "example_reasoning": "e",
                    "dependencies": [],
                    "retry_max": 3,
                    "llm_config": None,
                }
            ],
        }
    )
    instance.chat.completions.create = AsyncMock(return_value=_resp(fast_chain))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    gen = gen_mod.MAGEGenerator(config=cfg)

    results = await gen.generate("q", return_candidates=True)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], MAGEResult)
