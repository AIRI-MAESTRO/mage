"""Applicability gate for memory research (plan item P1.1).

After recall, ``MemoryResearchAgent`` runs an LLM judge that rates each hit's
applicability to the current query; hits below the threshold are dropped and the
rest are ordered best-first. The gate is non-fatal — judge failures keep the
unfiltered hits. These tests drive the agent with a directly-mocked LLM (no real
calls), so they need no optional dependency.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.memory_research_agent import MemoryResearchAgent, _summarize_structure
from mmar_mage.schemas import MemoryHit


def _agent(generate_json_return=None, generate_json_side_effect=None) -> MemoryResearchAgent:
    llm = MagicMock()
    if generate_json_side_effect is not None:
        llm.generate_json = AsyncMock(side_effect=generate_json_side_effect)
    else:
        llm.generate_json = AsyncMock(return_value=generate_json_return)
    return MemoryResearchAgent(llm)


def _hits(*ids: str) -> list[MemoryHit]:
    return [MemoryHit(entity_id=i, name=f"chain-{i}", score=0.5, abstract_structure={}) for i in ids]


# --------------------------------------------------------------------------- #
# _summarize_structure
# --------------------------------------------------------------------------- #
def test_summarize_prefers_step_type_sequence():
    assert _summarize_structure({"step_type_sequence": ["llm", "tool", "llm"]}) == "llm -> tool -> llm"


def test_summarize_falls_back_to_steps():
    out = _summarize_structure({"steps": [{"step_type": "llm"}, {"step_type": "transform"}]})
    assert out == "llm -> transform"


def test_summarize_empty_is_blank():
    assert _summarize_structure({}) == ""
    assert _summarize_structure(None) == ""


# --------------------------------------------------------------------------- #
# _filter_applicable
# --------------------------------------------------------------------------- #
async def test_gate_drops_hits_below_threshold():
    agent = _agent({"ratings": [{"entity_id": "a", "applicability": 0.9}, {"entity_id": "b", "applicability": 0.2}]})
    kept = await agent._filter_applicable("q", _hits("a", "b"), threshold=0.5)
    assert [h.entity_id for h in kept] == ["a"]  # b (0.2) dropped


async def test_gate_sorts_by_applicability_desc():
    agent = _agent({"ratings": [{"entity_id": "a", "applicability": 0.6}, {"entity_id": "b", "applicability": 0.95}]})
    kept = await agent._filter_applicable("q", _hits("a", "b"), threshold=0.5)
    assert [h.entity_id for h in kept] == ["b", "a"]  # best-first


async def test_gate_treats_unrated_hit_as_zero():
    # only "a" is rated; "b" omitted → scored 0 → dropped
    agent = _agent({"ratings": [{"entity_id": "a", "applicability": 0.8}]})
    kept = await agent._filter_applicable("q", _hits("a", "b"), threshold=0.5)
    assert [h.entity_id for h in kept] == ["a"]


async def test_gate_empty_ratings_keeps_all_hits():
    # judge returned nothing usable → don't silently drop everything
    agent = _agent({"ratings": []})
    kept = await agent._filter_applicable("q", _hits("a", "b"), threshold=0.5)
    assert [h.entity_id for h in kept] == ["a", "b"]


# --------------------------------------------------------------------------- #
# research(): gate wired in, non-fatal, toggleable
# --------------------------------------------------------------------------- #
async def test_research_gate_failure_is_nonfatal(mocker):
    from mmar_mage.memory import MemoryManager
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    config = MAGEConfig(api_key="t", enable_memory=True, enable_memory_research=True, max_retries=0)
    memory = MemoryManager(config)
    memory._available = True
    mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=_hits("a", "b"))

    # sub-queries OK, ratings RAISES, digest OK → gate failure must not abort research
    agent = _agent(
        generate_json_side_effect=[
            {"sub_queries": ["q1"]},
            RuntimeError("judge down"),
            {"digest": "recovered digest"},
        ]
    )
    domain = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")
    result = await agent.research("q", domain, memory, config)

    assert result.was_cold_start is False
    assert result.hits_count == 2  # original hits preserved despite gate failure
    assert result.digest == "recovered digest"


async def test_research_gate_disabled_skips_judge(mocker):
    from mmar_mage.memory import MemoryManager
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    config = MAGEConfig(
        api_key="t",
        enable_memory=True,
        enable_memory_research=True,
        enable_memory_applicability_gate=False,
        max_retries=0,
    )
    memory = MemoryManager(config)
    memory._available = True
    mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=_hits("a", "b"))

    # only 2 LLM calls when the gate is off: sub-queries + digest (no ratings call)
    agent = _agent(generate_json_side_effect=[{"sub_queries": ["q1"]}, {"digest": "d"}])
    domain = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")
    result = await agent.research("q", domain, memory, config)

    assert result.hits_count == 2
    assert agent._llm.generate_json.await_count == 2  # gate's judge call skipped


# --------------------------------------------------------------------------- #
# Second-wave recall (P1.1)
# --------------------------------------------------------------------------- #
async def test_second_wave_generates_gaps_recalls_and_merges(mocker):
    from mmar_mage.memory import MemoryManager
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    config = MAGEConfig(api_key="t", enable_memory=True, enable_memory_research=True, max_retries=0)
    memory = MemoryManager(config)
    memory._available = True
    # wave-2 recall returns two new chains
    isearch = mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=_hits("c", "d"))

    # gap sub-queries, then ratings for the wave-2 hits (both applicable)
    agent = _agent(
        generate_json_side_effect=[
            {"sub_queries": ["gap1", "gap2"]},
            {"ratings": [{"entity_id": "c", "applicability": 0.8}, {"entity_id": "d", "applicability": 0.9}]},
        ]
    )
    domain = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")

    merged = await agent._second_wave("q", domain, memory, config, _hits("a"))

    assert sorted(h.entity_id for h in merged) == ["a", "c", "d"]  # wave-1 'a' + new 'c','d'
    # the second recall is driven by the generated gap queries
    assert isearch.call_args.kwargs["extra_queries"] == ["gap1", "gap2"]


async def test_second_wave_skipped_on_cold_start(mocker):
    from mmar_mage.memory import MemoryManager
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    config = MAGEConfig(api_key="t", enable_memory=True, enable_memory_research=True, max_retries=0)
    memory = MemoryManager(config)
    memory._available = True
    mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=[])  # empty memory

    # only the sub-query call should happen: no gate (no hits), no second wave (raw_count==0)
    agent = _agent(generate_json_side_effect=[{"sub_queries": ["q1"]}])
    domain = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")
    result = await agent.research("q", domain, memory, config)

    assert result.was_cold_start is True
    assert agent._llm.generate_json.await_count == 1  # no gap-query call on a true cold start


async def test_no_second_wave_when_enough_hits(mocker):
    from mmar_mage.memory import MemoryManager
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    config = MAGEConfig(api_key="t", enable_memory=True, enable_memory_research=True, max_retries=0)
    memory = MemoryManager(config)
    memory._available = True
    isearch = mocker.patch.object(memory, "iterative_search", new_callable=AsyncMock, return_value=_hits("a", "b"))

    # gate keeps both (>= memory_min_applicable_hits=2) → no second wave
    agent = _agent(
        generate_json_side_effect=[
            {"sub_queries": ["q1"]},
            {"ratings": [{"entity_id": "a", "applicability": 0.9}, {"entity_id": "b", "applicability": 0.8}]},
            {"digest": "d"},
        ]
    )
    domain = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")
    result = await agent.research("q", domain, memory, config)

    assert result.hits_count == 2
    assert isearch.await_count == 1  # recall ran once; no second wave
