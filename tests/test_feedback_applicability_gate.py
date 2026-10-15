"""Applicability gate for recalled feedback (plan item P1.3).

`FeedbackRecallAgent` rates each quality-filtered feedback record's applicability
to the current query (LLM judge) and drops the irrelevant ones before the digest.
Records are addressed by list index, so missing `chain_entity_id` is fine. The
gate is non-fatal and toggleable. Tests use a directly-mocked LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.feedback_recall_agent import FeedbackRecallAgent
from mmar_mage.schemas import DomainAnalysis, MAGEConfig

DOMAIN = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")


def _agent(side_effect=None, return_value=None) -> FeedbackRecallAgent:
    llm = MagicMock()
    if side_effect is not None:
        llm.generate_json = AsyncMock(side_effect=side_effect)
    else:
        llm.generate_json = AsyncMock(return_value=return_value)
    return FeedbackRecallAgent(llm, MagicMock())


def _recs(n: int) -> list[dict]:
    return [{"chain_entity_id": f"e{i}", "pattern_used": "p", "quality_score": 0.8, "success": True} for i in range(n)]


def _agent_with_feedback(records: list[dict], side_effect) -> FeedbackRecallAgent:
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=side_effect)
    memory = MagicMock()
    memory.search_feedback = AsyncMock(return_value=records)
    return FeedbackRecallAgent(llm, memory)


# --------------------------------------------------------------------------- #
# _filter_applicable (unit)
# --------------------------------------------------------------------------- #
async def test_gate_drops_below_threshold():
    agent = _agent(return_value={"ratings": [{"id": "0", "applicability": 0.9}, {"id": "1", "applicability": 0.2}]})
    kept = await agent._filter_applicable("q", _recs(2), threshold=0.5)
    assert [r["chain_entity_id"] for r in kept] == ["e0"]


async def test_gate_sorts_by_applicability_desc():
    agent = _agent(return_value={"ratings": [{"id": "0", "applicability": 0.6}, {"id": "1", "applicability": 0.95}]})
    kept = await agent._filter_applicable("q", _recs(2), threshold=0.5)
    assert [r["chain_entity_id"] for r in kept] == ["e1", "e0"]


async def test_gate_empty_ratings_keeps_all():
    agent = _agent(return_value={"ratings": []})
    kept = await agent._filter_applicable("q", _recs(2), threshold=0.5)
    assert len(kept) == 2


async def test_gate_robust_to_missing_chain_entity_id():
    # records without chain_entity_id are still judged (addressed by index)
    agent = _agent(return_value={"ratings": [{"id": "0", "applicability": 0.9}]})
    kept = await agent._filter_applicable("q", [{"pattern_used": "p"}], threshold=0.5)
    assert len(kept) == 1


# --------------------------------------------------------------------------- #
# recall(): gate wired in, non-fatal, toggleable
# --------------------------------------------------------------------------- #
async def test_recall_gate_failure_is_nonfatal():
    # ratings RAISES, digest OK → gate failure keeps the quality-filtered feedback
    agent = _agent_with_feedback(_recs(2), side_effect=[RuntimeError("judge down"), {"digest": "d"}])
    result = await agent.recall("q", DOMAIN, MAGEConfig(feedback_min_quality=0.0))
    assert result.recalled_count == 2
    assert result.digest == "d"


async def test_recall_gate_drops_all_returns_empty():
    # judge rates everything below threshold → cold (empty digest), no digest call
    agent = _agent_with_feedback(
        _recs(2), side_effect=[{"ratings": [{"id": "0", "applicability": 0.1}, {"id": "1", "applicability": 0.0}]}]
    )
    result = await agent.recall("q", DOMAIN, MAGEConfig(feedback_min_quality=0.0))
    assert result.recalled_count == 0
    assert result.digest == ""


async def test_recall_gate_disabled_skips_judge():
    agent = _agent_with_feedback(_recs(2), side_effect=[{"digest": "d"}])  # only the digest call
    result = await agent.recall(
        "q", DOMAIN, MAGEConfig(enable_feedback_applicability_gate=False, feedback_min_quality=0.0)
    )
    assert result.recalled_count == 2
    assert agent._llm.generate_json.await_count == 1  # no gate judge call
