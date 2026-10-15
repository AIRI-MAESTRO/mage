"""Typed-entity persistence for reusable-knowledge payloads (plan item P1.18 item 3).

MemoryManager previously persisted templates/feedback/ideas by overloading the
``chain`` entity and recalled them with entity types the v0.2.3 server does not
recognize (``chain_template`` etc.), so recall silently returned nothing. This
slice routes the **template** round-trip through proper memory cards:

* ``_save_card`` → ``client.save_memory_card`` with a raw dict payload (+ ``kind``).
* ``_search_cards`` → ``client.search_hits`` (raw ``.content`` survives) using the
  configured hybrid-search mode.

The ``_search_cards`` / round-trip tests need the optional ``gigaevo_memory``
package and skip without it; the ``_save_card`` unit test does not.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig

try:
    import gigaevo_memory  # noqa: F401

    _HAS_GIGAEVO = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_GIGAEVO = False

_needs_gigaevo = pytest.mark.skipif(not _HAS_GIGAEVO, reason="gigaevo_memory not installed")


def _mgr(**cfg) -> MemoryManager:
    mgr = MemoryManager(MAGEConfig(enable_memory=True, **cfg))
    mgr._available = True
    return mgr


def _client_cm(mock_client: MagicMock) -> MagicMock:
    """Wrap a mock client in a context manager (matches ``with MemoryClient(...)``)."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _hit(content: dict, entity_id: str = "e", score: float = 0.0) -> types.SimpleNamespace:
    """Stand-in for gigaevo SearchHitData (raw ``.content`` dict + ``.score``)."""
    return types.SimpleNamespace(content=content, entity_id=entity_id, score=score)


# --------------------------------------------------------------------------- #
# _save_card (no gigaevo import needed — takes the client as an argument)
# --------------------------------------------------------------------------- #
def test_save_card_calls_save_memory_card_and_returns_id():
    mgr = _mgr(memory_namespace="proj-x")
    client = MagicMock()
    client.save_memory_card.return_value = types.SimpleNamespace(entity_id="card-1")

    out = mgr._save_card(
        client,
        {"kind": "mage_template", "x": 1},
        name="t",
        tags=["mage-template"],
        when_to_use="reuse",
    )

    assert out == "card-1"
    kwargs = client.save_memory_card.call_args.kwargs
    assert kwargs["memory_card"] == {"kind": "mage_template", "x": 1}
    assert kwargs["name"] == "t"
    assert kwargs["tags"] == ["mage-template"]
    assert kwargs["when_to_use"] == "reuse"
    assert kwargs["namespace"] == "proj-x"  # threaded from config


# --------------------------------------------------------------------------- #
# _search_cards routes through search_hits on the memory_card entity type
# --------------------------------------------------------------------------- #
@_needs_gigaevo
def test_search_cards_uses_search_hits_on_memory_cards():
    from gigaevo_memory import SearchType

    mgr = _mgr(memory_search_mode="hybrid")
    client = MagicMock()
    client.search_hits.return_value = []

    mgr._search_cards(client, "q", top_k=4)

    assert client.search.called is False  # uses search_hits, not search
    kwargs = client.search_hits.call_args.kwargs
    assert kwargs["entity_type"] == "memory_card"
    assert kwargs["search_type"] == SearchType.HYBRID
    assert kwargs["top_k"] == 4


# --------------------------------------------------------------------------- #
# Template round-trip via memory cards
# --------------------------------------------------------------------------- #
@_needs_gigaevo
async def test_save_template_persists_as_memory_card():
    mgr = _mgr()
    client = MagicMock()
    client.save_memory_card.return_value = types.SimpleNamespace(entity_id="tpl-1")

    chain = {"steps": [{"step_type": "llm"}, {"step_type": "tool"}]}
    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        out = await mgr.save_template(chain, domain="finance", task_type="analysis", quality_score=0.9)

    assert out == "tpl-1"
    client.save_chain.assert_not_called()  # no longer overloads chain
    content = client.save_memory_card.call_args.kwargs["memory_card"]
    assert content["kind"] == "mage_template"
    assert content["step_type_sequence"] == ["llm", "tool"]
    assert content["quality_score"] == 0.9
    assert "mage-template" in client.save_memory_card.call_args.kwargs["tags"]


@_needs_gigaevo
async def test_search_templates_filters_by_kind_and_parses_content():
    mgr = _mgr()
    client = MagicMock()
    # a real template card + an unrelated feedback card in the same results
    client.search_hits.return_value = [
        _hit(
            {
                "kind": "mage_template",
                "abstract_dag": [{"number": 1}],
                "step_type_sequence": ["llm"],
                "quality_score": 0.8,
            },
            entity_id="tpl-1",
            score=0.42,
        ),
        _hit({"kind": "mage_feedback", "feedback": {}}, entity_id="fb-1", score=0.99),
    ]

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        out = await mgr.search_templates(domain="finance", task_type="analysis", complexity="medium")

    assert len(out) == 1  # feedback card filtered out
    tpl = out[0]
    assert tpl["entity_id"] == "tpl-1"
    assert tpl["step_type_sequence"] == ["llm"]
    assert tpl["quality_score"] == 0.8
    assert tpl["score"] == 0.42


# --------------------------------------------------------------------------- #
# Feedback round-trip via memory cards
# --------------------------------------------------------------------------- #
@_needs_gigaevo
async def test_submit_feedback_persists_as_memory_card():
    mgr = _mgr()
    client = MagicMock()
    client.save_memory_card.return_value = types.SimpleNamespace(entity_id="fb-1")

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        await mgr.submit_feedback("chain-9", {"quality_score": 0.7, "success": False, "domain": "finance"})

    client.save_chain.assert_not_called()
    content = client.save_memory_card.call_args.kwargs["memory_card"]
    assert content["kind"] == "mage_feedback"
    assert content["chain_entity_id"] == "chain-9"
    assert content["quality_score"] == 0.7  # feedback fields flattened into content
    assert content["success"] is False


@_needs_gigaevo
async def test_search_feedback_filters_by_kind_and_parses_content():
    mgr = _mgr()
    client = MagicMock()
    client.search_hits.return_value = [
        _hit(
            {
                "kind": "mage_feedback",
                "chain_entity_id": "chain-9",
                "domain": "finance",
                "quality_score": 0.7,
                "success": False,
                "failure_reason": "timeout",
            },
            entity_id="fb-1",
            score=0.55,
        ),
        _hit({"kind": "mage_template"}, entity_id="tpl-1", score=0.9),
    ]

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        out = await mgr.search_feedback(domain="finance", task_type="analysis", top_k=5)

    assert len(out) == 1  # template card filtered out
    fb = out[0]
    assert fb["chain_entity_id"] == "chain-9"
    assert fb["quality_score"] == 0.7
    assert fb["success"] is False
    assert fb["failure_reason"] == "timeout"
    assert fb["score"] == 0.55


@_needs_gigaevo
async def test_save_ideas_persists_as_memory_card():
    mgr = _mgr()
    client = MagicMock()
    client.save_memory_card.return_value = types.SimpleNamespace(entity_id="idea-1")

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        out = await mgr.save_ideas([{"idea": "split into map-reduce"}], domain="data", query="aggregate logs")

    assert out == "idea-1"
    client.save_chain.assert_not_called()
    content = client.save_memory_card.call_args.kwargs["memory_card"]
    assert content["kind"] == "mage_ideas"
    assert content["query"] == "aggregate logs"


# --------------------------------------------------------------------------- #
# iterative_search: scored multi-query recall (P1.18 item 4)
# --------------------------------------------------------------------------- #
async def test_iterative_search_returns_empty_when_disabled():
    mgr = MemoryManager(MAGEConfig(enable_memory=False))
    mgr._available = True
    assert await mgr.iterative_search("q", ["concept"]) == []


@_needs_gigaevo
async def test_iterative_search_returns_scored_hits_dedup_and_sorted():
    mgr = _mgr(memory_search_mode="bm25")
    client = MagicMock()
    # two sub-queries (query + one concept); e1 recurs (kept at first/higher score)
    client.search_hits.side_effect = [
        [_hit({"name": "A"}, entity_id="e1", score=0.9), _hit({"name": "B"}, entity_id="e2", score=0.5)],
        [_hit({"name": "A"}, entity_id="e1", score=0.95), _hit({"name": "C"}, entity_id="e3", score=0.7)],
    ]

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        hits = await mgr.iterative_search("q", ["concept"], sub_queries_count=2, top_k=5, threshold=0.4)

    # NOT empty (the pre-fix code always returned []), deduped, sorted by score desc
    assert [h.entity_id for h in hits] == ["e1", "e3", "e2"]
    assert [h.score for h in hits] == [0.9, 0.7, 0.5]
    assert hits[0].name == "A"
    assert hits[0].abstract_structure == {"name": "A"}


@_needs_gigaevo
async def test_iterative_search_filters_below_threshold():
    mgr = _mgr(memory_search_mode="bm25")
    client = MagicMock()
    client.search_hits.side_effect = [
        [_hit({}, entity_id="hi", score=0.9), _hit({}, entity_id="lo", score=0.2)],
    ]

    with patch("gigaevo_memory.MemoryClient", return_value=_client_cm(client)):
        hits = await mgr.iterative_search("q", [], sub_queries_count=1, top_k=5, threshold=0.5)

    assert [h.entity_id for h in hits] == ["hi"]  # lo (0.2) filtered out
