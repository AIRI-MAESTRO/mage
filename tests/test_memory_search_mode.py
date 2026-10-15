"""Configurable hybrid-search recall in MemoryManager (plan item P1.18).

MemoryManager previously hard-coded ``SearchType.BM25`` at every recall site.
``_search`` now resolves the retrieval mode (bm25/vector/hybrid), passes
``hybrid_weights`` and ``namespace`` per config, and degrades a rejected
vector/hybrid request to BM25 so recall never silently fails.

The ``_search`` tests need the optional ``gigaevo_memory`` package (for the real
``SearchType`` enum) and skip without it; the config-default tests always run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig

try:
    from gigaevo_memory import SearchType

    _HAS_GIGAEVO = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_GIGAEVO = False

_needs_gigaevo = pytest.mark.skipif(not _HAS_GIGAEVO, reason="gigaevo_memory not installed")


def _mgr(**cfg) -> MemoryManager:
    mgr = MemoryManager(MAGEConfig(enable_memory=True, **cfg))
    mgr._available = True
    return mgr


# --------------------------------------------------------------------------- #
# Config defaults (no gigaevo needed)
# --------------------------------------------------------------------------- #
def test_default_search_mode_is_hybrid():
    cfg = MAGEConfig()
    assert cfg.memory_search_mode == "hybrid"
    assert cfg.memory_hybrid_weights == (0.5, 0.5)
    assert cfg.memory_namespace is None


# --------------------------------------------------------------------------- #
# _search behavior against a mocked client
# --------------------------------------------------------------------------- #
@_needs_gigaevo
def test_hybrid_mode_passes_search_type_and_weights():
    mgr = _mgr(memory_search_mode="hybrid", memory_hybrid_weights=(0.7, 0.3))
    client = MagicMock()
    client.search.return_value = ["r"]

    out = mgr._search(client, "q", entity_type="chain", top_k=3)

    assert out == ["r"]
    kwargs = client.search.call_args.kwargs
    assert kwargs["search_type"] == SearchType.HYBRID
    assert kwargs["hybrid_weights"] == (0.7, 0.3)
    assert kwargs["entity_type"] == "chain"
    assert kwargs["top_k"] == 3
    assert "namespace" not in kwargs  # unset by default


@_needs_gigaevo
def test_bm25_mode_omits_hybrid_weights():
    mgr = _mgr(memory_search_mode="bm25")
    client = MagicMock()
    client.search.return_value = []

    mgr._search(client, "q", entity_type="chain", top_k=1)

    kwargs = client.search.call_args.kwargs
    assert kwargs["search_type"] == SearchType.BM25
    assert "hybrid_weights" not in kwargs


@_needs_gigaevo
def test_vector_mode_resolves_to_vector():
    mgr = _mgr(memory_search_mode="vector")
    client = MagicMock()
    client.search.return_value = []

    mgr._search(client, "q", entity_type="chain", top_k=1)

    assert client.search.call_args.kwargs["search_type"] == SearchType.VECTOR


@_needs_gigaevo
def test_namespace_passed_when_set():
    mgr = _mgr(memory_namespace="proj-x")
    client = MagicMock()
    client.search.return_value = []

    mgr._search(client, "q", entity_type="chain", top_k=1)

    assert client.search.call_args.kwargs["namespace"] == "proj-x"


@_needs_gigaevo
def test_hybrid_rejection_falls_back_to_bm25():
    mgr = _mgr(memory_search_mode="hybrid")
    client = MagicMock()
    # first (hybrid) attempt raises; retry as bm25 succeeds
    client.search.side_effect = [RuntimeError("vector search disabled"), ["ok"]]

    out = mgr._search(client, "q", entity_type="chain", top_k=1)

    assert out == ["ok"]
    assert client.search.call_count == 2
    retry = client.search.call_args_list[1].kwargs
    assert retry["search_type"] == SearchType.BM25
    assert "hybrid_weights" not in retry


@_needs_gigaevo
def test_bm25_failure_propagates():
    mgr = _mgr(memory_search_mode="bm25")
    client = MagicMock()
    client.search.side_effect = RuntimeError("server down")

    with pytest.raises(RuntimeError):
        mgr._search(client, "q", entity_type="chain", top_k=1)


@_needs_gigaevo
def test_unknown_mode_falls_back_to_bm25():
    mgr = _mgr()
    # inject an out-of-Literal mode without triggering validation
    mgr._config = mgr._config.model_copy(update={"memory_search_mode": "weird"})
    client = MagicMock()
    client.search.return_value = []

    mgr._search(client, "q", entity_type="chain", top_k=1)

    assert client.search.call_args.kwargs["search_type"] == SearchType.BM25
