"""Tests for benchmark-related MemoryManager hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mmar_mage import MAGEConfig
from mmar_mage.memory import MemoryManager


def _config_with_memory(enabled: bool) -> MAGEConfig:
    return MAGEConfig(
        mode="fast",
        model="gpt-4o-test",
        api_key="k",
        enable_memory=enabled,
        memory_base_url="http://localhost:8002",
    )


class TestSaveBenchmark:
    async def test_no_op_when_memory_disabled(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=False))
        result = await mgr.save_benchmark(
            domain="finance",
            tasks=[{"task_id": "t1", "input": "x"}],
            best_chain={"steps": []},
            best_score=0.7,
            query="q",
        )
        assert result is None

    async def test_no_op_when_gigaevo_unavailable(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=True))
        # Force unavailable regardless of actual install state.
        mgr._available = False
        result = await mgr.save_benchmark(
            domain="finance",
            tasks=[],
            best_chain={"steps": []},
            best_score=0.0,
            query="q",
        )
        assert result is None

    async def test_save_call_uses_typed_memory_card(self) -> None:
        """P1.19: benchmarks persist as typed memory cards, not abused `chain` entities."""
        mgr = MemoryManager(_config_with_memory(enabled=True))
        mgr._available = True

        fake_client = MagicMock()
        fake_ref = MagicMock()
        fake_ref.entity_id = "ent_123"
        fake_client.save_memory_card.return_value = fake_ref
        fake_cm = MagicMock()
        fake_cm.__enter__ = MagicMock(return_value=fake_client)
        fake_cm.__exit__ = MagicMock(return_value=False)

        fake_module = MagicMock(MemoryClient=MagicMock(return_value=fake_cm))
        with patch.dict("sys.modules", {"gigaevo_memory": fake_module}):
            entity_id = await mgr.save_benchmark(
                domain="finance",
                tasks=[{"task_id": "t1", "input": "x", "rubric": ["r"]}],
                best_chain={"name": "best", "steps": []},
                best_score=0.85,
                query="Verify finance fact",
            )

        assert entity_id == "ent_123"
        fake_client.save_memory_card.assert_called_once()
        fake_client.save_chain.assert_not_called()  # no longer abuses the chain entity
        kwargs = fake_client.save_memory_card.call_args.kwargs
        assert kwargs["name"] == "mage_benchmark_finance"
        assert "finance" in kwargs["tags"]
        assert "mage-benchmark" in kwargs["tags"]
        payload = kwargs["memory_card"]
        assert payload["kind"] == "mage_benchmark"
        assert payload["domain"] == "finance"
        assert payload["best_score"] == 0.85
        assert payload["task_count"] == 1


class TestRecallBenchmarks:
    async def test_returns_empty_when_disabled(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=False))
        out = await mgr.recall_benchmarks(domain="finance")
        assert out == []

    async def test_returns_empty_when_search_fails(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=True))
        mgr._available = True
        fake_module = MagicMock()
        fake_module.MemoryClient.side_effect = RuntimeError("connection refused")
        fake_module.SearchType = MagicMock()
        with patch.dict("sys.modules", {"gigaevo_memory": fake_module}):
            out = await mgr.recall_benchmarks(domain="finance")
        assert out == []

    async def test_filters_by_kind_and_domain(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=True))
        mgr._available = True

        # P1.19: recall now reads memory-card hits (`.content` + `.score`), not `.chain`.
        good_hit = MagicMock()
        good_hit.entity_id = "ent_good"
        good_hit.score = 0.77
        good_hit.content = {
            "kind": "mage_benchmark",
            "domain": "finance",
            "tasks": [{"task_id": "t1", "input": "x"}],
            "query": "q",
            "best_score": 0.9,
        }
        bad_hit = MagicMock()
        bad_hit.entity_id = "ent_bad"
        bad_hit.score = 0.5
        bad_hit.content = {"kind": "mage_template", "domain": "finance"}  # wrong kind
        mismatch_hit = MagicMock()
        mismatch_hit.entity_id = "ent_other"
        mismatch_hit.score = 0.5
        mismatch_hit.content = {"kind": "mage_benchmark", "domain": "medicine", "tasks": []}

        fake_client = MagicMock()
        fake_client.search_hits.return_value = [good_hit, bad_hit, mismatch_hit]
        fake_cm = MagicMock()
        fake_cm.__enter__ = MagicMock(return_value=fake_client)
        fake_cm.__exit__ = MagicMock(return_value=False)
        fake_module = MagicMock(MemoryClient=MagicMock(return_value=fake_cm), SearchType=MagicMock())

        with patch.dict("sys.modules", {"gigaevo_memory": fake_module}):
            out = await mgr.recall_benchmarks(domain="finance", top_k=10)

        # Only the good hit survives the kind+domain filter.
        assert len(out) == 1
        assert out[0]["entity_id"] == "ent_good"
        assert out[0]["tasks"] == [{"task_id": "t1", "input": "x"}]
        assert out[0]["best_score"] == 0.9
        assert out[0]["score"] == 0.77  # relevance score now surfaced (P1.18 scored recall)


class TestBenchmarkRoundTrip:
    """Real-execution scenario: save → recall through one shared (fake) card store."""

    async def test_saved_benchmark_is_recalled(self) -> None:
        mgr = MemoryManager(_config_with_memory(enabled=True))
        mgr._available = True

        class _Store:
            def __init__(self) -> None:
                self.cards: list[dict] = []

            def save_memory_card(self, *, memory_card, name, tags, when_to_use=None, namespace=None):  # noqa: ANN001
                self.cards.append(memory_card)
                ref = MagicMock()
                ref.entity_id = f"card_{len(self.cards)}"
                return ref

            def search_hits(self, query, **kwargs):  # noqa: ANN001, ARG002
                hits = []
                for i, c in enumerate(self.cards):
                    h = MagicMock()
                    h.entity_id = f"card_{i + 1}"
                    h.content = c
                    h.score = 0.9
                    hits.append(h)
                return hits

            def __enter__(self):
                return self

            def __exit__(self, *a):  # noqa: ANN002
                return False

        store = _Store()
        fake_module = MagicMock(MemoryClient=MagicMock(return_value=store), SearchType=MagicMock())

        with patch.dict("sys.modules", {"gigaevo_memory": fake_module}):
            ent = await mgr.save_benchmark(
                domain="finance",
                tasks=[{"task_id": "t1", "input": "x", "rubric": ["r"]}],
                best_chain={"name": "best", "steps": []},
                best_score=0.85,
                query="Verify finance fact",
            )
            recalled = await mgr.recall_benchmarks(domain="finance", top_k=5)

        assert ent == "card_1"
        assert len(recalled) == 1
        assert recalled[0]["query"] == "Verify finance fact"
        assert recalled[0]["tasks"] == [{"task_id": "t1", "input": "x", "rubric": ["r"]}]
        assert recalled[0]["best_score"] == 0.85
