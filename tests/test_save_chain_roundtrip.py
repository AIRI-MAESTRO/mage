"""§9 P1 save_chain ↔ get_chain round-trip test.

Verifies the `CareChainMetadata`-shaped block written by
``MemoryManager.save_chain`` (§7.3) is present in the retrieved chain
when read back via a server-shaped ``MemoryClient.get_chain``. This
test stands in for the live-server integration test described in the
spec — uses an in-process fake `gigaevo_memory.MemoryClient` that
persists what `save_chain(chain=...)` sends and returns it on
`get_chain(entity_id=...)`, simulating server behaviour.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig


class _RoundTripMemoryClient:
    """Fake `gigaevo_memory.MemoryClient` simulating server persist+retrieve.

    Stores the dict passed to ``save_chain(chain=...)`` keyed by
    auto-assigned ``entity_id``; returns it through ``get_chain``.
    The returned object exposes a ``.chain`` attribute so the test
    matches what a real `gigaevo_memory` client would expose.
    """

    storage: dict[str, dict[str, Any]] = {}
    counter: int = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _RoundTripMemoryClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def save_chain(self, **kwargs: Any) -> Any:
        cls = type(self)
        cls.counter += 1
        eid = f"chain-{cls.counter}"
        cls.storage[eid] = dict(kwargs)
        ref = MagicMock()
        ref.entity_id = eid
        return ref

    def get_chain(self, entity_id: str) -> Any:
        cls = type(self)
        if entity_id not in cls.storage:
            raise KeyError(entity_id)
        record = cls.storage[entity_id]
        obj = MagicMock()
        obj.entity_id = entity_id
        obj.chain = record.get("chain")
        obj.name = record.get("name")
        obj.tags = record.get("tags")
        return obj


def _install_round_trip_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the in-process round-trip fake as `gigaevo_memory`."""
    _RoundTripMemoryClient.storage = {}
    _RoundTripMemoryClient.counter = 0
    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _RoundTripMemoryClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)


# ---------------------------------------------------------------------------
# The round-trip contract: every CareChainMetadata field survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_chain_get_chain_preserves_care_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`save_chain` → simulated server → `get_chain` preserves every
    CareChainMetadata field."""
    _install_round_trip_client(monkeypatch)

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    chain_dict = {
        "name": "Quarterly Demo",
        "max_workers": 2,
        "steps": [{"step_type": "llm", "number": 1, "title": "Read"}],
    }
    context_files = [
        {"path": "/tmp/q3.pdf", "sha256": "a" * 64, "size_bytes": 4096},
        {"path": "/tmp/q4.pdf", "sha256": "b" * 64, "size_bytes": 8192},
    ]
    mage_metadata = {
        "domain": "finance",
        "num_steps": 1,
        "model": "gpt-4o",
        "suggested_display_name": "Quarterly Analyst",
    }

    entity_id = await mgr.save_chain(
        chain_dict,
        domain="finance",
        name="quarterly-demo",
        task_description="Analyze quarterly revenue",
        context_files=context_files,
        mage_metadata=mage_metadata,
        display_name="Quarterly Analyst",
        description="Reads quarterly reports and extracts revenue trends.",
        evolution_meta={"step_critique_score": 0.82, "verification_passed": True},
    )
    assert entity_id == "chain-1"

    # Now simulate the server-side retrieve via `MemoryClient.get_chain`.
    # The §7.3 contract is: the persisted `chain` dict carries a
    # `metadata` block with the CareChainMetadata shape.
    with _RoundTripMemoryClient(base_url="http://t") as client:
        retrieved = client.get_chain(entity_id)

    persisted_chain = retrieved.chain
    assert persisted_chain is not None

    # ---- §7.3 CARE metadata block round-trips ----
    metadata = persisted_chain.get("metadata")
    assert isinstance(metadata, dict), "missing CareChainMetadata block"
    assert metadata["task_description"] == "Analyze quarterly revenue"
    assert metadata["generated_by"] == "mage"
    assert metadata["display_name"] == "Quarterly Analyst"
    assert metadata["description"] == "Reads quarterly reports and extracts revenue trends."

    # context_files round-trips with full structure
    assert metadata["context_files"] == context_files

    # mage_metadata round-trips with its dump
    assert metadata["mage_metadata"]["domain"] == "finance"
    assert metadata["mage_metadata"]["model"] == "gpt-4o"
    assert metadata["mage_metadata"]["suggested_display_name"] == "Quarterly Analyst"

    # §5 P2 evolution_meta block round-trips
    assert metadata["evolution_meta"]["step_critique_score"] == 0.82
    assert metadata["evolution_meta"]["verification_passed"] is True

    # The original chain payload survives alongside the metadata.
    assert persisted_chain.get("name") == "Quarterly Demo"
    assert persisted_chain.get("max_workers") == 2
    assert len(persisted_chain.get("steps") or []) == 1


@pytest.mark.asyncio
async def test_save_chain_get_chain_with_only_task_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even the minimum CARE payload (just `task_description`) survives the round-trip."""
    _install_round_trip_client(monkeypatch)

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    eid = await mgr.save_chain(
        {"name": "Mini", "steps": [{"step_type": "llm", "number": 1, "title": "A"}]},
        domain="general",
        task_description="hello world",
    )
    assert eid

    with _RoundTripMemoryClient(base_url="http://t") as client:
        got = client.get_chain(eid)

    metadata = got.chain.get("metadata")
    assert metadata is not None
    assert metadata["task_description"] == "hello world"
    assert metadata["generated_by"] == "mage"
    # Optional fields default to absent (we didn't pass them in)
    assert "context_files" not in metadata
    assert "display_name" not in metadata


@pytest.mark.asyncio
async def test_save_chain_backwards_compat_no_metadata_when_care_args_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When NO CARE kwargs are supplied, the stored chain has no metadata
    block — confirms legacy callers (save_template, save_ideas) aren't
    accidentally CARE-tagged."""
    _install_round_trip_client(monkeypatch)

    cfg = MAGEConfig(enable_memory=True, memory_base_url="http://t")
    mgr = MemoryManager(cfg)
    mgr._available = True

    eid = await mgr.save_chain(
        {"name": "Legacy", "steps": [{"step_type": "llm", "number": 1, "title": "A"}]},
        domain="finance",
    )
    assert eid

    with _RoundTripMemoryClient(base_url="http://t") as client:
        got = client.get_chain(eid)
    # The Memory server requires a top-level `version` + `metadata` block, so
    # save_chain injects defaults. The backwards-compat invariant is that NO
    # CARE-specific fields are added when CARE args are omitted — the metadata
    # block stays empty.
    assert got.chain.get("version")
    assert got.chain.get("metadata", {}) == {}
