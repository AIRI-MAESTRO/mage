"""Tests for §7.3 — CARE library-ready chain metadata persistence.

Covers:
- `_merge_care_chain_metadata()` direct dict-merge fallback (when
  ``gigaevo_client`` is not installed).
- `MemoryManager.save_chain()` extended signature: passes ``task_description``,
  ``context_files``, ``mage_metadata``, ``display_name``, ``description``
  into the persisted chain dict.
- Round-trip via ``MAGEGenerator.generate()``: query is propagated to the
  ``MemoryClient.save_chain(chain=...)`` argument under
  ``chain["metadata"]["task_description"]``.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.memory import MemoryManager, _merge_care_chain_metadata
from mmar_mage.schemas import MAGEConfig

from .conftest import SAMPLE_CARL_CHAIN, _make_openai_response

# ---------------------------------------------------------------------------
# _merge_care_chain_metadata — direct dict-merge fallback
# ---------------------------------------------------------------------------


class TestMergeCareChainMetadata:
    def test_no_fields_returns_input_unchanged(self) -> None:
        chain = {"name": "x", "steps": []}
        out = _merge_care_chain_metadata(
            chain,
            task_description=None,
            context_files=None,
            mage_metadata=None,
            display_name=None,
            description=None,
        )
        # When ALL fields are None we MUST skip the merge so legacy callers
        # (e.g. save_template, save_ideas) don't get a CARE block they
        # didn't ask for.
        assert out is chain
        assert "metadata" not in out

    def test_task_description_is_persisted(self) -> None:
        chain = {"name": "x", "steps": []}
        out = _merge_care_chain_metadata(
            chain,
            task_description="Analyze quarterly revenue",
            context_files=None,
            mage_metadata=None,
            display_name=None,
            description=None,
        )
        assert out["metadata"]["task_description"] == "Analyze quarterly revenue"
        assert out["metadata"]["generated_by"] == "mage"

    def test_context_files_persisted_as_list(self) -> None:
        chain = {"name": "x", "steps": []}
        files = [{"path": "/tmp/a.pdf", "sha256": "abc", "size_bytes": 100}]
        out = _merge_care_chain_metadata(
            chain,
            task_description="x",
            context_files=files,
            mage_metadata=None,
            display_name=None,
            description=None,
        )
        assert out["metadata"]["context_files"] == files

    def test_display_name_and_description_persisted(self) -> None:
        chain = {"name": "x", "steps": []}
        out = _merge_care_chain_metadata(
            chain,
            task_description="x",
            context_files=None,
            mage_metadata=None,
            display_name="Quarterly PDF Reader",
            description="Reads PDFs and extracts KPIs.",
        )
        assert out["metadata"]["display_name"] == "Quarterly PDF Reader"
        assert out["metadata"]["description"] == "Reads PDFs and extracts KPIs."

    def test_mage_metadata_block_persisted(self) -> None:
        chain = {"name": "x", "steps": []}
        mage_meta = {"domain": "finance", "model": "gpt-4o", "num_steps": 3}
        out = _merge_care_chain_metadata(
            chain,
            task_description="x",
            context_files=None,
            mage_metadata=mage_meta,
            display_name=None,
            description=None,
        )
        assert out["metadata"]["mage_metadata"] == mage_meta

    def test_existing_metadata_preserved(self) -> None:
        """If chain_dict already carries a `metadata` block, it survives."""
        chain = {"name": "x", "steps": [], "metadata": {"custom_field": "keep me"}}
        out = _merge_care_chain_metadata(
            chain,
            task_description="t",
            context_files=None,
            mage_metadata=None,
            display_name=None,
            description=None,
        )
        assert out["metadata"]["custom_field"] == "keep me"
        assert out["metadata"]["task_description"] == "t"

    def test_input_chain_not_mutated(self) -> None:
        """Merge must NOT mutate the input chain dict (shared by other callers)."""
        chain = {"name": "x", "steps": []}
        before = json.dumps(chain, sort_keys=True)
        _merge_care_chain_metadata(
            chain,
            task_description="t",
            context_files=None,
            mage_metadata=None,
            display_name=None,
            description=None,
        )
        assert json.dumps(chain, sort_keys=True) == before


# ---------------------------------------------------------------------------
# _merge_care_chain_metadata — prefers gigaevo_client.CareChainMetadata when
# present (shimmed via a fake module)
# ---------------------------------------------------------------------------


class _FakeContextFileRef:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _FakeCareChainMetadata:
    """Fake gigaevo_client.models.CareChainMetadata used to assert MAGE
    routes through the typed helper when it's available."""

    last_args: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_args = kwargs
        self._kwargs = kwargs

    def merge_into_content(self, chain_dict: dict[str, Any]) -> dict[str, Any]:
        out = dict(chain_dict)
        out["metadata"] = {"_via_helper": True, **self._kwargs}
        return out


def _install_fake_gigaevo_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_client")
    models_mod = types.ModuleType("gigaevo_client.models")
    models_mod.CareChainMetadata = _FakeCareChainMetadata  # type: ignore[attr-defined]
    models_mod.ContextFileRef = _FakeContextFileRef  # type: ignore[attr-defined]
    pkg.models = models_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_client", pkg)
    monkeypatch.setitem(sys.modules, "gigaevo_client.models", models_mod)


def test_typed_helper_used_when_gigaevo_client_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gigaevo_client(monkeypatch)
    _FakeCareChainMetadata.last_args = {}

    out = _merge_care_chain_metadata(
        {"name": "x", "steps": []},
        task_description="Analyze X",
        context_files=[{"path": "/p", "sha256": "h", "size_bytes": 1}],
        mage_metadata={"model": "gpt-4o"},
        display_name="X Reader",
        description="Reads X.",
    )

    assert out["metadata"]["_via_helper"] is True
    args = _FakeCareChainMetadata.last_args
    assert args["task_description"] == "Analyze X"
    assert args["generated_by"] == "mage"
    assert args["display_name"] == "X Reader"
    assert args["description"] == "Reads X."
    assert args["mage_metadata"] == {"model": "gpt-4o"}
    # ContextFileRef instances, not raw dicts
    assert len(args["context_files"]) == 1
    assert args["context_files"][0].path == "/p"


# ---------------------------------------------------------------------------
# MemoryManager.save_chain — extended signature
# ---------------------------------------------------------------------------


class _RecordingMemoryClient:
    """Stand-in for gigaevo_memory.MemoryClient that records save_chain args."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _RecordingMemoryClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def save_chain(self, **kwargs: Any) -> Any:
        type(self).last_kwargs = kwargs
        ref = MagicMock()
        ref.entity_id = "entity-xyz"
        return ref


def _install_fake_gigaevo_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _RecordingMemoryClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)


@pytest.mark.asyncio
async def test_save_chain_persists_care_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_gigaevo_memory(monkeypatch)

    config = MAGEConfig(enable_memory=True, memory_base_url="http://test")
    mgr = MemoryManager(config)
    mgr._available = True  # we installed a fake gigaevo_memory above

    _RecordingMemoryClient.last_kwargs = {}
    key = await mgr.save_chain(
        {"name": "Test Chain", "steps": [{"number": 1, "title": "A", "step_type": "llm"}]},
        domain="finance",
        task_description="Read this PDF",
        context_files=[{"path": "/p.pdf", "sha256": "h", "size_bytes": 4096}],
        mage_metadata={"domain": "finance", "model": "gpt-4o"},
        display_name="PDF Reader",
        description="Extracts revenue from PDFs.",
    )

    assert key == "entity-xyz"
    persisted = _RecordingMemoryClient.last_kwargs["chain"]
    assert persisted["metadata"]["task_description"] == "Read this PDF"
    assert persisted["metadata"]["display_name"] == "PDF Reader"
    assert persisted["metadata"]["description"] == "Extracts revenue from PDFs."
    assert persisted["metadata"]["mage_metadata"]["model"] == "gpt-4o"
    assert persisted["metadata"]["context_files"][0]["path"] == "/p.pdf"
    assert persisted["metadata"]["generated_by"] == "mage"


@pytest.mark.asyncio
async def test_save_chain_backwards_compatible_when_care_args_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no CARE args are passed, the persisted chain has no metadata block.

    Keeps existing callers (template/ideas/feedback saves) unchanged.
    """
    _install_fake_gigaevo_memory(monkeypatch)

    config = MAGEConfig(enable_memory=True, memory_base_url="http://test")
    mgr = MemoryManager(config)
    mgr._available = True

    _RecordingMemoryClient.last_kwargs = {}
    await mgr.save_chain(
        {"name": "Test", "steps": [{"number": 1, "title": "A", "step_type": "llm"}]},
        domain="finance",
    )

    persisted = _RecordingMemoryClient.last_kwargs["chain"]
    # save_chain injects the server-required `version` + `metadata` block;
    # the backwards-compat invariant is that the metadata stays empty (no
    # CARE-specific fields) when CARE args are omitted.
    assert persisted.get("version")
    assert persisted.get("metadata", {}) == {}


# ---------------------------------------------------------------------------
# Generator round-trip: query flows into persisted chain's metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_persists_query_as_task_description(monkeypatch: pytest.MonkeyPatch, mocker: Any) -> None:
    _install_fake_gigaevo_memory(monkeypatch)
    _RecordingMemoryClient.last_kwargs = {}

    # Mock the LLM (fast mode → single chain response)
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(
        mode="fast",
        model="gpt-4o-test",
        api_key="x",
        enable_memory=True,
        memory_base_url="http://test",
        max_retries=0,
    )
    gen = MAGEGenerator(cfg)
    # Force memory availability — the in-process fake we installed above.
    gen._memory._available = True

    files = [{"path": "/tmp/q.pdf", "sha256": "deadbeef", "size_bytes": 1024}]
    result = await gen.generate("Analyze a quarterly financial report", context_files=files)

    assert result.memory_key == "entity-xyz"
    persisted = _RecordingMemoryClient.last_kwargs["chain"]

    md = persisted["metadata"]
    assert md["task_description"] == "Analyze a quarterly financial report"
    assert md["context_files"] == files
    assert md["generated_by"] == "mage"
    # display_name + description come from the §7.2 suggested_* fields
    assert md["display_name"]  # non-empty heuristic value
    assert md["description"]
    # mage_metadata round-trips MAGEMetadata.model_dump()
    assert md["mage_metadata"]["model"] == "gpt-4o-test"
    assert md["mage_metadata"]["num_steps"] == 3


@pytest.mark.asyncio
async def test_generator_works_without_context_files(monkeypatch: pytest.MonkeyPatch, mocker: Any) -> None:
    """Calling generate() without context_files is the common case."""
    _install_fake_gigaevo_memory(monkeypatch)
    _RecordingMemoryClient.last_kwargs = {}

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(
        mode="fast",
        model="gpt-4o-test",
        api_key="x",
        enable_memory=True,
        memory_base_url="http://test",
        max_retries=0,
    )
    gen = MAGEGenerator(cfg)
    gen._memory._available = True

    result = await gen.generate("Any query")
    assert result.memory_key == "entity-xyz"

    md = _RecordingMemoryClient.last_kwargs["chain"]["metadata"]
    assert md["task_description"] == "Any query"
    # context_files key is omitted when not supplied (caller didn't pass any)
    assert "context_files" not in md or md["context_files"] is None
