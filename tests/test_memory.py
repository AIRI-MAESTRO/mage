"""Tests for the MemoryManager integration layer."""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig

SAMPLE_CHAIN_DICT: dict[str, Any] = {
    "name": "Test",
    "steps": [{"number": 1, "title": "A", "step_type": "llm"}],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryManagerDisabled:
    """Memory disabled scenarios."""

    @pytest.mark.asyncio
    async def test_save_returns_none_when_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        key = await mgr.save_chain(SAMPLE_CHAIN_DICT, domain="test")
        assert key is None

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        results = await mgr.search_similar("test")
        assert results == []


class TestMemoryManagerNotInstalled:
    """gigaevo-memory not installed scenarios."""

    @pytest.mark.asyncio
    async def test_save_returns_none_when_not_installed(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        # Force _available to False to simulate missing package
        mgr._available = False
        key = await mgr.save_chain(SAMPLE_CHAIN_DICT, domain="test")
        assert key is None

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_not_installed(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        results = await mgr.search_similar("test")
        assert results == []


class TestMemoryManagerSave:
    """Successful save scenarios with mocked gigaevo_memory."""

    @pytest.mark.asyncio
    async def test_save_chain_success(self, mocker: Any) -> None:
        """save_chain returns entity_id on success."""
        config = MAGEConfig(enable_memory=True, memory_tags=["extra"])
        mgr = MemoryManager(config)
        mgr._available = True

        mock_ref = MagicMock()
        mock_ref.entity_id = "mem-abc-123"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.save_chain.return_value = mock_ref

        mocker.patch("mmar_mage.memory.MemoryManager._do_save", return_value="mem-abc-123")

        key = await mgr.save_chain(SAMPLE_CHAIN_DICT, domain="finance")
        assert key == "mem-abc-123"

    @pytest.mark.asyncio
    async def test_save_chain_failure_returns_none(self, mocker: Any) -> None:
        """save_chain returns None on error (graceful degradation)."""
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = True

        mocker.patch(
            "mmar_mage.memory.MemoryManager._do_save",
            side_effect=Exception("Connection refused"),
        )

        key = await mgr.save_chain(SAMPLE_CHAIN_DICT, domain="test")
        assert key is None


class TestMemoryManagerAvailability:
    """Check availability detection."""

    def test_check_availability_false(self) -> None:
        """Returns False when neither memory SDK is importable."""
        with patch.dict("sys.modules", {"gigaevo_client": None, "gigaevo_memory": None}):
            assert MemoryManager._check_availability() is False

    def test_check_availability_true_with_client(self) -> None:
        """Returns True via the renamed ``gigaevo_client`` (0.3.0)."""
        with patch.dict(sys.modules, {"gigaevo_client": types.ModuleType("gigaevo_client")}):
            assert MemoryManager._check_availability() is True
