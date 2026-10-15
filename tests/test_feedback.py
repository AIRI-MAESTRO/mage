"""Tests for submit_feedback flow."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig, MAGEResult


class TestSubmitFeedback:
    """Tests for MAGEResult.submit_feedback()."""

    @pytest.mark.asyncio
    async def test_submit_feedback_no_memory_key(self) -> None:
        """submit_feedback is a no-op when memory_key is None."""
        result = MAGEResult(chain_json='{"steps":[]}', memory_key=None)
        result._config = MAGEConfig(enable_memory=True, api_key="test")
        await result.submit_feedback(quality_score=0.9, success=True)

    @pytest.mark.asyncio
    async def test_submit_feedback_no_config(self) -> None:
        """submit_feedback is a no-op when _config is None."""
        result = MAGEResult(chain_json='{"steps":[]}', memory_key="mem-123")
        await result.submit_feedback(quality_score=0.9, success=True)

    @pytest.mark.asyncio
    async def test_submit_feedback_calls_memory(self, mocker: Any) -> None:
        """submit_feedback delegates to MemoryManager.submit_feedback()."""
        config = MAGEConfig(enable_memory=True, api_key="test")
        result = MAGEResult(
            chain_json='{"steps":[]}',
            memory_key="mem-abc-123",
        )
        result._config = config

        mock_submit = mocker.patch(
            "mmar_mage.memory.MemoryManager.submit_feedback",
            new_callable=AsyncMock,
        )

        await result.submit_feedback(
            quality_score=0.8,
            success=True,
            failed_at_step=None,
            failure_reason=None,
        )

        mock_submit.assert_called_once_with(
            "mem-abc-123",
            {
                "quality_score": 0.8,
                "success": True,
                "failed_at_step": None,
                "failure_reason": None,
            },
        )

    @pytest.mark.asyncio
    async def test_submit_feedback_with_failure(self, mocker: Any) -> None:
        """submit_feedback includes failure details."""
        config = MAGEConfig(enable_memory=True, api_key="test")
        result = MAGEResult(
            chain_json='{"steps":[]}',
            memory_key="mem-456",
        )
        result._config = config

        mock_submit = mocker.patch(
            "mmar_mage.memory.MemoryManager.submit_feedback",
            new_callable=AsyncMock,
        )

        await result.submit_feedback(
            quality_score=0.3,
            success=False,
            failed_at_step=2,
            failure_reason="Timeout at step 2",
        )

        mock_submit.assert_called_once()
        call_args = mock_submit.call_args[0]
        assert call_args[1]["success"] is False
        assert call_args[1]["failed_at_step"] == 2


class TestMemoryManagerFeedback:
    """Tests for MemoryManager feedback methods."""

    @pytest.mark.asyncio
    async def test_submit_feedback_disabled(self) -> None:
        """submit_feedback is a no-op when memory disabled."""
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        await mgr.submit_feedback("mem-123", {"quality_score": 0.9})

    @pytest.mark.asyncio
    async def test_submit_feedback_not_available(self) -> None:
        """submit_feedback is a no-op when gigaevo-memory not installed."""
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        await mgr.submit_feedback("mem-123", {"quality_score": 0.9})

    @pytest.mark.asyncio
    async def test_get_capabilities_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        result = await mgr.get_capabilities()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_capabilities_not_available(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        result = await mgr.get_capabilities()
        assert result is None

    @pytest.mark.asyncio
    async def test_search_templates_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        result = await mgr.search_templates("finance", "analysis", "medium")
        assert result == []

    @pytest.mark.asyncio
    async def test_search_templates_not_available(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        result = await mgr.search_templates("finance", "analysis", "medium")
        assert result == []

    @pytest.mark.asyncio
    async def test_save_template_disabled(self) -> None:
        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        result = await mgr.save_template({}, "finance", "analysis", 0.9)
        assert result is None

    @pytest.mark.asyncio
    async def test_save_template_not_available(self) -> None:
        config = MAGEConfig(enable_memory=True)
        mgr = MemoryManager(config)
        mgr._available = False
        result = await mgr.save_template({}, "finance", "analysis", 0.9)
        assert result is None
