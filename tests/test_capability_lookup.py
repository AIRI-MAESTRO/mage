"""Tests for CapabilityLookupAgent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mmar_mage.agents.capability_lookup_agent import CapabilityContext, CapabilityLookupAgent
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig


class TestCapabilityContext:
    """Tests for CapabilityContext model."""

    def test_defaults(self) -> None:
        ctx = CapabilityContext()
        assert ctx.tools == []
        assert ctx.mcp_servers == []
        assert ctx.environment_id == "default"

    def test_custom_values(self) -> None:
        ctx = CapabilityContext(
            tools=[{"name": "fetch", "description": "Fetch data"}],
            mcp_servers=[{"server_name": "postgres"}],
            environment_id="prod",
        )
        assert len(ctx.tools) == 1
        assert len(ctx.mcp_servers) == 1
        assert ctx.environment_id == "prod"


class TestCapabilityLookupAgent:
    """Tests for CapabilityLookupAgent."""

    @pytest.mark.asyncio
    async def test_lookup_returns_empty_when_no_registry(self, mocker: Any) -> None:
        config = MAGEConfig(
            enable_memory=True,
            enable_capability_lookup=True,
            api_key="test",
        )
        memory = MemoryManager(config)
        mocker.patch.object(memory, "get_capabilities", new_callable=AsyncMock, return_value=None)

        agent = CapabilityLookupAgent(memory, config)
        result = await agent.lookup()

        assert isinstance(result, CapabilityContext)
        assert result.tools == []
        assert result.mcp_servers == []

    @pytest.mark.asyncio
    async def test_lookup_returns_capabilities(self, mocker: Any) -> None:
        config = MAGEConfig(
            enable_memory=True,
            enable_capability_lookup=True,
            capability_environment_id="test-env",
            api_key="test",
        )
        memory = MemoryManager(config)
        mocker.patch.object(
            memory,
            "get_capabilities",
            new_callable=AsyncMock,
            return_value={
                "tools": [{"name": "calculator", "description": "Math ops"}],
                "mcp_servers": [{"server_name": "postgres", "transport": "stdio"}],
            },
        )

        agent = CapabilityLookupAgent(memory, config)
        result = await agent.lookup()

        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "calculator"
        assert len(result.mcp_servers) == 1
        assert result.environment_id == "test-env"

    @pytest.mark.asyncio
    async def test_lookup_handles_exception(self, mocker: Any) -> None:
        config = MAGEConfig(enable_memory=True, api_key="test")
        memory = MemoryManager(config)
        mocker.patch.object(
            memory,
            "get_capabilities",
            new_callable=AsyncMock,
            side_effect=Exception("Connection failed"),
        )

        agent = CapabilityLookupAgent(memory, config)
        result = await agent.lookup()

        assert isinstance(result, CapabilityContext)
        assert result.tools == []
