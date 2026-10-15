"""Tests for TemplateAssemblerAgent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mmar_mage.agents.template_assembler_agent import TemplateAssemblerAgent, TemplateSkeleton
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig


class TestTemplateSkeleton:
    """Tests for TemplateSkeleton model."""

    def test_defaults(self) -> None:
        skel = TemplateSkeleton()
        assert skel.template_id is None
        assert skel.abstract_dag == []
        assert skel.step_type_sequence == []
        assert skel.quality_score == 0.0
        assert skel.success_rate == 0.0

    def test_custom_values(self) -> None:
        skel = TemplateSkeleton(
            template_id="tpl-123",
            abstract_dag=[{"number": 1, "step_type": "llm"}],
            step_type_sequence=["llm", "eval", "llm"],
            quality_score=0.9,
            success_rate=0.85,
        )
        assert skel.template_id == "tpl-123"
        assert len(skel.abstract_dag) == 1
        assert skel.quality_score == 0.9


class TestTemplateAssemblerAgent:
    """Tests for TemplateAssemblerAgent."""

    @pytest.mark.asyncio
    async def test_assemble_no_templates_synthesizes_skeleton(self, mocker: Any) -> None:
        # P1.3: a memory miss now yields a built-in skeleton (soft guidance), not None.
        config = MAGEConfig(enable_memory=True, enable_template_assembly=True, api_key="test")
        memory = MemoryManager(config)
        mocker.patch.object(memory, "search_templates", new_callable=AsyncMock, return_value=[])

        agent = TemplateAssemblerAgent(memory, config)
        result = await agent.assemble("finance", "analysis", "medium")

        assert result is not None
        # "analysis" → multi_source_analysis skeleton
        assert result.template_id == "builtin:multi_source_analysis"
        assert result.step_type_sequence  # non-empty, so the planner can use it
        assert result.abstract_dag
        assert result.quality_score == 0.0  # marked as unproven guidance

    @pytest.mark.asyncio
    async def test_assemble_selects_best(self, mocker: Any) -> None:
        config = MAGEConfig(enable_memory=True, enable_template_assembly=True, api_key="test")
        memory = MemoryManager(config)

        templates = [
            {
                "entity_id": "tpl-1",
                "abstract_dag": [{"number": 1}],
                "step_type_sequence": ["llm"],
                "quality_score": 0.7,
                "success_rate": 0.8,
            },
            {
                "entity_id": "tpl-2",
                "abstract_dag": [{"number": 1}, {"number": 2}],
                "step_type_sequence": ["llm", "eval"],
                "quality_score": 0.9,
                "success_rate": 0.9,
            },
        ]
        mocker.patch.object(memory, "search_templates", new_callable=AsyncMock, return_value=templates)

        agent = TemplateAssemblerAgent(memory, config)
        result = await agent.assemble("finance", "analysis", "complex")

        assert result is not None
        assert result.template_id == "tpl-2"
        assert result.quality_score == 0.9
        assert result.success_rate == 0.9

    @pytest.mark.asyncio
    async def test_assemble_handles_exception(self, mocker: Any) -> None:
        config = MAGEConfig(enable_memory=True, api_key="test")
        memory = MemoryManager(config)
        mocker.patch.object(
            memory,
            "search_templates",
            new_callable=AsyncMock,
            side_effect=Exception("Search failed"),
        )

        agent = TemplateAssemblerAgent(memory, config)
        result = await agent.assemble("finance", "analysis", "medium")

        assert result is None

    @pytest.mark.asyncio
    async def test_assemble_single_template(self, mocker: Any) -> None:
        config = MAGEConfig(enable_memory=True, api_key="test")
        memory = MemoryManager(config)

        templates = [
            {
                "entity_id": "tpl-only",
                "abstract_dag": [{"number": 1, "step_type": "llm"}],
                "step_type_sequence": ["llm"],
                "quality_score": 0.5,
                "success_rate": 0.6,
            },
        ]
        mocker.patch.object(memory, "search_templates", new_callable=AsyncMock, return_value=templates)

        agent = TemplateAssemblerAgent(memory, config)
        result = await agent.assemble("general", "qa", "simple")

        assert result is not None
        assert result.template_id == "tpl-only"
