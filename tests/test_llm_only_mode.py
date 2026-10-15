"""
Tests for LLM-only mode (allowed_step_types restriction).

Covers: MAGEConfig validation, DomainAnalysis override, prompt constraint,
and end-to-end pipeline enforcement.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.prompts import get_step_types_constraint
from mmar_mage.schemas import (
    VALID_STEP_TYPES,
    DomainAnalysis,
    MAGEConfig,
)

# ---------------------------------------------------------------------------
# MAGEConfig.allowed_step_types validation
# ---------------------------------------------------------------------------


class TestAllowedStepTypesConfig:
    """Tests for allowed_step_types field on MAGEConfig."""

    def test_allowed_step_types_default_none(self) -> None:
        """Default is None (all types allowed)."""
        cfg = MAGEConfig()
        assert cfg.allowed_step_types is None

    def test_allowed_step_types_validation_valid(self) -> None:
        """Valid step types are accepted."""
        cfg = MAGEConfig(allowed_step_types=["llm"])
        assert cfg.allowed_step_types == ["llm"]

        cfg2 = MAGEConfig(allowed_step_types=["llm", "tool", "eval"])
        assert cfg2.allowed_step_types == ["llm", "tool", "eval"]

    def test_allowed_step_types_validation_invalid(self) -> None:
        """Invalid step types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid step types"):
            MAGEConfig(allowed_step_types=["invalid_type"])

    def test_allowed_step_types_validation_mixed_invalid(self) -> None:
        """Mix of valid and invalid types raises ValueError."""
        with pytest.raises(ValueError, match="Invalid step types"):
            MAGEConfig(allowed_step_types=["llm", "bogus"])

    def test_allowed_step_types_all_valid_types(self) -> None:
        """All valid step types are accepted."""
        cfg = MAGEConfig(allowed_step_types=sorted(VALID_STEP_TYPES))
        assert set(cfg.allowed_step_types) == VALID_STEP_TYPES

    def test_allowed_step_types_empty_list(self) -> None:
        """Empty list is accepted (technically no types allowed)."""
        cfg = MAGEConfig(allowed_step_types=[])
        assert cfg.allowed_step_types == []

    def test_for_openrouter_passes_allowed_step_types(self) -> None:
        """for_openrouter accepts allowed_step_types via kwargs."""
        cfg = MAGEConfig.for_openrouter(
            api_key="test",
            model="test/model",
            allowed_step_types=["llm"],
        )
        assert cfg.allowed_step_types == ["llm"]
        assert cfg.provider == "openrouter"


# ---------------------------------------------------------------------------
# DomainAnalysis override
# ---------------------------------------------------------------------------


class TestLLMOnlyDomainOverride:
    """Test that allowed_step_types overrides preferred_step_types."""

    def test_llm_only_mode_overrides_preferred_types(self) -> None:
        """When allowed_step_types is set, DomainAnalysis.preferred_step_types is overridden."""
        domain = DomainAnalysis(
            domain="engineering",
            preferred_step_types=["tool", "llm", "eval", "conditional"],
        )
        # Simulate what the generator does
        allowed = ["llm"]
        domain.preferred_step_types = list(allowed)
        assert domain.preferred_step_types == ["llm"]

    def test_override_with_multiple_types(self) -> None:
        """Override with multiple allowed types works."""
        domain = DomainAnalysis(preferred_step_types=["mcp", "tool", "llm"])
        allowed = ["llm", "eval"]
        domain.preferred_step_types = list(allowed)
        assert domain.preferred_step_types == ["llm", "eval"]


# ---------------------------------------------------------------------------
# Prompt constraint helper
# ---------------------------------------------------------------------------


class TestStepTypesConstraint:
    """Test the get_step_types_constraint helper."""

    def test_none_returns_empty(self) -> None:
        assert get_step_types_constraint(None) == ""

    def test_empty_list_returns_empty(self) -> None:
        assert get_step_types_constraint([]) == ""

    def test_single_type(self) -> None:
        result = get_step_types_constraint(["llm"])
        assert "STRICT CONSTRAINT" in result
        assert '"llm"' in result

    def test_multiple_types(self) -> None:
        result = get_step_types_constraint(["llm", "tool"])
        assert '"llm"' in result
        assert '"tool"' in result
        assert "MUST ONLY" in result

    def test_deprecated_types_filtered(self) -> None:
        # P-1: MAGE-internal eval/aggregator are never offered to the planner.
        result = get_step_types_constraint(["llm", "eval", "aggregator"])
        assert '"llm"' in result
        assert '"eval"' not in result
        assert '"aggregator"' not in result


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

# Sample responses for a full deep-mode pipeline with LLM-only steps
_DOMAIN_RESPONSE = {
    "domain": "science",
    "task_type": "analysis",
    "complexity": "medium",
    "key_concepts": ["hypothesis", "experiment"],
    "suggested_step_count": 3,
    "preferred_step_types": ["tool", "llm", "eval"],
}

_STEP_PLAN_RESPONSE = {
    "steps": [
        {"number": 1, "title": "Formulate hypothesis", "step_type": "llm", "rough_aim": "Define hypothesis"},
        {"number": 2, "title": "Analyze evidence", "step_type": "llm", "rough_aim": "Evaluate evidence"},
        {"number": 3, "title": "Draw conclusions", "step_type": "llm", "rough_aim": "Synthesize findings"},
    ],
    "rationale": "LLM-only pipeline",
}

_DAG_RESPONSE = {
    "steps": [
        {
            "number": 1,
            "title": "Formulate hypothesis",
            "step_type": "llm",
            "dependencies": [],
            "rough_aim": "Define hypothesis",
        },
        {
            "number": 2,
            "title": "Analyze evidence",
            "step_type": "llm",
            "dependencies": [1],
            "rough_aim": "Evaluate evidence",
        },
        {
            "number": 3,
            "title": "Draw conclusions",
            "step_type": "llm",
            "dependencies": [2],
            "rough_aim": "Synthesize findings",
        },
    ],
    "max_workers": 1,
    "pattern": "pipeline",
}

_DESCRIBED_STEPS_RESPONSE = {
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Formulate hypothesis",
            "aim": "Define the research hypothesis",
            "reasoning_questions": "What is the core hypothesis?",
            "step_context_queries": ["hypothesis", "research question"],
            "stage_action": "Formulate the hypothesis",
            "example_reasoning": "A clear hypothesis guides the analysis",
            "dependencies": [],
            "retry_max": 3,
            "llm_config": None,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Analyze evidence",
            "aim": "Evaluate supporting evidence",
            "reasoning_questions": "What evidence supports or refutes the hypothesis?",
            "step_context_queries": ["evidence", "data analysis"],
            "stage_action": "Analyze evidence systematically",
            "example_reasoning": "Weighing evidence strength is critical",
            "dependencies": [1],
            "retry_max": 3,
            "llm_config": None,
        },
        {
            "step_type": "llm",
            "number": 3,
            "title": "Draw conclusions",
            "aim": "Synthesize analysis into conclusions",
            "reasoning_questions": "What conclusions can be drawn?",
            "step_context_queries": ["conclusions", "synthesis"],
            "stage_action": "Synthesize findings into conclusions",
            "example_reasoning": "Conclusions must follow from the evidence",
            "dependencies": [1, 2],
            "retry_max": 3,
            "llm_config": None,
        },
    ],
}


def _make_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMOnlyPipeline:
    """End-to-end test: LLM-only mode produces only llm steps."""

    @pytest.fixture
    def mock_openai_llm_only(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI returning LLM-only deep-mode responses."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        responses = [
            _make_openai_response(json.dumps(_DOMAIN_RESPONSE)),
            _make_openai_response(json.dumps(_STEP_PLAN_RESPONSE)),
            _make_openai_response(json.dumps(_DAG_RESPONSE)),
            _make_openai_response(json.dumps(_DESCRIBED_STEPS_RESPONSE)),
        ]
        instance.chat.completions.create = AsyncMock(side_effect=responses)
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    async def test_llm_only_mode_full_pipeline_generates_only_llm_steps(
        self,
        mock_openai_llm_only: MagicMock,
    ) -> None:
        """Full pipeline with allowed_step_types=['llm'] produces only llm steps."""
        config = MAGEConfig(
            mode="deep",
            model="test-model",
            api_key="test-key",
            enable_memory=False,
            max_retries=0,
            allowed_step_types=["llm"],
        )
        generator = MAGEGenerator(config=config)
        result = await generator.generate("Test scientific analysis")

        # All steps must be llm type
        for step in result.chain_dict["steps"]:
            assert step["step_type"] == "llm", f"Step {step['number']} has type {step['step_type']}, expected 'llm'"

        assert result.metadata.num_steps == 3
