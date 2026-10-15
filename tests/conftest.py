"""
Shared test fixtures for MAGE tests.

Provides mock LLM responses for every pipeline stage, a pre-configured
MAGEConfig, and helper factories.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.schemas import MAGEConfig

# ---------------------------------------------------------------------------
# Sample CARL chain used across tests
# ---------------------------------------------------------------------------

SAMPLE_CARL_CHAIN: dict[str, Any] = {
    "name": "Test Analysis Chain",
    "max_workers": 2,
    "timeout": 300.0,
    "search_config": {
        "strategy": "substring",
        "substring_config": {
            "case_sensitive": False,
            "min_word_length": 3,
            "max_matches_per_query": 5,
        },
    },
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Identify document type",
            "aim": "Determine the type of document",
            "reasoning_questions": "What type of document is this?",
            "step_context_queries": ["document", "type", "structure"],
            "stage_action": "Classify the document",
            "example_reasoning": "The document type determines the analysis approach",
            "dependencies": [],
            "retry_max": 3,
            "llm_config": None,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Extract key information",
            "aim": "Extract all key facts and data",
            "reasoning_questions": "What are the main facts presented?",
            "step_context_queries": ["facts", "data", "key information"],
            "stage_action": "Extract and categorize information",
            "example_reasoning": "Key facts form the basis for conclusions",
            "dependencies": [],
            "retry_max": 3,
            "llm_config": None,
        },
        {
            "step_type": "llm",
            "number": 3,
            "title": "Synthesize conclusions",
            "aim": "Form a structured summary",
            "reasoning_questions": "What conclusions follow from the extracted information?",
            "step_context_queries": ["conclusions", "summary", "results"],
            "stage_action": "Synthesize a coherent summary",
            "example_reasoning": "The summary should answer the user's original question",
            "dependencies": [1, 2],
            "retry_max": 3,
            "llm_config": None,
        },
    ],
}

SAMPLE_DOMAIN_ANALYSIS: dict[str, Any] = {
    "domain": "finance",
    "task_type": "analysis",
    "complexity": "medium",
    "key_concepts": ["financial report", "risks", "revenue"],
    "suggested_step_count": 3,
    "preferred_step_types": ["mcp", "transform", "llm", "eval"],
}

SAMPLE_STEP_PLAN: dict[str, Any] = {
    "steps": [
        {"number": 1, "title": "Analyze structure", "step_type": "llm", "rough_aim": "Understand document structure"},
        {"number": 2, "title": "Extract data", "step_type": "llm", "rough_aim": "Extract key financial data"},
        {"number": 3, "title": "Synthesize", "step_type": "llm", "rough_aim": "Form conclusions"},
    ],
    "rationale": "Standard analysis pipeline",
}

SAMPLE_DAG: dict[str, Any] = {
    "steps": [
        {
            "number": 1,
            "title": "Analyze structure",
            "step_type": "llm",
            "dependencies": [],
            "rough_aim": "Understand structure",
        },
        {"number": 2, "title": "Extract data", "step_type": "llm", "dependencies": [], "rough_aim": "Extract data"},
        {
            "number": 3,
            "title": "Synthesize",
            "step_type": "llm",
            "dependencies": [1, 2],
            "rough_aim": "Form conclusions",
        },
    ],
    "max_workers": 2,
    "pattern": "funnel",
}

SAMPLE_DESCRIBED_STEPS: dict[str, Any] = {
    "steps": SAMPLE_CARL_CHAIN["steps"],
}


# ---------------------------------------------------------------------------
# §9 P1 — exemplar steps for the agent-features types
# ---------------------------------------------------------------------------
#
# Each exemplar is a full CARLStepSchema-shape dict that round-trips
# cleanly through `validate_carl_json` and (after `to_carl_compat`)
# `validate_carl_json(strict_carl=True)`. The exemplars are deliberately
# self-contained so tests that need just one step type can import the
# single exemplar without dragging in the rest.

SAMPLE_STEP_AGENT_SKILL: dict[str, Any] = {
    "step_type": "agent_skill",
    "number": 1,
    "title": "Extract PDF tables",
    "aim": "Use the pdf skill to extract structured tables",
    "reasoning_questions": "Which sheets / tables are present?",
    "step_context_queries": ["pdf", "tables", "extraction"],
    "stage_action": "Run the official PDF skill on the input file",
    "example_reasoning": "Tables 2 and 5 contain revenue figures",
    "skill": "github://anthropics/skills/document-skills/pdf@main",
    "skill_execution_mode": "llm_agent",
    "skill_input_mapping": {"file_path": "$context.uploaded_pdf"},
    "skill_output_key": "pdf_tables",
    "skill_timeout": 120.0,
    "dependencies": [],
    "retry_max": 3,
}

SAMPLE_STEP_STRUCTURED_OUTPUT: dict[str, Any] = {
    "step_type": "structured_output",
    "number": 2,
    "title": "Coerce summary to schema",
    "aim": "Produce a JSON object matching the report schema",
    "reasoning_questions": "Does every required field have a value?",
    "step_context_queries": ["summary", "schema"],
    "stage_action": "Emit JSON matching the schema",
    "example_reasoning": "Map upstream prose into the schema fields",
    "structured_output_schema": {
        "type": "object",
        "properties": {
            "revenue_change_pct": {"type": "number"},
            "summary": {"type": "string"},
        },
        "required": ["revenue_change_pct", "summary"],
    },
    "structured_output_schema_name": "QuarterlyReport",
    "structured_output_input_source": "$history[-1]",
    "structured_output_instruction": "Return ONLY the JSON object",
    "structured_output_strict_json": True,
    "dependencies": [1],
    "retry_max": 3,
}

SAMPLE_STEP_AGENT_HANDOFF: dict[str, Any] = {
    "step_type": "agent_handoff",
    "number": 3,
    "title": "Delegate to risk-analysis sub-chain",
    "aim": "Hand off to the risk-analysis sub-chain with the extracted tables",
    "reasoning_questions": "What risk factors apply to these revenue trends?",
    "step_context_queries": ["risk", "delegate"],
    "stage_action": "Run the named risk-analysis sub-chain",
    "example_reasoning": "Risk chain produces a structured risk-report",
    "handoff_chain_path": "registry://chains/risk-analysis@v2",
    "handoff_input_mapping": {"tables": "$steps.2.output"},
    "handoff_output_key": "risk_report",
    "dependencies": [2],
    "retry_max": 3,
}

# A chain that exercises every step type listed in the §9 P1 spec.
# Used by `tests/test_step_type_round_trips.py` to verify each step
# type survives validate_carl_json + to_carl_compat + MAGEResult round-trips.
SAMPLE_HETEROGENEOUS_CHAIN: dict[str, Any] = {
    "name": "Heterogeneous demo",
    "max_workers": 1,
    "timeout": 600.0,
    "search_config": SAMPLE_CARL_CHAIN["search_config"],
    "steps": [
        SAMPLE_STEP_AGENT_SKILL,
        SAMPLE_STEP_STRUCTURED_OUTPUT,
        SAMPLE_STEP_AGENT_HANDOFF,
    ],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> MAGEConfig:
    """Default test config with memory disabled."""
    return MAGEConfig(
        mode="fast",
        model="gpt-4o-test",
        api_key="test-key-123",
        enable_memory=False,
        max_retries=0,
    )


@pytest.fixture
def deep_config() -> MAGEConfig:
    """Deep-mode test config with memory disabled."""
    return MAGEConfig(
        mode="deep",
        model="gpt-4o-test",
        api_key="test-key-123",
        enable_memory=False,
        max_retries=0,
    )


def _make_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content

    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_openai_fast(mocker: Any) -> MagicMock:
    """Mock AsyncOpenAI client returning a valid CARL chain JSON (fast mode)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance


@pytest.fixture
def mock_openai_deep(mocker: Any) -> MagicMock:
    """Mock AsyncOpenAI client returning sequential deep-mode responses."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value

    responses = [
        _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
        _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
        _make_openai_response(json.dumps(SAMPLE_DAG)),
        _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
    ]

    instance.chat.completions.create = AsyncMock(side_effect=responses)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance
