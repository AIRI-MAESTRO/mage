"""Tests for CARE library-ready naming (§7.2).

Covers:
- Direct unit tests on `_detect_capability_tags`, `_naming_heuristic`,
  `_coerce_naming_payload`.
- Integration: fast mode populates `suggested_*` via heuristic.
- Integration: deep mode populates `suggested_*` via LLM call.
- Integration: deep mode falls back to heuristic when the LLM call fails
  or `enable_suggested_naming_llm=False`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import (
    MAGEGenerator,
    _coerce_naming_payload,
    _detect_capability_tags,
    _naming_heuristic,
)
from mmar_mage.schemas import MAGEConfig, MAGEMetadata

from .conftest import (
    SAMPLE_CARL_CHAIN,
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# Pure unit tests — helpers
# ---------------------------------------------------------------------------


class TestNamingHeuristic:
    def test_default_shape(self) -> None:
        name, desc, tags = _naming_heuristic("Analyze quarterly revenue")
        assert name.startswith("General")
        assert "Analyze quarterly revenue" in name
        assert "analysis:" in desc.lower()
        # general/analysis are deduped, key concepts add nothing -> tags only
        # contains the task_type when domain is "general"
        assert "analysis" in tags

    def test_display_name_uses_domain_title_and_query_slice(self) -> None:
        name, _, _ = _naming_heuristic("Quarterly EBITDA review", domain="finance")
        assert name.startswith("Finance — ")
        assert "Quarterly EBITDA review" in name

    def test_display_name_capped_at_60_chars(self) -> None:
        long_query = "x" * 200
        name, _, _ = _naming_heuristic(long_query, domain="finance")
        assert len(name) <= 60

    def test_tags_include_capability_tags_and_concepts(self) -> None:
        _, _, tags = _naming_heuristic(
            "Read this PDF",
            domain="legal",
            task_type="extraction",
            key_concepts=["contract", "clauses"],
            capability_tags=["uses-pdf"],
        )
        assert "legal" in tags
        assert "extraction" in tags
        assert "contract" in tags
        assert "uses-pdf" in tags

    def test_empty_query(self) -> None:
        name, desc, tags = _naming_heuristic("", domain="finance")
        # Should not crash, should not produce trailing em-dash
        assert name == "Finance"
        assert "analysis" in desc.lower()
        assert "finance" in tags


class TestDetectCapabilityTags:
    def test_memory_step_detected(self) -> None:
        chain = {"steps": [{"step_type": "memory", "title": "Recall"}]}
        assert "uses-memory" in _detect_capability_tags(chain)

    def test_pdf_skill_detected(self) -> None:
        chain = {
            "steps": [
                {"step_type": "agent_skill", "title": "Extract", "skill_name": "pdf-reader"},
            ]
        }
        assert "uses-pdf" in _detect_capability_tags(chain)

    def test_pdf_detected_inside_step_config(self) -> None:
        chain = {
            "steps": [
                {
                    "step_type": "agent_skill",
                    "title": "Extract",
                    "step_config": {"skill_name": "pdf-reader"},
                }
            ]
        }
        assert "uses-pdf" in _detect_capability_tags(chain)

    def test_web_tool_detected(self) -> None:
        chain = {
            "steps": [
                {"step_type": "tool", "title": "Lookup", "tool_name": "web_search"},
            ]
        }
        assert "uses-web" in _detect_capability_tags(chain)

    def test_web_results_used_flag(self) -> None:
        chain = {"steps": [{"step_type": "llm", "title": "Analyze"}]}
        tags = _detect_capability_tags(chain, web_results_used=2)
        assert "uses-web" in tags

    def test_memory_hits_used_flag(self) -> None:
        chain = {"steps": [{"step_type": "llm", "title": "Analyze"}]}
        tags = _detect_capability_tags(chain, memory_hits_used=4)
        assert "uses-memory" in tags

    def test_no_capabilities(self) -> None:
        chain = {"steps": [{"step_type": "llm", "title": "Analyze", "aim": "Just think"}]}
        assert _detect_capability_tags(chain) == []


class TestCoerceNamingPayload:
    def test_valid_payload_returned_as_is(self) -> None:
        name, desc, tags = _coerce_naming_payload(
            {
                "display_name": "Financier PDF Assistant",
                "description": "Reads financial PDFs and extracts KPIs.",
                "tags": ["finance", "pdf", "extraction"],
            },
            fallback_display_name="Fallback",
            fallback_description="Fallback desc",
            fallback_tags=["uses-pdf"],
        )
        assert name == "Financier PDF Assistant"
        assert desc == "Reads financial PDFs and extracts KPIs."
        assert "finance" in tags
        assert "uses-pdf" in tags  # capability flag injected even if LLM omitted

    def test_missing_display_name_falls_back(self) -> None:
        name, _, _ = _coerce_naming_payload(
            {"description": "x", "tags": []},
            fallback_display_name="Heuristic Name",
            fallback_description="fb",
            fallback_tags=[],
        )
        assert name == "Heuristic Name"

    def test_display_name_strips_trailing_period(self) -> None:
        name, _, _ = _coerce_naming_payload(
            {"display_name": "Loud Name.", "description": "x", "tags": []},
            fallback_display_name="fb",
            fallback_description="fb",
            fallback_tags=[],
        )
        assert name == "Loud Name"

    def test_display_name_over_60_chars_truncated(self) -> None:
        name, _, _ = _coerce_naming_payload(
            {"display_name": "x" * 100, "description": "x", "tags": []},
            fallback_display_name="fb",
            fallback_description="fb",
            fallback_tags=[],
        )
        assert len(name) <= 60

    def test_tags_lowercased_and_kebab_cased(self) -> None:
        _, _, tags = _coerce_naming_payload(
            {"display_name": "x", "description": "x", "tags": ["FINANCE", "Use PDF", "use pdf"]},
            fallback_display_name="fb",
            fallback_description="fb",
            fallback_tags=[],
        )
        assert "finance" in tags
        assert "use-pdf" in tags
        # de-duplicated
        assert tags.count("use-pdf") == 1

    def test_bad_payload_falls_back_completely(self) -> None:
        name, desc, tags = _coerce_naming_payload(
            {"display_name": 123, "description": None, "tags": "not a list"},
            fallback_display_name="Heuristic Name",
            fallback_description="Heuristic Desc",
            fallback_tags=["heuristic-tag"],
        )
        assert name == "Heuristic Name"
        assert desc == "Heuristic Desc"
        assert tags == ["heuristic-tag"]


# ---------------------------------------------------------------------------
# MAGEMetadata field defaults
# ---------------------------------------------------------------------------


class TestMAGEMetadataNamingFields:
    def test_defaults(self) -> None:
        meta = MAGEMetadata()
        assert meta.suggested_display_name == ""
        assert meta.suggested_description == ""
        assert meta.suggested_tags == []

    def test_setting(self) -> None:
        meta = MAGEMetadata(
            suggested_display_name="Test Agent",
            suggested_description="Does a thing.",
            suggested_tags=["a", "b"],
        )
        assert meta.suggested_display_name == "Test Agent"
        assert meta.suggested_tags == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration — fast mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_mode_populates_suggested_naming(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Fast mode populates `suggested_*` via heuristic (no LLM call)."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Analyze a quarterly financial report")

    meta = result.metadata
    assert meta.suggested_display_name  # non-empty
    assert meta.suggested_description  # non-empty
    assert isinstance(meta.suggested_tags, list)
    # Fast mode uses domain="general" since no DomainAnalysis runs
    assert meta.suggested_display_name.startswith("General")
    # Fast-mode heuristic only sees one LLM call (the chain itself), so no
    # extra LLM call should have been made for naming.
    assert mock_openai_fast.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Integration — deep mode
# ---------------------------------------------------------------------------


def _mock_deep_with_naming(mocker: Any, naming_payload: dict[str, Any]) -> MagicMock:
    """Mock AsyncOpenAI with the 4 deep-mode responses + 1 naming response."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    responses = [
        _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
        _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
        _make_openai_response(json.dumps(SAMPLE_DAG)),
        _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        _make_openai_response(json.dumps(naming_payload)),
    ]
    instance.chat.completions.create = AsyncMock(side_effect=responses)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance


@pytest.mark.asyncio
async def test_deep_mode_naming_uses_llm(deep_config: MAGEConfig, mocker: Any) -> None:
    """Deep mode populates `suggested_*` via the LLM naming call."""
    _mock_deep_with_naming(
        mocker,
        {
            "display_name": "Financier Analyst",
            "description": "Reads quarterly reports and surfaces revenue trends.",
            "tags": ["finance", "analysis", "quarterly"],
        },
    )

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a quarterly financial report")

    meta = result.metadata
    assert meta.suggested_display_name == "Financier Analyst"
    assert meta.suggested_description.startswith("Reads quarterly reports")
    assert "finance" in meta.suggested_tags
    assert "quarterly" in meta.suggested_tags


@pytest.mark.asyncio
async def test_deep_mode_naming_falls_back_when_llm_fails(deep_config: MAGEConfig, mock_openai_deep: MagicMock) -> None:
    """Deep mode falls back to heuristic when the naming LLM call fails.

    The shared `mock_openai_deep` fixture only provides 4 responses — the
    naming call (5th) raises StopAsyncIteration, which `_generate_suggested_naming`
    catches and falls back to the heuristic.
    """
    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a quarterly financial report")

    meta = result.metadata
    # Heuristic uses DomainAnalysis.domain == "finance" (from SAMPLE_DOMAIN_ANALYSIS)
    assert meta.suggested_display_name.startswith("Finance")
    assert "Analyze a quarterly financial report" in meta.suggested_display_name
    assert meta.suggested_description  # heuristic populated description
    assert "finance" in meta.suggested_tags


@pytest.mark.asyncio
async def test_deep_mode_naming_llm_disabled(deep_config: MAGEConfig, mock_openai_deep: MagicMock) -> None:
    """When `enable_suggested_naming_llm=False`, no naming LLM call is made."""
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a quarterly financial report")

    meta = result.metadata
    # Only 4 LLM calls (domain, plan, dag, describe) — no 5th for naming
    assert mock_openai_deep.chat.completions.create.call_count == 4
    # Heuristic still populates the fields
    assert meta.suggested_display_name.startswith("Finance")
    assert meta.suggested_description
    assert "finance" in meta.suggested_tags


@pytest.mark.asyncio
async def test_deep_mode_naming_capability_tags_injected(deep_config: MAGEConfig, mocker: Any) -> None:
    """Capability tags detected from the chain are injected into final tags.

    Uses `mock_openai_deep` shape but stitches a memory step into the
    described-steps response so the chain ends up with a memory step,
    triggering `uses-memory`. The LLM naming response omits `uses-memory`;
    `_coerce_naming_payload` should inject it from the heuristic fallback.
    """
    # Chain dict with a memory step
    chain_with_memory = json.loads(json.dumps(SAMPLE_DESCRIBED_STEPS))
    chain_with_memory["steps"][0]["step_type"] = "memory"
    chain_with_memory["steps"][0]["memory_query"] = "context"

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    responses = [
        _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
        _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
        _make_openai_response(json.dumps(SAMPLE_DAG)),
        _make_openai_response(json.dumps(chain_with_memory)),
        _make_openai_response(
            json.dumps(
                {
                    "display_name": "Finance Recall",
                    "description": "x",
                    "tags": ["finance"],  # LLM forgot uses-memory
                }
            )
        ),
    ]
    instance.chat.completions.create = AsyncMock(side_effect=responses)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Recall finance reports")

    assert "uses-memory" in result.metadata.suggested_tags


# Sanity: the fixtures from conftest aren't broken — fast mode chain still
# matches.
def test_sample_chain_unchanged() -> None:
    assert SAMPLE_CARL_CHAIN["name"] == "Test Analysis Chain"
    assert len(SAMPLE_CARL_CHAIN["steps"]) == 3
