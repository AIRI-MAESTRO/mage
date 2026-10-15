"""§9 P1 golden test — `suggested_*` fields via a cassette-recorded LLM response.

Cassette pattern: a frozen JSON response captured from a real run of
the deep-mode `SUGGESTED_NAMING_PROMPT` LLM call. Replayed offline so
the test runs without network access. The asserted values are pinned
verbatim against this cassette — any drift in the wiring between
``LLMClient.generate_json`` and ``MAGEMetadata.suggested_*`` will be
caught by the byte-level comparison.

The cassette covers the deep-mode happy path used by CARE's
``SaveAgentModal``: a finance domain analysis + a quarterly-report
query → punchy display name, 1-2 sentence description, lowercase
kebab-case tags including the capability flag the chain
auto-detected.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# Cassette: a real-shape LLM response for the `SUGGESTED_NAMING_PROMPT`
# stage that CARE would receive on a finance-quarterly query. Pinned
# so any wiring drift between `_generate_suggested_naming` → MAGEResult
# fails this test loudly.
CASSETTE_NAMING_RESPONSE: dict[str, Any] = {
    "display_name": "Quarterly Revenue Analyst",
    "description": (
        "Reads quarterly revenue reports and returns a structured summary with year-over-year deltas and risk callouts."
    ),
    "tags": ["finance", "quarterly-analysis", "revenue", "structured-output"],
}


@pytest.mark.asyncio
async def test_suggested_naming_golden_cassette(deep_config: MAGEConfig, mocker: Any) -> None:
    """Deep-mode generation replays the cassette and pins MAGEMetadata.suggested_*."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    # Five LLM calls: domain, plan, dag, describe, naming.
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
            _make_openai_response(json.dumps(CASSETTE_NAMING_RESPONSE)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    # Default deep_config has enable_suggested_naming_llm=True — exactly
    # what we want to exercise.
    assert deep_config.enable_suggested_naming_llm is True

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Summarize Q3 2024 revenue with YoY deltas")

    md = result.metadata
    # ---- Golden assertions: pinned verbatim against the cassette ----
    assert md.suggested_display_name == CASSETTE_NAMING_RESPONSE["display_name"]
    assert md.suggested_description == CASSETTE_NAMING_RESPONSE["description"]
    # Tags may include capability flags appended by `_coerce_naming_payload`
    # (e.g. detected `uses-*`), but every cassette tag MUST survive.
    for tag in CASSETTE_NAMING_RESPONSE["tags"]:
        assert tag in md.suggested_tags, f"tag {tag!r} missing from {md.suggested_tags}"

    # Exactly 5 LLM calls fired (cassette covers each one).
    assert instance.chat.completions.create.call_count == 5


@pytest.mark.asyncio
async def test_suggested_naming_cassette_with_malformed_payload_falls_back(
    deep_config: MAGEConfig, mocker: Any
) -> None:
    """When the cassette is malformed (LLM regression), the heuristic
    fallback kicks in — display_name still non-empty, never None."""
    bad_cassette = {"display_name": 123, "description": None, "tags": "not a list"}
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
            _make_openai_response(json.dumps(bad_cassette)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Test")

    # Heuristic fallback produces a Finance-prefixed display name
    # (SAMPLE_DOMAIN_ANALYSIS has domain="finance").
    assert result.metadata.suggested_display_name.startswith("Finance")
    assert result.metadata.suggested_description
    assert isinstance(result.metadata.suggested_tags, list)
    assert len(result.metadata.suggested_tags) > 0
