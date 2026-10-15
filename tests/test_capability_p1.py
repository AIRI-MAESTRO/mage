"""Tests for §2 P1 — capability-driven step selection + external injection.

Covers:
- `_tokenize()` drops stop-words and short tokens
- `_score_capability_match()` returns higher scores for well-matched
  rough-aim/capability pairs and ~0 for unrelated ones
- `_score_capability_match()` blends pre-existing `relevance` with Jaccard
- `MAGEGenerator.generate(capabilities=...)` bypasses internal
  CapabilityLookupAgent — no Memory call, no SkillDiscoveryAgent invocation
- The injected context's `agent_skills` reach the StepPlanner via
  `skill_digest=...`
- A synthetic `capability_lookup_injected` stage entry appears in
  `deep_stages_completed` so callers can tell which path ran
- The new planner-prompt rule mentions agent_skill preference
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.capability_lookup_agent import (
    AgentSkillEntry,
    CapabilityContext,
    _score_capability_match,
    _tokenize,
)
from mmar_mage.generator import MAGEGenerator
from mmar_mage.prompts import STEP_PLANNER_PROMPT
from mmar_mage.schemas import MAGEConfig

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_drops_stopwords_and_short_tokens(self) -> None:
        toks = _tokenize("Extract a table from the PDF")
        # "a"/"the"/"from"/"extract" are stop-words; "of"/short tokens dropped
        assert "table" in toks
        assert "pdf" in toks
        assert "the" not in toks
        assert "a" not in toks
        # "extract" is in stop-list because it's planner-generic
        assert "extract" not in toks

    def test_lowercases(self) -> None:
        assert "table" in _tokenize("TABLE")

    def test_handles_punctuation(self) -> None:
        toks = _tokenize("Read PDF/DOCX files.")
        assert "pdf" in toks
        assert "docx" in toks
        assert "files" in toks

    def test_empty(self) -> None:
        assert _tokenize("") == set()


# ---------------------------------------------------------------------------
# _score_capability_match
# ---------------------------------------------------------------------------


class TestScoreCapabilityMatch:
    def _skill(self, **kwargs: Any) -> AgentSkillEntry:
        defaults = {"name": "x", "uri": "u://x"}
        defaults.update(kwargs)
        return AgentSkillEntry(**defaults)  # type: ignore[arg-type]

    def test_strong_match_scores_high(self) -> None:
        skill = self._skill(
            name="pdf",
            description="Extract tables and text from PDF documents",
            tags=["documents", "pdf"],
        )
        score = _score_capability_match("extract tables from a PDF report", skill)
        # Token-overlap Jaccard between filtered tokens — strong matches
        # land around 0.25-0.35 since the score is bounded by stop-word
        # filtering. The planner-prompt threshold is 0.5 (with relevance
        # boost in practice).
        assert score >= 0.2

    def test_strong_match_with_relevance_clears_planner_threshold(self) -> None:
        """When SkillDiscoveryAgent gave the skill a high relevance score,
        the combined match should exceed the documented 0.5 threshold."""
        skill = self._skill(
            name="pdf",
            description="Extract tables and text from PDF documents",
            tags=["documents", "pdf"],
            relevance=0.9,
        )
        score = _score_capability_match("extract tables from a PDF report", skill)
        assert score >= 0.5

    def test_unrelated_pair_scores_low(self) -> None:
        skill = self._skill(name="pdf", description="Extract tables from PDFs", tags=["documents"])
        score = _score_capability_match("summarize quarterly revenue trends", skill)
        # "quarterly", "revenue", "trends", "summarize" share nothing with PDF
        assert score < 0.1

    def test_empty_aim_returns_zero(self) -> None:
        skill = self._skill(name="pdf", description="x")
        assert _score_capability_match("", skill) == 0.0

    def test_empty_capability_text_returns_zero(self) -> None:
        skill = self._skill(name="", description="", tags=[], manifest_summary="")
        assert _score_capability_match("anything", skill) == 0.0

    def test_relevance_blended_into_score(self) -> None:
        skill_no_rel = self._skill(name="pdf", description="PDF tables", relevance=0.0)
        skill_with_rel = self._skill(name="pdf", description="PDF tables", relevance=1.0)
        aim = "extract a PDF tables"
        s1 = _score_capability_match(aim, skill_no_rel)
        s2 = _score_capability_match(aim, skill_with_rel)
        # Relevance contributes 0.3 weight → s2 should beat s1 by ~0.3
        assert s2 > s1
        assert (s2 - s1) == pytest.approx(0.3, abs=0.01)

    def test_score_capped_at_one(self) -> None:
        skill = self._skill(
            name="pdf",
            description="pdf",
            tags=["pdf"],
            relevance=1.0,
        )
        assert _score_capability_match("pdf pdf pdf", skill) <= 1.0


# ---------------------------------------------------------------------------
# STEP_PLANNER_PROMPT rule
# ---------------------------------------------------------------------------


def test_planner_prompt_contains_capability_rule() -> None:
    """The prompt now tells the LLM to reuse skills when matches exist."""
    assert "CAPABILITY-DRIVEN STEP SELECTION" in STEP_PLANNER_PROMPT
    assert "agent_skill" in STEP_PLANNER_PROMPT
    assert "MUST emit" in STEP_PLANNER_PROMPT  # strong rule


# ---------------------------------------------------------------------------
# External CapabilityContext injection
# ---------------------------------------------------------------------------


def _mock_deep_responses(mocker: Any) -> MagicMock:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance


@pytest.mark.asyncio
async def test_caller_supplied_capabilities_bypass_internal_lookup(deep_config: MAGEConfig, mocker: Any) -> None:
    """Passing `capabilities=` skips CapabilityLookupAgent entirely.

    With `enable_capability_lookup=True` on the config AND a
    caller-supplied context, the agent must NOT be instantiated — the
    caller's context is the single source of truth.
    """
    _mock_deep_responses(mocker)

    # Spy on CapabilityLookupAgent — should NOT be called.
    spy = mocker.patch("mmar_mage.generator.CapabilityLookupAgent")

    deep_config.enable_capability_lookup = True
    deep_config.enable_skill_discovery = True  # would normally fire too
    deep_config.enable_suggested_naming_llm = False

    injected = CapabilityContext(
        tools=[{"name": "calc"}],
        mcp_servers=[],
        agent_skills=[
            AgentSkillEntry(
                name="pdf",
                description="PDF helper",
                uri="github://anthropics/skills/pdf",
                source="discovered",
                relevance=0.9,
                why="Pre-installed by CARE",
            )
        ],
    )

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Extract PDF tables", capabilities=injected)

    # CapabilityLookupAgent NEVER instantiated
    spy.assert_not_called()
    # Stage label distinguishes the injected path
    stages = result.metadata.deep_stages_completed or []
    assert "capability_lookup_injected" in stages
    assert "capability_lookup" not in stages
    assert "skill_discovery" not in stages


@pytest.mark.asyncio
async def test_caller_supplied_capabilities_reach_planner(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's `agent_skills` appear in the StepPlanner's skill_digest."""
    captured: dict[str, Any] = {}

    async def fake_plan(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured["skill_digest"] = kwargs.get("skill_digest", "")
        captured["capability_context"] = kwargs.get("capability_context", "")
        from mmar_mage.schemas import PlannedStep, StepPlan

        return StepPlan(
            steps=[
                PlannedStep(number=1, title="A", step_type="llm", rough_aim="x"),
                PlannedStep(number=2, title="B", step_type="llm", rough_aim="y"),
            ],
            rationale="ok",
        )

    monkeypatch.setattr("mmar_mage.agents.StepPlanner.plan", fake_plan)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_suggested_naming_llm = False

    injected = CapabilityContext(
        tools=[{"name": "calculator"}],
        agent_skills=[
            AgentSkillEntry(
                name="pdf",
                description="PDF extraction skill",
                uri="github://anthropics/skills/pdf@main",
                source="discovered",
                relevance=0.88,
                why="Caller pre-installed",
            )
        ],
    )

    gen = MAGEGenerator(config=deep_config)
    await gen.generate("Extract tables from a PDF", capabilities=injected)

    digest = captured["skill_digest"]
    assert "pdf" in digest
    assert "github://anthropics/skills/pdf@main" in digest
    assert "0.88" in digest

    cap_text = captured["capability_context"]
    assert "calculator" in cap_text


@pytest.mark.asyncio
async def test_caller_supplied_empty_context_still_bypasses_lookup(deep_config: MAGEConfig, mocker: Any) -> None:
    """An explicit empty CapabilityContext still bypasses the internal agent.

    Useful when CARE wants to say "no capabilities at all" (e.g. for a
    deterministic test run) rather than letting MAGE auto-discover.
    """
    _mock_deep_responses(mocker)
    spy = mocker.patch("mmar_mage.generator.CapabilityLookupAgent")

    deep_config.enable_capability_lookup = True
    deep_config.enable_skill_discovery = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Anything", capabilities=CapabilityContext())

    spy.assert_not_called()
    stages = result.metadata.deep_stages_completed or []
    assert "capability_lookup_injected" in stages


@pytest.mark.asyncio
async def test_no_capabilities_kwarg_keeps_legacy_behavior(deep_config: MAGEConfig, mocker: Any) -> None:
    """Omitting `capabilities=` falls back to the auto-lookup path."""
    _mock_deep_responses(mocker)

    deep_config.enable_capability_lookup = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Test")

    stages = result.metadata.deep_stages_completed or []
    assert "capability_lookup" in stages
    assert "capability_lookup_injected" not in stages
