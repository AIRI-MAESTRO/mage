"""Tests for §7.1 — MAGEResult.to_care_dict / from_care_dict.

Covers:
- New `intermediate_artifacts` and `memory_keys` fields on MAGEResult
- `to_care_dict()` includes every field with stable schema_version
- `from_care_dict()` round-trips the result back losslessly
- Forward-compat: unknown `schema_version` warns but loads
- Missing optional keys default sanely
- Generator captures domain/plan/dag/described_steps into
  intermediate_artifacts in deep mode
- Generator's `memory_keys` collects chain + skills + reusable_steps +
  failure_cards entity ids
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig, MAGEMetadata, MAGEResult

from .conftest import (
    SAMPLE_CARL_CHAIN,
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# MAGEResult schema: new fields
# ---------------------------------------------------------------------------


class TestNewFields:
    def test_intermediate_artifacts_defaults_to_empty_dict(self) -> None:
        r = MAGEResult(chain_json='{"steps":[]}')
        assert r.intermediate_artifacts == {}

    def test_memory_keys_defaults_to_empty_dict(self) -> None:
        r = MAGEResult(chain_json='{"steps":[]}')
        assert r.memory_keys == {}

    def test_fields_accept_values(self) -> None:
        r = MAGEResult(
            chain_json='{"steps":[]}',
            intermediate_artifacts={"domain_analysis": {"domain": "finance"}},
            memory_keys={"chain": "abc", "skills": ["s1", "s2"]},
        )
        assert r.intermediate_artifacts["domain_analysis"]["domain"] == "finance"
        assert r.memory_keys["chain"] == "abc"
        assert r.memory_keys["skills"] == ["s1", "s2"]


# ---------------------------------------------------------------------------
# to_care_dict / from_care_dict
# ---------------------------------------------------------------------------


class TestToCareDict:
    def _sample_result(self) -> MAGEResult:
        return MAGEResult(
            chain_json=json.dumps(SAMPLE_CARL_CHAIN),
            chain_dict=SAMPLE_CARL_CHAIN,
            memory_key="chain-id-42",
            mode="deep",
            metadata=MAGEMetadata(
                domain="finance",
                num_steps=3,
                model="gpt-4o",
                step_critique_score=0.85,
                suggested_display_name="Finance Analyst",
            ),
            intermediate_artifacts={
                "domain_analysis": {"domain": "finance", "task_type": "analysis"},
                "step_plan": {"steps": [], "rationale": "ok"},
            },
            memory_keys={
                "chain": "chain-id-42",
                "skills": ["skill-1"],
                "failure_cards": [],
            },
        )

    def test_includes_schema_version(self) -> None:
        bundle = self._sample_result().to_care_dict()
        assert bundle["schema_version"] == 1

    def test_includes_all_top_level_keys(self) -> None:
        bundle = self._sample_result().to_care_dict()
        expected = {
            "schema_version",
            "mode",
            "chain_json",
            "chain_dict",
            "code",
            "memory_key",
            "metadata",
            "intermediate_artifacts",
            "memory_keys",
        }
        assert expected <= set(bundle.keys())

    def test_metadata_serializes_as_dict(self) -> None:
        bundle = self._sample_result().to_care_dict()
        assert isinstance(bundle["metadata"], dict)
        assert bundle["metadata"]["domain"] == "finance"
        assert bundle["metadata"]["step_critique_score"] == 0.85

    def test_is_json_serializable(self) -> None:
        bundle = self._sample_result().to_care_dict()
        # Round-trip through JSON to prove the bundle is hop-friendly.
        as_json = json.dumps(bundle)
        loaded = json.loads(as_json)
        assert loaded["memory_key"] == "chain-id-42"


class TestFromCareDict:
    def test_round_trip(self) -> None:
        original = MAGEResult(
            chain_json='{"steps":[]}',
            chain_dict={"steps": []},
            memory_key="x",
            mode="fast",
            metadata=MAGEMetadata(domain="finance", num_steps=2),
            intermediate_artifacts={"k": "v"},
            memory_keys={"chain": "x", "skills": ["s1"]},
        )
        bundle = original.to_care_dict()
        rebuilt = MAGEResult.from_care_dict(bundle)
        assert rebuilt.chain_json == original.chain_json
        assert rebuilt.memory_key == "x"
        assert rebuilt.mode == "fast"
        assert rebuilt.metadata.domain == "finance"
        assert rebuilt.intermediate_artifacts == {"k": "v"}
        assert rebuilt.memory_keys == {"chain": "x", "skills": ["s1"]}

    def test_round_trip_through_json(self) -> None:
        original = MAGEResult(
            chain_json='{"steps":[]}',
            chain_dict={"steps": []},
            metadata=MAGEMetadata(suggested_tags=["a", "b"]),
        )
        bundle_json = json.dumps(original.to_care_dict())
        rebuilt = MAGEResult.from_care_dict(json.loads(bundle_json))
        assert rebuilt.metadata.suggested_tags == ["a", "b"]

    def test_missing_optional_keys_default_sanely(self) -> None:
        """A minimal bundle (just chain_json) should load."""
        rebuilt = MAGEResult.from_care_dict({"chain_json": "{}"})
        assert rebuilt.chain_json == "{}"
        assert rebuilt.chain_dict == {}
        assert rebuilt.intermediate_artifacts == {}
        assert rebuilt.memory_keys == {}
        assert rebuilt.mode == "deep"

    def test_unknown_schema_version_loads_with_warning(self, caplog: Any) -> None:
        import logging

        with caplog.at_level(logging.WARNING, logger="mmar_mage.schemas"):
            r = MAGEResult.from_care_dict({"schema_version": 99, "chain_json": "{}"})
        assert r.chain_json == "{}"
        # The warning should have been emitted
        assert any("schema_version" in rec.message for rec in caplog.records)

    def test_non_dict_input_raises(self) -> None:
        with pytest.raises(TypeError):
            MAGEResult.from_care_dict("not a dict")  # type: ignore[arg-type]

    def test_accepts_metadata_as_object(self) -> None:
        """Tolerates a MAGEMetadata instance in the metadata slot too."""
        r = MAGEResult.from_care_dict(
            {
                "chain_json": "{}",
                "metadata": MAGEMetadata(num_steps=5),
            }
        )
        assert r.metadata.num_steps == 5


# ---------------------------------------------------------------------------
# Generator integration — intermediate_artifacts captured in deep mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_captures_intermediate_artifacts_in_deep_mode(deep_config: MAGEConfig, mocker: Any) -> None:
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

    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze quarterly")

    arts = result.intermediate_artifacts
    # Every successful deep stage left its dump behind.
    assert "domain_analysis" in arts
    assert arts["domain_analysis"]["domain"] == "finance"
    assert "step_plan" in arts
    assert len(arts["step_plan"]["steps"]) == 3
    assert "dag" in arts
    assert arts["dag"]["max_workers"] == 2
    assert "described_steps" in arts
    assert len(arts["described_steps"]) == 3


@pytest.mark.asyncio
async def test_generator_fast_mode_intermediate_artifacts_empty(
    config: MAGEConfig, mock_openai_fast: MagicMock
) -> None:
    """Fast mode has no per-stage agents → empty intermediate_artifacts."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")
    assert result.intermediate_artifacts == {}


@pytest.mark.asyncio
async def test_generator_to_care_dict_roundtrip_end_to_end(deep_config: MAGEConfig, mocker: Any) -> None:
    """The whole pipeline produces a bundle that round-trips."""
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

    deep_config.enable_suggested_naming_llm = False

    result = await MAGEGenerator(config=deep_config).generate("X")

    bundle = result.to_care_dict()
    # Serializable through JSON
    bundle_json = json.dumps(bundle)
    rebuilt = MAGEResult.from_care_dict(json.loads(bundle_json))

    assert rebuilt.metadata.num_steps == result.metadata.num_steps
    assert rebuilt.intermediate_artifacts == result.intermediate_artifacts
    assert rebuilt.chain_dict["name"] == result.chain_dict["name"]


# ---------------------------------------------------------------------------
# memory_keys captures chain + ecosystem entity ids
# ---------------------------------------------------------------------------


SAMPLE_CHAIN_WITH_AGENT_SKILL: dict[str, Any] = {
    "name": "Test",
    "steps": [
        {
            "step_type": "agent_skill",
            "number": 1,
            "title": "Read PDF",
            "aim": "Use the pdf skill",
            "reasoning_questions": "?",
            "step_context_queries": ["x"],
            "stage_action": "do",
            "example_reasoning": "y",
            "skill_name": "pdf",
            "skill": "github://anthropics/skills/pdf",
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Synthesize",
            "aim": "Synthesize a structured summary of the analysis",
            "reasoning_questions": "What conclusions follow?",
            "step_context_queries": ["summary"],
            "stage_action": "Produce a coherent summary",
            "example_reasoning": "Answers the question",
            "dependencies": [1],
            "retry_max": 3,
        },
    ],
}


@pytest.mark.asyncio
async def test_generator_memory_keys_collects_chain_and_skills(config: MAGEConfig, mocker: Any) -> None:
    """When skill provenance saves are on, the entity ids appear in memory_keys."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        return_value=_make_openai_response(json.dumps(SAMPLE_CHAIN_WITH_AGENT_SKILL))
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    config.enable_memory = False  # short-circuit save_chain → memory_key=None
    config.enable_skill_provenance_save = True
    config.enable_reusable_step_save = True

    gen = MAGEGenerator(config=config)
    mocker.patch.object(gen._memory, "save_agent_skill", new_callable=AsyncMock, return_value="skill-entity-A")
    mocker.patch.object(gen._memory, "save_reusable_step", new_callable=AsyncMock, return_value="step-entity-B")

    result = await gen.generate("Test query")

    # Chain memory_key is None (memory disabled), so memory_keys["chain"] is absent
    assert "chain" not in result.memory_keys
    # Skills + reusable_steps captured
    assert result.memory_keys["skills"] == ["skill-entity-A"]
    assert result.memory_keys["reusable_steps"] == ["step-entity-B"]


@pytest.mark.asyncio
async def test_generator_memory_keys_empty_when_features_off(config: MAGEConfig, mock_openai_fast: MagicMock) -> None:
    """Default: no ecosystem-save flags, no chain save → memory_keys is empty."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")
    assert result.memory_keys == {}
