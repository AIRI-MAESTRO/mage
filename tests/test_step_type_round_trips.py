"""Tests for §9 P1 — heterogeneous step-type round-trips.

Exercises every agent-features step type through MAGE's validation and
serialization layers:

- `validate_carl_json` accepts the chain with internal step types
- `to_carl_compat` produces a strict-CARL-loadable dict (re-nesting
  per-type config keys under `step_config`)
- `validate_carl_json(strict_carl=True)` passes on the exported form
- The whole flow round-trips through `MAGEResult.to_care_dict` /
  `from_care_dict` losslessly
- `_try_load_reasoning_chain` accepts the strict-CARL JSON when
  ``mmar_carl`` is importable (skip otherwise)

The exemplars come from `tests/conftest.py`:
``SAMPLE_STEP_AGENT_SKILL``, ``SAMPLE_STEP_STRUCTURED_OUTPUT``,
``SAMPLE_STEP_AGENT_HANDOFF``, plus the aggregate
``SAMPLE_HETEROGENEOUS_CHAIN``.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from mmar_mage.carl_export import carl_loadable_step_types, to_carl_compat
from mmar_mage.generator import validate_carl_json
from mmar_mage.schemas import CARLChainSchema, CARLStepSchema, MAGEMetadata, MAGEResult

from .conftest import (
    SAMPLE_HETEROGENEOUS_CHAIN,
    SAMPLE_STEP_AGENT_HANDOFF,
    SAMPLE_STEP_AGENT_SKILL,
    SAMPLE_STEP_STRUCTURED_OUTPUT,
)

# ---------------------------------------------------------------------------
# CARLStepSchema accepts each exemplar
# ---------------------------------------------------------------------------


class TestExemplarsValidate:
    def test_agent_skill_exemplar(self) -> None:
        step = CARLStepSchema.model_validate(SAMPLE_STEP_AGENT_SKILL)
        assert step.step_type == "agent_skill"
        assert step.skill_execution_mode == "llm_agent"
        assert step.skill_timeout == 120.0
        assert step.skill_output_key == "pdf_tables"

    def test_structured_output_exemplar(self) -> None:
        step = CARLStepSchema.model_validate(SAMPLE_STEP_STRUCTURED_OUTPUT)
        assert step.step_type == "structured_output"
        assert step.structured_output_strict_json is True
        assert step.structured_output_schema_name == "QuarterlyReport"
        assert "revenue_change_pct" in step.structured_output_schema["properties"]

    def test_agent_handoff_exemplar(self) -> None:
        step = CARLStepSchema.model_validate(SAMPLE_STEP_AGENT_HANDOFF)
        assert step.step_type == "agent_handoff"
        assert step.handoff_chain_path == "registry://chains/risk-analysis@v2"
        assert step.handoff_output_key == "risk_report"


# ---------------------------------------------------------------------------
# Heterogeneous chain validates + round-trips
# ---------------------------------------------------------------------------


class TestHeterogeneousChain:
    def test_validates_with_internal_step_types(self) -> None:
        validate_carl_json(SAMPLE_HETEROGENEOUS_CHAIN)

    def test_to_carl_compat_then_strict_validation(self) -> None:
        exported = to_carl_compat(SAMPLE_HETEROGENEOUS_CHAIN)
        validate_carl_json(exported, strict_carl=True)
        # `to_carl_compat` keeps MAGE's canonical nested form: agent_skill and
        # structured_output stay native; agent_handoff is the one type that
        # always translates to ``llm`` (CARL needs a runtime sub-chain for it).
        types_after = {s["step_type"] for s in exported["steps"]}
        assert types_after == {"agent_skill", "structured_output", "llm"}

    def test_to_carl_compat_nests_per_type_config(self) -> None:
        """`to_carl_compat` re-nests flat per-type keys under `step_config`."""
        exported = to_carl_compat(SAMPLE_HETEROGENEOUS_CHAIN)
        # agent_skill: skill_execution_mode / skill_input_mapping / etc.
        ag_step = next(s for s in exported["steps"] if s["step_type"] == "agent_skill")
        assert "step_config" in ag_step
        sc = ag_step["step_config"]
        assert sc.get("execution_mode") == "llm_agent" or "execution_mode" in sc
        # structured_output: schema / schema_name
        so_step = next(s for s in exported["steps"] if s["step_type"] == "structured_output")
        assert "step_config" in so_step
        assert "structured_output_schema" not in so_step  # flat key was nested

    def test_chain_schema_serializes_each_type(self) -> None:
        """Build a CARLChainSchema from typed step models and ensure
        no field gets silently dropped on serialization."""
        chain = CARLChainSchema(
            name="Test heterogeneous",
            steps=[
                CARLStepSchema.model_validate(SAMPLE_STEP_AGENT_SKILL),
                CARLStepSchema.model_validate(SAMPLE_STEP_STRUCTURED_OUTPUT),
                CARLStepSchema.model_validate(SAMPLE_STEP_AGENT_HANDOFF),
            ],
        )
        dumped = chain.model_dump(exclude_none=True)
        # Each step keeps its identifying per-type field
        skill_step = next(s for s in dumped["steps"] if s["step_type"] == "agent_skill")
        assert "skill" in skill_step
        so_step = next(s for s in dumped["steps"] if s["step_type"] == "structured_output")
        assert "structured_output_schema" in so_step
        ah_step = next(s for s in dumped["steps"] if s["step_type"] == "agent_handoff")
        assert "handoff_chain_path" in ah_step


# ---------------------------------------------------------------------------
# MAGEResult.to_care_dict / from_care_dict round-trip with each type
# ---------------------------------------------------------------------------


class TestCareBundleRoundTrip:
    def _result_with(self, step: dict[str, Any]) -> MAGEResult:
        chain = {
            "name": f"single-{step['step_type']}",
            "max_workers": 1,
            "timeout": 60.0,
            "steps": [step],
        }
        return MAGEResult(
            chain_json=json.dumps(chain),
            chain_dict=chain,
            mode="deep",
            metadata=MAGEMetadata(num_steps=1),
        )

    @pytest.mark.parametrize(
        "exemplar",
        [SAMPLE_STEP_AGENT_SKILL, SAMPLE_STEP_STRUCTURED_OUTPUT, SAMPLE_STEP_AGENT_HANDOFF],
    )
    def test_round_trip_preserves_each_step_type(self, exemplar: dict[str, Any]) -> None:
        original = self._result_with(exemplar)
        bundle = original.to_care_dict()
        # Bundle must JSON-serialize
        as_json = json.dumps(bundle)
        rebuilt = MAGEResult.from_care_dict(json.loads(as_json))
        assert rebuilt.chain_dict["steps"][0]["step_type"] == exemplar["step_type"]
        # And the per-type config keys survive
        for key in exemplar:
            assert key in rebuilt.chain_dict["steps"][0]
            assert rebuilt.chain_dict["steps"][0][key] == exemplar[key]

    def test_full_heterogeneous_chain_round_trip(self) -> None:
        result = MAGEResult(
            chain_json=json.dumps(SAMPLE_HETEROGENEOUS_CHAIN),
            chain_dict=SAMPLE_HETEROGENEOUS_CHAIN,
            mode="deep",
            metadata=MAGEMetadata(num_steps=3),
            intermediate_artifacts={"domain_analysis": {"domain": "finance"}},
            memory_keys={"chain": "abc", "skills": ["sk1"]},
        )
        bundle_json = json.dumps(result.to_care_dict())
        rebuilt = MAGEResult.from_care_dict(json.loads(bundle_json))
        assert len(rebuilt.chain_dict["steps"]) == 3
        types = [s["step_type"] for s in rebuilt.chain_dict["steps"]]
        assert types == ["agent_skill", "structured_output", "agent_handoff"]
        assert rebuilt.memory_keys["skills"] == ["sk1"]
        assert rebuilt.intermediate_artifacts["domain_analysis"]["domain"] == "finance"


# ---------------------------------------------------------------------------
# Optional: mmar_carl ReasoningChain.from_json acceptance
# ---------------------------------------------------------------------------


def _mmar_carl_available() -> bool:
    try:
        importlib.import_module("mmar_carl")
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _mmar_carl_available(),
    reason="mmar_carl not installed; CARL round-trip skipped",
)
def test_strict_carl_chain_loads_via_reasoning_chain() -> None:
    """When mmar_carl is installed, the strict-CARL export loads cleanly.

    This pairs with the §9 P1 "round-trip via ReasoningChain.from_json"
    item — the rest of that test (executing the chain) requires both
    `use_typed_steps=True` support and a configured runtime, both of
    which sit outside MAGE.

    The heterogeneous sample includes ``agent_skill``; agent-features types
    reconstruct into TYPED step classes, so the chain loads via
    ``from_dict(use_typed_steps=True)`` (the legacy ``from_json`` path builds the
    unified ``StepDescription`` whose config union predates them). A step type the
    installed CARL doesn't know at all still skips the round-trip.
    """
    from mmar_carl import ReasoningChain  # type: ignore[import-untyped]

    exported = to_carl_compat(SAMPLE_HETEROGENEOUS_CHAIN)
    loadable = carl_loadable_step_types()
    unsupported = {s["step_type"] for s in exported["steps"]} - loadable
    if unsupported:
        pytest.skip(f"installed mmar_carl predates step type(s): {sorted(unsupported)}")

    chain_json = json.dumps(exported, ensure_ascii=False)
    # Agent-features types (agent_skill, ...) reconstruct into TYPED step classes whose
    # configs the legacy unified StepDescription union doesn't accept — so they load via
    # from_dict(use_typed_steps=True), not the legacy from_json() path.
    chain = ReasoningChain.from_dict(json.loads(chain_json), use_typed_steps=True)
    assert chain is not None
    assert len(chain.steps) == len(exported["steps"])
