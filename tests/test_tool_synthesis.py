"""Tool/skill synthesis — plan item P1.17, Phase A (ToolSpec, no code execution).

Covers the detector + spec generator + planner-hint formatting with a directly
mocked LLM, plus the real-execution path: ``generate()`` with
``enable_tool_synthesis=True`` runs stage 0.6 and the proposed tool reaches the
StepPlanner prompt. Phases B (sandboxed codegen) and C (skill synthesis) are out
of scope here — this module generates and runs NO code.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.tool_spec_generator import (
    ToolNeed,
    ToolSpec,
    ToolSpecGenerator,
    format_tool_spec_for_planner,
)
from mmar_mage.schemas import DomainAnalysis, MAGEConfig, PlannedStep, StepPlan

DOMAIN = DomainAnalysis(domain="data-engineering")

NEED_YES = {"needed": True, "description": "parse fixed-width COBOL records to dicts", "rationale": "llm can't"}
NEED_NO = {"needed": False}
SPEC = {
    "name": "parse_cobol_records",
    "signature": "parse_cobol_records(raw: bytes, layout: dict) -> list[dict]",
    "docstring": "Parse fixed-width COBOL records into dicts using a layout map.",
    "input_schema": {"raw": "bytes", "layout": "dict"},
    "output_schema": {"records": "list[dict]"},
    "implementation_hint": "struct.unpack per field offsets",
    "dependencies": [],
}


def _llm(json_side: list[Any]) -> MagicMock:
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=json_side)
    return llm


# --------------------------------------------------------------------------- #
# detect_need / generate_spec
# --------------------------------------------------------------------------- #
async def test_detect_need_parses_positive():
    need = await ToolSpecGenerator(_llm([NEED_YES])).detect_need("q", DOMAIN)
    assert need.needed is True
    assert "COBOL" in need.description


async def test_detect_need_failure_is_non_fatal():
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=RuntimeError("down"))
    need = await ToolSpecGenerator(llm).detect_need("q", DOMAIN)
    assert need.needed is False


async def test_generate_spec_parses_all_fields():
    spec = await ToolSpecGenerator(_llm([SPEC])).generate_spec(ToolNeed(needed=True, description="d"), "q", DOMAIN)
    assert spec is not None
    assert spec.name == "parse_cobol_records"
    assert "list[dict]" in spec.signature
    assert spec.input_schema == {"raw": "bytes", "layout": "dict"}


async def test_generate_spec_missing_name_returns_none():
    spec = await ToolSpecGenerator(_llm([{"signature": "f()"}])).generate_spec(ToolNeed(needed=True), "q", DOMAIN)
    assert spec is None


# --------------------------------------------------------------------------- #
# synthesize (combined entry point)
# --------------------------------------------------------------------------- #
async def test_synthesize_returns_spec_when_needed():
    gen = ToolSpecGenerator(_llm([NEED_YES, SPEC]))
    spec = await gen.synthesize("parse these COBOL files", DOMAIN)
    assert spec is not None
    assert spec.name == "parse_cobol_records"


async def test_synthesize_skips_spec_when_not_needed():
    llm = _llm([NEED_NO])
    spec = await ToolSpecGenerator(llm).synthesize("summarize this text", DOMAIN)
    assert spec is None
    assert llm.generate_json.await_count == 1  # only the detector ran, no spec call


async def test_synthesize_needed_but_no_description_returns_none():
    spec = await ToolSpecGenerator(_llm([{"needed": True, "description": ""}])).synthesize("q", DOMAIN)
    assert spec is None


# --------------------------------------------------------------------------- #
# format_tool_spec_for_planner
# --------------------------------------------------------------------------- #
def test_format_marks_tool_as_not_existing():
    text = format_tool_spec_for_planner(ToolSpec(**SPEC))
    assert "does NOT exist yet" in text
    assert "parse_cobol_records" in text
    assert "list[dict]" in text


def test_format_empty_spec_returns_empty_string():
    assert format_tool_spec_for_planner(ToolSpec(name="")) == ""


# --------------------------------------------------------------------------- #
# Real-execution scenario — generate() stage 0.6 surfaces the proposal
# --------------------------------------------------------------------------- #
def _planning_only_gen(json_side: list[Any]):
    from mmar_mage.generator import MAGEGenerator

    cfg = MAGEConfig(
        api_key="t",
        mode="deep",
        enable_memory=False,
        enable_domain_analysis=False,
        enable_topology_selection=False,  # isolate the tool-synthesis stage
        enable_tool_synthesis=True,
        enable_step_planning=True,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        simplicity_bias=0.0,
        max_retries=0,
    )
    gen = MAGEGenerator(config=cfg)
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate_json = AsyncMock(side_effect=json_side)
    plan = StepPlan(
        steps=[PlannedStep(number=1, title="Parse", step_type="tool", rough_aim="parse")],
        rationale="r",
    )
    gen._llm.generate_structured = AsyncMock(return_value=plan)
    return gen


async def test_generate_surfaces_proposed_tool_to_planner():
    gen = _planning_only_gen([NEED_YES, SPEC])
    result = await gen.generate("parse these COBOL files into JSON")

    assert "tool_synthesis" in result.metadata.deep_stages_completed
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Proposed synthesized tool" in planner_prompt
    assert "parse_cobol_records" in planner_prompt


async def test_generate_no_tool_when_not_needed():
    gen = _planning_only_gen([NEED_NO])
    result = await gen.generate("summarize this paragraph")

    assert "tool_synthesis" not in result.metadata.deep_stages_completed  # nothing proposed
    planner_prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Proposed synthesized tool" not in planner_prompt
