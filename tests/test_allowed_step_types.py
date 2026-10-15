"""Enforce allowed_step_types at planner output — plan item P2.8.

The ``allowed_step_types`` constraint was prompt-only; ``StepPlanner`` never
validated the LLM's output, so a disallowed type slipped through and
``carl_export`` silently downgraded it to ``llm`` (lossy). ``StepPlanner.plan``
now validates the output: re-prompt on violation (up to ``max_type_retries``),
then coerce any leftover to a safe allowed type. These tests drive the real
``plan()`` path with a mocked LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.step_planner import StepPlanner
from mmar_mage.schemas import DomainAnalysis, PlannedStep, StepPlan

DOMAIN = DomainAnalysis(domain="general")


def _plan(*types: str) -> StepPlan:
    return StepPlan(
        steps=[PlannedStep(number=i + 1, title=f"S{i + 1}", step_type=t, rough_aim="x") for i, t in enumerate(types)],
        rationale="r",
    )


def _planner(side_effect: list) -> StepPlanner:
    llm = MagicMock()
    llm.generate_structured = AsyncMock(side_effect=side_effect)
    return StepPlanner(llm)


async def _run(planner: StepPlanner, allowed: list[str] | None, **kw) -> StepPlan:
    return await planner.plan("q", DOMAIN, allowed_step_types=allowed, **kw)


# --------------------------------------------------------------------------- #
# No enforcement when unconstrained / already valid
# --------------------------------------------------------------------------- #
async def test_no_allowed_means_no_enforcement():
    planner = _planner([_plan("agent_skill", "llm")])
    out = await _run(planner, None)
    assert [s.step_type for s in out.steps] == ["agent_skill", "llm"]
    assert planner._llm.generate_structured.await_count == 1


async def test_allowed_satisfied_no_retry():
    planner = _planner([_plan("llm", "llm")])
    out = await _run(planner, ["llm"])
    assert [s.step_type for s in out.steps] == ["llm", "llm"]
    assert planner._llm.generate_structured.await_count == 1  # no re-prompt


# --------------------------------------------------------------------------- #
# Re-prompt on violation
# --------------------------------------------------------------------------- #
async def test_violation_reprompts_then_accepts():
    planner = _planner([_plan("agent_skill", "llm"), _plan("llm", "llm")])
    out = await _run(planner, ["llm"])
    assert [s.step_type for s in out.steps] == ["llm", "llm"]
    assert planner._llm.generate_structured.await_count == 2  # initial + 1 re-prompt


async def test_reprompt_feedback_lists_violations():
    planner = _planner([_plan("agent_skill"), _plan("llm")])
    await _run(planner, ["llm"])
    # the re-prompt's system prompt names the offending type
    second_call = planner._llm.generate_structured.await_args_list[1]
    assert "agent_skill" in second_call.kwargs["system_prompt"]
    assert "INVALID" in second_call.kwargs["system_prompt"]


# --------------------------------------------------------------------------- #
# Coercion after exhausting retries
# --------------------------------------------------------------------------- #
async def test_coerces_to_llm_after_retries():
    # always returns a disallowed type → initial + 2 retries, then coerce
    planner = _planner([_plan("agent_skill", "tool")] * 3)
    out = await _run(planner, ["llm"], max_type_retries=2)
    assert [s.step_type for s in out.steps] == ["llm", "llm"]
    assert planner._llm.generate_structured.await_count == 3


async def test_coerces_to_first_allowed_when_llm_not_allowed():
    planner = _planner([_plan("agent_skill")] * 3)
    out = await _run(planner, ["tool", "mcp"], max_type_retries=2)
    assert out.steps[0].step_type == "mcp"  # sorted(["tool","mcp"])[0] == "mcp"


async def test_partial_violation_only_coerces_offenders():
    planner = _planner([_plan("llm", "agent_skill", "llm")] * 3)
    out = await _run(planner, ["llm"], max_type_retries=2)
    assert [s.step_type for s in out.steps] == ["llm", "llm", "llm"]


async def test_zero_retries_coerces_immediately():
    planner = _planner([_plan("agent_skill")])
    out = await _run(planner, ["llm"], max_type_retries=0)
    assert out.steps[0].step_type == "llm"
    assert planner._llm.generate_structured.await_count == 1  # no re-prompt, straight to coerce
