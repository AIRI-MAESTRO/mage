"""Anti-pattern catalogue — plan item P2.6.

A ``STEP_PLANNER_ANTI_PATTERNS_PROMPT`` is appended to the planner system prompt
to steer away from over-engineering (needless supervisor/debate/parallel_sampling,
runaway step counts, checkpoint spam). The real-execution check confirms the
catalogue reaches the planner's LLM call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.step_planner import StepPlanner
from mmar_mage.prompts import STEP_PLANNER_ANTI_PATTERNS_PROMPT
from mmar_mage.schemas import DomainAnalysis, StepPlan


# --------------------------------------------------------------------------- #
# The constant
# --------------------------------------------------------------------------- #
def test_catalogue_covers_the_key_anti_patterns():
    text = STEP_PLANNER_ANTI_PATTERNS_PROMPT.lower()
    assert "supervisor" in text
    assert "parallel_sampling" in text
    assert "debate" in text
    assert "checkpoint" in text
    assert "6+" in STEP_PLANNER_ANTI_PATTERNS_PROMPT or "fewest steps" in text


def test_catalogue_has_no_format_placeholders():
    # it's concatenated AFTER STEP_PLANNER_PROMPT.format(), but guard anyway:
    # no stray single braces that a future .format() would choke on.
    assert "{" not in STEP_PLANNER_ANTI_PATTERNS_PROMPT
    assert "}" not in STEP_PLANNER_ANTI_PATTERNS_PROMPT


# --------------------------------------------------------------------------- #
# Real-execution scenario — the catalogue reaches the planner LLM call
# --------------------------------------------------------------------------- #
async def test_planner_prompt_includes_anti_patterns():
    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value=StepPlan(steps=[], rationale="ok"))
    planner = StepPlanner(llm)

    await planner.plan("compare three options", DomainAnalysis(domain="general"))

    system_prompt = llm.generate_structured.await_args.kwargs["system_prompt"]
    assert "Avoid these anti-patterns" in system_prompt
    assert "Don't use `supervisor`" in system_prompt
    # composes with the rest of the planner prompt (not a standalone replacement)
    assert "reasoning step planner" in system_prompt
