"""T1 — planner prompt enforces: one tool step = exactly one tool call."""

from __future__ import annotations

from mmar_mage.prompts import STEP_PLANNER_ANTI_PATTERNS_PROMPT, STEP_PLANNER_PROMPT


def test_planner_prompt_states_one_tool_one_call() -> None:
    p = STEP_PLANNER_PROMPT
    assert "EXACTLY ONE external call" in p
    assert "CANNOT chain" in p
    # the composite live example (IP -> geo -> weather) is spelled out
    assert "geolocate that IP" in p
    # the "then call" violation marker is named
    assert '"then call"' in p


def test_anti_patterns_flags_composite_tool_step() -> None:
    a = STEP_PLANNER_ANTI_PATTERNS_PROMPT
    assert "one tool step = one" in a
    assert "THREE dependent tool steps" in a
