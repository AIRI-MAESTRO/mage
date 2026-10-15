"""T19 — planner prefers a single ready tool over a hand-rolled raw-HTTP chain."""

from __future__ import annotations

from mmar_mage.prompts import STEP_PLANNER_ANTI_PATTERNS_PROMPT


def test_planner_prompt_prefers_ready_tool_over_http_chain() -> None:
    low = STEP_PLANNER_ANTI_PATTERNS_PROMPT.lower()
    assert "web_search" in low  # the concrete ready tool the run-096 planner ignored
    assert "http_request" in low
    # The rule now matches the tool to the OUTPUT shape: web_search returns prose, so
    # named fields come from a follow-up structured_output step — not a hand-rolled chain.
    assert "structured_output" in low


def test_rule_does_not_contradict_atomic_split() -> None:
    # T19 must coexist with T1's "one tool step = one external call": the rule
    # explicitly carves out genuine N-call composites so it isn't read as "merge calls".
    low = STEP_PLANNER_ANTI_PATTERNS_PROMPT.lower()
    assert "one listed tool that covers the task is one step" in low
    assert "n real external calls" in low
