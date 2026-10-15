"""T2 — step count by atomic external calls (analyzer example + planner lower-bound)."""

from __future__ import annotations

from mmar_mage.prompts import DOMAIN_ANALYZER_PROMPT, STEP_PLANNER_PROMPT


def test_analyzer_counts_atomic_external_calls() -> None:
    p = DOMAIN_ANALYZER_PROMPT
    assert "COUNT ATOMIC EXTERNAL CALLS" in p
    # Example 5: composite live query → 4 steps, medium complexity
    assert "Example 5" in p
    assert "weather where my IP is" in p
    assert '"preferred_step_types": ["tool", "tool", "tool", "llm"]' in p


def test_planner_step_count_is_lower_bound() -> None:
    p = STEP_PLANNER_PROMPT
    assert "LOWER-BOUND hint" in p
    assert "one tool/mcp call per step" in p
    # the old hard "should match" wording is gone
    assert "should match the suggested_step_count" not in p
