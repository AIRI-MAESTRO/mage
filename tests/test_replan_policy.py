"""Default RE-PLAN policy generation — deepened + shape-corrected (plan item P2.3).

The default policy now (a) emits the *real* ``mmar_carl.ReplanPolicy`` shape — the
old one nested checker settings under ``config`` where the schema expects them
flat, so every setting was silently dropped on load, and wrote ``max_replans`` to
a non-existent top-level field — and (b) scales depth + budgets with chain risk.
The real-execution check round-trips the emitted policy through the real
``mmar_carl.ReplanPolicy`` and asserts the settings survive.
"""

from __future__ import annotations

from typing import Any

import pytest

from mmar_mage.generator import _build_default_replan_policy, _chain_replan_risk


def _chain(specs: list[tuple[int, str]], checkpoint: bool = True, checkpoint_idx: int = 0) -> dict[str, Any]:
    steps = [
        {"number": i + 1, "step_type": t, "title": f"S{i + 1}", "aim": "do", "dependencies": []}
        for i, (_, t) in enumerate(specs)
    ]
    if checkpoint and steps:
        steps[checkpoint_idx]["checkpoint"] = True
        steps[checkpoint_idx]["checkpoint_name"] = "primary"
    return {"steps": steps}


# --------------------------------------------------------------------------- #
# Gating
# --------------------------------------------------------------------------- #
class TestGating:
    def test_no_checkpoint_returns_none(self) -> None:
        assert _build_default_replan_policy(_chain([(1, "llm"), (2, "llm")], checkpoint=False)) is None

    def test_falsey_checkpoint_field_ignored(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "step_type": "llm", "checkpoint": False},
                {"number": 2, "step_type": "llm", "checkpoint": None},
            ]
        }
        assert _build_default_replan_policy(chain) is None


# --------------------------------------------------------------------------- #
# Corrected shape (regression guards for the silent-config-loss bug)
# --------------------------------------------------------------------------- #
class TestCorrectedShape:
    def test_with_checkpoint_produces_policy(self) -> None:
        policy = _build_default_replan_policy(_chain([(1, "agent_skill"), (2, "llm")]))
        assert policy is not None
        assert policy["source"] == "mage_default"
        assert policy["aggregation"]["strategy"] == "any"
        assert len(policy["checkers"]) == 1

    def test_checker_fields_are_flat_not_nested(self) -> None:
        checker = _build_default_replan_policy(_chain([(1, "tool"), (2, "llm")]))["checkers"][0]
        assert "config" not in checker  # the old nesting silently dropped everything
        assert checker["type"] == "rule_based"
        assert checker["trigger_on_failed_step"] is True
        assert checker["action_on_failure"] == "replan_from_checkpoint"
        assert checker["rollback_target_on_failure"]["target_type"] == "nearest_checkpoint"

    def test_custom_action(self) -> None:
        policy = _build_default_replan_policy(_chain([(1, "llm")]), action_on_failure="restart_chain")
        assert policy["checkers"][0]["action_on_failure"] == "restart_chain"

    def test_max_replans_reaches_budget_not_dropped(self) -> None:
        # old shape wrote a top-level "max_replans" that CARL silently dropped
        policy = _build_default_replan_policy(_chain([(1, "llm"), (2, "llm")]), max_replans=10)
        assert "max_replans" not in policy
        assert policy["budgets"]["max_replans_per_chain"] == 10


# --------------------------------------------------------------------------- #
# Risk-based depth + budget scaling
# --------------------------------------------------------------------------- #
class TestRiskDepth:
    def test_risk_scoring(self) -> None:
        assert _chain_replan_risk([{"step_type": "llm"}, {"step_type": "llm"}]) == 0
        assert _chain_replan_risk([{"step_type": "tool"}, {"step_type": "mcp"}]) == 2
        assert _chain_replan_risk([{"step_type": "debate"}]) == 2
        assert _chain_replan_risk([{"step_type": "llm"}] * 6) == 1

    def test_low_risk_single_checker_any(self) -> None:
        policy = _build_default_replan_policy(_chain([(1, "llm"), (2, "llm"), (3, "llm")]))
        assert len(policy["checkers"]) == 1
        assert policy["aggregation"]["strategy"] == "any"

    def test_high_risk_adds_llm_judge_and_voting(self) -> None:
        # debate (2) + 2 tools (2) + len>=6 (1) = 5 → escalate
        policy = _build_default_replan_policy(
            _chain([(1, "debate"), (2, "tool"), (3, "tool"), (4, "llm"), (5, "llm"), (6, "mcp"), (7, "llm")])
        )
        assert [c["type"] for c in policy["checkers"]] == ["rule_based", "llm"]
        assert policy["aggregation"]["strategy"] == "mandatory_plus_k_of_rest"
        assert policy["aggregation"]["mandatory_checkers"] == ["auto-replan-on-failure"]

    def test_budgets_scale_with_length(self) -> None:
        short = _build_default_replan_policy(_chain([(1, "llm"), (2, "llm")]))
        long = _build_default_replan_policy(_chain([(i, "llm") for i in range(1, 11)]))
        assert long["budgets"]["max_replans_per_chain"] >= short["budgets"]["max_replans_per_chain"]
        assert long["budgets"]["max_visits_per_checkpoint"] >= short["budgets"]["max_visits_per_checkpoint"]


# --------------------------------------------------------------------------- #
# Real-execution scenario — round-trip through the real mmar_carl ReplanPolicy
# --------------------------------------------------------------------------- #
class TestCarlRoundTrip:
    def test_settings_preserved_on_load(self) -> None:
        carl = pytest.importorskip("mmar_carl")
        policy = _build_default_replan_policy(_chain([(1, "tool"), (2, "llm"), (3, "llm")]), max_replans=4)
        loaded = carl.ReplanPolicy.model_validate(policy)

        rule = loaded.checkers[0]
        assert rule.action_on_failure.value == "replan_from_checkpoint"  # was silently lost before
        assert rule.rollback_target_on_failure.target_type.value == "nearest_checkpoint"
        assert rule.feedback_on_failure  # non-empty
        assert loaded.budgets.max_replans_per_chain == 4

    def test_high_risk_policy_loads(self) -> None:
        carl = pytest.importorskip("mmar_carl")
        policy = _build_default_replan_policy(
            _chain([(1, "debate"), (2, "tool"), (3, "tool"), (4, "mcp"), (5, "llm"), (6, "llm")])
        )
        loaded = carl.ReplanPolicy.model_validate(policy)
        assert len(loaded.checkers) == 2
        assert loaded.aggregation.strategy.value == "mandatory_plus_k_of_rest"
