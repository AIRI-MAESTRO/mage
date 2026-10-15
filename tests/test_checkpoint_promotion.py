"""Risk-driven checkpoint promotion → enables the default ReplanPolicy (P1.14)."""

from __future__ import annotations

from mmar_mage.generator import _build_default_replan_policy, _promote_checkpoints


def _steps(specs: list[tuple[int, str, list[int]]]) -> list[dict]:
    return [{"number": n, "step_type": t, "dependencies": d} for n, t, d in specs]


def test_promotes_latest_synthesizer_merge_step():
    # diamond: 1 → (2, 3) → 4 ; step 4 merges two branches
    steps = _steps([(1, "llm", []), (2, "llm", [1]), (3, "llm", [1]), (4, "llm", [2, 3])])
    _promote_checkpoints(steps)
    assert steps[3]["checkpoint"] is True
    assert steps[3]["checkpoint_name"] == "primary"
    assert not any(s.get("checkpoint") for s in steps[:3])  # only the merge step


def test_promotes_latest_when_multiple_merges():
    steps = _steps([(1, "llm", []), (2, "llm", []), (3, "llm", [1, 2]), (4, "llm", [1, 2, 3])])
    _promote_checkpoints(steps)
    assert steps[3]["checkpoint"] is True  # step 4 (latest merge), not step 3
    assert steps[2].get("checkpoint") is None


def test_promotes_first_tool_in_long_linear_chain():
    # no merge (all single-dep), >=4 steps → first external call
    steps = _steps([(1, "llm", []), (2, "tool", [1]), (3, "tool", [2]), (4, "llm", [3])])
    _promote_checkpoints(steps)
    assert steps[1]["checkpoint"] is True  # first tool step
    assert steps[1]["checkpoint_name"] == "primary"


def test_no_promotion_in_short_linear_chain():
    steps = _steps([(1, "llm", []), (2, "llm", [1]), (3, "llm", [2])])  # 3 steps, no merge, no tool
    _promote_checkpoints(steps)
    assert not any(s.get("checkpoint") for s in steps)


def test_noop_when_a_checkpoint_already_exists():
    steps = _steps([(1, "llm", []), (2, "llm", [1]), (3, "llm", [1, 2])])
    steps[0]["checkpoint"] = True  # pre-existing checkpoint (e.g. set by StepDescriber)
    _promote_checkpoints(steps)
    assert steps[0]["checkpoint"] is True
    assert steps[2].get("checkpoint") is None  # merge step NOT additionally promoted


def test_promotion_enables_default_replan_policy():
    steps = _steps([(1, "llm", []), (2, "llm", [1]), (3, "llm", [1]), (4, "llm", [2, 3])])
    chain = {"steps": steps}

    assert _build_default_replan_policy(chain) is None  # no checkpoint yet → no policy

    _promote_checkpoints(steps)
    policy = _build_default_replan_policy(chain)
    assert policy is not None
    assert policy["source"] == "mage_default"
