"""T10 — replan reanimated: care_default enables it; 3-step composites get a checkpoint."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from mmar_mage.generator import _build_default_replan_policy, _promote_checkpoints


def _steps(specs: list[tuple[int, str, list[int]]]) -> list[dict[str, Any]]:
    return [{"number": n, "step_type": t, "dependencies": d} for n, t, d in specs]


def test_three_step_composite_gets_a_checkpoint() -> None:
    # IP(tool) -> geolocate(tool) -> advise(llm): the run-096 class, 3 steps
    steps = _steps([(1, "tool", []), (2, "tool", [1]), (3, "llm", [2])])
    _promote_checkpoints(steps)
    assert steps[0]["checkpoint"] is True  # first external call is the rollback point
    assert steps[0]["checkpoint_name"] == "primary"


def test_three_step_composite_enables_policy() -> None:
    steps = _steps([(1, "tool", []), (2, "tool", [1]), (3, "llm", [2])])
    chain = {"steps": steps}
    assert _build_default_replan_policy(chain) is None  # no checkpoint yet → no policy
    _promote_checkpoints(steps)
    policy = _build_default_replan_policy(chain)
    assert policy is not None and policy["enabled"] is True


def test_two_step_chain_still_no_checkpoint() -> None:
    steps = _steps([(1, "tool", []), (2, "llm", [1])])  # too short to be a composite
    _promote_checkpoints(steps)
    assert not any(s.get("checkpoint") for s in steps)


def test_care_default_enables_replan() -> None:
    mage = tomllib.loads(Path("configs/care_default.toml").read_text(encoding="utf-8"))["mage"]
    assert mage["enable_replan_generation"] is True
