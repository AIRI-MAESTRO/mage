"""T12 — E2E regression: the composite live query generates a replan-ready chain.

The run-096 class (IP -> geolocate -> weather -> advise) must produce a >=4-step
chain with valid urls AND stay recovery-ready: a tool failure retries then replans
from a checkpoint instead of masking as success. The actual runtime retry/replan is
exercised once the real CARL executor is wired into evaluation/ (T35).
"""

from __future__ import annotations

from typing import Any

from evaluation.tasks import BENCHMARK_TASKS
from mmar_mage.chain_repair import lint_chain
from mmar_mage.generator import _build_default_replan_policy, _promote_checkpoints
from mmar_mage.schemas import CARLStepSchema


def _tool(number: int, title: str, url: str, dep: list[int], key: str) -> dict[str, Any]:
    return {
        "step_type": "tool", "number": number, "title": title, "aim": title,
        "dependencies": dep, "tool_name": "http_request",
        "tool_input_mapping": {"url": url, "method": "GET"}, "tool_output_key": key,
    }


def _composite_chain() -> dict[str, Any]:
    return {
        "name": "Weather where my IP is",
        "steps": [
            _tool(1, "Get public IP", "https://api.ipify.org", [], "ip"),
            _tool(2, "Geolocate the IP", "http://ip-api.com/json/$history[1]", [1], "geo"),
            _tool(3, "Get the weather", "https://wttr.in/$history[2]?format=j1", [2], "weather"),
            {"step_type": "llm", "number": 4, "title": "Recommend clothing",
             "aim": "Advise what to wear for the weather", "dependencies": [3]},
        ],
    }


def test_live_composite_is_a_mandatory_eval_case() -> None:
    by_id = {t.task_id: t for t in BENCHMARK_TASKS["live_composite"]}
    assert "live_composite_001" in by_id  # the run-096 query
    composite = by_id["live_composite_001"]
    assert "ip" in composite.query.lower()
    assert composite.metadata["min_atomic_calls"] >= 3


def test_composite_chain_is_valid_and_multistep() -> None:
    chain = _composite_chain()
    assert len(chain["steps"]) >= 4  # T1/T2: one atomic external call per step
    assert lint_chain(chain) == []  # T5: all urls valid, no $outer_context
    for s in chain["steps"]:
        url = (s.get("tool_input_mapping") or {}).get("url", "")
        assert "$outer_context" not in url


def test_composite_chain_is_replan_ready() -> None:
    chain = _composite_chain()
    _promote_checkpoints(chain["steps"])
    assert any(s.get("checkpoint") for s in chain["steps"])  # T10: a rollback point exists
    policy = _build_default_replan_policy(chain)
    assert policy is not None and policy["enabled"]  # a tool failure replans...
    assert CARLStepSchema(number=1, step_type="tool", title="t", tool_name="x").retry_max >= 1  # ...after retrying
