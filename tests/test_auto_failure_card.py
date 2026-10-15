"""T34 — auto failure-cards for masked / explicit tool errors in a run."""

from __future__ import annotations

from unittest.mock import AsyncMock

from mmar_mage.memory import MemoryManager, find_masked_tool_errors
from mmar_mage.schemas import MAGEConfig


def test_detects_explicit_and_masked_tool_errors() -> None:
    steps = [
        {"step_type": "tool", "number": 1, "tool_name": "http_request", "success": True, "output": "185.189.46.80"},
        {"step_type": "tool", "number": 2, "tool_name": "http_request", "success": True,
         "output": "GET https://...: Invalid IDNA hostname"},  # masked: success but error output
        {"step_type": "tool", "number": 3, "tool_name": "geo", "success": False, "error": "connection refused"},
        {"step_type": "llm", "number": 4, "success": True, "output": "the request raised an exception earlier"},
    ]
    errs = find_masked_tool_errors(steps)
    assert {e["number"] for e in errs} == {2, 3}  # 1 is clean, 4 is llm prose (not a tool)
    by_num = {e["number"]: e for e in errs}
    assert by_num[2]["masked"] is True  # success=true + error-looking output (the run-096 class)
    assert by_num[3]["masked"] is False  # explicit success=false


def test_detector_robust_on_junk() -> None:
    assert find_masked_tool_errors(None) == []
    assert find_masked_tool_errors([]) == []
    assert find_masked_tool_errors(["not a dict", 42]) == []


async def test_save_failure_cards_noop_when_disabled() -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_failure_card_save=False))
    steps = [{"step_type": "tool", "number": 1, "success": False, "error": "boom"}]
    assert await mgr.save_failure_cards_for_run("q", step_results=steps) == []


async def test_save_failure_cards_writes_one_per_error() -> None:
    mgr = MemoryManager(MAGEConfig(api_key="test", enable_failure_card_save=True))
    mgr.save_failure_card = AsyncMock(side_effect=["k1", "k2"])  # type: ignore[method-assign]
    steps = [
        {"step_type": "tool", "number": 2, "success": True, "output": "Invalid IDNA hostname"},  # masked
        {"step_type": "tool", "number": 3, "success": False, "error": "timeout"},  # explicit
        {"step_type": "tool", "number": 1, "success": True, "output": "ok 200"},  # clean → no card
    ]
    keys = await mgr.save_failure_cards_for_run("weather where my ip", domain="general", step_results=steps)
    assert keys == ["k1", "k2"]
    assert mgr.save_failure_card.await_count == 2
    reasons = [c.kwargs["reason"] for c in mgr.save_failure_card.await_args_list]
    assert any("masked tool error" in r for r in reasons)  # the masked one is labelled
