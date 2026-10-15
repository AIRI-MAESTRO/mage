"""T17 — extract a run's tool-step results into session-memory facts."""

from __future__ import annotations

from mmar_mage.memory import extract_run_facts


def test_extracts_successful_tool_outputs() -> None:
    steps = [
        {"step_type": "tool", "number": 1, "tool_output_key": "ip", "success": True, "output": "185.189.46.80"},
        {"step_type": "tool", "number": 2, "tool_output_key": "country", "success": True,
         "output": {"country": "Sweden"}},
        {"step_type": "llm", "number": 3, "success": True, "output": "you should wear a coat"},  # not a tool
    ]
    facts = extract_run_facts(steps)
    assert facts["ip"] == "185.189.46.80"
    assert "Sweden" in facts["country"]
    assert "step_3" not in facts and len(facts) == 2  # the llm step is skipped


def test_skips_failed_and_empty_steps() -> None:
    steps = [
        {"step_type": "tool", "number": 1, "success": False, "output": "boom"},  # failed
        {"step_type": "tool", "number": 2, "error": "timeout", "output": "x"},  # has an error field
        {"step_type": "tool", "number": 3, "success": True, "output": ""},  # empty output
        {"step_type": "tool", "number": 4, "success": True, "output": "185.1.2.3"},  # the only fact
    ]
    assert extract_run_facts(steps) == {"step_4": "185.1.2.3"}  # keyed by step_N when no output key


def test_truncates_and_handles_junk() -> None:
    assert extract_run_facts(None) == {}
    assert extract_run_facts(["not a dict", 42]) == {}
    big = {"step_type": "tool", "number": 1, "success": True, "output": "y" * 1000}
    assert len(extract_run_facts([big], max_chars=100)["step_1"]) == 100
