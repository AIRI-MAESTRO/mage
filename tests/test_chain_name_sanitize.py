"""T16 — chain name + description never leak a memory / context block."""

from __future__ import annotations

from mmar_mage.generator import _clean_query_for_name, _naming_heuristic

_POLLUTED = (
    "## What I remember about you (long-term memory)\n"
    "- role: ML researcher\n"
    "- location: Stockholm\n"
    "what is my ip address?"
)


def test_clean_query_strips_memory_block() -> None:
    out = _clean_query_for_name(_POLLUTED)
    assert out == "what is my ip address?"  # only the real task survives
    low = out.lower()
    assert "remember about you" not in low and "long-term memory" not in low and "role:" not in low


def test_clean_query_keeps_plain_query() -> None:
    assert _clean_query_for_name("какой у меня ip") == "какой у меня ip"


def test_clean_query_all_memory_is_empty() -> None:
    # whole query is a memory dump → empty (caller falls back to the domain title)
    assert _clean_query_for_name("## What I remember about you (long-term memory)\n- role: ML r") == ""


def test_naming_heuristic_does_not_leak_memory() -> None:
    name, description, _tags = _naming_heuristic(_POLLUTED, domain="general", task_type="qa")
    blob = (name + " " + description).lower()
    assert "remember about you" not in blob  # the run-096 leak
    assert "long-term memory" not in blob
    assert "role:" not in blob
    assert "what is my ip" in name.lower()  # the real task is in the name
