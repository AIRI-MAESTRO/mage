"""T18 — live data > memory: planner anti-pattern rule + a lint that flags a tool
step whose aim presumes the personal fact it should fetch live."""

from __future__ import annotations

from mmar_mage.chain_repair import lint_chain
from mmar_mage.prompts import STEP_PLANNER_ANTI_PATTERNS_PROMPT


def test_planner_prompt_warns_against_presuming_live_facts() -> None:
    # snapshot of the prompt rule (DoD: prompt edits covered by a prompt snapshot test)
    low = STEP_PLANNER_ANTI_PATTERNS_PROMPT.lower()
    assert "the user is in russia" in low  # the concrete run-096 example
    assert "discovers it live" in low
    assert "$history" in STEP_PLANNER_ANTI_PATTERNS_PROMPT


def test_lint_flags_aim_presuming_a_personal_fact() -> None:
    chain = {"steps": [{
        "step_type": "tool", "number": 1, "title": "Weather",
        "tool_name": "http_request", "tool_input_mapping": {"url": "https://wttr.in/"},
        "aim": "Get the weather since the user is in Moscow",
    }]}
    assert any("presumes a personal fact" in i for i in lint_chain(chain))


def test_lint_allows_a_clean_fetch_aim() -> None:
    chain = {"steps": [{
        "step_type": "tool", "number": 1, "title": "Geolocate",
        "tool_name": "http_request", "tool_input_mapping": {"url": "http://ip-api.com/json/$history[1]"},
        "aim": "Geolocate the IP from step 1 to determine the country",
    }]}
    assert lint_chain(chain) == []  # determines the value, no presumption


def test_lint_reads_rough_aim_too() -> None:
    chain = {"steps": [{
        "step_type": "tool", "number": 2, "title": "x",
        "tool_name": "http_request", "tool_input_mapping": {"url": "https://x/"},
        "rough_aim": "fetch data; your country is Sweden",
    }]}
    assert any("presumes a personal fact" in i for i in lint_chain(chain))
