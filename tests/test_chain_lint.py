"""T5 — deterministic tool-step lint before export + lint→repair routing."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.chain_repair import lint_chain, validate_or_repair
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig


def _tool_step(**over: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "step_type": "tool", "number": 1, "title": "Fetch", "aim": "fetch the data",
        "dependencies": [], "tool_name": "http_request",
        "tool_input_mapping": {"url": "https://api.ipify.org", "method": "GET"},
        "retry_max": 3, "reasoning_questions": "", "step_context_queries": [],
        "stage_action": "", "example_reasoning": "", "llm_config": None,
    }
    s.update(over)
    return s


def test_lint_clean_chain() -> None:
    assert lint_chain({"steps": [_tool_step()]}) == []
    assert lint_chain({"steps": [_tool_step(tool_input_mapping={"url": "$history[-1]"})]}) == []


def test_lint_flags_outer_context_in_url() -> None:
    issues = lint_chain({"steps": [_tool_step(tool_input_mapping={"url": "$outer_context", "method": "GET"})]})
    assert len(issues) == 1 and "$outer_context" in issues[0]


def test_lint_flags_free_text_url() -> None:
    issues = lint_chain({"steps": [_tool_step(tool_input_mapping={"url": "какая погода"})]})
    assert len(issues) == 1 and "free text" in issues[0]


def test_lint_flags_empty_tool_name_and_url() -> None:
    assert any("empty tool_name" in i for i in lint_chain({"steps": [_tool_step(tool_name="")]}))
    assert any("is empty" in i for i in lint_chain({"steps": [_tool_step(tool_input_mapping={"url": ""})]}))


def test_lint_unknown_tool() -> None:
    issues = lint_chain({"steps": [_tool_step(tool_name="frobnicate")]}, known_tools={"http_request", "web_search"})
    assert any("not a registered tool" in i for i in issues)
    assert lint_chain({"steps": [_tool_step()]}, known_tools={"http_request"}) == []


def test_lint_reads_compat_nested_form() -> None:
    step = {
        "step_type": "tool", "number": 1,
        "step_config": {"tool_name": "http_request", "input_mapping": {"url": "$outer_context"}},
    }
    assert any("$outer_context" in i for i in lint_chain({"steps": [step]}))


async def test_validate_or_repair_routes_lint_to_repair(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    base = {
        "name": "T", "max_workers": 1, "timeout": 300.0,
        "search_config": {
            "strategy": "substring",
            "substring_config": {"case_sensitive": False, "min_word_length": 3, "max_matches_per_query": 5},
        },
    }
    buggy = {**base, "steps": [_tool_step(tool_input_mapping={"url": "$outer_context", "method": "GET"})]}
    fixed = {**base, "steps": [_tool_step()]}
    llm.generate_json = AsyncMock(return_value=fixed)  # type: ignore[method-assign]

    out = await validate_or_repair(buggy, query="weather", llm=llm)
    assert lint_chain(out) == []  # repaired chain passes the lint
    llm.generate_json.assert_called_once()  # routed to exactly one repair pass


# --- transform lint: fake LLM-style extraction in a mechanical transform --------


def test_lint_flags_placeholder_transform() -> None:
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {"tool_name": "web_search"}},
        {"number": 2, "step_type": "transform", "dependencies": [1], "step_config": {
            "transform_type": "format",
            "expression": '{"temperature": "{{extracted from $history[0]}}"}',
        }},
    ]}
    issues = lint_chain(chain)
    assert any("non-executable syntax" in i and "step 2" in i for i in issues)


def test_lint_flags_history_slice_transform() -> None:
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {"tool_name": "current_datetime"}},
        {"number": 2, "step_type": "transform", "dependencies": [1], "step_config": {
            "transform_type": "format",
            "expression": '{"date": "$history[0][:10]"}',
        }},
    ]}
    assert any("non-executable syntax" in i for i in lint_chain(chain))


def test_lint_allows_clean_transform() -> None:
    # Legitimate format: whole-value substitution + a single-brace {key} template.
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {"tool_name": "get_ip"}},
        {"number": 2, "step_type": "transform", "dependencies": [1], "step_config": {
            "transform_type": "format",
            "input_key": "$history[0]",
            "expression": "http://ip-api.com/json/{ip}?fields=country,city",
        }},
    ]}
    assert lint_chain(chain) == []


# --- whole-value rule: a $-ref must BE the value, not be spliced into a string ---


def test_lint_flags_spliced_history_ref() -> None:
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {
            "tool_name": "web_search",
            "input_mapping": {"query": "$history[0] погода сейчас"},  # spliced!
        }},
    ]}
    assert any("splices a $-reference" in i for i in lint_chain(chain))


def test_lint_flags_spliced_outer_context() -> None:
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {
            "tool_name": "web_search",
            "input_mapping": {"query": "weather in $outer_context today"},
        }},
    ]}
    assert any("splices a $-reference" in i for i in lint_chain(chain))


def test_lint_allows_whole_value_refs() -> None:
    chain = {"steps": [
        {"number": 1, "step_type": "tool", "step_config": {
            "tool_name": "web_search",
            "input_mapping": {"query": "$outer_context"},  # whole value — OK
        }},
        {"number": 2, "step_type": "tool", "dependencies": [1], "step_config": {
            "tool_name": "fetch_url",
            "input_mapping": {"url": "$history[0]"},  # whole value — OK
        }},
    ]}
    assert lint_chain(chain) == []
