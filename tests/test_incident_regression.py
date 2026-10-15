"""T36 — incident regression for run 096 (the composite "weather where my IP is").

Parts (a) the describer never emitting ``$outer_context`` in a url, (b) the T5
deterministic lint, and (c) the T8 stage-retry are covered by
``test_describer_tool_signatures.py`` / ``test_tool_step_schema_validator.py``,
``test_chain_lint.py`` and ``test_stage_retry.py`` respectively. This file adds
(d): the end-to-end composite query — the exact one that failed in run 096 —
driven through the deep pipeline with mocked stage responses, asserting the
incident-class invariants now hold; plus a consolidated guard that the run-096
describer output can never ship.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from mmar_mage.chain_repair import lint_chain
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import CARLStepSchema, MAGEConfig

COMPOSITE_QUERY = "какая погода там где какой у меня ай пи страны и что мне стоит надеть?"

# What the FIXED pipeline should emit: complexity=medium / 4 atomic steps (T1/T2),
# one external call per step, real URLs — vs run 096 (complexity=simple / 2 steps,
# url="$outer_context").
_DOMAIN = {
    "domain": "general", "task_type": "qa", "complexity": "medium",
    "key_concepts": ["ip", "geolocation", "weather", "clothing"],
    "suggested_step_count": 4, "preferred_step_types": ["tool", "tool", "tool", "llm"],
}
_PLAN = {
    "steps": [
        {"number": 1, "title": "Get public IP", "step_type": "tool", "rough_aim": "Fetch the caller's public IP"},
        {"number": 2, "title": "Geolocate the IP", "step_type": "tool", "rough_aim": "Look up the location"},
        {"number": 3, "title": "Get the weather", "step_type": "tool", "rough_aim": "Fetch weather for the location"},
        {"number": 4, "title": "Recommend clothing", "step_type": "llm", "rough_aim": "Advise what to wear"},
    ],
    "rationale": "One atomic external call per step",
}
_DAG = {
    "steps": [
        {"number": 1, "title": "Get public IP", "step_type": "tool", "dependencies": [], "rough_aim": "Fetch IP"},
        {"number": 2, "title": "Geolocate the IP", "step_type": "tool", "dependencies": [1], "rough_aim": "Geolocate"},
        {"number": 3, "title": "Get the weather", "step_type": "tool", "dependencies": [2], "rough_aim": "Weather"},
        {"number": 4, "title": "Recommend clothing", "step_type": "llm", "dependencies": [3], "rough_aim": "Advise"},
    ],
    "max_workers": 1, "pattern": "pipeline",
}


def _tool(number: int, title: str, aim: str, url: str, dep: list[int], key: str) -> dict[str, Any]:
    return {
        "step_type": "tool", "number": number, "title": title, "aim": aim,
        "dependencies": dep, "tool_name": "http_request",
        "tool_input_mapping": {"url": url, "method": "GET"}, "tool_output_key": key,
        "reasoning_questions": "", "step_context_queries": [], "stage_action": "", "example_reasoning": "",
    }


_DESCRIBED = {
    "steps": [
        _tool(1, "Get public IP", "Fetch the caller's public IP", "https://api.ipify.org", [], "public_ip"),
        _tool(2, "Geolocate the IP", "Look up the location", "http://ip-api.com/json/", [1], "geo"),
        _tool(3, "Get the weather", "Fetch current weather", "https://wttr.in/?format=j1", [2], "weather"),
        {
            "step_type": "llm", "number": 4, "title": "Recommend clothing",
            "aim": "Advise what to wear for the weather", "dependencies": [3],
            "reasoning_questions": "What does the weather imply for clothing?",
            "step_context_queries": ["weather", "clothing"],
            "stage_action": "Give a concise recommendation", "example_reasoning": "Match clothing to conditions",
        },
    ],
}


def _resp(content: dict[str, Any]) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    response = MagicMock()
    response.choices = [choice]
    return response


async def test_composite_query_produces_valid_multistep_chain(mocker: Any) -> None:
    """(d) The run-096 composite query flows through the deep pipeline to a valid
    multi-step chain with no ``$outer_context`` in any url field."""
    instance = mocker.patch("mmar_mage.llm.AsyncOpenAI").return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[_resp(_DOMAIN), _resp(_PLAN), _resp(_DAG), _resp(_DESCRIBED)]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(
        config=MAGEConfig(mode="deep", model="gpt-4o-test", api_key="k", enable_memory=False, max_retries=0)
    )
    result = await gen.generate(COMPOSITE_QUERY)

    steps = result.chain_dict["steps"]
    # T2: a composite live query yields a multi-step plan, not run 096's 2-step chain
    assert len(steps) >= 4
    # T3/T5/T23: nothing ships that would GET https://<the question text>
    assert lint_chain(result.chain_dict) == []
    for step in steps:
        cfg = step.get("step_config") or {}
        mapping = cfg.get("input_mapping") or step.get("tool_input_mapping") or {}
        for field, value in mapping.items():
            if "url" in field.lower():
                assert value != "$outer_context"
                assert "http" in str(value)  # a real URL, not free text


def test_run096_bad_describer_output_cannot_ship() -> None:
    """(a)+(b) The exact run-096 describer output (url=$outer_context) is rejected
    by both the T5 lint and the T23 schema validator."""
    bad = _tool(1, "Fetch", "fetch the weather", "$outer_context", [], "x")
    assert lint_chain({"steps": [bad]})  # T5 lint flags it
    with pytest.raises(ValidationError):  # T23 schema rejects it
        CARLStepSchema.model_validate(bad)
