"""Tool-step generation guards (MAGE_TODO A1-A6).

Two layers:
1. Lock the prompt guidance that (A1) forbids putting a tool name in
   ``step_type``, (A2) shows a worked tool-step example, (A3/A4) routes
   live-data tasks to tools, and (A5) lists the CARL-loadable final types.
2. End-to-end (fast mode, mocked LLM): a generated ``tool`` step must
   validate + survive CARL export with its ``tool_name`` intact — a
   regression guard for the ``step_type: "web_search"`` glitch.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage import prompts
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


def test_prompts_carry_tool_step_guidance() -> None:
    # A1 — step_type is a literal; the tool's name goes in tool_name.
    for name in ("STEP_PLANNER_PROMPT", "STEP_DESCRIBER_PROMPT", "FAST_MODE_SYSTEM_PROMPT"):
        text = getattr(prompts, name)
        assert "tool_name" in text, name
        assert "NEVER a step_type" in text or "NEVER a `step_type`" in text or "never a step_type" in text.lower(), name
    # A2 — worked web_search tool example in describer + fast.
    assert "web_search" in prompts.STEP_DESCRIBER_PROMPT
    assert "web_search" in prompts.FAST_MODE_SYSTEM_PROMPT
    # A3 — live-data -> tool rule in the planner.
    assert "LIVE / EXTERNAL DATA" in prompts.STEP_PLANNER_PROMPT
    # A4 — domain analyzer prefers tool for live-data.
    assert "LIVE / EXTERNAL DATA" in prompts.DOMAIN_ANALYZER_PROMPT
    # A5 — CARL-loadable final types are spelled out.
    assert "CARL-loadable FINAL step types" in prompts.STEP_DESCRIBER_PROMPT


_TOOL_CHAIN: dict = {
    "name": "Live lookup",
    "max_workers": 1,
    "timeout": 300.0,
    "search_config": {
        "strategy": "substring",
        "substring_config": {"case_sensitive": False, "min_word_length": 3, "max_matches_per_query": 5},
    },
    "steps": [
        {
            "step_type": "tool",
            "number": 1,
            "title": "Search the web",
            "aim": "Fetch current info for the query",
            "tool_name": "web_search",
            "tool_input_mapping": {"query": "latest"},
            "tool_output_key": "results",
            "tool_timeout": 30,
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Answer from results",
            "aim": "Format a precise answer from the fetched results",
            "reasoning_questions": "Which fetched result is freshest and on-topic?",
            "step_context_queries": ["latest", "current"],
            "stage_action": "Summarize the freshest matching result",
            "example_reasoning": "Prefer the freshest dated source over prior assumptions",
            "dependencies": [1],
            "retry_max": 3,
            "llm_config": None,
        },
    ],
}


def _resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    r = MagicMock()
    r.choices = [choice]
    return r


@pytest.mark.asyncio
async def test_fast_mode_generates_valid_tool_step(config: MAGEConfig, mocker) -> None:  # noqa: ANN001
    """A planned tool step must validate + keep step_type 'tool' + tool_name
    (not collapse to the 'web_search'-as-step_type glitch)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    inst = mock_cls.return_value
    inst.chat.completions.create = AsyncMock(return_value=_resp(json.dumps(_TOOL_CHAIN)))
    inst.models.list = AsyncMock(return_value=MagicMock(data=[]))

    gen = MAGEGenerator(config=config)
    result = await gen.generate("What's the latest single by some artist?")

    chain = result.chain_dict
    s1 = chain["steps"][0]
    assert s1["step_type"] == "tool"  # literal "tool" — never the tool's name
    # tool_name survives (flat during gen, nested under step_config after export)
    tool_name = s1.get("tool_name") or (s1.get("step_config") or {}).get("tool_name")
    assert tool_name == "web_search"
