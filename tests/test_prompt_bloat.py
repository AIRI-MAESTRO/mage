"""T15 — prompt-bloat audit: digest caps in all planner channels + bounded tool sigs."""

from __future__ import annotations

from typing import Any

from mmar_mage.agents.step_planner import StepPlanner, _truncate_digest
from mmar_mage.llm import LLMClient
from mmar_mage.prompts import render_tool_signatures
from mmar_mage.schemas import DomainAnalysis, MAGEConfig, PlannedStep, StepPlan


def test_truncate_digest_caps_to_max_chars() -> None:
    assert _truncate_digest("short", 1200) == "short"
    big = "lorem ipsum " * 500  # 6000 chars
    out = _truncate_digest(big, 1200)
    assert len(out) <= 1300  # cap + a small elision marker
    assert _truncate_digest(big, 0) == big  # 0 disables the cap


def test_render_tool_signatures_bounded() -> None:
    tools = [
        {"name": f"tool_{i}", "signature": f"tool_{i}(arg_a, arg_b, arg_c)", "description": "x" * 120}
        for i in range(60)
    ]
    out = render_tool_signatures(tools, max_chars=2400)
    assert len(out) <= 2400 + 250  # header + omission note add a little
    assert "more tool(s) available" in out  # the omission is noted
    assert len(render_tool_signatures(tools, max_chars=0)) > 2400  # unbounded keeps everything


def test_render_tool_signatures_small_chain_unchanged() -> None:
    out = render_tool_signatures([{"name": "http_request", "signature": "http_request(url, method)"}])
    assert "http_request(url, method)" in out
    assert "more tool(s)" not in out  # nothing omitted for a small set


async def test_planner_caps_all_digest_channels(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    captured: dict[str, str] = {}

    async def fake_gen(*args: Any, **kwargs: Any) -> StepPlan:
        strings = [a for a in args if isinstance(a, str)] + [v for v in kwargs.values() if isinstance(v, str)]
        captured["prompt"] = "\n".join(strings)
        return StepPlan(steps=[PlannedStep(number=1, title="x", step_type="llm", rough_aim="a")], rationale="r")

    mocker.patch.object(llm, "generate_structured", fake_gen)
    big = "lorem ipsum " * 500  # 6000 chars — far over max_digest_chars
    await StepPlanner(llm).plan(
        "q", DomainAnalysis(),
        memory_digest=big, web_digest=big, capability_context=big,
        template_skeleton=big, skill_digest=big, max_digest_chars=1200,
    )
    prompt = captured["prompt"]
    # If ANY of the five channels were uncapped, the full 6000-char block would appear.
    assert big not in prompt
    assert "lorem ipsum" in prompt  # the (truncated) digests are still present
