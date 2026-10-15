"""T3 — describer receives tool signatures; $outer_context only for text fields."""

from __future__ import annotations

from typing import Any

from mmar_mage.agents.step_describer import StepDescriber
from mmar_mage.llm import LLMClient
from mmar_mage.prompts import STEP_DESCRIBER_PROMPT, render_tool_signatures
from mmar_mage.schemas import DAGStep, DAGStructure, MAGEConfig

_TOOLS = [
    {"name": "http_request", "signature": "http_request(url: str, method: str = 'GET')", "description": "HTTP call"},
    {"name": "ipify", "input_schema": {"format": "str"}},
    {"name": ""},  # nameless → skipped
]
_STEP = {
    "step_type": "tool", "number": 1, "title": "t", "aim": "a", "reasoning_questions": "q",
    "step_context_queries": ["x"], "stage_action": "s", "example_reasoning": "e",
    "dependencies": [], "retry_max": 3, "llm_config": None, "tool_name": "http_request",
}


def test_render_tool_signatures() -> None:
    out = render_tool_signatures(_TOOLS)
    assert "http_request(url: str, method: str = 'GET')" in out
    assert "ipify(format)" in out  # synthesized from input_schema when no signature
    assert "url field needs a real URL or a $history" in out
    assert render_tool_signatures([]) == ""
    assert render_tool_signatures(None) == ""


def test_describer_prompt_forbids_outer_context_in_url() -> None:
    p = STEP_DESCRIBER_PROMPT
    assert "Placeholder discipline by FIELD TYPE" in p
    assert "$outer_context" in p
    assert "url-shaped field" in p


async def test_describe_injects_tool_signatures(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    captured: dict[str, str] = {}

    async def _fake(*, system_prompt: str, user_prompt: str, stage: str, **_kw: Any) -> dict[str, Any]:
        captured["sp"] = system_prompt
        return {"steps": [_STEP]}

    llm.generate_json = _fake  # type: ignore[method-assign]
    dag = DAGStructure(
        pattern="pipeline",
        max_workers=1,
        steps=[DAGStep(number=1, title="Fetch", step_type="tool", dependencies=[], rough_aim="fetch")],
    )
    out = await StepDescriber(llm).describe("q", dag, tools=_TOOLS)
    assert len(out) == 1
    assert "Available tool signatures" in captured["sp"]
    assert "http_request(url: str" in captured["sp"]
