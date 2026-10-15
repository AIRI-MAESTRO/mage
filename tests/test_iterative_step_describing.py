"""Iterative step describing (P1.5).

When ``iterative=True`` the StepDescriber generates one step at a time, feeding
the fully-described prior steps into each step's prompt so step N can reference
step N-1's real output. asyncio_mode=auto → no @mark.asyncio needed.
"""

from __future__ import annotations

from typing import Any

from mmar_mage.agents.step_describer import StepDescriber
from mmar_mage.schemas import DAGStep, DAGStructure


class _FakeLLM:
    """Records each user_prompt and returns canned generate_json responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate_json(self, system_prompt: str = "", user_prompt: str = "", **_: Any) -> dict[str, Any]:
        self.prompts.append(user_prompt)
        return self._responses.pop(0)


def _two_step_dag() -> DAGStructure:
    return DAGStructure(
        steps=[
            DAGStep(number=1, title="A", rough_aim="produce X"),
            DAGStep(number=2, title="B", dependencies=[1], rough_aim="use X"),
        ]
    )


async def test_iterative_feeds_prior_step_into_next_prompt():
    fake = _FakeLLM(
        [
            {"steps": [{"step_type": "llm", "number": 1, "title": "A", "aim": "STEP1_AIM_MARKER"}]},
            {"steps": [{"step_type": "llm", "number": 2, "title": "B", "aim": "STEP2_AIM"}]},
        ]
    )
    steps = await StepDescriber(fake).describe("do X", _two_step_dag(), iterative=True)
    assert len(steps) == 2
    assert len(fake.prompts) == 2  # one LLM call per step
    # step 2's prompt SAW step 1's fully-described aim; step 1's did not
    assert "STEP1_AIM_MARKER" in fake.prompts[1]
    assert "STEP1_AIM_MARKER" not in fake.prompts[0]


async def test_iterative_skips_empty_response():
    fake = _FakeLLM(
        [
            {"steps": [{"step_type": "llm", "number": 1, "title": "A", "aim": "a"}]},
            {"steps": []},  # LLM returned nothing for step 2 → skipped, not crash
        ]
    )
    steps = await StepDescriber(fake).describe("q", _two_step_dag(), iterative=True)
    assert len(steps) == 1


async def test_batch_mode_is_a_single_call():
    fake = _FakeLLM(
        [
            {
                "steps": [
                    {"step_type": "llm", "number": 1, "title": "A", "aim": "a"},
                    {"step_type": "llm", "number": 2, "title": "B", "aim": "b"},
                ]
            }
        ]
    )
    steps = await StepDescriber(fake).describe("q", _two_step_dag(), iterative=False)
    assert len(steps) == 2
    assert len(fake.prompts) == 1  # batch = single LLM call (default behaviour)
