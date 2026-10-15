"""Few-shot examples for advanced step types — plan item P2.5.

The describer prompt had only type *definitions*, no concrete examples. We added
``ADVANCED_STEP_EXAMPLES`` (single source of truth) rendered into
``STEP_DESCRIBER_PROMPT``. These tests prove the examples are CORRECT — each
carries its required ``step_config`` fields (P2.1) and exports to a CARL-valid
step — so the few-shots can never teach the model a wrong field name.
"""

from __future__ import annotations

from typing import Any

import pytest

from mmar_mage.carl_export import missing_step_config_fields, to_carl_compat
from mmar_mage.generator import validate_carl_json
from mmar_mage.prompts import ADVANCED_STEP_EXAMPLES, STEP_DESCRIBER_PROMPT


def _ids(examples: list[dict[str, Any]]) -> list[str]:
    return [e["step_type"] for e in examples]


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_covers_priority_advanced_types():
    covered = set(_ids(ADVANCED_STEP_EXAMPLES))
    assert {"evaluation", "parallel_sampling", "agent_skill", "supervisor", "debate"} <= covered


def test_examples_rendered_into_describer_prompt():
    assert "Worked examples for advanced step types" in STEP_DESCRIBER_PROMPT
    # representative required fields from the examples appear verbatim
    for marker in ("evaluation_target_step", "sampling_n", "supervisor_routes", "debate_agents", '"skill"'):
        assert marker in STEP_DESCRIBER_PROMPT


# --------------------------------------------------------------------------- #
# Each example is a complete, valid step (the point of P2.5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("example", ADVANCED_STEP_EXAMPLES, ids=_ids(ADVANCED_STEP_EXAMPLES))
def test_example_has_all_required_config_fields(example: dict[str, Any]):
    # P2.1's reliability check must find nothing missing on a textbook example
    assert missing_step_config_fields({"steps": [example]}) == {}


@pytest.mark.parametrize("example", ADVANCED_STEP_EXAMPLES, ids=_ids(ADVANCED_STEP_EXAMPLES))
def test_example_exports_to_valid_carl(example: dict[str, Any]):
    # normalize deps so the single-step DAG is acyclic + self-contained
    step = {**example, "number": 1, "dependencies": []}
    exported = to_carl_compat({"name": "c", "max_workers": 1, "steps": [step]})
    validate_carl_json(exported, strict_carl=True)  # must not raise


# --------------------------------------------------------------------------- #
# Real-execution scenario — the describer call carries the examples
# --------------------------------------------------------------------------- #
async def test_describer_call_includes_examples():
    from unittest.mock import AsyncMock, MagicMock

    from mmar_mage.agents.step_describer import StepDescriber
    from mmar_mage.schemas import DAGStep, DAGStructure

    llm = MagicMock()
    # capture the system prompt the describer sends
    llm.generate_json = AsyncMock(return_value={"steps": []})
    llm.generate_structured = AsyncMock(return_value=MagicMock(steps=[]))

    dag = DAGStructure(
        steps=[DAGStep(number=1, title="A", step_type="llm", dependencies=[], rough_aim="do")],
        max_workers=1,
        pattern="pipeline",
    )
    describer = StepDescriber(llm)
    try:
        await describer.describe("q", dag, None)
    except Exception:
        pass  # we only care about the captured prompt

    calls = llm.generate_json.await_args_list + llm.generate_structured.await_args_list
    assert calls, "describer made no LLM call"
    system_prompts = [c.kwargs.get("system_prompt", "") for c in calls]
    assert any("Worked examples for advanced step types" in sp for sp in system_prompts)


def test_describer_requires_output_contract():
    """Every step must declare what it RETURNS and in what FORMAT — not just
    the activity. Locks the 'self-contained contract' guidance so a future
    edit can't quietly drop the per-step output spec."""
    p = STEP_DESCRIBER_PROMPT
    assert "SELF-CONTAINED CONTRACT" in p
    assert "what to RETURN and in what FORMAT" in p
    assert "output form" in p  # aim + stage_action both demand the output form
