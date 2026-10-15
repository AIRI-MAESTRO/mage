"""Reliable emission of new step types — plan item P2.1.

The describer-prompt corpus already lists every agent-features step type, the
``CARL_FIELD_MAP`` already nests them (P0.1), and ``test_carl_round_trip.py``
already round-trips them (P0.2). The remaining reliability gap this covers: a
typed step emitted WITHOUT its essential ``step_config`` fields exports to a
degraded no-op. ``missing_step_config_fields`` flags those deterministically.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.carl_export import (
    REQUIRED_STEP_CONFIG_FIELDS,
    missing_step_config_fields,
    to_carl_compat,
)


def _chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"name": "c", "steps": list(steps)}


def _step(n: int, step_type: str, **fields: Any) -> dict[str, Any]:
    return {"number": n, "title": f"S{n}", "step_type": step_type, "aim": "do", "dependencies": [], **fields}


# --------------------------------------------------------------------------- #
# missing_step_config_fields
# --------------------------------------------------------------------------- #
def test_llm_steps_never_flagged():
    assert missing_step_config_fields(_chain(_step(1, "llm"), _step(2, "transform"))) == {}


def test_evaluation_missing_both_fields_flagged():
    out = missing_step_config_fields(_chain(_step(1, "evaluation")))
    assert out == {1: ["evaluation_target_step", "evaluation_criteria"]}


def test_evaluation_complete_passes():
    step = _step(1, "evaluation", evaluation_target_step=1, evaluation_criteria=["min_words:5"])
    assert missing_step_config_fields(_chain(step)) == {}


def test_evaluation_partial_flags_only_missing():
    step = _step(1, "evaluation", evaluation_target_step=2)  # criteria missing
    assert missing_step_config_fields(_chain(step)) == {1: ["evaluation_criteria"]}


def test_parallel_sampling_requires_sampling_n():
    assert missing_step_config_fields(_chain(_step(1, "parallel_sampling"))) == {1: ["sampling_n"]}
    assert missing_step_config_fields(_chain(_step(1, "parallel_sampling", sampling_n=3))) == {}


def test_agent_skill_requires_skill_uri():
    assert missing_step_config_fields(_chain(_step(1, "agent_skill"))) == {1: ["skill"]}
    ok = _step(1, "agent_skill", skill="github://anthropics/skills/pdf@main")
    assert missing_step_config_fields(_chain(ok)) == {}


def test_mcp_resource_requires_server_and_uri():
    out = missing_step_config_fields(_chain(_step(1, "mcp_resource", mcp_server_name="fs")))
    assert out == {1: ["mcp_resource_uri"]}


def test_blank_values_count_as_missing():
    # empty string / empty list are "blank", not "present"
    step = _step(1, "evaluation", evaluation_target_step="", evaluation_criteria=[])
    assert missing_step_config_fields(_chain(step)) == {1: ["evaluation_target_step", "evaluation_criteria"]}


def test_field_present_in_nested_step_config_counts():
    step = _step(1, "agent_skill")
    step["step_config"] = {"skill": "local://x"}
    assert missing_step_config_fields(_chain(step)) == {}


def test_reports_per_step_number():
    out = missing_step_config_fields(
        _chain(
            _step(1, "llm"),
            _step(2, "parallel_sampling"),
            _step(3, "debate"),
        )
    )
    assert out == {2: ["sampling_n"], 3: ["debate_agents"]}


def test_required_map_keys_are_known_types():
    # every required-field type must be exportable (covered by to_carl_compat)
    for step_type in REQUIRED_STEP_CONFIG_FIELDS:
        chain = _chain(_step(1, step_type))
        # export must not raise even when fields are missing (degrades, not crashes)
        to_carl_compat(chain)


# --------------------------------------------------------------------------- #
# Real-execution scenario — the check runs inside generate()
# --------------------------------------------------------------------------- #
async def test_generator_warns_on_incomplete_typed_step(caplog: Any):
    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    # A fast-mode chain with an evaluation step missing its target/criteria.
    chain = _chain(_step(1, "llm"), _step(2, "evaluation", dependencies=[1]))
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", mode="fast"))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps(chain))

    with caplog.at_level(logging.WARNING):
        result = await gen.generate("evaluate the answer")

    # generation still succeeds (graceful) and the gap is logged (P2.1)
    assert result.chain_dict is not None
    assert any("missing required config" in r.message for r in caplog.records)
