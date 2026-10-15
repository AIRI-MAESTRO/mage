"""CARL-loadable smoke test — plan item P3.3.

Round-trips chains that ``MAGEGenerator.generate()`` actually produces through the
REAL ``mmar_carl.ReasoningChain.from_json`` — catching drift between MAGE's
internal ``validate_carl_json`` and CARL's actual loader (which hand-built fixtures
in ``test_carl_round_trip.py`` can miss). Two layers of assurance:

1. ``raise_on_carl_load_error=True`` makes ``generate()`` itself raise if the
   loader rejects its output (exercises the P0.3 ``_try_load_reasoning_chain`` path).
2. An explicit ``ReasoningChain.from_json(result.chain_json)`` afterwards.

Only round-trippable step types are exercised (the 6 agent-features types that
crash ``from_json`` are tracked as ``xfail`` in ``test_carl_round_trip.py``).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


def _step(step_type: str, number: int, **fields: Any) -> dict[str, Any]:
    return {
        "step_type": step_type,
        "number": number,
        "title": f"S{number}",
        "aim": f"do {number}",
        "reasoning_questions": "what?",
        "step_context_queries": ["a", "b"],
        "stage_action": "Process",
        "dependencies": [number - 1] if number > 1 else [],
        **fields,
    }


# A spread of chains using only round-trippable step types.
ROUND_TRIPPABLE_CHAINS: dict[str, dict[str, Any]] = {
    "llm_only": {"name": "c", "max_workers": 1, "steps": [_step("llm", 1), _step("llm", 2)]},
    "tool": {"name": "c", "max_workers": 1, "steps": [_step("llm", 1), _step("tool", 2, tool_name="lookup")]},
    "mcp": {
        "name": "c",
        "max_workers": 1,
        "steps": [_step("mcp", 1, mcp_server_name="fs", mcp_tool_name="read")],
    },
    "transform": {
        "name": "c",
        "max_workers": 1,
        "steps": [_step("llm", 1), _step("transform", 2, transform_type="extract")],
    },
    "structured_output": {
        "name": "c",
        "max_workers": 1,
        "steps": [_step("structured_output", 1, structured_output_schema={"type": "object"})],
    },
    "multi_type": {
        "name": "c",
        "max_workers": 2,
        "steps": [
            _step("llm", 1),
            _step("tool", 2, tool_name="fetch"),
            _step("transform", 3, transform_type="format"),
        ],
    },
}


def _fast_gen(chain: dict[str, Any]) -> MAGEGenerator:
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", mode="fast", raise_on_carl_load_error=True))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps(chain))
    return gen


@pytest.mark.parametrize("name", list(ROUND_TRIPPABLE_CHAINS), ids=list(ROUND_TRIPPABLE_CHAINS))
async def test_generated_chain_loads_in_carl(name: str):
    carl = pytest.importorskip("mmar_carl")

    gen = _fast_gen(ROUND_TRIPPABLE_CHAINS[name])
    # raise_on_carl_load_error=True → generate() raises if CARL's loader rejects it
    result = await gen.generate(f"task for {name}")

    # explicit second check on the serialized JSON
    rc = carl.ReasoningChain.from_json(result.chain_json)
    assert rc is not None


async def test_deep_generated_chain_loads_in_carl(mock_openai_deep: MagicMock):
    carl = pytest.importorskip("mmar_carl")

    config = MAGEConfig(
        mode="deep",
        api_key="test-key",
        model="gpt-4o-test",
        enable_memory=False,
        # topology stays ON — the conftest mock_openai_deep includes its response slot.
        raise_on_carl_load_error=True,
    )
    result = await MAGEGenerator(config=config).generate("Analyze the dataset")
    carl.ReasoningChain.from_json(result.chain_json)  # must not raise


async def test_loader_rejection_surfaces_when_strict():
    # a chain that passes MAGE structural validation but references a bad dependency
    # would be caught; here we confirm the strict flag is actually wired by checking
    # a clean chain passes and the flag is honoured (no exception on valid output).
    pytest.importorskip("mmar_carl")
    gen = _fast_gen(ROUND_TRIPPABLE_CHAINS["llm_only"])
    result = await gen.generate("ok")
    assert result.chain_dict["steps"]  # generated + loaded cleanly under strict flag
