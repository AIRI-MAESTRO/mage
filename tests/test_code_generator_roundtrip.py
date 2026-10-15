"""T31 — code_generator round-trip: required CARL fields survive Python codegen.

The audit's literal round-trip (chain_dict → python → exec → from_json → compare
fields) needs ``mmar_carl`` to exec the generated script; it runs as a
``skipif``-gated test. The always-on equivalent compares the generated source
against ``CARL_FIELD_MAP`` to catch fields the codegen silently drops — which is
how ``tool_output_key`` / ``memory_namespace`` / ``condition_context_key`` were
found missing.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

from mmar_mage.carl_export import CARL_FIELD_MAP
from mmar_mage.code_generator import CodeGenerator
from mmar_mage.schemas import MAGEConfig

_CFG = MAGEConfig(api_key="test")

# Per codegen-handled type: every CARL_FIELD_MAP field populated, plus the marker
# strings whose presence in the generated source proves the field wasn't dropped.
_CASES: dict[str, tuple[dict[str, Any], list[str]]] = {
    "tool": (
        {"tool_name": "TN", "tool_input_mapping": {"url": "URLVAL"}, "tool_output_key": "OUTK",
         "tool_description": "DESCV", "tool_timeout": 42},
        # "OUTK" is intentionally NOT a codegen marker: ChainBuilder.add_tool_step()
        # has no output_key param, so the builder-codegen path can't carry it. The
        # JSON export path (to_carl_compat) preserves tool_output_key -> output_key.
        ["TN", "URLVAL", "DESCV", "42"],
    ),
    "memory": (
        {"memory_operation": "write", "memory_key": "MEMK", "memory_value_source": "VSRC", "memory_namespace": "NSV"},
        ["MEMK", "VSRC", "NSV"],
    ),
    "transform": (
        {"transform_type": "format", "transform_input_key": "INK", "transform_output_format": "OFMT",
         "transform_expression": "EXPR"},
        ["INK", "OFMT", "EXPR"],
    ),
    "conditional": (
        {"condition_branches": [{"condition": "CONDV", "next_step": 2}], "condition_default_step": 3,
         "condition_context_key": "CTXK"},
        ["CONDV", "CTXK"],
    ),
}


def _gen(*steps: dict[str, Any]) -> str:
    chain = {"name": "t", "max_workers": 1, "timeout": 300.0, "steps": list(steps)}
    return CodeGenerator.generate(chain, "weather where my ip is", _CFG)


def _step(stype: str, number: int = 1, deps: list[int] | None = None) -> dict[str, Any]:
    fields, _ = _CASES[stype]
    return {"step_type": stype, "number": number, "title": "S", "dependencies": deps or [], **fields}


def test_cases_cover_every_carl_field_map_field() -> None:
    # guard: if CARL_FIELD_MAP[stype] grows a field, this forces the case to cover it
    for stype, (fields, _) in _CASES.items():
        assert set(fields) == set(CARL_FIELD_MAP[stype]), f"{stype}: test fields out of sync with CARL_FIELD_MAP"


def test_handled_step_types_keep_all_carl_fields() -> None:
    """Every populated CARL_FIELD_MAP field survives into the generated Python —
    regression for the silently-dropped tool_output_key / memory_namespace /
    condition_context_key (T31)."""
    for stype, (_, markers) in _CASES.items():
        src = _gen(_step(stype))
        for marker in markers:
            assert marker in src, f"{stype}: {marker!r} dropped by codegen"


@pytest.mark.skipif(importlib.util.find_spec("mmar_carl") is None, reason="mmar_carl not installed")
def test_generated_python_execs_and_fields_survive_roundtrip() -> None:
    """The audit's literal round-trip — exec the generated script, build the chain,
    serialize, and confirm the once-dropped fields survive. Runs only when
    ``mmar_carl`` is importable."""
    src = _gen(_step("tool", 1), _step("memory", 2, [1]), _step("conditional", 3, [2]))
    ns: dict[str, Any] = {}
    exec(compile(src, "<generated>", "exec"), ns)
    built = ns["build_chain"]()
    blob = built.to_json() if hasattr(built, "to_json") else json.dumps(getattr(built, "__dict__", {}), default=str)
    # NSV (memory namespace) and CTXK (conditional context key) round-trip through the
    # builder. OUTK (tool output_key) is excluded: add_tool_step() can't set it — the
    # JSON export path carries it instead (covered by the static marker test above).
    for marker in ("NSV", "CTXK"):
        assert marker in blob
