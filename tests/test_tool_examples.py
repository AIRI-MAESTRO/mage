"""WP-8 — generated-tool examples run as sandbox unit tests + drive the repair loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.agents.tool_implementation_generator import ToolImplementationGenerator
from mmar_mage.agents.tool_spec_generator import ToolExample, ToolSpec
from mmar_mage.generated_tool_index import load_tool_index
from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig
from mmar_mage.security import validate_tool_source

_EXAMPLES = [{"input": {"a": 2, "b": 3}, "expected_output": 5, "comparison": "exact"}]
_SCHEMA = {"a": "int", "b": "int"}


def test_validate_rejects_wrong_output_via_examples() -> None:
    report = validate_tool_source(
        "def add(a, b):\n    return 0\n", "add", allowed_imports=set(), input_schema=_SCHEMA, examples=_EXAMPLES
    )
    assert report.ok is False  # smoke run passes but the example fails
    assert "Example cases failed" in report.feedback()


def test_validate_accepts_correct_output() -> None:
    report = validate_tool_source(
        "def add(a, b):\n    return a + b\n", "add", allowed_imports=set(), input_schema=_SCHEMA, examples=_EXAMPLES
    )
    assert report.ok is True


def test_no_examples_keeps_smoke_only_behaviour() -> None:
    # without examples, a non-crashing tool still validates (back-compat)
    report = validate_tool_source(
        "def add(a, b):\n    return 0\n", "add", allowed_imports=set(), input_schema=_SCHEMA
    )
    assert report.ok is True


async def test_repair_loop_fixes_wrong_impl(mocker: Any, tmp_path: Path) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    wrong = "def add(a, b):\n    return 0\n"
    right = "def add(a, b):\n    return a + b\n"
    llm.generate = AsyncMock(side_effect=[wrong, right])  # type: ignore[method-assign]

    spec = ToolSpec(
        name="add",
        signature="add(a: int, b: int) -> int",
        input_schema=_SCHEMA,
        examples=[ToolExample(input={"a": 2, "b": 3}, expected_output=5, comparison="exact")],
    )
    gen = ToolImplementationGenerator(llm, tools_dir=tmp_path, max_attempts=3)
    impl = await gen.generate(spec)

    assert impl is not None
    assert impl.attempts == 2  # wrong impl rejected by examples, correct one on retry
    assert "a + b" in impl.source
    assert list(tmp_path.glob("add_*_tests.json"))  # regression set persisted


async def test_persistently_wrong_impl_is_never_registered(mocker: Any, tmp_path: Path) -> None:
    """When every attempt fails its example cases, generate() gives up and ships
    NOTHING — no module, no regression set, no index entry (never register raw)."""
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    llm.generate = AsyncMock(return_value="def add(a, b):\n    return 0\n")  # type: ignore[method-assign]

    spec = ToolSpec(
        name="add",
        signature="add(a: int, b: int) -> int",
        input_schema=_SCHEMA,
        examples=[ToolExample(input={"a": 2, "b": 3}, expected_output=5, comparison="exact")],
    )
    impl = await ToolImplementationGenerator(llm, tools_dir=tmp_path, max_attempts=2).generate(spec)

    assert impl is None  # gave up after max_attempts
    assert not list(tmp_path.glob("add_*.py"))  # no module written
    assert not list(tmp_path.glob("add_*_tests.json"))  # no regression set
    assert load_tool_index(tmp_path) == []  # not registered
