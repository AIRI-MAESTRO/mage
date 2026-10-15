"""WP-9 — generated tools are registered + surfaced by the capability lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from mmar_mage.generated_tool_index import find_tool, load_tool_index, register_tool


def test_register_load_find(tmp_path: Path) -> None:
    register_tool(
        tmp_path,
        {
            "name": "sum_csv_column",
            "description": "sum the values in a CSV column",
            "signature": "sum_csv_column(csv: str, col: str) -> float",
            "path": "/x/sum.py",
        },
    )
    idx = load_tool_index(tmp_path)
    assert len(idx) == 1 and idx[0]["name"] == "sum_csv_column"
    assert find_tool(tmp_path, name="sum_csv_column") is not None
    found = find_tool(tmp_path, need="sum the values in a column")  # ≥2 content words overlap
    assert found is not None and found["name"] == "sum_csv_column"
    assert find_tool(tmp_path, name="other") is None
    assert find_tool(tmp_path, need="translate text into french") is None


def test_upsert_by_name(tmp_path: Path) -> None:
    register_tool(tmp_path, {"name": "t", "description": "first"})
    register_tool(tmp_path, {"name": "t", "description": "second"})
    idx = load_tool_index(tmp_path)
    assert len(idx) == 1 and idx[0]["description"] == "second"


def test_missing_index_is_empty(tmp_path: Path) -> None:
    assert load_tool_index(tmp_path) == []
    assert find_tool(tmp_path, name="x") is None


async def test_capability_lookup_surfaces_generated_tools(mocker: Any, tmp_path: Path) -> None:
    from mmar_mage.agents.capability_lookup_agent import CapabilityLookupAgent
    from mmar_mage.schemas import MAGEConfig

    register_tool(
        tmp_path,
        {"name": "sum_csv_column", "description": "sum csv column", "signature": "f()", "path": "/x.py"},
    )
    mocker.patch("mmar_mage.agents.capability_lookup_agent.DEFAULT_TOOLS_DIR", tmp_path)

    memory = mocker.MagicMock()
    memory.get_capabilities = AsyncMock(return_value=None)  # no memory registry
    agent = CapabilityLookupAgent(memory, MAGEConfig(api_key="test"))

    ctx = await agent.lookup(query="sum a csv column")
    assert "sum_csv_column" in [t.get("name") for t in ctx.tools]
    assert any(t.get("source") == "generated" for t in ctx.tools)


async def test_detect_need_skips_synth_when_registered(mocker: Any, tmp_path: Path) -> None:
    from mmar_mage.agents.tool_spec_generator import ToolSpecGenerator
    from mmar_mage.llm import LLMClient
    from mmar_mage.schemas import DomainAnalysis, MAGEConfig

    register_tool(
        tmp_path,
        {"name": "sum_csv_column", "description": "sum the values in a CSV column", "signature": "f()"},
    )
    mocker.patch("mmar_mage.agents.tool_spec_generator.DEFAULT_TOOLS_DIR", tmp_path)
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    llm = LLMClient(MAGEConfig(api_key="test"))
    # the LLM detector says a tool IS needed …
    llm.generate_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"needed": True, "description": "sum the values in a csv column", "rationale": "x"}
    )
    need = await ToolSpecGenerator(llm).detect_need("sum a csv column", DomainAnalysis(domain="data"))
    # … but a registered tool already covers it → no synthesis
    assert need.needed is False
    assert need.reused_existing is True


async def test_generated_tools_augment_memory_tools(mocker: Any, tmp_path: Path) -> None:
    """T29: generated tools are ADDED to the capability context alongside the
    memory-registry tools (not replacing them), carrying source/signature/path."""
    from mmar_mage.agents.capability_lookup_agent import CapabilityLookupAgent
    from mmar_mage.schemas import MAGEConfig

    register_tool(
        tmp_path,
        {"name": "geo_lookup", "description": "ip -> location", "signature": "geo(ip: str) -> dict", "path": "/g.py"},
    )
    mocker.patch("mmar_mage.agents.capability_lookup_agent.DEFAULT_TOOLS_DIR", tmp_path)

    memory = mocker.MagicMock()
    memory.get_capabilities = AsyncMock(
        return_value={"tools": [{"name": "http_request", "description": "http"}], "mcp_servers": []}
    )
    agent = CapabilityLookupAgent(memory, MAGEConfig(api_key="test"))

    ctx = await agent.lookup(query="geolocate an ip")
    names = [t.get("name") for t in ctx.tools]
    assert "http_request" in names  # the memory-registry tool is kept …
    gen = next(t for t in ctx.tools if t.get("name") == "geo_lookup")  # … and the generated one is added
    assert gen["source"] == "generated"  # tagged so the planner can tell them apart
    assert gen["signature"] == "geo(ip: str) -> dict"
    assert gen["path"] == "/g.py"
