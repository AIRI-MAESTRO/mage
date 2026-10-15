"""Orchestration primitives — plan item P2.2 (cache + MCP transports/headers slice).

Covers the two deterministic, schema-verified primitives landed for P2.2:
- ``cache: StepCache`` memoization (flat ``step_cache_ttl`` → top-level ``cache``;
  auto-marking of read-only MCP-resource steps),
- MCP ``http``/``sse`` transports + ``headers`` (auth) in the exported server config.

Real-execution checks validate the exported shapes against the real
``mmar_carl`` ``MCPServerConfig`` and round-trip a cached ``tool`` step through
``ReasoningChain.from_json``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mmar_mage.carl_export import (
    carl_mcp_transports,
    mark_cacheable_steps,
    to_carl_compat,
)


def _step(step_type: str, **fields: Any) -> dict[str, Any]:
    return {"number": 1, "step_type": step_type, "title": "S", "aim": "do", "dependencies": [], **fields}


def _chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"name": "c", "max_workers": 1, "steps": list(steps)}


# --------------------------------------------------------------------------- #
# cache: StepCache
# --------------------------------------------------------------------------- #
def test_flat_ttl_becomes_top_level_cache():
    out = to_carl_compat(_chain(_step("tool", tool_name="lookup", step_cache_ttl=600)))
    step = out["steps"][0]
    assert step["cache"] == {"ttl": 600}
    assert "step_cache_ttl" not in step  # flat field consumed


def test_no_ttl_no_cache():
    out = to_carl_compat(_chain(_step("tool", tool_name="lookup")))
    assert "cache" not in out["steps"][0]


def test_mark_cacheable_marks_mcp_resource_only():
    chain = _chain(
        _step("mcp_resource", mcp_server_name="fs", mcp_resource_uri="file://x"),
        {"number": 2, "step_type": "tool", "tool_name": "write", "dependencies": []},
    )
    mark_cacheable_steps(chain, default_ttl=120)
    assert chain["steps"][0]["step_cache_ttl"] == 120  # read-only resource → cached
    assert "step_cache_ttl" not in chain["steps"][1]  # tool (maybe side-effecting) → not auto-cached


def test_mark_cacheable_respects_existing_ttl():
    chain = _chain(_step("mcp_resource", mcp_server_name="fs", mcp_resource_uri="file://x", step_cache_ttl=999))
    mark_cacheable_steps(chain, default_ttl=120)
    assert chain["steps"][0]["step_cache_ttl"] == 999  # not overwritten


def test_cached_tool_round_trips_via_real_carl():
    pytest.importorskip("mmar_carl")
    from mmar_carl import ReasoningChain

    exported = to_carl_compat(_chain(_step("tool", tool_name="lookup", step_cache_ttl=300)))
    rc = ReasoningChain.from_json(json.dumps(exported))  # must not raise
    assert rc is not None


# --------------------------------------------------------------------------- #
# MCP transports + headers
# --------------------------------------------------------------------------- #
def test_mcp_http_transport_and_headers_exported():
    step = _step(
        "mcp",
        mcp_server_name="docs",
        mcp_server_transport="http",
        mcp_server_url="https://example.com/mcp",
        mcp_server_headers={"Authorization": "Bearer tok"},
        mcp_tool_name="search",
    )
    server = to_carl_compat(_chain(step))["steps"][0]["step_config"]["server"]
    assert server["transport"] == "http"
    assert server["url"] == "https://example.com/mcp"
    assert server["headers"] == {"Authorization": "Bearer tok"}


def test_mcp_server_config_loads_into_real_carl():
    pytest.importorskip("mmar_carl")
    from mmar_carl import MCPServerConfig

    step = _step(
        "mcp",
        mcp_server_name="docs",
        mcp_server_transport="sse",
        mcp_server_url="https://example.com/sse",
        mcp_server_headers={"X-API-Key": "k"},
        mcp_tool_name="q",
    )
    server = to_carl_compat(_chain(step))["steps"][0]["step_config"]["server"]
    loaded = MCPServerConfig.model_validate(server)
    # ``sse`` is an HTTP streaming transport: it loads natively on CARLs that
    # support it, otherwise it degrades to ``http`` so the server still loads.
    expected_transport = "sse" if "sse" in carl_mcp_transports() else "http"
    assert loaded.transport == expected_transport
    assert loaded.headers == {"X-API-Key": "k"}


def test_stdio_mcp_has_no_headers():
    step = _step(
        "mcp", mcp_server_name="local", mcp_server_transport="stdio", mcp_server_command="srv", mcp_tool_name="t"
    )
    server = to_carl_compat(_chain(step))["steps"][0]["step_config"]["server"]
    assert server["transport"] == "stdio"
    assert "headers" not in server  # none provided → not emitted


# --------------------------------------------------------------------------- #
# Real-execution scenario — generate() auto-caches read-only steps
# --------------------------------------------------------------------------- #
async def test_generator_auto_caches_mcp_resource():
    from unittest.mock import AsyncMock, MagicMock

    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    chain = _chain(
        {"number": 1, "step_type": "llm", "title": "A", "aim": "do", "dependencies": []},
        _step("mcp_resource", number=2, mcp_server_name="fs", mcp_resource_uri="file://x", dependencies=[1]),
    )
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", mode="fast", enable_step_caching=True))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps(chain))

    result = await gen.generate("read the config resource")
    resource_step = next(s for s in result.chain_dict["steps"] if s["step_type"] == "mcp_resource")
    assert resource_step.get("cache") == {"ttl": 300}
