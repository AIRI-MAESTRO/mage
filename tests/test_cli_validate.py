"""Tests for §8 P3 — `mage validate` CLI subcommand.

Covers:
- Valid chain JSON → exit 0 with green panel
- Missing file → exit 1
- Malformed JSON → exit 1
- Structurally invalid CARL chain → exit 1 with error in panel
- `--no-strict` skips the strict-CARL pass
- `--check-capabilities` against an in-process MemoryManager mock
- Soft-warning when an `agent_skill` URI isn't in the registry
- `--check-carl` and `_try_load_reasoning_chain` integration when
  `mmar_carl` is missing (the typical local-dev path)
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from mmar_mage.cli import _collect_agent_skill_uris, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Sample chains
# ---------------------------------------------------------------------------


SAMPLE_VALID_CHAIN: dict[str, Any] = {
    "name": "Valid demo",
    "max_workers": 2,
    "timeout": 300.0,
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Read",
            "aim": "Read the input",
            "reasoning_questions": "?",
            "step_context_queries": ["x"],
            "stage_action": "do",
            "example_reasoning": "y",
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Summarize",
            "aim": "Summarize the result",
            "reasoning_questions": "?",
            "step_context_queries": ["x"],
            "stage_action": "do",
            "example_reasoning": "y",
            "dependencies": [1],
            "retry_max": 3,
        },
    ],
}


SAMPLE_CHAIN_WITH_AGENT_SKILL: dict[str, Any] = {
    "name": "PDF demo",
    "max_workers": 1,
    "timeout": 300.0,
    "steps": [
        {
            "step_type": "agent_skill",
            "number": 1,
            "title": "Read PDF",
            "aim": "Use the pdf skill",
            "reasoning_questions": "?",
            "step_context_queries": ["x"],
            "stage_action": "do",
            "example_reasoning": "y",
            "skill_name": "pdf",
            "skill": "github://anthropics/skills/pdf",
            "dependencies": [],
            "retry_max": 3,
        },
    ],
}


# ---------------------------------------------------------------------------
# _collect_agent_skill_uris
# ---------------------------------------------------------------------------


class TestCollectAgentSkillUris:
    def test_flat_shape(self) -> None:
        uris = _collect_agent_skill_uris(SAMPLE_CHAIN_WITH_AGENT_SKILL)
        assert uris == {"github://anthropics/skills/pdf"}

    def test_nested_step_config_shape(self) -> None:
        chain = {
            "steps": [
                {
                    "step_type": "agent_skill",
                    "title": "x",
                    "step_config": {"skill": "github://x/y"},
                }
            ]
        }
        assert _collect_agent_skill_uris(chain) == {"github://x/y"}

    def test_skips_non_agent_skill_steps(self) -> None:
        assert _collect_agent_skill_uris(SAMPLE_VALID_CHAIN) == set()

    def test_empty_steps(self) -> None:
        assert _collect_agent_skill_uris({"steps": []}) == set()

    def test_missing_steps_key(self) -> None:
        assert _collect_agent_skill_uris({}) == set()


# ---------------------------------------------------------------------------
# `mage validate` happy paths
# ---------------------------------------------------------------------------


def _write_chain(tmp_path: Path, chain: dict[str, Any]) -> Path:
    p = tmp_path / "chain.json"
    p.write_text(json.dumps(chain), encoding="utf-8")
    return p


def test_validate_accepts_valid_chain(tmp_path: Path) -> None:
    p = _write_chain(tmp_path, SAMPLE_VALID_CHAIN)
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "validate_carl_json (internal types) passed" in result.output
    assert "validate_carl_json (strict CARL) passed" in result.output


def test_validate_no_strict_skips_strict_pass(tmp_path: Path) -> None:
    """A chain using a MAGE-internal step type (e.g. `eval`) fails
    strict but passes when --no-strict."""
    chain = {
        "name": "uses eval",
        "max_workers": 1,
        "timeout": 60.0,
        "steps": [
            {
                "step_type": "eval",
                "number": 1,
                "title": "Inline eval",
                "aim": "Score the chain",
                "reasoning_questions": "?",
                "step_context_queries": ["x"],
                "stage_action": "do",
                "example_reasoning": "y",
                "dependencies": [],
                "retry_max": 3,
            }
        ],
    }
    p = _write_chain(tmp_path, chain)

    # Strict (default) fails
    strict_result = runner.invoke(app, ["validate", str(p)])
    assert strict_result.exit_code == 1
    assert "FAIL" in strict_result.output

    # --no-strict passes
    loose_result = runner.invoke(app, ["validate", str(p), "--no-strict"])
    assert loose_result.exit_code == 0, loose_result.output


# ---------------------------------------------------------------------------
# `mage validate` error paths
# ---------------------------------------------------------------------------


def test_validate_missing_file_returns_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "does-not-exist.json")])
    assert result.exit_code == 1
    assert "File not found" in result.output


def test_validate_malformed_json_returns_nonzero(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 1
    assert "Not valid JSON" in result.output


def test_validate_invalid_chain_structure_returns_nonzero(tmp_path: Path) -> None:
    # Step 1 depends on non-existent step 99 → validate_carl_json raises
    bad = {
        "name": "bad",
        "steps": [
            {
                "step_type": "llm",
                "number": 1,
                "title": "Lonely step",
                "aim": "x",
                "dependencies": [99],
            }
        ],
    }
    p = _write_chain(tmp_path, bad)
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "non-existent step" in result.output


# ---------------------------------------------------------------------------
# --check-capabilities
# ---------------------------------------------------------------------------


class _FakeMemoryClient:
    """Records lookups against an in-process registry."""

    registered_skills: dict[str, dict[str, Any]] = {}

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _FakeMemoryClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def search(self, **_: Any) -> list[Any]:
        # Used by `MemoryManager.get_capabilities` — return an entity
        # with tools/mcp_servers/agent_skills set from
        # `registered_skills`.
        first = MagicMock()
        first.tools = []
        first.mcp_servers = []
        # NB: legacy `get_capabilities` doesn't read agent_skills; the
        # full capability surface comes via the v2 client path. We
        # leave agent_skills=[] here and rely on the `_install_fake_*`
        # helpers to populate the right paths per test.
        return [first]


def _install_fake_v2_client(monkeypatch: pytest.MonkeyPatch, *, available_uris: set[str]) -> None:
    """Install a fake `gigaevo_client.MemoryClient` that returns the
    given URIs from `find_capability_matches`."""

    class _FakeV2:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def __enter__(self) -> _FakeV2:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def find_capability_matches(self, rough_aim: str, top_k: int) -> list[dict[str, Any]]:
            return [
                {"name": f"skill-{i}", "uri": uri, "relevance": 0.9} for i, uri in enumerate(sorted(available_uris))
            ]

    pkg = types.ModuleType("gigaevo_client")
    pkg.MemoryClient = _FakeV2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_client", pkg)


def _install_fake_gigaevo_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in `gigaevo_memory.MemoryClient` so `MemoryManager._available` is True."""
    pkg = types.ModuleType("gigaevo_memory")
    pkg.MemoryClient = _FakeMemoryClient  # type: ignore[attr-defined]
    pkg.SearchType = MagicMock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_memory", pkg)


def test_validate_capability_check_reports_known_uris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every agent_skill URI is registered → green OK."""
    _install_fake_gigaevo_memory(monkeypatch)
    _install_fake_v2_client(
        monkeypatch,
        available_uris={"github://anthropics/skills/pdf"},
    )

    p = _write_chain(tmp_path, SAMPLE_CHAIN_WITH_AGENT_SKILL)
    result = runner.invoke(
        app,
        [
            "validate",
            str(p),
            "--check-capabilities",
            "--memory-base-url",
            "http://test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "every agent_skill URI exists" in result.output


def test_validate_capability_check_warns_on_missing_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown agent_skill URI → exit 0 with yellow warning."""
    _install_fake_gigaevo_memory(monkeypatch)
    _install_fake_v2_client(monkeypatch, available_uris=set())  # registry empty

    p = _write_chain(tmp_path, SAMPLE_CHAIN_WITH_AGENT_SKILL)
    result = runner.invoke(
        app,
        [
            "validate",
            str(p),
            "--check-capabilities",
            "--memory-base-url",
            "http://test",
        ],
    )
    # Warnings don't fail the exit code — the check is "soft" per the spec.
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "agent_skill URI not found" in result.output
    assert "github://anthropics/skills/pdf" in result.output


def test_validate_capability_check_noop_when_no_agent_skill_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No agent_skill steps → check is a no-op note."""
    _install_fake_gigaevo_memory(monkeypatch)

    p = _write_chain(tmp_path, SAMPLE_VALID_CHAIN)
    result = runner.invoke(app, ["validate", str(p), "--check-capabilities"])
    assert result.exit_code == 0
    assert "no agent_skill steps" in result.output


# ---------------------------------------------------------------------------
# --check-carl when mmar_carl is missing
# ---------------------------------------------------------------------------


def test_validate_check_carl_notes_when_mmar_carl_missing(tmp_path: Path) -> None:
    """In this dev env `mmar_carl` is not installed — the check is skipped
    (logged as a dim note) and exit code is still 0."""
    p = _write_chain(tmp_path, SAMPLE_VALID_CHAIN)
    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == 0
    assert "mmar_carl not installed" in result.output or "ReasoningChain" in result.output
