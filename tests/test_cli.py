"""Tests for the MAGE CLI (typer)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from mmar_mage.cli import app

from .conftest import SAMPLE_CARL_CHAIN, _make_openai_response

runner = CliRunner()


@pytest.fixture
def _mock_openai_cli(mocker: Any) -> MagicMock:
    """Mock AsyncOpenAI for CLI tests (fast mode)."""
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    return instance


class TestCLIVersion:
    """Tests for the 'version' command."""

    def test_version_output(self) -> None:
        from mmar_mage import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "mmar-mage" in result.stdout
        assert __version__ in result.stdout  # tracks the package version


class TestCLIGenerate:
    """Tests for the 'generate' command."""

    def test_generate_fast(self, _mock_openai_cli: MagicMock) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "Analyze a document",
                "--mode",
                "fast",
                "--api-key",
                "test-key",
                "--no-memory",
            ],
        )
        assert result.exit_code == 0
        assert "steps" in result.stdout.lower() or "CARL" in result.stdout or "Step" in result.stdout

    def test_generate_invalid_mode(self) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "Test",
                "--mode",
                "invalid",
                "--api-key",
                "test-key",
            ],
        )
        assert result.exit_code == 1

    def test_generate_with_output_file(self, _mock_openai_cli: MagicMock, tmp_path: Any) -> None:
        out = tmp_path / "chain.json"
        result = runner.invoke(
            app,
            [
                "generate",
                "Test query",
                "--mode",
                "fast",
                "--api-key",
                "test-key",
                "--no-memory",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0
        assert out.exists()
        chain = json.loads(out.read_text())
        assert "steps" in chain
