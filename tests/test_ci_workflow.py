"""T37 — the CI workflow runs the quality gates on a python matrix + a carl-roundtrip job."""

from __future__ import annotations

from pathlib import Path

_CI = Path(".github/workflows/ci.yml")


def test_ci_runs_quality_gates_on_pr() -> None:
    ci = _CI.read_text(encoding="utf-8")
    assert "pull_request" in ci  # runs on PRs
    assert "ruff check" in ci and "mypy" in ci and "pytest" in ci  # lint + typecheck + test


def test_ci_uses_pyproject_python_matrix() -> None:
    ci = _CI.read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in pyproject
    assert '"3.12"' in ci and '"3.13"' in ci  # matrix covers the supported pythons


def test_ci_has_nonblocking_carl_roundtrip_job() -> None:
    ci = _CI.read_text(encoding="utf-8")
    assert "carl-roundtrip" in ci
    assert "test_code_generator_roundtrip.py" in ci  # the round-trip suite
    assert "continue-on-error: true" in ci  # mmar-carl optional → must not block the merge
    assert "-rxXs" in ci  # the xfail / skip report
