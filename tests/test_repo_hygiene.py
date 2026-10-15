"""T41 — repo hygiene: .gitignore keeps cruft out + `make package` makes a clean cut."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_gitignore_excludes_archive_cruft() -> None:
    gi = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in (".venv/", "__MACOSX/", "dist/", "__pycache__/", ".DS_Store"):
        assert pattern in gi, f".gitignore is missing {pattern}"


def test_makefile_has_clean_package_target() -> None:
    mk = Path("Makefile").read_text(encoding="utf-8")
    assert "\npackage:" in mk  # the target exists
    assert "package" in mk.splitlines()[0]  # declared in .PHONY
    assert "git archive" in mk  # built from git-tracked files only — clean by construction


def test_cruft_not_tracked() -> None:
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, OSError):
        pytest.skip("git not available")
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = out.stdout.splitlines()
    assert not any(line.startswith(".venv/") or "__MACOSX" in line for line in tracked)
