"""Skill synthesis — plan item P1.17 Phase C.

Covers the SKILL.md manifest writer/parser (round-tripping through the real
``mmar_carl`` loader when installed) and the ``SkillSynthesizer`` (authoring,
unsafe-script dropping, loader validation) with a directly-mocked LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.skill_synthesizer import SkillSynthesizer, SynthesizedSkill
from mmar_mage.schemas import DomainAnalysis
from mmar_mage.skills.manifest import (
    SkillManifestData,
    parse_skill_md,
    render_skill_md,
    validate_with_carl,
    write_skill,
)

DOMAIN = DomainAnalysis(domain="data", task_type="transform", complexity="medium")

SKILL_NO_SCRIPT = {
    "name": "normalize_addresses",
    "description": "Normalize postal addresses to a canonical form.",
    "instructions": "# Normalize Addresses\n\n1. Parse the address.\n2. Canonicalize fields.\n3. Return JSON.",
    "allowed_tools": "Read Write",
    "script": None,
}
SAFE_SCRIPT = "def normalize(addr):\n    return addr.strip().lower()\n"
UNSAFE_SCRIPT = "import os\ndef wipe():\n    os.system('rm -rf /')\n"


# --------------------------------------------------------------------------- #
# manifest render / parse round-trip
# --------------------------------------------------------------------------- #
def test_render_parse_round_trip():
    m = SkillManifestData(
        name="csv_cleaner",
        description="Clean CSVs.",
        instructions="# CSV\n\nDo the thing.",
        allowed_tools="Read Write Bash(python:*)",
        metadata={"version": "1.0"},
    )
    back = parse_skill_md(render_skill_md(m))
    assert back.name == "csv_cleaner"
    assert back.description == "Clean CSVs."
    assert back.allowed_tools == "Read Write Bash(python:*)"
    assert back.metadata == {"version": "1.0"}
    assert "Do the thing." in back.instructions


def test_render_uses_hyphenated_allowed_tools_key():
    md = render_skill_md(SkillManifestData(name="x", description="d", allowed_tools="Read"))
    assert "allowed-tools: Read" in md  # AgentSkills convention


def test_parse_rejects_missing_frontmatter():
    import pytest

    with pytest.raises(ValueError):
        parse_skill_md("no frontmatter here")


def test_write_skill_and_carl_round_trip(tmp_path: Path):
    m = SkillManifestData(name="greeter", description="Greets.", instructions="# Greet\n\nSay hi.")
    skill_dir = write_skill(m, tmp_path, scripts={"run.py": "def run():\n    return 'hi'\n"})
    assert (skill_dir / "SKILL.md").exists()
    assert (skill_dir / "scripts" / "run.py").exists()
    # real mmar_carl loader accepts it (or skips cleanly if not installed)
    assert validate_with_carl(skill_dir) is None


# --------------------------------------------------------------------------- #
# SkillSynthesizer
# --------------------------------------------------------------------------- #
def _synth(return_value: Any, tmp: Path) -> SkillSynthesizer:
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value=return_value)
    return SkillSynthesizer(llm, skills_dir=tmp)


async def test_synthesize_writes_instructions_only_skill(tmp_path: Path):
    skill = await _synth(SKILL_NO_SCRIPT, tmp_path).synthesize("normalize addresses", DOMAIN)
    assert isinstance(skill, SynthesizedSkill)
    assert skill.name == "normalize_addresses"
    assert skill.has_script is False
    assert skill.uri.startswith("local://")
    # written + loader-valid
    assert (Path(skill.skill_dir) / "SKILL.md").exists()
    assert validate_with_carl(Path(skill.skill_dir)) is None


async def test_synthesize_includes_safe_script(tmp_path: Path):
    payload = {**SKILL_NO_SCRIPT, "name": "with_script", "script": {"filename": "norm.py", "source": SAFE_SCRIPT}}
    skill = await _synth(payload, tmp_path).synthesize("q", DOMAIN)
    assert skill is not None
    assert skill.has_script is True
    assert (Path(skill.skill_dir) / "scripts" / "norm.py").exists()


async def test_synthesize_drops_unsafe_script_but_keeps_skill(tmp_path: Path):
    payload = {**SKILL_NO_SCRIPT, "name": "danger", "script": {"filename": "wipe.py", "source": UNSAFE_SCRIPT}}
    skill = await _synth(payload, tmp_path).synthesize("q", DOMAIN)
    assert skill is not None  # skill still authored
    assert skill.has_script is False  # unsafe script dropped
    assert not (Path(skill.skill_dir) / "scripts" / "wipe.py").exists()


async def test_synthesize_no_name_returns_none(tmp_path: Path):
    skill = await _synth({"description": "x", "instructions": "y"}, tmp_path).synthesize("q", DOMAIN)
    assert skill is None


async def test_synthesize_llm_failure_returns_none(tmp_path: Path):
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=RuntimeError("down"))
    skill = await SkillSynthesizer(llm, skills_dir=tmp_path).synthesize("q", DOMAIN)
    assert skill is None
