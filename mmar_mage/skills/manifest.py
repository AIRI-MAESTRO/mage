"""SKILL.md manifest writer + parser (plan item P1.17 Phase C).

Produces and round-trips the AgentSkills ``SKILL.md`` format that
``mmar_carl``'s ``SkillLoader`` consumes (``models/agent_skill.py``'s
``SkillManifest``): YAML-ish frontmatter delimited by ``---`` lines, followed by a
markdown body (the ``instructions``). Frontmatter is parsed with the stdlib only
(matching the loader — keys at column 0, ``key: value``, hyphens normalized to
underscores); ``scripts``/``references``/``assets`` are discovered from subdirs,
not the frontmatter.

``validate_with_carl`` optionally re-parses a written skill through the real
``mmar_carl`` loader when it's installed — the strongest round-trip check.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Frontmatter keys promoted to first-class manifest fields (everything else →
# metadata), mirroring mmar_carl's SkillManifest.known_fields.
_KNOWN_FIELDS = ("name", "description", "license", "compatibility", "allowed_tools")
_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)")


class SkillManifestData(BaseModel):
    """MAGE-side mirror of ``mmar_carl``'s ``SkillManifest`` authoring surface."""

    name: str
    description: str
    instructions: str = ""
    allowed_tools: str | None = None  # raw "Bash(git:*) Read Write" string
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


def render_skill_md(manifest: SkillManifestData) -> str:
    """Render a :class:`SkillManifestData` to SKILL.md text (frontmatter + body)."""
    lines = ["---", f"name: {manifest.name}", f"description: {manifest.description}"]
    if manifest.license:
        lines.append(f"license: {manifest.license}")
    if manifest.compatibility:
        lines.append(f"compatibility: {manifest.compatibility}")
    if manifest.allowed_tools:
        # AgentSkills convention uses the hyphenated key; the loader normalizes it.
        lines.append(f"allowed-tools: {manifest.allowed_tools}")
    for key, value in manifest.metadata.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    lines.append(manifest.instructions.strip())
    return "\n".join(lines).rstrip() + "\n"


def parse_skill_md(text: str) -> SkillManifestData:
    """Parse SKILL.md text back into a :class:`SkillManifestData`.

    Mirrors ``mmar_carl``'s stdlib frontmatter parser (keys at column 0,
    hyphens→underscores, lowercased; unknown keys → ``metadata``).

    Raises
    ------
    ValueError
        If the frontmatter delimiters are missing.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must start with a '---' frontmatter delimiter")
    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        raise ValueError("SKILL.md is missing the closing '---' frontmatter delimiter")

    frontmatter: dict[str, str] = {}
    current_key: str | None = None
    current_val: list[str] = []

    def _flush() -> None:
        nonlocal current_key, current_val
        if current_key is not None:
            frontmatter[current_key] = "\n".join(current_val).strip()
        current_key, current_val = None, []

    for line in lines[1:closing]:
        m = _KEY_RE.match(line)
        if m:
            _flush()
            current_key = m.group(1).lower().replace("-", "_")
            current_val = [m.group(2)]
        elif (line.startswith("  ") or line.startswith("\t")) and current_key is not None:
            current_val.append(line.strip())
    _flush()

    instructions = "\n".join(lines[closing + 1 :]).strip()
    metadata = {k: v for k, v in frontmatter.items() if k not in _KNOWN_FIELDS}

    return SkillManifestData(
        name=frontmatter.get("name", ""),
        description=frontmatter.get("description", ""),
        instructions=instructions,
        allowed_tools=frontmatter.get("allowed_tools"),
        license=frontmatter.get("license"),
        compatibility=frontmatter.get("compatibility"),
        metadata=metadata,
    )


def write_skill(manifest: SkillManifestData, base_dir: Path, scripts: dict[str, str] | None = None) -> Path:
    """Write a skill directory: ``<base_dir>/<name>/SKILL.md`` (+ ``scripts/*``).

    Returns the skill directory path. ``scripts`` maps a filename (e.g.
    ``"run.py"``) to its source; each is written under ``scripts/`` so the loader
    discovers it.
    """
    skill_dir = base_dir / manifest.name
    (skill_dir).mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(render_skill_md(manifest), encoding="utf-8")
    if scripts:
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for fname, source in scripts.items():
            (scripts_dir / fname).write_text(source, encoding="utf-8")
    return skill_dir


def validate_with_carl(skill_dir: Path) -> str | None:
    """Round-trip a written skill through the real ``mmar_carl`` loader, if installed.

    Returns ``None`` when the skill loads cleanly (or when ``mmar_carl`` is absent —
    a skipped check, not a failure), else an error string.
    """
    try:
        from mmar_carl.models.agent_skill import AgentSkillSource
        from mmar_carl.skill_loader import SkillLoader
    except ImportError:
        logger.debug("validate_with_carl: mmar_carl not installed — skipping")
        return None
    try:
        manifest = SkillLoader().load_sync(AgentSkillSource(path=str(skill_dir)))
        if not manifest.name or not manifest.instructions:
            return "loaded manifest missing name/instructions"
        return None
    except Exception as exc:
        return f"mmar_carl rejected the skill: {exc}"
