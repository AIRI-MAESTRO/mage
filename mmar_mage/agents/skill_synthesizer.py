"""SkillSynthesizer — author a full AgentSkill for a query (plan item P1.17 Phase C).

When neither the registry nor a synthesized tool covers a task, MAGE can author a
complete *skill*: a ``SKILL.md`` (instructions an agent follows) plus an optional
helper script, written to ``~/.mage/generated_skills/<name>/`` and validated by
round-tripping through ``mmar_carl``'s real loader. The chain integrates it via an
``agent_skill`` step whose source points at the local directory.

Any generated helper script is statically sandbox-scanned (no ``eval``/``exec``/
``subprocess``/network); a script that fails the scan is dropped and the skill is
still produced instructions-only. Gated behind ``enable_tool_synthesis`` upstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from ..llm import LLMClient
from ..prompts import SKILL_SYNTHESIZER_PROMPT
from ..schemas import DomainAnalysis
from ..security import scan_source
from ..skills.manifest import SkillManifestData, validate_with_carl, write_skill

logger = logging.getLogger(__name__)

STAGE = "skill_synthesis"

# User-config dir — NOT the repo. Generated skills are never committed.
DEFAULT_SKILLS_DIR = Path.home() / ".mage" / "generated_skills"


class SynthesizedSkill(BaseModel):
    """A written, loader-validated skill."""

    name: str
    description: str
    skill_dir: str
    uri: str  # local://<dir> — MAGE's local-skill URI convention
    has_script: bool = False


class SkillSynthesizer:
    """Author + validate a complete AgentSkill from a query.

    Parameters
    ----------
    llm : LLMClient
        Client for the single authoring call.
    skills_dir : Path | None
        Output directory (defaults to ``~/.mage/generated_skills``).
    """

    def __init__(self, llm: LLMClient, skills_dir: Path | None = None) -> None:
        self._llm = llm
        self._dir = skills_dir or DEFAULT_SKILLS_DIR

    async def synthesize(self, query: str, domain: DomainAnalysis) -> SynthesizedSkill | None:
        """Author a skill for ``query``; return ``None`` on failure (non-fatal).

        Writes ``<skills_dir>/<name>/SKILL.md`` (+ ``scripts/<file>`` when a safe
        helper script is produced) and validates the result through ``mmar_carl``'s
        loader when installed.
        """
        context = (
            f"User query: {query}\n\n"
            f"Domain: {domain.domain} (task_type={domain.task_type}, complexity={domain.complexity})"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=SKILL_SYNTHESIZER_PROMPT,
                user_prompt=context,
                stage=STAGE,
            )
        except Exception as exc:
            logger.warning("SkillSynthesizer: authoring call failed (non-fatal): %s", exc)
            return None

        name = str(data.get("name", "")).strip()
        instructions = str(data.get("instructions", "")).strip()
        if not name or not instructions:
            logger.info("SkillSynthesizer: model returned no usable skill")
            return None

        manifest = SkillManifestData(
            name=name,
            description=str(data.get("description", "")).strip() or name,
            instructions=instructions,
            allowed_tools=(str(data.get("allowed_tools", "")).strip() or None),
        )

        scripts = self._safe_script(data.get("script"))

        try:
            skill_dir = write_skill(manifest, self._dir, scripts=scripts)
        except Exception as exc:
            logger.warning("SkillSynthesizer: failed to write skill %r: %s", name, exc)
            return None

        err = validate_with_carl(skill_dir)
        if err is not None:
            logger.warning("SkillSynthesizer: skill %r failed loader validation: %s", name, err)
            return None

        logger.info("SkillSynthesizer: authored skill %r at %s (script=%s)", name, skill_dir, bool(scripts))
        return SynthesizedSkill(
            name=name,
            description=manifest.description,
            skill_dir=str(skill_dir),
            uri=f"local://{skill_dir}",
            has_script=bool(scripts),
        )

    @staticmethod
    def _safe_script(script: object) -> dict[str, str] | None:
        """Return ``{filename: source}`` for a script that passes the static scan, else ``None``.

        A proposed script with banned imports/calls is dropped (the skill is still
        authored instructions-only) — code is never written unscanned.
        """
        if not isinstance(script, dict):
            return None
        filename = str(script.get("filename", "")).strip()
        source = script.get("source")
        if not filename or not isinstance(source, str) or not source.strip():
            return None
        violations = scan_source(source)  # no declared deps → network/process banned
        if violations:
            logger.info(
                "SkillSynthesizer: dropping unsafe script %r: %s", filename, "; ".join(str(v) for v in violations)
            )
            return None
        if not filename.endswith(".py"):
            filename += ".py"
        return {filename: source}
