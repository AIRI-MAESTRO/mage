"""ToolSpecGenerator — synthesize a *proposed* tool spec for a query (plan item P1.17, Phase A).

Today ``CapabilityLookupAgent`` / ``SkillDiscoveryAgent`` return empty when nothing
in the registry matches, so a query needing a tool that doesn't exist (a custom
transform, a niche parser) silently falls back to generic ``llm`` steps. Phase A
closes the *detection + specification* half of the gap **without executing any
code**: it decides whether a new tool is warranted and, if so, produces a
``ToolSpec`` text artifact that is surfaced to the StepPlanner as a *proposed*
(synthesized-not-existing) capability.

Phase B (sandboxed implementation generation) and Phase C (full skill synthesis)
build on this; they are intentionally out of scope here — no code is generated or
run by this module.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..generated_tool_index import DEFAULT_TOOLS_DIR, find_tool
from ..llm import LLMClient
from ..prompts import TOOL_NEED_DETECTOR_PROMPT, TOOL_SPEC_PROMPT
from ..schemas import DomainAnalysis

logger = logging.getLogger(__name__)

STAGE = "tool_synthesis"


class ToolNeed(BaseModel):
    """Detector verdict: does the query need a tool that doesn't exist yet?"""

    needed: bool = False
    description: str = ""  # what the missing tool should do
    rationale: str = ""
    reused_existing: bool = False  # WP-9: a previously-generated tool already covers it


class ToolExample(BaseModel):
    """One input→output behaviour case for a synthesized tool (WP-8).

    ``comparison`` controls how ``expected_output`` is matched against the actual
    return value: exact equality, substring/membership (``in``), or just the
    return type's name.
    """

    input: dict[str, Any] = Field(default_factory=dict)
    expected_output: Any = None
    comparison: Literal["exact", "contains", "type_only"] = "exact"


class ToolSpec(BaseModel):
    """A proposed tool, as a text artifact (no implementation in Phase A).

    Parameters
    ----------
    name : str
        ``snake_case`` identifier.
    signature : str
        Python-style call signature, e.g. ``extract_tables(pdf_path: str) -> list[dict]``.
    docstring : str
        One-paragraph description of behavior.
    input_schema / output_schema : dict
        JSON-schema-ish descriptions of inputs and outputs.
    implementation_hint : str
        How it would be implemented (libraries, approach) — guidance for Phase B.
    dependencies : list[str]
        Third-party packages the implementation would need.
    """

    name: str
    signature: str = ""
    docstring: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    implementation_hint: str = ""
    dependencies: list[str] = Field(default_factory=list)
    # WP-8: 2–4 behaviour cases generated with the spec; the sandbox runs them as
    # unit tests so a tool that "doesn't crash" but returns garbage is rejected.
    examples: list[ToolExample] = Field(default_factory=list)


def format_tool_spec_for_planner(spec: ToolSpec) -> str:
    """Render a proposed tool as a planner hint, clearly marked as not-yet-existing.

    Empty string when ``spec`` is falsy, so callers can pass it through verbatim.
    """
    if spec is None or not spec.name:
        return ""
    deps = ", ".join(spec.dependencies) if spec.dependencies else "none"
    return (
        "A tool for this task does NOT exist yet. MAGE proposes SYNTHESIZING it "
        "(not available in any registry — treat as a design proposal):\n"
        f"- name: {spec.name}\n"
        f"- signature: {spec.signature}\n"
        f"- purpose: {spec.docstring}\n"
        f"- dependencies: {deps}\n"
        "If a step would use this tool, plan a `tool` step named after it and note "
        "in the step that the tool is synthesized. Otherwise, ignore this proposal "
        "and use generic reasoning steps."
    )


class ToolSpecGenerator:
    """Detect a missing-tool need and specify it (Phase A — text only).

    Parameters
    ----------
    llm : LLMClient
        Client for the (at most two) generation calls.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def detect_need(self, query: str, domain: DomainAnalysis) -> ToolNeed:
        """Decide whether the query needs a tool that generic steps can't cover.

        Never raises — on failure returns ``ToolNeed(needed=False)`` so the caller
        proceeds without a proposal.
        """
        import json

        context = (
            f"User query: {query}\n\nDomain analysis:\n{json.dumps(domain.model_dump(), ensure_ascii=False, indent=2)}"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=TOOL_NEED_DETECTOR_PROMPT,
                user_prompt=context,
                stage=STAGE,
            )
        except Exception as exc:
            logger.debug("ToolSpecGenerator.detect_need failed (non-fatal): %s", exc)
            return ToolNeed(needed=False)
        verdict = ToolNeed(
            needed=bool(data.get("needed", False)),
            description=str(data.get("description", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
        )
        # WP-9: skip synthesis when a previously-generated tool already covers the
        # need — the capability lookup surfaces it to the planner instead.
        if verdict.needed and find_tool(DEFAULT_TOOLS_DIR, need=f"{verdict.description} {query}") is not None:
            logger.info("ToolSpecGenerator: an existing generated tool covers the need; skipping synthesis")
            verdict.needed = False
            verdict.reused_existing = True
        return verdict

    async def generate_spec(self, need: ToolNeed, query: str, domain: DomainAnalysis) -> ToolSpec | None:
        """Fill a :class:`ToolSpec` from a detected need. ``None`` on failure / no name."""
        context = (
            f"User query: {query}\n\n"
            f"Tool need: {need.description}\n"
            f"Why it's needed: {need.rationale}\n"
            f"Domain: {domain.domain}"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=TOOL_SPEC_PROMPT,
                user_prompt=context,
                stage=STAGE,
            )
        except Exception as exc:
            logger.debug("ToolSpecGenerator.generate_spec failed (non-fatal): %s", exc)
            return None

        name = str(data.get("name", "")).strip()
        if not name:
            return None
        try:
            return ToolSpec(
                name=name,
                signature=str(data.get("signature", "")).strip(),
                docstring=str(data.get("docstring", "")).strip(),
                input_schema=dict(data.get("input_schema", {}) or {}),
                output_schema=dict(data.get("output_schema", {}) or {}),
                implementation_hint=str(data.get("implementation_hint", "")).strip(),
                dependencies=[str(d) for d in (data.get("dependencies", []) or [])],
            )
        except Exception as exc:
            logger.debug("ToolSpecGenerator: malformed spec dropped: %s", exc)
            return None

    async def synthesize(self, query: str, domain: DomainAnalysis) -> ToolSpec | None:
        """Detect → (if needed) specify. Returns ``None`` when no tool is warranted.

        The single entry point used by the generator. Never raises.
        """
        need = await self.detect_need(query, domain)
        if not need.needed or not need.description:
            logger.info("ToolSpecGenerator: no new tool needed for this query")
            return None
        spec = await self.generate_spec(need, query, domain)
        if spec is not None:
            logger.info("ToolSpecGenerator: proposed synthesized tool %r", spec.name)
        return spec
