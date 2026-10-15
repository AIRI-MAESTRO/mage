"""Curated catalog of known public AgentSkills.

This is a deliberately minimal registry — we ship the best-known public
skills (Anthropic's official set + a few community ones) so cold-start
generation has something concrete to suggest without a web round-trip.

The registry is data-only: entries here describe WHAT a skill does and
WHERE to fetch it via URI. The actual fetching/loading is the runtime's
job (handled by ``mmar_carl.skill_resolver`` at chain execution time).

Adding a skill:
    KnownSkill(
        name="my-skill",
        description="One-line summary of what it does.",
        capabilities=["task1", "task2"],
        uri="github://owner/repo/path/to/skill@main",
        domains=["domain1"],
        execution_modes_supported=["llm_agent", "script"],
    )

The default registry can be augmented at runtime via
``SkillRegistry.add(...)`` or by passing a custom :class:`SkillRegistry`
into :class:`SkillDiscoveryAgent`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KnownSkill(BaseModel):
    """A single entry in the skill catalog.

    Designed to be small but informative enough for an LLM to rank by
    relevance to a user query without fetching the skill body.
    """

    name: str
    """Human-readable name, e.g. 'pdf', 'docx', 'pptx'."""

    description: str
    """One-sentence description (shown to the matching LLM)."""

    capabilities: list[str] = Field(default_factory=list)
    """Bullet list of what tasks this skill handles."""

    uri: str
    """Resolvable URI — github://, https://, module://, or local://."""

    domains: list[str] = Field(default_factory=list)
    """Domain hints used for fast pre-filtering, e.g. ['documents', 'web']."""

    execution_modes_supported: list[Literal["llm", "script", "hybrid", "subagent", "llm_agent"]] = Field(
        default_factory=list
    )
    """Which AgentSkillExecutionMode values this skill works with."""

    requires_pip: list[str] = Field(default_factory=list)
    """Extra Python packages the skill needs (mapped to skill_extra_pip)."""

    source: Literal["builtin", "user", "discovered"] = "builtin"
    """Where this entry came from — builtin (registry), user (added at runtime),
    or discovered (returned by a web search and cached)."""


# ---------------------------------------------------------------------------
# Built-in catalog
# ---------------------------------------------------------------------------
#
# Anthropic's official skills repository (https://github.com/anthropics/skills)
# is the canonical seed. We mirror the top-level set here.

_BUILTIN_SKILLS: list[KnownSkill] = [
    # ---- Anthropic official skills ----
    KnownSkill(
        name="pdf",
        description="Extract, manipulate, and produce PDF files: text extraction, "
        "table extraction, form filling, splitting, merging, OCR.",
        capabilities=[
            "extract text from PDF",
            "extract tables from PDF",
            "fill PDF forms",
            "merge / split PDFs",
            "convert PDF to images / markdown",
            "OCR scanned PDFs",
        ],
        uri="github://anthropics/skills/skills/pdf@main",
        domains=["documents", "extraction"],
        execution_modes_supported=["llm_agent", "script"],
        requires_pip=["pdfplumber", "pypdf"],
    ),
    KnownSkill(
        name="docx",
        description="Read, edit, and create Word documents: paragraphs, tables, "
        "headings, page numbers, find-and-replace, track changes.",
        capabilities=[
            "read DOCX content",
            "create Word documents from scratch",
            "edit existing DOCX files",
            "tables of contents and headings",
            "track changes and comments",
        ],
        uri="github://anthropics/skills/skills/docx@main",
        domains=["documents"],
        execution_modes_supported=["llm_agent"],
        requires_pip=["python-docx"],
    ),
    KnownSkill(
        name="pptx",
        description="Create, parse, and edit PowerPoint presentations: slides, layouts, speaker notes, charts.",
        capabilities=[
            "create slide decks",
            "extract text from PPTX",
            "edit existing presentations",
            "speaker notes",
            "templates and layouts",
        ],
        uri="github://anthropics/skills/skills/pptx@main",
        domains=["documents", "presentations"],
        execution_modes_supported=["llm_agent"],
        requires_pip=["python-pptx"],
    ),
    KnownSkill(
        name="xlsx",
        description="Open, edit, and create spreadsheets (.xlsx, .xlsm, .csv, .tsv): "
        "formulas, charts, formatting, cleaning messy tabular data.",
        capabilities=[
            "read XLSX/CSV/TSV",
            "compute formulas",
            "format and chart spreadsheets",
            "clean messy tabular data",
            "convert between tabular formats",
        ],
        uri="github://anthropics/skills/skills/xlsx@main",
        domains=["documents", "data"],
        execution_modes_supported=["llm_agent"],
        requires_pip=["openpyxl"],
    ),
    KnownSkill(
        name="skill-creator",
        description="Author and improve AgentSkills themselves: scaffolding new "
        "skills from scratch, optimizing descriptions, running evals.",
        capabilities=[
            "create a new skill from scratch",
            "edit existing skills",
            "benchmark skill performance",
            "optimize a skill's description for triggering accuracy",
        ],
        uri="github://anthropics/skills/skills/skill-creator@main",
        domains=["meta", "tooling"],
        execution_modes_supported=["llm_agent"],
    ),
    KnownSkill(
        name="mcp-builder",
        description="Build and edit Model Context Protocol (MCP) servers.",
        capabilities=[
            "scaffold a new MCP server",
            "add tools/resources to an MCP server",
            "edit existing MCP servers",
        ],
        uri="github://anthropics/skills/skills/mcp-builder@main",
        domains=["mcp", "tooling"],
        execution_modes_supported=["llm_agent"],
    ),
    KnownSkill(
        name="webdev",
        description="Web frontend tasks: editing HTML/CSS/JS, debugging, refactoring.",
        capabilities=[
            "edit HTML/CSS/JS",
            "frontend debugging",
            "component refactors",
        ],
        uri="github://anthropics/skills/webdev@main",
        domains=["code", "frontend"],
        execution_modes_supported=["llm_agent"],
    ),
    KnownSkill(
        name="canva",
        description="Generate and edit design assets via Canva's API.",
        capabilities=["create designs", "edit designs", "export designs"],
        uri="github://anthropics/skills/canva@main",
        domains=["design", "creative"],
        execution_modes_supported=["llm_agent"],
    ),
    # ---- Community skills (commonly cited) ----
    KnownSkill(
        name="brave-search",
        description="Web search using the Brave Search API.",
        capabilities=["web search", "news search", "site-restricted search"],
        uri="github://badlogic/pi-skills/brave-search@HEAD",
        domains=["web", "search"],
        execution_modes_supported=["script", "llm_agent"],
    ),
]


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class SkillRegistry(BaseModel):
    """Mutable in-memory catalog of skills.

    Wraps a list of :class:`KnownSkill` with simple search helpers.
    """

    skills: list[KnownSkill] = Field(default_factory=list)

    def add(self, skill: KnownSkill) -> None:
        """Add or replace a skill by name (last write wins)."""
        self.skills = [s for s in self.skills if s.name != skill.name]
        self.skills.append(skill)

    def get(self, name: str) -> KnownSkill | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def by_domain(self, domain: str) -> list[KnownSkill]:
        return [s for s in self.skills if domain in s.domains]

    def by_uri_prefix(self, prefix: str) -> list[KnownSkill]:
        return [s for s in self.skills if s.uri.startswith(prefix)]

    def search_substring(self, query: str, top_k: int = 10) -> list[KnownSkill]:
        """Cheap deterministic search — substring match against name + capabilities + description.

        Returns up to ``top_k`` matches sorted by a coarse relevance score
        (sum of substring hits, with stronger weight on name matches).
        Useful as a pre-filter before the LLM reranks.
        """
        q = query.lower().strip()
        if not q:
            return list(self.skills[:top_k])

        # Tokenise the query — multi-word queries match any token.
        tokens = [t for t in q.split() if len(t) >= 3]
        if not tokens:
            tokens = [q]

        scored: list[tuple[int, KnownSkill]] = []
        for s in self.skills:
            hay = " ".join([s.name.lower(), s.description.lower(), " ".join(s.capabilities).lower()])
            score = 0
            for tok in tokens:
                if tok in s.name.lower():
                    score += 3
                if tok in hay:
                    score += 1
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def __len__(self) -> int:
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills)


# Default singleton — call as ``from mmar_mage.skills import default_registry``
default_registry = SkillRegistry(skills=list(_BUILTIN_SKILLS))
