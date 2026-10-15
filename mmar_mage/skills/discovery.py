"""SkillDiscoveryAgent — find relevant AgentSkills for a user query.

Pipeline:

1. **Decide** — quick LLM gate: does the task plausibly benefit from a
   packaged skill, or is it pure reasoning? If pure reasoning → return
   empty (cheap path).
2. **Pre-filter** — substring search over the local
   :class:`SkillRegistry` for fast candidates.
3. **LLM rerank** — score candidates against the query, return top-K with
   reasons.
4. **Web fallback** (optional, off by default) — if local match is weak
   AND ``enable_web_discovery=True`` AND a web search key is configured,
   query ``"agent skill {keywords} site:github.com"`` via
   :class:`WebResearchAgent`, parse GitHub URIs out of the digest, and
   feed them through the same LLM reranker.

All failure modes are non-fatal — returns
:class:`SkillSearchResult` with empty ``recommendations`` if anything
breaks. The generation pipeline always proceeds.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from ..agents.web_research_agent import WebResearchAgent
from ..llm import LLMClient
from ..schemas import DomainAnalysis, MAGEConfig
from .registry import KnownSkill, SkillRegistry, default_registry

logger = logging.getLogger(__name__)

STAGE = "skill_discovery"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class SkillRecommendation(BaseModel):
    """One recommended skill for the query."""

    name: str
    uri: str
    description: str
    why: str = ""
    """Why the LLM thinks this skill fits — passed into StepPlanner prompt."""
    relevance: float = 0.0
    """0-1 score; higher = more relevant."""
    source: str = "registry"
    """'registry' | 'web' — for logging / debugging."""


class SkillSearchResult(BaseModel):
    """Output of :meth:`SkillDiscoveryAgent.discover`."""

    recommendations: list[SkillRecommendation] = Field(default_factory=list)
    used_web_fallback: bool = False
    candidates_considered: int = 0
    digest: str = ""
    """Short prose summary fed into StepPlanner as additional context."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SKILL_RELEVANCE_GATE_PROMPT = """\
You are a skill-routing gate. Given a user task, decide whether the task \
plausibly benefits from any packaged "AgentSkill" (a reusable script + \
LLM instructions bundle for things like PDF / DOCX / PPTX manipulation, \
web search, MCP server building, design-asset generation).

Return JSON:
{
  "needs_skill": true | false,
  "reason": "<one short sentence>",
  "hint_keywords": ["...", "..."]
}

Rules:
- Pure analytical / reasoning / synthesis tasks (no I/O of binary or rich-format \
data, no external scripts) → needs_skill=false.
- If the task mentions PDFs, Word docs, spreadsheets, slide decks, MCP, web \
search, code editing, design assets → needs_skill=true.
- hint_keywords: 3-6 short tokens to drive a registry search.
- Return ONLY valid JSON.
"""


SKILL_RERANK_PROMPT = """\
You are a skill recommender. Given a user task and a shortlist of candidate \
AgentSkills (with descriptions), pick at most 3 that genuinely fit the task. \
For each pick, give a one-sentence "why" and a relevance score 0-1.

Return JSON:
{
  "picks": [
    {"name": "<skill name>", "why": "<short reason>", "relevance": 0.0-1.0}
  ]
}

Rules:
- Be strict: do NOT recommend skills that only tangentially fit.
- If none of the candidates clearly fit, return an empty "picks" array.
- A score >= 0.7 means "strongly recommend"; 0.5-0.7 means "consider";
  below 0.5 should not be returned at all.
- Return ONLY valid JSON.
"""


# Regex for github://owner/repo[/subpath][@ref] URIs in web digests.
_GITHUB_URI_RE = re.compile(r"(github://[\w./@-]+)|(https?://github\.com/[\w.-]+/[\w.-]+)")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SkillDiscoveryAgent:
    """LLM-driven skill recommender backed by a registry + optional web search."""

    def __init__(
        self,
        llm: LLMClient,
        registry: SkillRegistry | None = None,
        web_agent: WebResearchAgent | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry or default_registry
        self._web = web_agent

    async def discover(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        config: MAGEConfig,
        *,
        max_recommendations: int = 3,
        enable_web_discovery: bool = False,
    ) -> SkillSearchResult:
        """Run the discovery pipeline. Always returns a result; never raises."""
        # Stage 1 — gate.
        try:
            gate = await self._gate_relevance(query, domain_analysis)
        except Exception as exc:
            logger.debug("SkillDiscoveryAgent: gate failed (%s) — skipping skills", exc)
            return SkillSearchResult()

        if not gate.get("needs_skill"):
            return SkillSearchResult(digest="No packaged skill needed for this task.")

        hint_keywords: list[str] = list(gate.get("hint_keywords") or [])
        query_for_search = (query + " " + " ".join(hint_keywords)).strip()

        # Stage 2 — registry pre-filter.
        candidates = self._registry.search_substring(query_for_search, top_k=10)
        if not candidates:
            # Fall back to the full catalog so the LLM has something to choose from.
            candidates = list(self._registry.skills)
        candidates_considered = len(candidates)

        # Stage 3 — LLM rerank.
        try:
            picks_from_registry = await self._rerank(query, candidates)
        except Exception as exc:
            logger.debug("SkillDiscoveryAgent: rerank failed (%s)", exc)
            picks_from_registry = []

        recommendations: list[SkillRecommendation] = []
        for pick in picks_from_registry:
            skill = self._registry.get(pick.get("name", ""))
            if skill is None:
                continue
            recommendations.append(
                SkillRecommendation(
                    name=skill.name,
                    uri=skill.uri,
                    description=skill.description,
                    why=str(pick.get("why", "")),
                    relevance=float(pick.get("relevance", 0.0)),
                    source="registry",
                )
            )

        used_web = False

        # Stage 4 — web fallback when registry result is weak.
        weak = not recommendations or max((r.relevance for r in recommendations), default=0.0) < 0.5
        if enable_web_discovery and weak and self._web is not None and config.web_search_api_key:
            try:
                web_recommendations = await self._discover_via_web(query, domain_analysis, hint_keywords, config)
                if web_recommendations:
                    used_web = True
                    recommendations.extend(web_recommendations)
                    # P3.8 — cache the web-discovered skills into the registry
                    # so the NEXT discovery of the same need is a registry hit
                    # (no repeated web-search cost).
                    self._cache_web_discoveries(web_recommendations, hint_keywords)
            except Exception as exc:
                logger.debug("SkillDiscoveryAgent: web discovery failed (%s)", exc)

        # Sort and trim.
        recommendations.sort(key=lambda r: r.relevance, reverse=True)
        recommendations = recommendations[:max_recommendations]
        digest = self._build_digest(recommendations)

        return SkillSearchResult(
            recommendations=recommendations,
            used_web_fallback=used_web,
            candidates_considered=candidates_considered,
            digest=digest,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_web_discoveries(
        self,
        web_recommendations: list[SkillRecommendation],
        hint_keywords: list[str],
    ) -> None:
        """P3.8 — persist web-discovered skills into the registry as
        ``source="discovered"`` so a later discovery of the same need is a
        registry hit instead of another web search. The hint keywords seed
        the skill's ``capabilities`` so ``search_substring`` can re-find it.
        Skips names already in the registry (builtin / previously cached).
        """
        cached: list[str] = []
        for rec in web_recommendations:
            name = (rec.name or "").strip()
            if not name or self._registry.get(name) is not None:
                continue
            self._registry.add(
                KnownSkill(
                    name=name,
                    description=(rec.description or rec.why or name),
                    uri=rec.uri or "",
                    capabilities=list(hint_keywords or []),
                    source="discovered",
                )
            )
            cached.append(name)
        if cached:
            logger.info(
                "SkillDiscoveryAgent: cached %d web-discovered skill(s) for reuse: %s",
                len(cached),
                ", ".join(cached),
            )

    async def _gate_relevance(self, query: str, domain_analysis: DomainAnalysis) -> dict[str, Any]:
        prompt = (
            f"User task: {query}\n\n"
            f"Domain: {domain_analysis.domain}; "
            f"task_type: {domain_analysis.task_type}; "
            f"complexity: {domain_analysis.complexity}\n"
        )
        data = await self._llm.generate_json(
            system_prompt=SKILL_RELEVANCE_GATE_PROMPT,
            user_prompt=prompt,
            stage=STAGE,
        )
        return data or {}

    async def _rerank(self, query: str, candidates: list[KnownSkill]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        catalog_block = "\n".join(
            f"- {s.name}: {s.description} (uri: {s.uri}; capabilities: {', '.join(s.capabilities[:5])})"
            for s in candidates
        )
        prompt = f"User task:\n{query}\n\nCandidate skills:\n{catalog_block}\n"
        data = await self._llm.generate_json(
            system_prompt=SKILL_RERANK_PROMPT,
            user_prompt=prompt,
            stage=STAGE,
        )
        picks = data.get("picks", []) if isinstance(data, dict) else []
        return picks if isinstance(picks, list) else []

    async def _discover_via_web(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        hint_keywords: list[str],
        config: MAGEConfig,
    ) -> list[SkillRecommendation]:
        """Use WebResearchAgent to find candidate GitHub skill repos."""
        if self._web is None:
            return []

        # Augment domain analysis with skill-discovery keywords so the
        # web search includes "agent skill" / "site:github.com" hints.
        aug_concepts = (
            list(domain_analysis.key_concepts)
            + hint_keywords
            + [
                "AgentSkills",
                "SKILL.md",
                "site:github.com",
            ]
        )
        aug_domain = domain_analysis.model_copy(update={"key_concepts": aug_concepts})
        web_result = await self._web.research(query, aug_domain, config)
        if not web_result.digest:
            return []

        # Extract candidate URIs from the digest + sources.
        text = (web_result.digest + " " + " ".join(web_result.sources)).strip()
        uri_candidates = _extract_skill_uris(text)
        if not uri_candidates:
            return []

        # Build provisional KnownSkill entries for rerank.
        provisional: list[KnownSkill] = []
        for uri in uri_candidates[:8]:
            provisional.append(
                KnownSkill(
                    name=_name_from_uri(uri),
                    description=f"Discovered via web search; URI: {uri}",
                    capabilities=hint_keywords,
                    uri=uri,
                    domains=[domain_analysis.domain],
                    source="discovered",
                )
            )

        try:
            picks = await self._rerank(query, provisional)
        except Exception as exc:
            logger.debug("SkillDiscoveryAgent: web rerank failed (%s)", exc)
            return []

        recs: list[SkillRecommendation] = []
        for pick in picks:
            match = next((p for p in provisional if p.name == pick.get("name")), None)
            if match is None:
                continue
            recs.append(
                SkillRecommendation(
                    name=match.name,
                    uri=match.uri,
                    description=match.description,
                    why=str(pick.get("why", "")),
                    relevance=float(pick.get("relevance", 0.0)),
                    source="web",
                )
            )
        return recs

    @staticmethod
    def _build_digest(recs: list[SkillRecommendation]) -> str:
        if not recs:
            return "No packaged skills recommended for this task."
        lines = ["Recommended packaged skills (use as `agent_skill` steps with `skill` = URI):"]
        for r in recs:
            lines.append(f"- {r.name} (relevance={r.relevance:.2f}): {r.why}")
            lines.append(f"    URI: {r.uri}")
            lines.append(f"    Description: {r.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# URI extraction helpers
# ---------------------------------------------------------------------------


def _extract_skill_uris(text: str) -> list[str]:
    """Pull github:// and https://github.com/... URIs from arbitrary text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _GITHUB_URI_RE.finditer(text):
        uri = match.group(0).rstrip(").,;'\"")
        # Normalise https github URLs to github:// form when possible
        if uri.startswith("https://github.com/"):
            path = uri.removeprefix("https://github.com/").rstrip("/")
            parts = path.split("/")
            if len(parts) >= 2:
                normalised = f"github://{parts[0]}/{parts[1]}"
                if len(parts) > 2:
                    normalised += "/" + "/".join(parts[2:])
                uri = normalised
        if uri not in seen:
            seen.add(uri)
            out.append(uri)
    return out


def _name_from_uri(uri: str) -> str:
    """Derive a short name from a URI for use in registry display."""
    tail = uri.rstrip("/").split("/")[-1]
    return tail.split("@")[0] or uri


# ---------------------------------------------------------------------------
# P6.2 — always-on FREE keyword surfacing (no LLM gate, no web)
# ---------------------------------------------------------------------------


def surface_skills(
    query: str,
    domain_analysis: DomainAnalysis | None = None,
    *,
    registry: SkillRegistry | None = None,
    top_k: int = 5,
) -> list[KnownSkill]:
    """Substring-match ``query`` + the domain's concepts against the skill
    registry — PURE + deterministic, NO LLM call and NO web. This is the
    "push" side of P6.2: the planner ALWAYS sees relevant packaged skills,
    instead of them being hidden behind the (LLM) discovery gate. The LLM
    rerank + web fallback in :class:`SkillDiscoveryAgent` stay opt-in.
    """
    reg = registry or default_registry
    parts = [query or ""]
    if domain_analysis is not None:
        parts.append(getattr(domain_analysis, "domain", "") or "")
        parts.extend(getattr(domain_analysis, "key_concepts", None) or [])
    search = " ".join(p for p in parts if p).strip()
    if not search:
        return []
    return reg.search_substring(search, top_k=top_k)


def format_surfaced_skills(skills: list[KnownSkill]) -> str:
    """Render surfaced skills as a planner-prompt block so the model knows it
    may use them as ``agent_skill`` steps. Empty list → ``""``."""
    if not skills:
        return ""
    lines = [
        "Relevant packaged skills available (use as `agent_skill` steps with "
        "`skill` = the URI):",
    ]
    lines.extend(f"- {s.name}: {s.description} (skill={s.uri})" for s in skills)
    return "\n".join(lines)
