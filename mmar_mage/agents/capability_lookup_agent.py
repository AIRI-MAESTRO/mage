"""
CapabilityLookupAgent — single source of truth for the planner's capability context.

Folds three previously-separate signals into one ``CapabilityContext``
so the planner sees ONE list of agent skills instead of overlapping
``capability_lookup`` + ``skill_discovery`` outputs (§2 P0):

1. Tools / MCP servers from a ``capability_registry`` memory entity
   (legacy behaviour, queried via ``MemoryClient.get_capabilities``).
2. AgentSkill memory entities ranked by ``MemoryClient.find_capability_matches``
   (when a rough aim — i.e. the user query + domain — is supplied).
3. Locally-installed skills catalogued via
   ``mmar_carl.skill_loader.SkillLoader.catalog_all`` when present.
4. (Optional) Recommendations from :class:`SkillDiscoveryAgent` when the
   caller hands the agent an :class:`LLMClient`.

Every external integration is lazy-imported and gracefully degraded:
missing ``gigaevo_client`` / missing ``mmar_carl.skill_loader`` /
SkillDiscoveryAgent failure are all non-fatal — the lookup simply
returns whatever it could collect. Deduplication is by SHA256 of the
skill manifest so a skill installed locally AND promoted to Memory only
appears once.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..generated_tool_index import DEFAULT_TOOLS_DIR, load_tool_index
from ..memory import MemoryManager
from ..schemas import DomainAnalysis, MAGEConfig

if TYPE_CHECKING:
    from ..llm import LLMClient
    from ..skills.discovery import SkillSearchResult
    from .web_research_agent import WebResearchAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentSkill entry (the new shape carried in CapabilityContext.agent_skills)
# ---------------------------------------------------------------------------


class AgentSkillEntry(BaseModel):
    """One AgentSkill surfaced into the planner's capability context.

    Field shape matches what CARE's Memory side stores under the
    ``agent_skill`` entity_type (see ``~/Development/care/PREPARE.md``
    §1.3). All fields except ``name``/``uri`` are optional so this also
    accommodates locally-discovered and web-discovered skills which
    don't carry a manifest summary.
    """

    name: str
    description: str = ""
    uri: str
    manifest_summary: str = ""
    sha256: str = ""
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    source: str = "memory"
    """``memory`` | ``local`` | ``discovered`` — origin of this entry."""
    relevance: float = 0.0
    """0-1 score from the ranker (or 0 when not ranked)."""
    why: str = ""
    """Optional rationale forwarded into the planner prompt."""


class CapabilityContext(BaseModel):
    """Unified planner context: tools, MCP servers, and agent skills.

    Parameters
    ----------
    tools : list[dict[str, Any]]
        Available tool definitions (from ``capability_registry`` memory).
    mcp_servers : list[dict[str, Any]]
        Available MCP server definitions (from ``capability_registry`` memory).
    agent_skills : list[AgentSkillEntry]
        AgentSkills available for this run, deduplicated across memory /
        local catalog / discovery sources by SHA256 of SKILL.md.
    environment_id : str
        Environment identifier used for the legacy ``capability_registry`` lookup.
    """

    tools: list[dict[str, Any]] = Field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    agent_skills: list[AgentSkillEntry] = Field(default_factory=list)
    environment_id: str = "default"

    def skills_digest(self) -> str:
        """Render ``agent_skills`` into a planner-prompt-friendly digest.

        Returns an empty string when no skills are present so callers can
        unconditionally include the digest with no extra check.
        """
        if not self.agent_skills:
            return ""
        lines = ["Recommended packaged skills (use as `agent_skill` steps with `skill` = URI):"]
        for s in self.agent_skills:
            score = f"relevance={s.relevance:.2f}" if s.relevance else "relevance=n/a"
            lines.append(f"- {s.name} ({score}, source={s.source}): {s.why or s.description}")
            lines.append(f"    URI: {s.uri}")
            if s.description and s.description != s.why:
                lines.append(f"    Description: {s.description}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Capability matching (§2 P1) — rank skills against a sub-goal
# ---------------------------------------------------------------------------


# Common words we drop before scoring — without this the score is dominated
# by stop-word overlap (every sub-goal matches every skill via "the", "a").
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
        "you",
        "your",
        "do",
        "does",
        "did",
        "use",
        "using",
        "make",
        "made",
        "create",
        "extract",
        "step",
        "data",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lower-case, alpha-split, drop stop-words.

    Deliberately simple: deterministic, dependency-free, fast enough to
    call per-step. Real semantic matching is left to the LLM rerank in
    :class:`SkillDiscoveryAgent`; this helper is a planner-side gate
    for the *planner-prompt rule* ("when a strong match exists, prefer
    agent_skill over a custom LLM step").
    """
    if not text:
        return set()
    out: set[str] = set()
    word = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tok = "".join(word)
                if len(tok) >= 3 and tok not in _STOPWORDS:
                    out.add(tok)
                word = []
    if word:
        tok = "".join(word)
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _score_capability_match(rough_aim: str, capability: AgentSkillEntry) -> float:
    """Score a capability's relevance to a sub-goal.

    Combines token-overlap (Jaccard) over the rough aim against the
    capability's name/description/tags/manifest summary with a small
    boost from the capability's own ``relevance`` score (when one is
    present from the discovery / memory rerank).

    Returns
    -------
    float
        Score in ``[0, 1]``. ``>= 0.5`` means "prefer agent_skill over
        a custom LLM step" — the planner-prompt rule uses this threshold.
    """
    aim_tokens = _tokenize(rough_aim)
    if not aim_tokens:
        return 0.0

    cap_text = " ".join(
        filter(
            None,
            [
                capability.name,
                capability.description,
                " ".join(capability.tags),
                capability.manifest_summary,
            ],
        )
    )
    cap_tokens = _tokenize(cap_text)
    if not cap_tokens:
        return 0.0

    overlap = aim_tokens & cap_tokens
    union = aim_tokens | cap_tokens
    jaccard = len(overlap) / len(union) if union else 0.0

    # Up-weight when the aim shares strong, low-frequency signals with
    # the capability. 0.7 * jaccard + 0.3 * pre-existing relevance keeps
    # the score in [0, 1] while letting upstream ranker confidence
    # nudge close calls.
    return min(1.0, 0.7 * jaccard + 0.3 * max(0.0, min(1.0, capability.relevance)))


# ---------------------------------------------------------------------------
# Capability lookup orchestrator
# ---------------------------------------------------------------------------


class CapabilityLookupAgent:
    """Look up available capabilities from every configured source.

    Parameters
    ----------
    memory_manager : MemoryManager
        Memory manager for tool/MCP/agent-skill queries.
    config : MAGEConfig
        MAGE configuration.
    llm : LLMClient | None
        Optional LLM client. When supplied AND
        ``config.enable_skill_discovery=True``, an internal
        :class:`SkillDiscoveryAgent` runs as part of ``lookup()`` and its
        recommendations are merged into ``agent_skills``.
    web_agent : WebResearchAgent | None
        Optional web research agent for the skill-discovery web
        fallback.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        config: MAGEConfig,
        *,
        llm: LLMClient | None = None,
        web_agent: WebResearchAgent | None = None,
    ) -> None:
        self._memory = memory_manager
        self._config = config
        self._llm = llm
        self._web = web_agent

    async def lookup(
        self,
        *,
        query: str | None = None,
        domain_analysis: DomainAnalysis | None = None,
    ) -> CapabilityContext:
        """Build a unified ``CapabilityContext``.

        Parameters
        ----------
        query : str | None
            User query — used as the rough aim for memory skill matching
            and as input to :class:`SkillDiscoveryAgent`.
        domain_analysis : DomainAnalysis | None
            Domain context — concepts feed the skill ranker; required to
            invoke SkillDiscoveryAgent.

        Returns
        -------
        CapabilityContext
            All four signals merged. Returns a context with only
            ``environment_id`` populated when every source fails (graceful
            degradation — generation is never blocked by a capability
            lookup failure).
        """
        env_id = self._config.capability_environment_id

        # 1. Legacy capability_registry lookup (tools + mcp_servers).
        tools, mcp_servers = await self._lookup_tools_and_mcp(env_id)

        # 1b. Locally-generated tools (WP-9) — surface tools synthesized + validated
        # on a prior run so the planner reuses them instead of re-synthesizing.
        tools = tools + self._fetch_generated_tools()

        # 2. AgentSkills from memory (gigaevo-client). Best-effort.
        memory_skills: list[AgentSkillEntry] = []
        if self._config.enable_memory_skill_lookup and query is not None:
            rough_aim = self._rough_aim(query, domain_analysis)
            memory_skills = await self._fetch_memory_skills(rough_aim)

        # 3. Locally-installed skills via mmar_carl.SkillLoader. Best-effort.
        local_skills: list[AgentSkillEntry] = []
        if self._config.enable_local_skill_catalog:
            local_skills = self._fetch_local_skills()

        # 4. SkillDiscoveryAgent (built-in registry + optional web fallback).
        discovered_skills: list[AgentSkillEntry] = []
        if (
            self._config.enable_skill_discovery
            and self._llm is not None
            and query is not None
            and domain_analysis is not None
        ):
            discovered_skills = await self._fetch_discovered_skills(query, domain_analysis)

        # Merge + dedupe. Order matters: memory wins ties over local,
        # local wins over discovered — the "more authoritative" source
        # keeps its metadata.
        agent_skills = _dedupe_skills(memory_skills + local_skills + discovered_skills)

        return CapabilityContext(
            tools=tools,
            mcp_servers=mcp_servers,
            agent_skills=agent_skills,
            environment_id=env_id,
        )

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    async def _lookup_tools_and_mcp(self, env_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            raw = await self._memory.get_capabilities(environment_id=env_id)
        except Exception as exc:
            logger.warning("CapabilityLookupAgent: tools/mcp lookup failed (non-fatal): %s", exc)
            return [], []
        if raw is None:
            logger.info("CapabilityLookupAgent: no registry found for env=%s", env_id)
            return [], []
        return list(raw.get("tools", [])), list(raw.get("mcp_servers", []))

    def _fetch_generated_tools(self) -> list[dict[str, Any]]:
        """Read the local generated-tool index (WP-9). Best-effort, never raises."""
        try:
            entries = load_tool_index(DEFAULT_TOOLS_DIR)
        except Exception as exc:  # noqa: BLE001 — non-fatal
            logger.warning("CapabilityLookupAgent: generated-tool index read failed: %s", exc)
            return []
        return [
            {
                "name": e.get("name", ""),
                "description": e.get("description", ""),
                "signature": e.get("signature", ""),
                "source": "generated",
                "path": e.get("path", ""),
            }
            for e in entries
            if e.get("name")
        ]

    @staticmethod
    def _rough_aim(query: str, domain_analysis: DomainAnalysis | None) -> str:
        if domain_analysis is None:
            return query.strip()
        concepts = " ".join((domain_analysis.key_concepts or [])[:5])
        return f"{query.strip()} | domain={domain_analysis.domain} | {concepts}".strip()

    async def _fetch_memory_skills(self, rough_aim: str) -> list[AgentSkillEntry]:
        """Pull AgentSkill entries from memory via gigaevo-client.

        Lazy-import so the absence of ``gigaevo_client`` is non-fatal.
        Best-effort: a missing module or any exception yields an empty
        list. ``MemoryClient.find_capability_matches`` is the helper
        shipped in gigaevo-client §4 (see ~/Development/care/PREPARE.md).
        """
        try:
            from gigaevo_client import MemoryClient
        except ImportError:
            logger.debug("CapabilityLookupAgent: gigaevo_client not installed; skipping memory AgentSkill lookup")
            return []

        top_k = self._config.memory_skill_top_k
        try:
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                if not hasattr(client, "find_capability_matches"):
                    logger.debug(
                        "CapabilityLookupAgent: gigaevo_client.MemoryClient lacks "
                        "find_capability_matches; skipping (older SDK)"
                    )
                    return []
                matches = client.find_capability_matches(rough_aim, top_k=top_k)
        except Exception as exc:
            logger.warning("CapabilityLookupAgent: memory skill lookup failed: %s", exc)
            return []

        out: list[AgentSkillEntry] = []
        for m in matches or []:
            try:
                out.append(_memory_match_to_entry(m))
            except Exception as exc:  # noqa: BLE001 — defensive against payload drift
                logger.debug("CapabilityLookupAgent: malformed memory skill payload: %s", exc)
        return out

    def _fetch_local_skills(self) -> list[AgentSkillEntry]:
        """Pull locally-installed skills via mmar_carl.SkillLoader.

        Lazy-import so absence is non-fatal.
        """
        try:
            from mmar_carl.skill_loader import SkillLoader
        except ImportError:
            logger.debug("CapabilityLookupAgent: mmar_carl.skill_loader not installed; skipping local catalog")
            return []

        try:
            loader = SkillLoader()
            raw = loader.catalog_all(payloads=True)
        except Exception as exc:
            logger.warning("CapabilityLookupAgent: local skill catalog failed: %s", exc)
            return []

        out: list[AgentSkillEntry] = []
        for item in raw or []:
            try:
                out.append(_local_skill_to_entry(item))
            except Exception as exc:  # noqa: BLE001
                logger.debug("CapabilityLookupAgent: malformed local skill payload: %s", exc)
        return out

    async def _fetch_discovered_skills(self, query: str, domain_analysis: DomainAnalysis) -> list[AgentSkillEntry]:
        """Invoke the (now-internal) SkillDiscoveryAgent."""
        # Lazy import to break the previously-circular skills→agents wiring.
        from ..skills.discovery import SkillDiscoveryAgent

        # Caller (`lookup`) only invokes this method when `self._llm is not None`,
        # so the assert is a type-narrowing no-op that can never fire at runtime.
        assert self._llm is not None
        try:
            discoverer = SkillDiscoveryAgent(
                self._llm,
                web_agent=self._web if self._config.enable_skill_web_discovery else None,
            )
            result: SkillSearchResult = await discoverer.discover(
                query,
                domain_analysis,
                self._config,
                max_recommendations=self._config.skill_discovery_max_recommendations,
                enable_web_discovery=self._config.enable_skill_web_discovery,
            )
        except Exception as exc:
            logger.warning("CapabilityLookupAgent: skill discovery failed: %s", exc)
            return []

        return [
            AgentSkillEntry(
                name=r.name,
                description=r.description,
                uri=r.uri,
                why=r.why,
                relevance=r.relevance,
                source="discovered",
            )
            for r in result.recommendations
        ]


# ---------------------------------------------------------------------------
# Helpers — payload conversion + dedup
# ---------------------------------------------------------------------------


def _memory_match_to_entry(m: Any) -> AgentSkillEntry:
    """Map a gigaevo-client capability match to our internal entry shape.

    Accepts either an object with attributes or a dict — gigaevo-client's
    shape has evolved, so we defensively probe both.
    """

    def field(key: str, default: Any = "") -> Any:
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)

    return AgentSkillEntry(
        name=str(field("name") or "unknown-skill"),
        description=str(field("description") or ""),
        uri=str(field("uri") or ""),
        manifest_summary=str(field("manifest_summary") or ""),
        sha256=str(field("sha256") or ""),
        tags=list(field("tags") or []),
        allowed_tools=list(field("allowed_tools") or []),
        source="memory",
        relevance=float(field("relevance", 0.0) or 0.0),
        why=str(field("why") or ""),
    )


def _local_skill_to_entry(item: Any) -> AgentSkillEntry:
    """Map a mmar_carl SkillLoader catalog entry to our internal entry shape."""

    def field(key: str, default: Any = "") -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    return AgentSkillEntry(
        name=str(field("name") or "unknown-skill"),
        description=str(field("description") or ""),
        uri=str(field("uri") or field("path") or ""),
        manifest_summary=str(field("manifest_summary") or ""),
        sha256=str(field("sha256") or ""),
        tags=list(field("tags") or []),
        allowed_tools=list(field("allowed_tools") or []),
        source="local",
    )


def _dedupe_skills(skills: list[AgentSkillEntry]) -> list[AgentSkillEntry]:
    """Dedupe by SHA256 of SKILL.md, falling back to (uri, name) when absent.

    Preserves input order — earlier sources (memory > local > discovered)
    win ties. Entries with empty SHA256 are deduped on (uri, name) to
    avoid silently dropping discovered skills (which never carry a SHA).
    """
    out: list[AgentSkillEntry] = []
    seen: set[str] = set()
    for s in skills:
        if s.sha256:
            key = f"sha256:{s.sha256}"
        else:
            key = f"id:{s.uri}|{s.name}"
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
