"""Tests for §2 P0 — unified CapabilityContext.

Covers:
- `CapabilityContext.agent_skills` field shape
- `CapabilityLookupAgent.lookup()` returns a unified context
- Memory AgentSkill source (mocked gigaevo_client)
- Local skill catalog source (mocked mmar_carl.SkillLoader)
- SkillDiscoveryAgent folded in: invoked internally, recommendations land
  in `agent_skills` with source="discovered"
- Dedup by SHA256 of SKILL.md — memory entries beat local ones in ties
- Generator wires the unified context: `skill_discovery` is NO LONGER a
  separate stage; the planner sees one skills digest from
  `CapabilityContext.skills_digest()`
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.capability_lookup_agent import (
    AgentSkillEntry,
    CapabilityContext,
    CapabilityLookupAgent,
    _dedupe_skills,
    _local_skill_to_entry,
    _memory_match_to_entry,
)
from mmar_mage.generator import MAGEGenerator
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import DomainAnalysis, MAGEConfig
from mmar_mage.skills.discovery import SkillRecommendation, SkillSearchResult

from .conftest import (
    SAMPLE_DAG,
    SAMPLE_DESCRIBED_STEPS,
    SAMPLE_DOMAIN_ANALYSIS,
    SAMPLE_STEP_PLAN,
    _make_openai_response,
)

# ---------------------------------------------------------------------------
# AgentSkillEntry + CapabilityContext schema
# ---------------------------------------------------------------------------


class TestAgentSkillEntry:
    def test_defaults(self) -> None:
        s = AgentSkillEntry(name="pdf", uri="github://anthropics/skills/pdf")
        assert s.description == ""
        assert s.sha256 == ""
        assert s.tags == []
        assert s.allowed_tools == []
        assert s.source == "memory"
        assert s.relevance == 0.0


class TestCapabilityContextDigest:
    def test_empty_skills_returns_empty_string(self) -> None:
        assert CapabilityContext().skills_digest() == ""

    def test_digest_includes_name_uri_relevance_source(self) -> None:
        ctx = CapabilityContext(
            agent_skills=[
                AgentSkillEntry(
                    name="pdf",
                    description="Extract PDF data",
                    uri="github://anthropics/skills/pdf@main",
                    relevance=0.85,
                    why="Task mentions PDFs",
                    source="discovered",
                )
            ]
        )
        digest = ctx.skills_digest()
        assert "pdf" in digest
        assert "0.85" in digest
        assert "github://anthropics/skills/pdf@main" in digest
        assert "discovered" in digest


# ---------------------------------------------------------------------------
# Payload conversion helpers
# ---------------------------------------------------------------------------


class TestPayloadConversion:
    def test_memory_match_from_dict(self) -> None:
        e = _memory_match_to_entry(
            {
                "name": "docx",
                "description": "Word docs",
                "uri": "github://anthropics/skills/docx",
                "sha256": "deadbeef",
                "tags": ["documents"],
                "allowed_tools": ["read_file"],
                "manifest_summary": "Extract / produce DOCX files",
                "relevance": 0.7,
                "why": "Task mentions docx",
            }
        )
        assert e.name == "docx"
        assert e.sha256 == "deadbeef"
        assert e.allowed_tools == ["read_file"]
        assert e.relevance == pytest.approx(0.7)
        assert e.source == "memory"

    def test_memory_match_from_object(self) -> None:
        obj = MagicMock()
        obj.name = "pdf"
        obj.uri = "github://x/y"
        obj.sha256 = "abc123"
        obj.description = ""
        obj.manifest_summary = ""
        obj.tags = []
        obj.allowed_tools = []
        obj.relevance = 0.5
        obj.why = ""
        e = _memory_match_to_entry(obj)
        assert e.name == "pdf"
        assert e.sha256 == "abc123"
        assert e.relevance == pytest.approx(0.5)

    def test_local_skill_source_label(self) -> None:
        e = _local_skill_to_entry({"name": "internal-tool", "uri": "local:///opt/skills/tool"})
        assert e.source == "local"
        assert e.uri == "local:///opt/skills/tool"

    def test_local_skill_falls_back_to_path_field(self) -> None:
        """Some SkillLoader payloads use `path` instead of `uri`."""
        e = _local_skill_to_entry({"name": "x", "path": "/opt/skills/x"})
        assert e.uri == "/opt/skills/x"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedupSkills:
    def test_sha256_dedup_keeps_first(self) -> None:
        skills = [
            AgentSkillEntry(name="pdf-memory", uri="u1", sha256="abc", source="memory"),
            AgentSkillEntry(name="pdf-local", uri="u2", sha256="abc", source="local"),
        ]
        out = _dedupe_skills(skills)
        assert len(out) == 1
        # Memory came first, so its name wins
        assert out[0].name == "pdf-memory"
        assert out[0].source == "memory"

    def test_no_sha_falls_back_to_uri_plus_name(self) -> None:
        skills = [
            AgentSkillEntry(name="x", uri="u1", source="discovered"),
            AgentSkillEntry(name="x", uri="u1", source="discovered"),  # exact dup
            AgentSkillEntry(name="x", uri="u2", source="discovered"),  # diff uri
        ]
        out = _dedupe_skills(skills)
        assert len(out) == 2
        assert {s.uri for s in out} == {"u1", "u2"}

    def test_mixed_sha_and_no_sha_coexist(self) -> None:
        skills = [
            AgentSkillEntry(name="a", uri="u1", sha256="aaa", source="memory"),
            AgentSkillEntry(name="b", uri="u2", source="discovered"),
        ]
        out = _dedupe_skills(skills)
        assert len(out) == 2

    def test_empty_input(self) -> None:
        assert _dedupe_skills([]) == []


# ---------------------------------------------------------------------------
# CapabilityLookupAgent — tools + MCP from memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_returns_tools_and_mcp_when_present() -> None:
    config = MAGEConfig(enable_memory=False, capability_environment_id="env-1")
    mgr = MemoryManager(config)

    async def fake_get(environment_id: str = "default") -> dict[str, Any]:
        assert environment_id == "env-1"
        return {"tools": [{"name": "calc"}], "mcp_servers": [{"name": "git"}]}

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    agent = CapabilityLookupAgent(mgr, config)
    ctx = await agent.lookup()
    assert ctx.environment_id == "env-1"
    assert ctx.tools == [{"name": "calc"}]
    assert ctx.mcp_servers == [{"name": "git"}]
    assert ctx.agent_skills == []


@pytest.mark.asyncio
async def test_lookup_graceful_when_memory_returns_none() -> None:
    config = MAGEConfig(enable_memory=False)
    mgr = MemoryManager(config)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]
    agent = CapabilityLookupAgent(mgr, config)
    ctx = await agent.lookup()
    assert ctx.tools == []
    assert ctx.mcp_servers == []


@pytest.mark.asyncio
async def test_lookup_graceful_when_memory_raises() -> None:
    config = MAGEConfig(enable_memory=False)
    mgr = MemoryManager(config)

    async def boom(environment_id: str = "default") -> None:
        raise RuntimeError("memory down")

    mgr.get_capabilities = boom  # type: ignore[method-assign]
    agent = CapabilityLookupAgent(mgr, config)
    ctx = await agent.lookup()
    # Tools/mcp empty, but the lookup itself didn't raise — graceful.
    assert ctx.tools == []


# ---------------------------------------------------------------------------
# CapabilityLookupAgent — memory AgentSkill source (gigaevo_client mock)
# ---------------------------------------------------------------------------


class _FakeMemoryClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> _FakeMemoryClient:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def find_capability_matches(self, rough_aim: str, top_k: int) -> list[dict[str, Any]]:
        # rough_aim should weave in query + domain hints
        assert "Read this PDF" in rough_aim
        return [
            {
                "name": "pdf",
                "description": "Read PDFs",
                "uri": "github://anthropics/skills/pdf",
                "sha256": "shared-sha",
                "tags": ["documents"],
                "allowed_tools": ["read_file"],
                "relevance": 0.9,
            }
        ]


def _install_fake_gigaevo_client(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("gigaevo_client")
    pkg.MemoryClient = _FakeMemoryClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gigaevo_client", pkg)


@pytest.mark.asyncio
async def test_lookup_pulls_memory_agent_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_gigaevo_client(monkeypatch)
    cfg = MAGEConfig(enable_memory=False, enable_memory_skill_lookup=True)
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    agent = CapabilityLookupAgent(mgr, cfg)
    ctx = await agent.lookup(
        query="Read this PDF",
        domain_analysis=DomainAnalysis(domain="documents", key_concepts=["pdf"]),
    )
    assert len(ctx.agent_skills) == 1
    s = ctx.agent_skills[0]
    assert s.name == "pdf"
    assert s.source == "memory"
    assert s.sha256 == "shared-sha"
    assert s.relevance == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# CapabilityLookupAgent — local catalog source (mmar_carl mock)
# ---------------------------------------------------------------------------


class _FakeSkillLoader:
    def catalog_all(self, payloads: bool = False) -> list[dict[str, Any]]:
        assert payloads is True
        return [
            {
                "name": "local-extractor",
                "description": "User-installed extractor",
                "uri": "local:///opt/skills/extractor",
                "sha256": "local-sha",
                "tags": ["custom"],
                "allowed_tools": [],
                "manifest_summary": "",
            }
        ]


def _install_fake_mmar_carl_skill_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("mmar_carl")
    skill_loader_mod = types.ModuleType("mmar_carl.skill_loader")
    skill_loader_mod.SkillLoader = _FakeSkillLoader  # type: ignore[attr-defined]
    pkg.skill_loader = skill_loader_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mmar_carl", pkg)
    monkeypatch.setitem(sys.modules, "mmar_carl.skill_loader", skill_loader_mod)


@pytest.mark.asyncio
async def test_lookup_pulls_local_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_mmar_carl_skill_loader(monkeypatch)
    cfg = MAGEConfig(enable_memory=False, enable_local_skill_catalog=True)
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    agent = CapabilityLookupAgent(mgr, cfg)
    ctx = await agent.lookup()
    assert len(ctx.agent_skills) == 1
    s = ctx.agent_skills[0]
    assert s.name == "local-extractor"
    assert s.source == "local"
    assert s.sha256 == "local-sha"


@pytest.mark.asyncio
async def test_lookup_dedupes_memory_and_local_by_sha256(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skill installed locally AND promoted to memory appears once."""
    _install_fake_gigaevo_client(monkeypatch)

    class _LoaderSharedSHA:
        def catalog_all(self, payloads: bool = False) -> list[dict[str, Any]]:
            return [
                {
                    "name": "pdf-local-copy",
                    "uri": "local:///opt/skills/pdf",
                    "sha256": "shared-sha",  # same as memory entry
                }
            ]

    pkg = types.ModuleType("mmar_carl")
    sl_mod = types.ModuleType("mmar_carl.skill_loader")
    sl_mod.SkillLoader = _LoaderSharedSHA  # type: ignore[attr-defined]
    pkg.skill_loader = sl_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mmar_carl", pkg)
    monkeypatch.setitem(sys.modules, "mmar_carl.skill_loader", sl_mod)

    cfg = MAGEConfig(
        enable_memory=False,
        enable_memory_skill_lookup=True,
        enable_local_skill_catalog=True,
    )
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    agent = CapabilityLookupAgent(mgr, cfg)
    ctx = await agent.lookup(
        query="Read this PDF",
        domain_analysis=DomainAnalysis(domain="documents", key_concepts=["pdf"]),
    )
    # Exactly one entry, and memory wins (richer metadata).
    assert len(ctx.agent_skills) == 1
    assert ctx.agent_skills[0].source == "memory"
    assert ctx.agent_skills[0].sha256 == "shared-sha"


# ---------------------------------------------------------------------------
# CapabilityLookupAgent — folded SkillDiscoveryAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_invokes_skill_discovery_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MAGEConfig(enable_memory=False, enable_skill_discovery=True)
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    # Stub the SkillDiscoveryAgent class as imported inside the agent
    class _FakeDiscoverer:
        def __init__(self, llm: Any, web_agent: Any = None) -> None:
            pass

        async def discover(
            self,
            query: str,
            domain_analysis: DomainAnalysis,
            config: MAGEConfig,
            *,
            max_recommendations: int = 3,
            enable_web_discovery: bool = False,
        ) -> SkillSearchResult:
            return SkillSearchResult(
                recommendations=[
                    SkillRecommendation(
                        name="pdf",
                        uri="github://anthropics/skills/pdf",
                        description="PDF helper",
                        why="task mentions PDFs",
                        relevance=0.92,
                        source="registry",
                    )
                ]
            )

    monkeypatch.setattr("mmar_mage.skills.discovery.SkillDiscoveryAgent", _FakeDiscoverer)

    # Pass a sentinel llm (just needs to be non-None)
    agent = CapabilityLookupAgent(mgr, cfg, llm=MagicMock())
    ctx = await agent.lookup(
        query="Extract data from a PDF",
        domain_analysis=DomainAnalysis(domain="documents", key_concepts=["pdf"]),
    )
    assert len(ctx.agent_skills) == 1
    s = ctx.agent_skills[0]
    assert s.source == "discovered"
    assert s.relevance == pytest.approx(0.92)
    assert s.why == "task mentions PDFs"


@pytest.mark.asyncio
async def test_lookup_skips_skill_discovery_when_llm_missing() -> None:
    """No LLM means no SkillDiscoveryAgent — but lookup still works."""
    cfg = MAGEConfig(enable_memory=False, enable_skill_discovery=True)
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    agent = CapabilityLookupAgent(mgr, cfg)  # no llm=
    ctx = await agent.lookup(
        query="Extract a PDF",
        domain_analysis=DomainAnalysis(domain="documents"),
    )
    assert ctx.agent_skills == []


@pytest.mark.asyncio
async def test_lookup_swallows_discovery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MAGEConfig(enable_memory=False, enable_skill_discovery=True)
    mgr = MemoryManager(cfg)

    async def fake_get(environment_id: str = "default") -> None:
        return None

    mgr.get_capabilities = fake_get  # type: ignore[method-assign]

    class _ExplodingDiscoverer:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def discover(self, *a: Any, **kw: Any) -> SkillSearchResult:
            raise RuntimeError("LLM down")

    monkeypatch.setattr("mmar_mage.skills.discovery.SkillDiscoveryAgent", _ExplodingDiscoverer)

    agent = CapabilityLookupAgent(mgr, cfg, llm=MagicMock())
    # No raise.
    ctx = await agent.lookup(
        query="x",
        domain_analysis=DomainAnalysis(),
    )
    assert ctx.agent_skills == []


# ---------------------------------------------------------------------------
# Generator integration — only ONE capability_lookup stage now
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_emits_capability_lookup_not_separate_skill_discovery(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enable_skill_discovery=True, only `capability_lookup` stage fires.

    Pre-§2-P0, both `capability_lookup` and `skill_discovery` would
    appear in `deep_stages_completed`. Now `skill_discovery` is folded
    into `capability_lookup` and only the latter shows up.
    """

    # Stub SkillDiscoveryAgent so it doesn't try to call the LLM
    class _NoOpDiscoverer:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def discover(self, *a: Any, **kw: Any) -> SkillSearchResult:
            return SkillSearchResult()

    monkeypatch.setattr("mmar_mage.skills.discovery.SkillDiscoveryAgent", _NoOpDiscoverer)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_STEP_PLAN)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_skill_discovery = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a quarterly PDF report")

    stages = result.metadata.deep_stages_completed or []
    assert "capability_lookup" in stages
    assert "skill_discovery" not in stages  # standalone stage gone


@pytest.mark.asyncio
async def test_generator_passes_unified_skill_digest_to_planner(
    deep_config: MAGEConfig, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single skills digest reaches StepPlanner via skill_digest=..."""

    class _OneRecDiscoverer:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def discover(self, *a: Any, **kw: Any) -> SkillSearchResult:
            return SkillSearchResult(
                recommendations=[
                    SkillRecommendation(
                        name="pdf",
                        uri="github://anthropics/skills/pdf",
                        description="PDF helper",
                        why="task mentions PDFs",
                        relevance=0.85,
                        source="registry",
                    )
                ]
            )

    monkeypatch.setattr("mmar_mage.skills.discovery.SkillDiscoveryAgent", _OneRecDiscoverer)

    # Capture the StepPlanner.plan call args
    captured: dict[str, Any] = {}

    async def fake_plan(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured["skill_digest"] = kwargs.get("skill_digest", "")
        # Build a minimal StepPlan result
        from mmar_mage.schemas import PlannedStep, StepPlan

        return StepPlan(
            steps=[
                PlannedStep(number=1, title="A", step_type="llm", rough_aim="x"),
                PlannedStep(number=2, title="B", step_type="llm", rough_aim="y"),
                PlannedStep(number=3, title="C", step_type="llm", rough_aim="z"),
            ],
            rationale="ok",
        )

    monkeypatch.setattr("mmar_mage.agents.StepPlanner.plan", fake_plan)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_openai_response(json.dumps(SAMPLE_DOMAIN_ANALYSIS)),
            _make_openai_response(json.dumps(SAMPLE_DAG)),
            _make_openai_response(json.dumps(SAMPLE_DESCRIBED_STEPS)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    deep_config.enable_skill_discovery = True
    deep_config.enable_suggested_naming_llm = False

    gen = MAGEGenerator(config=deep_config)
    await gen.generate("Extract data from a PDF")

    digest = captured["skill_digest"]
    assert "pdf" in digest
    assert "github://anthropics/skills/pdf" in digest
    assert "0.85" in digest
