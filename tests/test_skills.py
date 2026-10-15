"""Tests for skill discovery and registry."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage import (
    DomainAnalysis,
    KnownSkill,
    MAGEConfig,
    SkillDiscoveryAgent,
    SkillRegistry,
    default_registry,
)
from mmar_mage.skills.discovery import _extract_skill_uris, _name_from_uri


def _resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestSkillRegistry:
    def test_default_registry_has_official_skills(self) -> None:
        names = {s.name for s in default_registry}
        # The Anthropic Skills repo official set is the seed.
        assert {"pdf", "docx", "pptx", "xlsx"}.issubset(names)

    def test_get_by_name(self) -> None:
        pdf = default_registry.get("pdf")
        assert pdf is not None
        assert pdf.uri.startswith("github://anthropics/skills/")

    def test_add_replaces_existing(self) -> None:
        reg = SkillRegistry(
            skills=[
                KnownSkill(name="x", description="old", uri="local:///old"),
            ]
        )
        reg.add(KnownSkill(name="x", description="new", uri="local:///new"))
        assert len(reg) == 1
        assert reg.get("x").uri == "local:///new"  # type: ignore[union-attr]

    def test_by_domain_filter(self) -> None:
        doc_skills = default_registry.by_domain("documents")
        names = {s.name for s in doc_skills}
        assert "pdf" in names
        assert "docx" in names

    def test_substring_search_finds_pdf(self) -> None:
        results = default_registry.search_substring("extract tables from pdf")
        assert len(results) > 0
        assert results[0].name == "pdf"  # highest-scoring match should be pdf

    def test_substring_search_no_match_returns_empty(self) -> None:
        results = default_registry.search_substring("zzzzzzzzz", top_k=5)
        assert results == []

    def test_substring_search_short_query_falls_back_to_whole_string(self) -> None:
        # All tokens < 3 chars, but the full query should still match.
        results = default_registry.search_substring("docx")
        assert len(results) >= 1
        assert any(s.name == "docx" for s in results)


# ---------------------------------------------------------------------------
# URI extraction helpers
# ---------------------------------------------------------------------------


class TestUriExtraction:
    def test_github_scheme(self) -> None:
        text = "Check out github://anthropics/skills/skills/pdf@main for details."
        uris = _extract_skill_uris(text)
        assert "github://anthropics/skills/skills/pdf@main" in uris

    def test_https_normalised_to_github_scheme(self) -> None:
        text = "See https://github.com/anthropics/skills"
        uris = _extract_skill_uris(text)
        # https://github.com/owner/repo → github://owner/repo
        assert any(u.startswith("github://anthropics/skills") for u in uris)

    def test_dedupes(self) -> None:
        text = "github://a/b/c github://a/b/c https://github.com/a/b https://github.com/a/b"
        uris = _extract_skill_uris(text)
        # All four references collapse — github:// and https:// normalise to same scheme.
        assert len(uris) <= 2

    def test_name_from_uri(self) -> None:
        assert _name_from_uri("github://anthropics/skills/skills/pdf@main") == "pdf"
        assert _name_from_uri("local:///abs/path/to/my-skill") == "my-skill"
        assert _name_from_uri("github://a/b") == "b"


# ---------------------------------------------------------------------------
# Discovery agent — gate path
# ---------------------------------------------------------------------------


class TestDiscoveryGate:
    async def test_gate_skips_for_pure_reasoning(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # Gate says no — discovery returns empty without reranking.
        instance.chat.completions.create = AsyncMock(
            return_value=_resp(json.dumps({"needs_skill": False, "reason": "pure reasoning"}))
        )
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="m", api_key="k", enable_memory=False)
        agent = SkillDiscoveryAgent(LLMClient(cfg))

        result = await agent.discover(
            "Prove that the sum of two odd numbers is even.",
            DomainAnalysis(domain="math"),
            cfg,
        )
        assert result.recommendations == []
        # Only one LLM call — the gate. No rerank.
        assert instance.chat.completions.create.await_count == 1

    async def test_gate_triggers_rerank_for_pdf_task(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # 1st call = gate (needs_skill=true), 2nd = rerank picks pdf.
        gate = _resp(json.dumps({"needs_skill": True, "hint_keywords": ["pdf", "extract"]}))
        rerank = _resp(
            json.dumps({"picks": [{"name": "pdf", "why": "PDF extraction is exactly this task", "relevance": 0.95}]})
        )
        instance.chat.completions.create = AsyncMock(side_effect=[gate, rerank])
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="m", api_key="k", enable_memory=False)
        agent = SkillDiscoveryAgent(LLMClient(cfg))
        result = await agent.discover(
            "Extract tables from a PDF document and produce JSON.",
            DomainAnalysis(domain="documents"),
            cfg,
            max_recommendations=3,
        )
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.name == "pdf"
        assert rec.relevance == 0.95
        assert rec.source == "registry"
        assert "PDF" in result.digest

    async def test_gate_failure_returns_empty_gracefully(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        # Gate returns malformed JSON → discover returns empty without crashing.
        instance.chat.completions.create = AsyncMock(return_value=_resp("not-json"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="m", api_key="k", enable_memory=False)
        agent = SkillDiscoveryAgent(LLMClient(cfg))
        result = await agent.discover("any task", DomainAnalysis(), cfg)
        assert result.recommendations == []

    async def test_rerank_returns_no_picks_when_nothing_matches(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        gate = _resp(json.dumps({"needs_skill": True, "hint_keywords": ["nothingrelevant"]}))
        rerank = _resp(json.dumps({"picks": []}))
        instance.chat.completions.create = AsyncMock(side_effect=[gate, rerank])
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="m", api_key="k", enable_memory=False)
        agent = SkillDiscoveryAgent(LLMClient(cfg))
        result = await agent.discover("obscure task", DomainAnalysis(), cfg)
        assert result.recommendations == []
        # rerank was called → candidates were considered
        assert result.candidates_considered > 0


# ---------------------------------------------------------------------------
# Discovery agent — web fallback
# ---------------------------------------------------------------------------


class TestDiscoveryWebFallback:
    async def test_web_fallback_used_when_registry_weak(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value

        gate = _resp(json.dumps({"needs_skill": True, "hint_keywords": ["custom"]}))
        # Registry rerank returns nothing matching well.
        empty_rerank = _resp(json.dumps({"picks": []}))
        # Web rerank returns a candidate from the discovered URI.
        web_rerank = _resp(
            json.dumps({"picks": [{"name": "custom-skill", "why": "Match found on github", "relevance": 0.8}]})
        )
        # WebResearchAgent calls happen in between — sub-query gen + 1 web call + digest summarize.
        sub_queries = _resp(json.dumps({"sub_queries": ["custom skill site:github.com"]}))
        web_digest = _resp(
            json.dumps({"digest": "Found a relevant skill: github://owner/custom-skill@main with custom capabilities."})
        )

        instance.chat.completions.create = AsyncMock(
            side_effect=[gate, empty_rerank, sub_queries, web_digest, web_rerank]
        )
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.agents.web_research_agent import WebResearchAgent
        from mmar_mage.llm import LLMClient

        # Patch _search_web to avoid a real HTTP call.
        async def _fake_search(self, query, *, provider, api_key, max_results):  # noqa: ARG001
            return [{"title": "custom-skill", "url": "https://github.com/owner/custom-skill", "content": "..."}]

        mocker.patch.object(WebResearchAgent, "_search_web", _fake_search)

        cfg = MAGEConfig(
            mode="fast",
            model="m",
            api_key="k",
            enable_memory=False,
            web_search_api_key="dummy-key",
            web_min_sourced_claims=0,  # this test exercises the skill web-fallback, not the P1.2 web second wave
        )
        llm = LLMClient(cfg)
        agent = SkillDiscoveryAgent(llm, web_agent=WebResearchAgent(llm))
        result = await agent.discover(
            "do something obscure",
            DomainAnalysis(domain="other"),
            cfg,
            enable_web_discovery=True,
        )
        assert result.used_web_fallback is True
        assert len(result.recommendations) >= 1
        assert any(r.source == "web" for r in result.recommendations)

    async def test_web_discovered_skill_is_cached_for_reuse(self) -> None:
        """P3.8 — a web-discovered skill is cached into the registry, so the
        second discovery of the same need is a registry hit (no web call)."""
        from unittest.mock import AsyncMock, MagicMock

        from mmar_mage.skills.discovery import SkillRecommendation
        from mmar_mage.skills.registry import KnownSkill

        # A non-empty registry (an empty one is falsy → falls back to the
        # builtins) whose lone skill never matches the query.
        reg = SkillRegistry(
            skills=[
                KnownSkill(
                    name="zzz_filler", description="unrelated filler",
                    capabilities=["filler"], uri="local://filler", source="builtin",
                )
            ]
        )
        agent = SkillDiscoveryAgent(MagicMock(), registry=reg, web_agent=object())
        agent._gate_relevance = AsyncMock(
            return_value={"needs_skill": True, "hint_keywords": ["custom", "obscure"]},
        )

        async def fake_rerank(query: str, candidates: list) -> list:  # noqa: ARG001
            # Only the cached web skill is a real match (the filler never is).
            return [
                {"name": c.name, "why": "m", "relevance": 0.9}
                for c in candidates
                if c.name == "custom-skill"
            ]

        agent._rerank = fake_rerank
        web_calls: list[int] = []

        async def fake_web(query, domain, hints, config):  # noqa: ANN001, ARG001
            web_calls.append(1)
            return [
                SkillRecommendation(
                    name="custom-skill",
                    uri="github://owner/custom-skill@main",
                    description="Handles obscure custom tasks",
                    relevance=0.8,
                    source="web",
                )
            ]

        agent._discover_via_web = fake_web
        cfg = MAGEConfig(
            mode="fast", model="m", api_key="k", enable_memory=False,
            web_search_api_key="dummy-key",
        )
        domain = DomainAnalysis(domain="other")

        # 1st discovery — empty registry → web fallback → caches the skill.
        r1 = await agent.discover(
            "do something obscure custom", domain, cfg, enable_web_discovery=True,
        )
        assert r1.used_web_fallback is True
        assert len(web_calls) == 1
        cached = reg.get("custom-skill")
        assert cached is not None and cached.source == "discovered"

        # 2nd discovery (same need) — registry hit, NO second web call.
        r2 = await agent.discover(
            "do something obscure custom", domain, cfg, enable_web_discovery=True,
        )
        assert len(web_calls) == 1  # cache hit — web fallback not re-run
        assert r2.used_web_fallback is False
        assert any(rec.name == "custom-skill" for rec in r2.recommendations)

    async def test_web_disabled_when_flag_off(self, mocker: Any) -> None:
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        gate = _resp(json.dumps({"needs_skill": True, "hint_keywords": []}))
        empty_rerank = _resp(json.dumps({"picks": []}))
        instance.chat.completions.create = AsyncMock(side_effect=[gate, empty_rerank])
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        from mmar_mage.agents.web_research_agent import WebResearchAgent
        from mmar_mage.llm import LLMClient

        cfg = MAGEConfig(mode="fast", model="m", api_key="k", enable_memory=False, web_search_api_key="x")
        llm = LLMClient(cfg)
        agent = SkillDiscoveryAgent(llm, web_agent=WebResearchAgent(llm))
        result = await agent.discover("task", DomainAnalysis(), cfg, enable_web_discovery=False)
        # Web search should NOT have been called — only 2 LLM calls.
        assert instance.chat.completions.create.await_count == 2
        assert result.used_web_fallback is False


# ---------------------------------------------------------------------------
# P6.2 — always-on FREE keyword surfacing (push)
# ---------------------------------------------------------------------------


class TestSurfaceSkills:
    def test_surface_skills_keyword_no_llm(self) -> None:
        from mmar_mage.skills import surface_skills

        # PURE keyword surfacing — takes NO LLM, hits the registry only.
        hits = surface_skills("extract tables from a pdf and merge pdfs")
        assert "pdf" in {s.name for s in hits}

    def test_surface_skills_uses_domain_concepts(self) -> None:
        from mmar_mage.schemas import DomainAnalysis
        from mmar_mage.skills import surface_skills

        dom = DomainAnalysis(domain="presentations", key_concepts=["slides", "deck"])
        hits = surface_skills("make it", dom)  # concepts carry the match
        assert "pptx" in {s.name for s in hits}

    def test_surface_skills_empty_query(self) -> None:
        from mmar_mage.skills import surface_skills

        assert surface_skills("") == []

    def test_format_surfaced_skills_renders_agent_skill_block(self) -> None:
        from mmar_mage.skills import format_surfaced_skills, surface_skills

        block = format_surfaced_skills(surface_skills("extract text from a pdf"))
        assert "agent_skill" in block
        assert "skill=github://" in block
        assert format_surfaced_skills([]) == ""


# ---------------------------------------------------------------------------
# P6.2 — needs-analysis (pull): the model declares structured requirements
# ---------------------------------------------------------------------------


class TestNeedsAnalysis:
    async def test_agent_emits_structured_requirements(self) -> None:
        from mmar_mage.agents.needs_analysis import NeedsAnalysisAgent
        from mmar_mage.schemas import NeedsAnalysis

        llm = MagicMock()
        llm.generate_structured = AsyncMock(
            return_value=NeedsAnalysis(
                needs_skills=["pptx", "pdf"],
                needs_services=["gmail"],
                needs_from_user=["target language"],
                needs_credentials=["GMAIL_TOKEN"],
            )
        )
        out = await NeedsAnalysisAgent(llm).analyze(
            "build a deck from my pdf and email it",
        )
        assert out.needs_skills == ["pptx", "pdf"]
        assert out.needs_services == ["gmail"]
        assert out.needs_from_user == ["target language"]
        assert out.needs_credentials == ["GMAIL_TOKEN"]
        # used generate_structured against the NeedsAnalysis model
        assert llm.generate_structured.await_args.kwargs["response_model"] is NeedsAnalysis
