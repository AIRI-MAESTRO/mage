"""WP-4 — research stages 0a–0e run concurrently, not sequentially."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mmar_mage.agents import MemoryResearchAgent, WebResearchAgent
from mmar_mage.agents.capability_lookup_agent import CapabilityContext, CapabilityLookupAgent
from mmar_mage.agents.feedback_recall_agent import FeedbackDigest, FeedbackRecallAgent
from mmar_mage.agents.template_assembler_agent import TemplateAssemblerAgent
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import DomainAnalysis, MAGEConfig, MemoryResearchResult, WebResearchResult


async def test_research_stages_run_concurrently(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    config = MAGEConfig(
        mode="deep",
        api_key="test",
        enable_memory_research=True,
        enable_web_research=True,
        enable_capability_lookup=True,
        enable_template_assembly=True,
        enable_feedback_loop=True,
    )
    gen = MAGEGenerator(config=config)

    async def _mem(*_a: Any, **_k: Any) -> MemoryResearchResult:
        await asyncio.sleep(0.1)
        return MemoryResearchResult()

    async def _web(*_a: Any, **_k: Any) -> WebResearchResult:
        await asyncio.sleep(0.1)
        return WebResearchResult()

    async def _cap(*_a: Any, **_k: Any) -> CapabilityContext:
        await asyncio.sleep(0.1)
        return CapabilityContext()

    async def _tpl(*_a: Any, **_k: Any) -> None:
        await asyncio.sleep(0.1)
        return None  # assemble legitimately returns None — the stage still counts

    async def _fb(*_a: Any, **_k: Any) -> FeedbackDigest:
        await asyncio.sleep(0.1)
        return FeedbackDigest()

    mocker.patch.object(MemoryResearchAgent, "research", _mem)
    mocker.patch.object(WebResearchAgent, "research", _web)
    mocker.patch.object(CapabilityLookupAgent, "lookup", _cap)
    mocker.patch.object(TemplateAssemblerAgent, "assemble", _tpl)
    mocker.patch.object(FeedbackRecallAgent, "recall", _fb)

    stages: list[str] = []
    da = DomainAnalysis(domain="d", task_type="analysis", complexity="medium")
    t0 = time.perf_counter()
    out = await gen._run_research_stages("q", da, capabilities=None, plan_will_run=True, stages_completed=stages)
    elapsed = time.perf_counter() - t0

    # 5 × 0.1s sequential = 0.5s; concurrent ≈ 0.1s.
    assert elapsed < 0.25, f"research stages ran sequentially: {elapsed:.3f}s"
    # stages_completed is deterministic regardless of completion order
    assert stages == [
        "memory_research",
        "web_research",
        "capability_lookup",
        "template_assembly",
        "feedback_recall",
    ]
    assert isinstance(out.memory_result, MemoryResearchResult)
    assert out.template_skel is None


async def test_injected_capabilities_bypass_gather(mocker: Any) -> None:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    gen = MAGEGenerator(config=MAGEConfig(mode="deep", api_key="test", enable_capability_lookup=True))
    stages: list[str] = []
    injected = CapabilityContext()
    out = await gen._run_research_stages(
        "q", DomainAnalysis(domain="d"), capabilities=injected, plan_will_run=True, stages_completed=stages
    )
    assert out.capability_ctx is injected
    assert "capability_lookup_injected" in stages
    assert "capability_lookup" not in stages  # the computed path did not run
