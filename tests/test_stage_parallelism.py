"""T43 — research stages run concurrently with topology selection (overlap)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.capability_lookup_agent import CapabilityContext
from mmar_mage.agents.feedback_recall_agent import FeedbackDigest
from mmar_mage.generator import MAGEGenerator, _ResearchOutcome
from mmar_mage.schemas import MAGEConfig, MemoryResearchResult, WebResearchResult


def _empty_research() -> _ResearchOutcome:
    return _ResearchOutcome(
        memory_result=MemoryResearchResult(),
        web_result=WebResearchResult(),
        capability_ctx=CapabilityContext(),
        template_skel=None,
        feedback_digest=FeedbackDigest(),
        memory_hits_used=0,
        web_results_used=0,
        was_cold_start=True,
        feedback_recalled=None,
    )


def _resp(content: dict[str, Any]) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    r = MagicMock()
    r.choices = [choice]
    return r


_DOMAIN = {"domain": "general", "task_type": "qa", "complexity": "medium", "key_concepts": [],
           "suggested_step_count": 2, "preferred_step_types": ["llm", "llm"]}
_PLAN = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "rough_aim": "a"},
                   {"number": 2, "title": "B", "step_type": "llm", "rough_aim": "b"}], "rationale": "x"}
_DAG = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "dependencies": [], "rough_aim": "a"},
                  {"number": 2, "title": "B", "step_type": "llm", "dependencies": [1], "rough_aim": "b"}],
        "max_workers": 1, "pattern": "pipeline"}
_DESCRIBED = {"steps": [{"step_type": "llm", "number": 1, "title": "A", "aim": "a", "dependencies": []},
                        {"step_type": "llm", "number": 2, "title": "B", "aim": "b", "dependencies": [1]}]}


async def test_research_overlaps_topology(mocker: Any) -> None:
    instance = mocker.patch("mmar_mage.llm.AsyncOpenAI").return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[_resp(_DOMAIN), _resp(_PLAN), _resp(_DAG), _resp(_DESCRIBED)]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    research_started = asyncio.Event()
    events: list[str] = []

    async def fake_research(self: Any, *a: Any, **k: Any) -> _ResearchOutcome:
        events.append("research_start")
        research_started.set()
        await asyncio.sleep(0)
        return _empty_research()

    async def fake_select(self: Any, *a: Any, **k: Any) -> Any:
        events.append("topology_start")
        # If research truly overlaps it has started concurrently and this resolves;
        # a sequential pipeline (research after topology) would block here → timeout.
        await asyncio.wait_for(research_started.wait(), timeout=2.0)
        events.append("topology_saw_research_in_flight")
        return SimpleNamespace(best=None, candidates=[])

    mocker.patch.object(MAGEGenerator, "_run_research_stages", fake_research)
    mocker.patch("mmar_mage.generator.TopologySelectorAgent.select", fake_select)

    gen = MAGEGenerator(config=MAGEConfig(api_key="test", mode="deep", enable_topology_selection=True))
    result = await gen.generate("compose a multi-step thing")

    assert result.chain_dict["steps"]  # the pipeline still completed end-to-end
    assert "topology_saw_research_in_flight" in events  # research was in-flight while topology ran
    assert events.index("research_start") < events.index("topology_saw_research_in_flight")
