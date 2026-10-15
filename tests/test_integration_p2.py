"""Integration tests for §9 P2 — three end-to-end scenarios.

Each test drives `MAGEGenerator` end-to-end and asserts the **outcome**
(not just wiring) for one §9 P2 spec item:

1. **Capability-aware planning** — with `CapabilityContext` carrying a
   `pdf` skill, the planner emits an `agent_skill` step for a
   "extract data from a PDF" query.
2. **Streaming** — `on_llm_chunk` fires with non-empty deltas when the
   underlying API returns `stream=True` chunks.
3. **Cancellation** — `cancel.set()` mid-stage halts within one
   LLM-call worth of latency (i.e. no further LLM calls happen once
   the event is observed).

These are intentionally separate from the existing unit / wiring
tests in `test_capability_*.py`, `test_streaming_progress.py`,
`test_cancel_retry.py`. They verify the **integration contract** the
TODO spec describes, not just per-component behaviour.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.capability_lookup_agent import (
    AgentSkillEntry,
    CapabilityContext,
)
from mmar_mage.exceptions import MAGECancelledError
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

# ---------------------------------------------------------------------------
# Shared deep-mode fixtures
# ---------------------------------------------------------------------------


SAMPLE_DOMAIN = {
    "domain": "documents",
    "task_type": "extraction",
    "complexity": "medium",
    "key_concepts": ["pdf", "tables"],
    "suggested_step_count": 2,
    "preferred_step_types": [],
}


def _make_resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    r = MagicMock()
    r.choices = [choice]
    return r


def _make_chunk(content: str | None) -> Any:
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


class _AsyncChunkIterator:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self._i = 0

    def __aiter__(self) -> _AsyncChunkIterator:
        return self

    async def __anext__(self) -> Any:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


# ---------------------------------------------------------------------------
# §9 P2 / 1 — capability-aware planning (pdf skill → agent_skill step)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capability_aware_planning_emits_agent_skill_for_pdf_query(
    mocker: Any,
) -> None:
    """With a PDF skill in `CapabilityContext`, the planner's response
    (mocked) selects `agent_skill` for a PDF-extraction sub-goal and that
    selection flows through to the final chain."""
    plan_with_skill = {
        "steps": [
            {
                "number": 1,
                "title": "Extract tables from PDF",
                "step_type": "agent_skill",
                "rough_aim": "Use the PDF skill to pull tables out of the input file",
            },
            {
                "number": 2,
                "title": "Summarize",
                "step_type": "llm",
                "rough_aim": "Summarize the extracted tables",
            },
        ],
        "rationale": "Use the available pdf skill; LLM only for final synthesis.",
    }
    dag = {
        "steps": [
            {
                "number": 1,
                "title": "Extract tables from PDF",
                "step_type": "agent_skill",
                "dependencies": [],
                "rough_aim": "x",
            },
            {
                "number": 2,
                "title": "Summarize",
                "step_type": "llm",
                "dependencies": [1],
                "rough_aim": "y",
            },
        ],
        "max_workers": 1,
        "pattern": "pipeline",
    }
    described = {
        "steps": [
            {
                "step_type": "agent_skill",
                "number": 1,
                "title": "Extract tables",
                "aim": "Run the pdf skill on the input file",
                "reasoning_questions": "?",
                "step_context_queries": ["pdf"],
                "stage_action": "extract",
                "example_reasoning": "y",
                "skill_name": "pdf",
                "skill": "github://anthropics/skills/pdf",
                "dependencies": [],
                "retry_max": 3,
            },
            {
                "step_type": "llm",
                "number": 2,
                "title": "Summarize",
                "aim": "Summarize the table contents",
                "reasoning_questions": "what?",
                "step_context_queries": ["summary"],
                "stage_action": "summarize",
                "example_reasoning": "z",
                "dependencies": [1],
                "retry_max": 3,
            },
        ]
    }

    captured_planner_kwargs: dict[str, Any] = {}

    async def fake_plan(self: Any, query: str, domain: Any, **kwargs: Any) -> Any:
        captured_planner_kwargs.update(kwargs)
        from mmar_mage.schemas import StepPlan

        return StepPlan.model_validate(plan_with_skill)

    mocker.patch("mmar_mage.agents.StepPlanner.plan", fake_plan)

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(
        side_effect=[
            _make_resp(json.dumps(SAMPLE_DOMAIN)),
            _make_resp(json.dumps(dag)),
            _make_resp(json.dumps(described)),
        ]
    )
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(
        mode="deep",
        model="test",
        api_key="x",
        enable_memory=False,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    injected_caps = CapabilityContext(
        agent_skills=[
            AgentSkillEntry(
                name="pdf",
                description="Extract tables and text from PDF documents",
                uri="github://anthropics/skills/pdf",
                tags=["documents"],
                relevance=0.92,
                why="task mentions PDFs",
                source="discovered",
            )
        ]
    )

    gen = MAGEGenerator(config=cfg)
    result = await gen.generate(
        "Extract data from a PDF report",
        capabilities=injected_caps,
    )

    # 1. The planner saw the skill in the digest
    skill_digest = captured_planner_kwargs.get("skill_digest", "")
    assert "pdf" in skill_digest
    assert "github://anthropics/skills/pdf" in skill_digest

    # 2. The final chain contains an `agent_skill` step (MAGE keeps its
    # canonical nested form; agent-features types are not downgraded on export).
    step_types = [s["step_type"] for s in result.chain_dict["steps"]]
    assert "agent_skill" in step_types

    # 3. That step references the PDF skill URI
    skill_step = next(s for s in result.chain_dict["steps"] if s["step_type"] == "agent_skill")
    # `skill` field survives `to_carl_compat` either flat or under `step_config`
    skill_uri = skill_step.get("skill") or (skill_step.get("step_config") or {}).get("skill")
    assert "pdf" in (skill_uri or "").lower()


# ---------------------------------------------------------------------------
# §9 P2 / 2 — streaming integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_fires_on_llm_chunk_with_nonempty_deltas(mocker: Any) -> None:
    """Drive a full deep-mode generation with `enable_streaming=True` and
    a stream-capable mock; assert `on_llm_chunk` fires with every delta
    for the streamed StepDescriber stage."""
    domain_response = _make_resp(json.dumps(SAMPLE_DOMAIN))
    plan_response = _make_resp(
        json.dumps(
            {
                "steps": [
                    {"number": 1, "title": "Read", "step_type": "llm", "rough_aim": "x"},
                    {"number": 2, "title": "Synth", "step_type": "llm", "rough_aim": "y"},
                ],
                "rationale": "ok",
            }
        )
    )
    dag_response = _make_resp(
        json.dumps(
            {
                "steps": [
                    {"number": 1, "title": "Read", "step_type": "llm", "dependencies": [], "rough_aim": "x"},
                    {"number": 2, "title": "Synth", "step_type": "llm", "dependencies": [1], "rough_aim": "y"},
                ],
                "max_workers": 1,
                "pattern": "pipeline",
            }
        )
    )
    described_full = json.dumps(
        {
            "steps": [
                {
                    "step_type": "llm",
                    "number": i,
                    "title": f"Step {i}",
                    "aim": f"do {i}",
                    "reasoning_questions": "?",
                    "step_context_queries": ["x"],
                    "stage_action": "do",
                    "example_reasoning": "y",
                    "dependencies": [] if i == 1 else [1],
                    "retry_max": 3,
                }
                for i in (1, 2)
            ]
        }
    )

    # 3 stream chunks for the StepDescriber response — each non-empty.
    third = len(described_full) // 3
    streamed_chunks = [
        _make_chunk(described_full[:third]),
        _make_chunk(described_full[third : 2 * third]),
        _make_chunk(described_full[2 * third :]),
    ]

    seq_state = {"i": 0}
    non_stream_responses = [domain_response, plan_response, dag_response]

    async def fake_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return _AsyncChunkIterator(streamed_chunks)
        idx = seq_state["i"]
        seq_state["i"] += 1
        return non_stream_responses[idx]

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    class _RecordingCb:
        def __init__(self) -> None:
            self.chunks: list[tuple[str, str]] = []

        def on_stage_start(self, stage: str) -> None:
            pass

        def on_stage_complete(self, stage: str, result: Any) -> None:
            pass

        def on_error(self, stage: str, error: Exception) -> None:
            pass

        def on_llm_chunk(self, stage: str, delta: str) -> None:
            self.chunks.append((stage, delta))

    cb = _RecordingCb()
    cfg = MAGEConfig(
        mode="deep",
        model="test",
        api_key="x",
        enable_memory=False,
        enable_streaming=True,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    gen = MAGEGenerator(config=cfg, progress=cb)  # type: ignore[arg-type]
    await gen.generate("Analyze something")

    # `on_llm_chunk` fired 3 times, all from step_describing, with non-empty deltas.
    assert len(cb.chunks) == 3
    assert {stage for stage, _ in cb.chunks} == {"step_describing"}
    assert all(delta for _, delta in cb.chunks)

    # The reassembled stream content reproduces the StepDescriber JSON
    # response — proving deltas weren't lost.
    reassembled = "".join(delta for _, delta in cb.chunks)
    assert reassembled == described_full


# ---------------------------------------------------------------------------
# §9 P2 / 3 — cancellation mid-stage stops within one LLM call's latency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_mid_stage_stops_within_one_llm_call(mocker: Any) -> None:
    """Set the cancel event while the FIRST LLM call is in flight. The
    current call completes, but the next stage's `_check_cancel` fires
    immediately — no further LLM calls happen."""
    cancel = asyncio.Event()

    n_calls = 0

    async def fake_create(**kwargs: Any) -> Any:
        nonlocal n_calls
        n_calls += 1
        # First LLM call: simulate the user pressing Esc DURING this call.
        if n_calls == 1:
            # Fire cancel while we're "in" this LLM call.
            cancel.set()
            return _make_resp(json.dumps(SAMPLE_DOMAIN))
        # Should NEVER reach the second LLM call.
        return _make_resp(json.dumps({"steps": [], "rationale": "should not see"}))

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cfg = MAGEConfig(
        mode="deep",
        model="test",
        api_key="x",
        enable_memory=False,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    gen = MAGEGenerator(config=cfg)

    t0 = time.monotonic()
    with pytest.raises(MAGECancelledError) as exc_info:
        await gen.generate("Analyze quarterly", cancel=cancel)
    elapsed = time.monotonic() - t0

    # The cancel was observed at the next stage's pre-check.
    assert exc_info.value.stage == "step_planning"
    # Exactly ONE LLM call ran — the one in flight when cancel was set;
    # the next stage aborted before its LLM call.
    assert n_calls == 1
    # Stops well within one LLM-call worth of latency (mocks return
    # instantly, so this is effectively zero; bound it loosely).
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_cancel_set_before_any_stage_aborts_with_zero_llm_calls(
    mocker: Any,
) -> None:
    """A pre-set cancel never invokes the LLM at all — startup check fires."""
    n_calls = 0

    async def fake_create(**kwargs: Any) -> Any:
        nonlocal n_calls
        n_calls += 1
        return _make_resp(json.dumps(SAMPLE_DOMAIN))

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=fake_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    cancel = asyncio.Event()
    cancel.set()

    cfg = MAGEConfig(
        mode="deep",
        model="test",
        api_key="x",
        enable_memory=False,
        enable_suggested_naming_llm=False,
        max_retries=0,
    )
    gen = MAGEGenerator(config=cfg)
    with pytest.raises(MAGECancelledError) as exc_info:
        await gen.generate("anything", cancel=cancel)
    assert exc_info.value.stage == "startup"
    assert n_calls == 0
