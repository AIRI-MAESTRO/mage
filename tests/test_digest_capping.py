"""Tests for research-digest length capping in the StepPlanner (plan item P1.6).

Unbounded memory/web/skill/capability/template digests previously got appended
verbatim to the planner prompt, bloating it and diluting the query. These tests
cover the ``_truncate_digest`` helper and verify the real assembly path inside
``StepPlanner.plan`` caps each digest before it reaches the LLM.
"""

from __future__ import annotations

from mmar_mage.agents.step_planner import DEFAULT_MAX_DIGEST_CHARS, StepPlanner, _truncate_digest
from mmar_mage.schemas import DomainAnalysis, PlannedStep, StepPlan


class _RecordingLLM:
    """Minimal stand-in for LLMClient that records the assembled user prompt."""

    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    async def generate_structured(self, *, system_prompt, user_prompt, response_model, stage):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return StepPlan(steps=[PlannedStep(number=1, title="Step")])


# --------------------------------------------------------------------------- #
# _truncate_digest unit tests
# --------------------------------------------------------------------------- #
def test_short_text_unchanged():
    assert _truncate_digest("short digest", 1200) == "short digest"


def test_empty_text_unchanged():
    assert _truncate_digest("", 1200) == ""


def test_long_text_capped_with_marker():
    text = "x" * 5000
    out = _truncate_digest(text, 1000)
    assert out.endswith("…[truncated]")
    # head is bounded by max_chars; total only exceeds it by the short marker
    assert len(out) <= 1000 + len(" …[truncated]")
    assert len(out) < len(text)


def test_truncation_is_on_word_boundary():
    text = "word " * 200  # 1000 chars of repeated whole words
    out = _truncate_digest(text, 23)
    head = out.split(" …[truncated]")[0]
    # no partial word at the cut: every token is the complete "word"
    assert head.split()[-1] == "word"
    assert "wor…" not in out


def test_zero_or_negative_disables_capping():
    text = "y" * 5000
    assert _truncate_digest(text, 0) == text
    assert _truncate_digest(text, -1) == text


def test_no_whitespace_still_capped():
    # a single giant token (no spaces) must still be bounded, not returned whole
    text = "z" * 5000
    out = _truncate_digest(text, 100)
    assert out.endswith("…[truncated]")
    assert len(out) <= 100 + len(" …[truncated]")


# --------------------------------------------------------------------------- #
# Real assembly path: StepPlanner.plan caps digests before the LLM call
# --------------------------------------------------------------------------- #
def _domain() -> DomainAnalysis:
    return DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")


async def test_planner_caps_oversized_memory_digest():
    llm = _RecordingLLM()
    planner = StepPlanner(llm)
    huge = ("relevant fact " * 1000) + "UNIQUE_TAIL_SENTINEL"
    await planner.plan("q", _domain(), memory_digest=huge, max_digest_chars=200)

    assert llm.user_prompt is not None
    # the elision marker is present and the far tail did not survive
    assert "…[truncated]" in llm.user_prompt
    assert "UNIQUE_TAIL_SENTINEL" not in llm.user_prompt


async def test_planner_caps_every_digest_channel():
    llm = _RecordingLLM()
    planner = StepPlanner(llm)
    big = lambda tag: ("blah " * 1000) + tag  # noqa: E731 - terse test helper
    await planner.plan(
        "q",
        _domain(),
        memory_digest=big("MEM_TAIL"),
        web_digest=big("WEB_TAIL"),
        capability_context=big("CAP_TAIL"),
        template_skeleton=big("TPL_TAIL"),
        skill_digest=big("SKILL_TAIL"),
        max_digest_chars=200,
    )
    full = (llm.system_prompt or "") + (llm.user_prompt or "")
    # none of the far tails survive in either the system or user prompt
    for tail in ("MEM_TAIL", "WEB_TAIL", "CAP_TAIL", "TPL_TAIL", "SKILL_TAIL"):
        assert tail not in full


async def test_planner_disabled_cap_keeps_full_digest():
    llm = _RecordingLLM()
    planner = StepPlanner(llm)
    huge = ("relevant fact " * 1000) + "UNIQUE_TAIL_SENTINEL"
    await planner.plan("q", _domain(), memory_digest=huge, max_digest_chars=0)

    assert llm.user_prompt is not None
    assert "UNIQUE_TAIL_SENTINEL" in llm.user_prompt  # not truncated
    assert "…[truncated]" not in llm.user_prompt


def test_default_cap_constant_matches_config_default():
    from mmar_mage.schemas import MAGEConfig

    assert MAGEConfig().max_digest_chars == DEFAULT_MAX_DIGEST_CHARS
