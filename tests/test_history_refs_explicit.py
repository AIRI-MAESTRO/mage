"""P-6 — prompts prescribe explicit $history[<dep number>] over positional $history[-1]."""

from __future__ import annotations

from mmar_mage.prompts import (
    CHAIN_EDIT_PROMPT,
    LLM_REPAIR_PROMPT,
    STEP_DESCRIBER_TYPE_SECTIONS,
    build_step_describer_prompt,
)


def test_type_sections_prescribe_explicit_history_number() -> None:
    tool = STEP_DESCRIBER_TYPE_SECTIONS["tool"]
    assert "$history[2]" in tool  # explicit-number example
    assert "fallback" in tool  # -1 demoted to fallback

    so = STEP_DESCRIBER_TYPE_SECTIONS["structured_output"]
    assert "$history[2]" in so
    assert "fallback" in so

    skill = STEP_DESCRIBER_TYPE_SECTIONS["agent_skill"]
    assert "$history[<dependency step number>]" in skill
    assert "$history[-1]" not in skill  # the bare positional default is gone


def test_describer_base_prompt_has_history_rule() -> None:
    p = build_step_describer_prompt({"tool"})
    assert "History references" in p
    assert "defeat cache reuse" in p


def test_edit_and_repair_prompts_have_history_rule() -> None:
    for prompt in (CHAIN_EDIT_PROMPT, LLM_REPAIR_PROMPT):
        assert "$history[-1]" in prompt  # named as the thing to avoid
        assert "$history[<number" in prompt  # explicit form prescribed
        assert "fallback" in prompt
