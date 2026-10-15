"""P-5 — grader/judge anti-keyword-stuffing guards."""

from __future__ import annotations

from mmar_mage.benchmark.schemas import ChainScore, SynthesizedTask
from mmar_mage.evolve.grader import ChainGrader
from mmar_mage.prompts import CHAIN_JUDGE_PROMPT, CHAIN_JUDGE_REAL_OUTPUT_PROMPT

_TASK = SynthesizedTask(task_id="t1", input="analyze the report", rubric=["mentions revenue and EBITDA margin"])


def test_structural_rubric_ignores_stuffing_outside_title_aim() -> None:
    # keywords in title/aim → counted toward rubric coverage
    legit = {
        "steps": [
            {"title": "Analyze revenue", "aim": "Assess EBITDA and margin trends", "example_reasoning": "x"},
            {"title": "Synthesize", "aim": "Write the report", "example_reasoning": "y"},
        ]
    }
    # same keywords ONLY in example_reasoning (title/aim generic) → NOT counted
    stuffed = {
        "steps": [
            {"title": "Step one", "aim": "Do the thing", "example_reasoning": "revenue EBITDA margin " * 3},
            {"title": "Step two", "aim": "Do another thing", "example_reasoning": "revenue EBITDA margin"},
        ]
    }
    assert ChainGrader._structural_grade(legit, _TASK) > ChainGrader._structural_grade(stuffed, _TASK)


def test_suspect_gaming_flags_stuffed_chain() -> None:
    normal = {
        "steps": [
            {"title": "Analyze revenue", "aim": "Assess EBITDA", "example_reasoning": "Look at the income statement."},
            {"title": "Synthesize", "aim": "Write report", "example_reasoning": "Summarize the findings clearly."},
        ]
    }
    assert ChainGrader._suspect_gaming(normal, _TASK) is False

    stuffed = {
        "steps": [
            {
                "title": "x",
                "aim": "y",
                "example_reasoning": "revenue ebitda margin " * 30,
                "step_context_queries": ["revenue", "ebitda", "margin"] * 10,
            }
        ]
    }
    assert ChainGrader._suspect_gaming(stuffed, _TASK) is True


def test_chain_score_has_suspect_gaming_field() -> None:
    assert ChainScore().suspect_gaming is False
    assert ChainScore(suspect_gaming=True).suspect_gaming is True


def test_judge_prompts_warn_against_keyword_stuffing() -> None:
    assert "keyword-stuffed" in CHAIN_JUDGE_PROMPT
    assert "NEGATIVE signal" in CHAIN_JUDGE_PROMPT
    assert "keyword-stuffed" in CHAIN_JUDGE_REAL_OUTPUT_PROMPT
