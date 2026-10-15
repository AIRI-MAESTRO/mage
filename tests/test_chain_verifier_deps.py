"""ChainVerifier independent-answer context includes dependency steps (P1.12).

The CoVe answer pass already runs one independent LLM call per question; this
verifies the improvement that the answer context now also carries the step's
dependency steps, so cross-step questions (dependencies / ordering /
consistency) can actually be judged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.chain_verifier import ChainVerifier, VerificationQuestion
from mmar_mage.schemas import DomainAnalysis

_CHAIN = {
    "name": "c",
    "steps": [
        {"number": 1, "title": "Fetch raw data", "step_type": "tool", "aim": "get data", "dependencies": []},
        {"number": 2, "title": "Analyze the data", "step_type": "llm", "aim": "analyze", "dependencies": [1]},
    ],
}


async def test_answer_context_includes_target_and_dependency_steps():
    q_resp = {"questions": [{"step_number": 2, "question": "Does step 2 correctly use step 1?", "aspect": "logical"}]}
    a_resp = {"answer": "yes", "passed": True, "explanation": "ok"}
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=[q_resp, a_resp])

    verifier = ChainVerifier(llm)
    await verifier.verify("Analyze the data", _CHAIN, DomainAnalysis())

    # call 0 = question generation; call 1 = the independent answer for the question
    assert llm.generate_json.await_count == 2  # one answer call per question (independent)
    answer_prompt = llm.generate_json.call_args_list[1].kwargs["user_prompt"]
    assert "Analyze the data" in answer_prompt  # the step under review
    assert "Fetch raw data" in answer_prompt  # its dependency step (#1) — the P1.12 improvement
    assert "Steps it depends on" in answer_prompt


async def test_answer_question_omits_deps_section_when_no_dependencies():
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value={"answer": "yes", "passed": True})
    verifier = ChainVerifier(llm)

    q = VerificationQuestion(step_number=1, question="Is the aim clear?", aspect="clarity")
    await verifier._answer_question("Q", _CHAIN, q)

    prompt = llm.generate_json.call_args.kwargs["user_prompt"]
    assert "Fetch raw data" in prompt  # the target step itself
    assert "Steps it depends on" not in prompt  # step 1 has no dependencies


async def test_answer_question_is_independent_per_question():
    # two questions → two separate answer calls (no batching)
    q_resp = {
        "questions": [
            {"step_number": 1, "question": "Q1?", "aspect": "factual"},
            {"step_number": 2, "question": "Q2?", "aspect": "logical"},
        ]
    }
    llm = MagicMock()
    llm.generate_json = AsyncMock(
        side_effect=[q_resp, {"passed": True, "answer": "y"}, {"passed": True, "answer": "y"}]
    )

    verifier = ChainVerifier(llm)
    result = await verifier.verify("Q", _CHAIN, DomainAnalysis())

    assert llm.generate_json.await_count == 3  # 1 generate + 2 independent answers
    assert len(result.answers) == 2
