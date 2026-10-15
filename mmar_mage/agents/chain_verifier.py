"""
ChainVerifier — generates verification questions, answers them independently,
and fixes issues found in the chain.

Implements Chain of Verification (CoVe): generate questions → answer each
independently → compare with chain content → fix discrepancies.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..prompts import CHAIN_VERIFICATION_PROMPT, VERIFICATION_ANSWER_PROMPT, VERIFICATION_FIXER_PROMPT
from ..schemas import DomainAnalysis

logger = logging.getLogger(__name__)

STAGE = "chain_verification"


class VerificationQuestion(BaseModel):
    """A single verification question targeting a chain step.

    Parameters
    ----------
    step_number : int
        Step being verified.
    question : str
        The verification question.
    aspect : str
        Verification aspect (factual, logical, completeness, consistency).
    """

    step_number: int
    question: str
    aspect: str = "factual"


class VerificationResult(BaseModel):
    """Result of the verification process.

    Parameters
    ----------
    questions : list[VerificationQuestion]
        Generated verification questions.
    answers : list[dict]
        Answers with pass/fail status.
    issues_found : list[dict]
        Issues identified with fix suggestions.
    chain_passed : bool
        Whether the chain passed all verifications.
    """

    questions: list[VerificationQuestion] = Field(default_factory=list)
    answers: list[dict] = Field(default_factory=list)
    issues_found: list[dict] = Field(default_factory=list)
    chain_passed: bool = True


class ChainVerifier:
    """Verify chain quality through independent question-answer verification.

    Parameters
    ----------
    llm : LLMClient
        LLM client for generation.

    Examples
    --------
    >>> verifier = ChainVerifier(llm_client)
    >>> result = await verifier.verify(query, chain_dict, domain_analysis)
    >>> if not result.chain_passed:
    ...     chain_dict = await verifier.fix(chain_dict, result)
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def verify(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain_analysis: DomainAnalysis,
        max_questions: int = 3,
    ) -> VerificationResult:
        """Generate verification questions, answer them, identify issues.

        Parameters
        ----------
        query : str
            The user's original text query.
        chain_dict : dict[str, Any]
            Serialized CARL chain dictionary.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        max_questions : int
            Maximum verification questions per step.

        Returns
        -------
        VerificationResult
            Verification results with questions, answers, and issues.
        """
        logger.info("ChainVerifier: verifying chain with %d steps", len(chain_dict.get("steps", [])))

        # Step 1: Generate verification questions
        prompt = CHAIN_VERIFICATION_PROMPT.format(max_questions=max_questions)
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Chain:\n{json.dumps(chain_dict, ensure_ascii=False, indent=2)}"
        )

        q_data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=prompt,
            user_prompt=context,
            stage=STAGE,
        )

        questions: list[VerificationQuestion] = []
        for raw_q in q_data.get("questions", []):
            questions.append(VerificationQuestion.model_validate(raw_q))

        # Step 2: Answer each question independently (CoVe). Each question gets
        # its own LLM call scoped to the step under review PLUS its dependency
        # steps, so cross-step questions (dependencies/consistency) can be judged.
        answers: list[dict] = []
        issues_found: list[dict] = []

        for q in questions:
            try:
                a_data = await self._answer_question(query, chain_dict, q)
                passed = a_data.get("passed", True)
                answers.append(
                    {
                        "question": q.question,
                        "answer": a_data.get("answer", ""),
                        "passed": passed,
                    }
                )
                if not passed:
                    issues_found.append(
                        {
                            "step_number": q.step_number,
                            "issue": q.question,
                            "fix_suggestion": a_data.get("explanation", ""),
                        }
                    )
            except Exception as exc:
                logger.warning("ChainVerifier: failed to answer question: %s", exc)
                answers.append({"question": q.question, "answer": "error", "passed": True})

        chain_passed = len(issues_found) == 0

        result = VerificationResult(
            questions=questions,
            answers=answers,
            issues_found=issues_found,
            chain_passed=chain_passed,
        )

        logger.info(
            "ChainVerifier: %d questions, %d issues found, passed=%s",
            len(questions),
            len(issues_found),
            chain_passed,
        )
        return result

    async def fix(
        self,
        chain_dict: dict[str, Any],
        verification_result: VerificationResult,
    ) -> dict[str, Any]:
        """Fix identified issues in the chain.

        Parameters
        ----------
        chain_dict : dict[str, Any]
            Original chain dictionary.
        verification_result : VerificationResult
            Verification result with issues to fix.

        Returns
        -------
        dict[str, Any]
            Fixed chain dictionary.
        """
        if not verification_result.issues_found:
            return chain_dict

        logger.info("ChainVerifier: fixing %d issues", len(verification_result.issues_found))

        issues_text = json.dumps(verification_result.issues_found, ensure_ascii=False, indent=2)
        prompt = VERIFICATION_FIXER_PROMPT.format(issues=issues_text)

        fixed: dict[str, Any] = await self._llm.generate_json(
            system_prompt=prompt,
            user_prompt=json.dumps(chain_dict, ensure_ascii=False, indent=2),
            stage=STAGE,
        )

        logger.info("ChainVerifier: fixes applied")
        return fixed

    async def _answer_question(
        self,
        query: str,
        chain_dict: dict[str, Any],
        q: VerificationQuestion,
    ) -> dict[str, Any]:
        """Independently answer one verification question (P1.12).

        Builds a cold context: the query, the step under review, and the steps
        it depends on (so dependency / ordering / consistency questions can be
        verified) — then asks the independent-answerer LLM.
        """
        target = self._get_step(chain_dict, q.step_number)
        dep_steps = [s for n in (target.get("dependencies") or []) if (s := self._get_step(chain_dict, n))]

        parts = [
            f"User query: {query}",
            f"Verification question ({q.aspect}): {q.question}",
            f"Step under review (#{q.step_number}):\n{json.dumps(target, ensure_ascii=False, indent=2)}",
        ]
        if dep_steps:
            parts.append(f"Steps it depends on:\n{json.dumps(dep_steps, ensure_ascii=False, indent=2)}")

        return await self._llm.generate_json(
            system_prompt=VERIFICATION_ANSWER_PROMPT,
            user_prompt="\n\n".join(parts),
            stage=STAGE,
        )

    @staticmethod
    def _get_step(chain_dict: dict[str, Any], step_number: int) -> dict[str, Any]:
        """Extract a step by number from a chain dict.

        Parameters
        ----------
        chain_dict : dict[str, Any]
            Chain dictionary.
        step_number : int
            Step number to find.

        Returns
        -------
        dict[str, Any]
            Step dict, or empty dict if not found.
        """
        for step in chain_dict.get("steps", []):
            if step.get("number") == step_number:
                return step
        return {}
