"""ChainGrader — Stage 3 of evolve mode.

Scores an :class:`Individual` (a candidate chain) against the synthesized
benchmark. Two-tier grading:

1. **Structural rubric** — fast, deterministic checks against the chain
   structure (does it have steps that could plausibly satisfy each rubric
   item? does it use the right step types?).
2. **LLM-judge** — runs the chain "in the model's head" (it walks the
   steps, predicts each step's contribution, then judges the final answer
   against ``expected_output`` + rubric).

Why no real execution? MVP: real CARL execution requires API keys, mocking,
and time. LLM-judge is a noisy-but-cheap reward signal that's good enough
to drive evolutionary search. Real execution can replace step (2) later
without changing the engine API — just call ``ChainExecutor`` and feed its
output into the same judge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, cast

from ..llm import LLMClient
from ..prompts import (
    CHAIN_FLATTEN_EXECUTION_PROMPT,
    CHAIN_JUDGE_PROMPT,
    CHAIN_JUDGE_REAL_OUTPUT_PROMPT,
)
from .benchmark_synthesizer import chain_dict_summary_for_judge
from .schemas import (
    BenchmarkSynthesisResult,
    ChainScore,
    EvolveConfig,
    SynthesizedTask,
)

logger = logging.getLogger(__name__)

STAGE = "chain_grading"


class ChainGrader:
    """Grade chains against a synthesized benchmark."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def grade(
        self,
        chain_dict: dict[str, Any],
        benchmark: BenchmarkSynthesisResult,
        evolve_config: EvolveConfig,
    ) -> ChainScore:
        """Score one chain on the whole benchmark."""
        if not benchmark.tasks:
            return ChainScore(overall=0.0, judge_notes="empty benchmark")

        # T26: tasks are independent — grade them concurrently (bounded by a
        # semaphore) instead of awaiting one at a time. gather preserves order,
        # so the assembled dicts / notes are identical to the sequential version.
        sem = asyncio.Semaphore(max(1, evolve_config.grading_concurrency))
        results = await asyncio.gather(
            *(self._grade_one_task(chain_dict, task, evolve_config, sem) for task in benchmark.tasks)
        )

        per_task_scores: dict[str, float] = {}
        per_task_failures: dict[str, str] = {}
        notes: list[str] = []
        for task_id, task_score, failure, note in results:
            per_task_scores[task_id] = task_score
            if failure is not None:
                per_task_failures[task_id] = failure
            if note is not None:
                notes.append(note)

        overall = sum(per_task_scores.values()) / len(per_task_scores) if per_task_scores else 0.0
        suspect_gaming = any(self._suspect_gaming(chain_dict, t) for t in benchmark.tasks)
        return ChainScore(
            overall=round(overall, 3),
            per_task_scores=per_task_scores,
            per_task_failures=per_task_failures,
            judge_notes=" | ".join(notes[:5]),
            suspect_gaming=suspect_gaming,
        )

    async def _grade_one_task(
        self,
        chain_dict: dict[str, Any],
        task: SynthesizedTask,
        evolve_config: EvolveConfig,
        sem: asyncio.Semaphore,
    ) -> tuple[str, float, str | None, str | None]:
        """Score one benchmark task (T26). Independent per task, so :meth:`grade`
        runs these under ``asyncio.gather`` bounded by ``sem``. Returns
        ``(task_id, task_score, failure_or_None, note_or_None)``."""
        async with sem:
            structural_score = 0.0
            judge_score = 0.0
            failure: str | None = None
            note: str | None = None

            if evolve_config.use_structural_rubric:
                structural_score = self._structural_grade(chain_dict, task)

            if evolve_config.use_llm_judge:
                try:
                    if evolve_config.execution_model:
                        # Real-execution path: run chain, judge actual output.
                        actual_output = await self._execute_chain(chain_dict, task, evolve_config)
                        judge_score, judge_note = await self._llm_judge_real_output(
                            actual_output, task, evolve_config.grader_temperature
                        )
                    else:
                        # Forecast path: judge predicts what the chain would produce.
                        judge_score, judge_note = await self._llm_judge(
                            chain_dict, task, evolve_config.grader_temperature
                        )
                    if judge_note:
                        note = f"{task.task_id}: {judge_note}"
                except Exception as exc:
                    logger.warning("ChainGrader: LLM judge failed for %s: %s", task.task_id, exc)
                    failure = f"judge error: {exc}"
                    judge_score = 0.0

            # Weighted combo: judge dominates (0.7), structural is a sanity prior (0.3).
            if evolve_config.use_structural_rubric and evolve_config.use_llm_judge:
                task_score = 0.3 * structural_score + 0.7 * judge_score
            elif evolve_config.use_llm_judge:
                task_score = judge_score
            else:
                task_score = structural_score

            task_score = round(task_score, 3)
            if failure is None and task_score < 0.5:
                failure = f"score={task_score:.2f}; rubric_unmet_or_judge_low"
            return task.task_id, task_score, failure, note

    # ------------------------------------------------------------------
    # Structural rubric — fast, deterministic.
    # ------------------------------------------------------------------

    @staticmethod
    def _structural_grade(chain_dict: dict[str, Any], task: SynthesizedTask) -> float:
        """Deterministic structural prior.

        Heuristics:
        - Chain has at least 2 steps and a final synthesis-shaped step.
        - Chain mentions task-specific keywords from the rubric in any step's
          aim / reasoning_questions / step_context_queries.
        - For tasks with structured ``expected_output`` (JSON-shaped), reward
          chains containing a ``structured_output`` step.
        """
        steps = chain_dict.get("steps", []) or []
        if not steps:
            return 0.0

        score = 0.0
        max_score = 0.0

        # 1. Has multiple steps (basic sanity).
        max_score += 1.0
        if len(steps) >= 2:
            score += 1.0

        # 2. Rubric keyword coverage.
        if task.rubric:
            max_score += 1.0
            # P-5: search title + aim ONLY, not the whole step JSON. Stuffing
            # rubric words into example_reasoning / step_context_queries must
            # not inflate the structural score (reward-hacking guard).
            haystack = " ".join(f"{s.get('title', '')} {s.get('aim', '')}" for s in steps).lower()
            keywords = ChainGrader._rubric_keywords(task)
            if keywords:
                hits = sum(1 for kw in keywords if kw in haystack)
                score += min(1.0, hits / len(keywords))

        # 3. Structured-output reward when reference looks like JSON.
        if task.expected_output and task.expected_output.strip().startswith(("{", "[")):
            max_score += 1.0
            if any(s.get("step_type") == "structured_output" for s in steps):
                score += 1.0
            else:
                score += 0.3  # partial — chain may emit JSON via plain llm

        if max_score == 0:
            return 0.5  # no signal — neutral
        return score / max_score

    @staticmethod
    def _rubric_keywords(task: SynthesizedTask) -> list[str]:
        """Rough content nouns from the rubric (alpha words >= 4 chars)."""
        kws: list[str] = []
        for r in task.rubric or []:
            for word in str(r).lower().replace("=", " ").replace("'", " ").split():
                if len(word) >= 4 and word.isalpha():
                    kws.append(word)
        return kws

    @staticmethod
    def _suspect_gaming(chain_dict: dict[str, Any], task: SynthesizedTask, *, threshold: float = 0.12) -> bool:
        """Keyword-stuffing detector (P-5) — monitoring only, never changes the score.

        Flags chains whose full step JSON has an abnormally high density of
        rubric keywords relative to its length — the reward-hacking signature of
        stuffing rubric words into example_reasoning / step_context_queries.
        """
        steps = chain_dict.get("steps", []) or []
        keywords = set(ChainGrader._rubric_keywords(task))
        if not steps or not keywords:
            return False
        tokens = re.findall(r"[a-z]+", json.dumps(steps, ensure_ascii=False).lower())
        if len(tokens) < 20:
            return False
        occurrences = sum(1 for t in tokens if t in keywords)
        return occurrences / len(tokens) > threshold

    # ------------------------------------------------------------------
    # Chain execution (stub v1) — flattened-prompt one-shot run.
    # ------------------------------------------------------------------

    async def _execute_chain(
        self,
        chain_dict: dict[str, Any],
        task: SynthesizedTask,
        evolve_config: EvolveConfig,
    ) -> str:
        """Run the chain on the task input by flattening it into one prompt.

        Mirrors :mod:`evaluation.chain_executor`'s stub-v1 strategy: build a
        numbered reasoning plan from each step's ``aim`` and let a single LLM
        call execute it. Returns the raw assistant response.
        """
        steps = chain_dict.get("steps", []) or []
        plan_lines: list[str] = []
        for i, step in enumerate(steps, start=1):
            aim = step.get("aim") or step.get("title") or f"Step {i}"
            plan_lines.append(f"{i}. {aim}")
        plan = "\n".join(plan_lines) if plan_lines else "1. Analyze and answer."

        user_prompt = CHAIN_FLATTEN_EXECUTION_PROMPT.format(plan=plan, input=task.input)
        try:
            # Bypass per-stage overrides — execution model is configured
            # explicitly via EvolveConfig.execution_model.
            response = await self._llm._client.chat.completions.create(  # noqa: SLF001
                # `_execute_chain` is only reached when `execution_model` is set
                # (guarded by the caller), so it is never None here.
                model=cast(str, evolve_config.execution_model),
                messages=[{"role": "user", "content": user_prompt}],
                temperature=evolve_config.execution_temperature,
                max_tokens=evolve_config.execution_max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.debug("ChainGrader._execute_chain failed for %s: %s", task.task_id, exc)
            return ""

    async def _llm_judge_real_output(
        self,
        actual_output: str,
        task: SynthesizedTask,
        temperature: float,
    ) -> tuple[float, str]:
        """Judge the actual chain response against the rubric / expected output."""
        if not actual_output:
            return 0.0, "empty execution output"

        user_prompt = (
            f"Synthesized task:\n"
            f"  task_id: {task.task_id}\n"
            f"  input: {task.input}\n"
            f"  expected_output: {task.expected_output if task.expected_output else '<open-ended>'}\n"
            f"  rubric: {json.dumps(task.rubric, ensure_ascii=False)}\n\n"
            f"Actual chain response:\n{actual_output}\n"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=CHAIN_JUDGE_REAL_OUTPUT_PROMPT,
                user_prompt=user_prompt,
                stage=STAGE,
            )
        except Exception as exc:
            logger.debug("ChainGrader._llm_judge_real_output JSON failure: %s", exc)
            return 0.0, "judge JSON parse failed"

        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        notes = str(data.get("notes", "")).strip()
        return round(score, 3), notes

    # ------------------------------------------------------------------
    # LLM-judge — chain walk-through + verdict (forecast mode).
    # ------------------------------------------------------------------

    async def _llm_judge(
        self,
        chain_dict: dict[str, Any],
        task: SynthesizedTask,
        temperature: float,
    ) -> tuple[float, str]:
        chain_summary = chain_dict_summary_for_judge(chain_dict)

        user_prompt = (
            f"Synthesized task:\n"
            f"  task_id: {task.task_id}\n"
            f"  input: {task.input}\n"
            f"  expected_output: {task.expected_output if task.expected_output else '<open-ended>'}\n"
            f"  rubric: {json.dumps(task.rubric, ensure_ascii=False)}\n\n"
            f"Candidate chain (one line per step):\n{chain_summary}\n\n"
            "Step 1 (silently): mentally walk the chain on the input — what would each "
            "step plausibly produce, given the aim and reasoning questions? "
            "Step 2: estimate the final output the chain would emit.\n"
            "Step 3: grade the final output against the rubric and expected_output."
        )

        try:
            data = await self._llm.generate_json(
                system_prompt=CHAIN_JUDGE_PROMPT,
                user_prompt=user_prompt,
                stage=STAGE,
            )
        except Exception as exc:
            logger.debug("ChainGrader._llm_judge JSON failure: %s", exc)
            return 0.0, "judge JSON parse failed"

        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        notes = str(data.get("notes", "")).strip()
        return round(score, 3), notes
