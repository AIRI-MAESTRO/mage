"""
Baseline prompt generation and execution for evaluation.

Prompt generation is deterministic — templates are filled with the task query.
Baseline execution sends those prompts to LLMs via OpenRouter and captures
responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .config import EvalConfig
from .tasks import BenchmarkTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ZERO_SHOT_TEMPLATE = "{query}"

COT_TEMPLATE = "{query}\n\nLet's think step by step."

PLAN_AND_SOLVE_TEMPLATE = (
    "{query}\n\n"
    "Let's first understand the problem and devise a plan to solve it. "
    "Then, let's carry out the plan and solve the problem step by step."
)

BASELINE_TEMPLATES: dict[str, str] = {
    "zero_shot": ZERO_SHOT_TEMPLATE,
    "cot": COT_TEMPLATE,
    "plan_and_solve": PLAN_AND_SOLVE_TEMPLATE,
}


class BaselineResult(BaseModel):
    """Result of generating a single baseline prompt.

    Parameters
    ----------
    task : BenchmarkTask
        The benchmark task that was processed.
    baseline_type : str
        Baseline strategy (``"zero_shot"``, ``"cot"``, ``"plan_and_solve"``).
    prompt : str
        The complete prompt text.
    metadata : dict[str, Any]
        Extra metadata.
    """

    task: BenchmarkTask
    baseline_type: str
    prompt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def generate_baselines(
    config: EvalConfig,
    tasks: list[BenchmarkTask],
) -> list[BaselineResult]:
    """Generate baseline prompts for each task × baseline_type.

    Baselines are deterministic — they construct the prompt from a template
    without calling any LLM.

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    tasks : list[BenchmarkTask]
        Benchmark tasks to process.

    Returns
    -------
    list[BaselineResult]
        One result per (baseline_type, task) combination.
    """
    results: list[BaselineResult] = []

    for baseline_type in config.baseline_types:
        template = BASELINE_TEMPLATES.get(baseline_type)
        if template is None:
            logger.warning("Unknown baseline type: %s — skipping", baseline_type)
            continue

        for task in tasks:
            prompt = template.format(query=task.query)
            results.append(
                BaselineResult(
                    task=task,
                    baseline_type=baseline_type,
                    prompt=prompt,
                    metadata={"benchmark": task.benchmark, "task_id": task.task_id},
                )
            )

    logger.info("Generated %d baseline prompts", len(results))
    return results


def save_baselines(results: list[BaselineResult], output_dir: str | Path) -> None:
    """Save baseline prompts to disk.

    Parameters
    ----------
    results : list[BaselineResult]
        Baseline generation results.
    output_dir : str | Path
        Root output directory.

    Notes
    -----
    Directory structure::

        output_dir/
        └── baselines/
            └── <baseline_type>/
                └── <benchmark>/
                    ├── <task_id>.txt          # the prompt
                    └── <task_id>_meta.json    # metadata
    """
    base = Path(output_dir) / "baselines"

    for result in results:
        task_dir = base / result.baseline_type / result.task.benchmark
        task_dir.mkdir(parents=True, exist_ok=True)

        # Prompt text
        prompt_path = task_dir / f"{result.task.task_id}.txt"
        prompt_path.write_text(result.prompt, encoding="utf-8")

        # Metadata
        meta = {
            "baseline_type": result.baseline_type,
            "benchmark": result.task.benchmark,
            "task_id": result.task.task_id,
            "metadata": result.metadata,
        }
        meta_path = task_dir / f"{result.task.task_id}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved %d baseline prompts to %s", len(results), base)


# ---------------------------------------------------------------------------
# Baseline execution — send prompts to LLMs
# ---------------------------------------------------------------------------


class BaselineExecution(BaseModel):
    """Result of executing a baseline prompt against a model."""

    task: BenchmarkTask
    execution_model: str
    baseline_type: str
    prompt: str
    response: str = ""
    execution_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    tokens_used: int | None = None


async def execute_baselines(
    config: EvalConfig,
    tasks: list[BenchmarkTask],
    models: list[str] | None = None,
) -> list[BaselineExecution]:
    """Execute each baseline prompt x task x model via OpenRouter.

    For each ``(task, execution_model, baseline_type)`` triple the prompt is
    constructed from templates and sent to the model.  Concurrency is capped by
    ``config.baseline_concurrency``.

    Args:
        config: Evaluation configuration.
        tasks: Benchmark tasks to process.
        models: Override default execution_models. If None, uses config.execution_models.
    """
    api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or "no-key"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url,
        http_client=httpx.AsyncClient(timeout=httpx.Timeout(120.0)),
        default_headers={
            "HTTP-Referer": "https://github.com/Glazkoff/carl-mage",
            "X-Title": "carl-mage",
        },
    )

    target_models = models if models is not None else config.execution_models

    semaphore = asyncio.Semaphore(config.baseline_concurrency)

    async def _run_one(
        task: BenchmarkTask,
        model: str,
        baseline_type: str,
        prompt: str,
    ) -> BaselineExecution:
        async with semaphore:
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.baseline_max_tokens,
                    temperature=config.temperature,
                )
                content = response.choices[0].message.content or ""
                tokens = None
                if response.usage:
                    tokens = response.usage.total_tokens
                elapsed = time.perf_counter() - t0
                return BaselineExecution(
                    task=task,
                    execution_model=model,
                    baseline_type=baseline_type,
                    prompt=prompt,
                    response=content,
                    execution_time_seconds=elapsed,
                    tokens_used=tokens,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.warning("Baseline exec failed (%s/%s/%s): %s", model, baseline_type, task.task_id, exc)
                return BaselineExecution(
                    task=task,
                    execution_model=model,
                    baseline_type=baseline_type,
                    prompt=prompt,
                    response="",
                    execution_time_seconds=elapsed,
                    success=False,
                    error=str(exc),
                )

    coros: list[Any] = []
    for model in target_models:
        for baseline_type in config.baseline_types:
            template = BASELINE_TEMPLATES.get(baseline_type)
            if template is None:
                continue
            for task in tasks:
                prompt = template.format(query=task.query)
                coros.append(_run_one(task, model, baseline_type, prompt))

    results = await asyncio.gather(*coros)
    logger.info("Executed %d baseline prompts", len(results))
    return list(results)


def save_baseline_executions(
    results: list[BaselineExecution],
    output_dir: str | Path,
) -> None:
    """Save baseline execution results to disk.

    Layout::

        output_dir/baseline_executions/<baseline_type>/<execution_model_slug>/<benchmark>/
            <task_id>.txt           — the LLM response
            <task_id>_prompt.txt    — the prompt sent
            <task_id>_meta.json     — metadata (time, tokens, success, error)
    """
    base = Path(output_dir) / "baseline_executions"

    for result in results:
        execution_model_slug = result.execution_model.replace("/", "--")
        task_dir = base / result.baseline_type / execution_model_slug / result.task.benchmark
        task_dir.mkdir(parents=True, exist_ok=True)

        # Response text
        (task_dir / f"{result.task.task_id}.txt").write_text(result.response, encoding="utf-8")

        # Prompt text
        (task_dir / f"{result.task.task_id}_prompt.txt").write_text(result.prompt, encoding="utf-8")

        # Metadata
        meta = {
            "execution_model": result.execution_model,
            "baseline_type": result.baseline_type,
            "benchmark": result.task.benchmark,
            "task_id": result.task.task_id,
            "execution_time_seconds": result.execution_time_seconds,
            "success": result.success,
            "error": result.error,
            "tokens_used": result.tokens_used,
        }
        (task_dir / f"{result.task.task_id}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info("Saved %d baseline executions to %s", len(results), base)
