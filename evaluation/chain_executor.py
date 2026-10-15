"""
Chain execution for evaluation — load saved CARL chains and run them on execution models.

**Stub executor v1**: Since a full CARL chain runtime is out of scope, this
executor flattens each chain into a single prompt (using step aims as a
reasoning guide) and sends it to the execution model as one LLM call.  This
gives a reasonable proxy to measure chain quality before the real orchestrator
is built.

TODO: Replace stub executor with full CARL chain runtime integration.
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
from pydantic import BaseModel

from .config import EvalConfig
from .tasks import BenchmarkTask

logger = logging.getLogger(__name__)


class ChainExecution(BaseModel):
    """Result of executing a saved CARL chain on an execution model.

    Parameters
    ----------
    task : BenchmarkTask
        The benchmark task that was executed.
    generation_model : str
        Which model created the chain.
    execution_model : str
        Which model is running the chain now.
    chain_json_path : str
        Path to the chain JSON file on disk.
    response : str
        Final answer from chain execution.
    steps_completed : int
        Number of steps in the chain that were included in the prompt.
    execution_time_seconds : float
        Wall-clock execution time.
    success : bool
        Whether execution succeeded.
    error : str | None
        Error message if execution failed.
    """

    task: BenchmarkTask
    generation_model: str
    execution_model: str
    chain_json_path: str
    response: str = ""
    steps_completed: int = 0
    execution_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None


def _model_slug(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return model.replace("/", "--")


def _build_chain_prompt(task: BenchmarkTask, chain_dict: dict[str, Any]) -> tuple[str, int]:
    """Build a flattened prompt from a CARL chain (stub executor v1).

    Extracts step aims from the chain and constructs a single prompt that
    asks the execution model to follow the reasoning plan.

    Returns
    -------
    tuple[str, int]
        The prompt text and the number of steps included.
    """
    steps = chain_dict.get("steps", [])
    step_lines: list[str] = []
    for i, step in enumerate(steps, 1):
        aim = step.get("aim", step.get("title", f"Step {i}"))
        step_lines.append(f"{i}. {aim}")

    steps_text = "\n".join(step_lines) if step_lines else "1. Analyze the question and provide an answer."

    prompt = (
        f"Follow this reasoning plan to answer the question.\n\n"
        f"QUESTION: {task.query}\n\n"
        f"REASONING PLAN:\n{steps_text}\n\n"
        f"Now work through the plan step by step and give the final answer."
    )
    return prompt, len(steps)


async def execute_chains(
    config: EvalConfig,
    tasks: list[BenchmarkTask],
    artifacts_dir: Path,
    models: list[str] | None = None,
) -> list[ChainExecution]:
    """Load saved chains from disk and execute them on execution models.

    **Stub executor v1**: Flattens each chain into a concatenated prompt
    (step aims as reasoning guide) and sends it to the execution model as
    a single LLM call.

    TODO: Replace with full CARL chain runtime orchestrator.

    For each (task x generation_model x execution_model):
      1. Load chain JSON from artifacts_dir/chains/<gen_model>/<benchmark>/<task_id>.json
      2. Build a flattened prompt from step aims
      3. Send to execution_model as a single LLM call
      4. Record final response, timing

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    tasks : list[BenchmarkTask]
        Benchmark tasks to execute chains for.
    artifacts_dir : Path
        Root directory containing saved chain artifacts.
    models : list[str] | None
        Override execution models. If None, uses config.execution_models.

    Returns
    -------
    list[ChainExecution]
        One result per (task, generation_model, execution_model) combination.
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

    chains_dir = artifacts_dir / "chains"
    if not chains_dir.exists():
        logger.warning("No chains directory found at %s", chains_dir)
        return []

    # Discover available generation model directories
    gen_model_dirs = [d for d in chains_dir.iterdir() if d.is_dir()]

    # T35: real-runtime CARL clients, built lazily per execution model.
    carl_clients: dict[str, Any] = {}

    def _carl_client(exec_model: str) -> Any:
        if exec_model not in carl_clients:
            from evaluation.carl_executor import build_eval_llm_client

            carl_clients[exec_model] = build_eval_llm_client(config, exec_model)
        return carl_clients[exec_model]

    async def _run_one(
        task: BenchmarkTask,
        gen_model: str,
        exec_model: str,
        chain_dict: dict[str, Any],
        chain_json_path: str,
    ) -> ChainExecution:
        if config.use_carl_runtime:
            # T35: the REAL DAG runtime — tool steps actually execute (against
            # the deterministic mock-HTTP tool set), so evaluation catches the
            # 09.06 incident class the flattened stub was blind to.
            from evaluation.carl_executor import (
                execute_chain_via_carl,
                register_benchmark_tools,
            )

            async with semaphore:
                t0 = time.perf_counter()
                success, output, steps_done, error = await execute_chain_via_carl(
                    chain_dict,
                    task.query,
                    api=_carl_client(exec_model),
                    register_tools=register_benchmark_tools,
                )
                elapsed = time.perf_counter() - t0
            return ChainExecution(
                task=task,
                generation_model=gen_model,
                execution_model=exec_model,
                chain_json_path=chain_json_path,
                response=output,
                steps_completed=steps_done,
                execution_time_seconds=round(elapsed, 3),
                success=success,
                error=error,
            )
        prompt, steps_count = _build_chain_prompt(task, chain_dict)
        async with semaphore:
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=exec_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.baseline_max_tokens,
                    temperature=config.temperature,
                )
                content = response.choices[0].message.content or ""
                elapsed = time.perf_counter() - t0
                return ChainExecution(
                    task=task,
                    generation_model=gen_model,
                    execution_model=exec_model,
                    chain_json_path=chain_json_path,
                    response=content,
                    steps_completed=steps_count,
                    execution_time_seconds=round(elapsed, 3),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.warning(
                    "Chain exec failed (%s→%s/%s): %s",
                    gen_model,
                    exec_model,
                    task.task_id,
                    exc,
                )
                return ChainExecution(
                    task=task,
                    generation_model=gen_model,
                    execution_model=exec_model,
                    chain_json_path=chain_json_path,
                    response="",
                    steps_completed=0,
                    execution_time_seconds=round(elapsed, 3),
                    success=False,
                    error=str(exc),
                )

    coros: list[Any] = []

    for gen_dir in gen_model_dirs:
        # Reverse the slug to get the model name
        gen_model = gen_dir.name.replace("--", "/")

        for task in tasks:
            chain_path = gen_dir / task.benchmark / f"{task.task_id}.json"
            if not chain_path.exists():
                logger.debug("No chain file for %s/%s at %s", gen_model, task.task_id, chain_path)
                continue

            try:
                chain_dict = json.loads(chain_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load chain %s: %s", chain_path, exc)
                for exec_model in target_models:
                    coros.append(
                        _make_error_result(task, gen_model, exec_model, str(chain_path), str(exc)),
                    )
                continue

            for exec_model in target_models:
                coros.append(
                    _run_one(task, gen_model, exec_model, chain_dict, str(chain_path)),
                )

    if not coros:
        logger.warning("No chain artifacts found to execute")
        return []

    results = await asyncio.gather(*coros)
    logger.info("Executed %d chain runs", len(results))
    return list(results)


async def _make_error_result(
    task: BenchmarkTask,
    gen_model: str,
    exec_model: str,
    chain_json_path: str,
    error: str,
) -> ChainExecution:
    """Create an error ChainExecution result (async for gather compatibility)."""
    return ChainExecution(
        task=task,
        generation_model=gen_model,
        execution_model=exec_model,
        chain_json_path=chain_json_path,
        response="",
        steps_completed=0,
        execution_time_seconds=0.0,
        success=False,
        error=error,
    )


def build_execution_prompt_from_category(
    category_chain: dict[str, Any],
    task: BenchmarkTask,
) -> tuple[str, int]:
    """Build a prompt for an execution model from a category-level abstract chain.

    Combines the abstract reasoning plan with the actual question so the
    execution model can follow the plan step-by-step.

    Parameters
    ----------
    category_chain : dict[str, Any]
        The abstract category chain dictionary.
    task : BenchmarkTask
        The benchmark task to answer.

    Returns
    -------
    tuple[str, int]
        The prompt text and the number of steps in the plan.
    """
    steps = category_chain.get("steps", [])
    step_lines: list[str] = []
    for i, step in enumerate(steps, 1):
        aim = step.get("aim", step.get("title", f"Step {i}"))
        reasoning_qs = step.get("reasoning_questions", "")
        action = step.get("stage_action", "")

        parts = [f"Step {i}: {aim}"]
        if reasoning_qs:
            parts.append(f"   Key questions to think about: {reasoning_qs}")
        if action:
            parts.append(f"   Process: {action}")
        step_lines.append("\n".join(parts))

    if step_lines:
        plan_text = "\n\n".join(step_lines)
    else:
        plan_text = "Step 1: Analyze the question and provide an answer."

    prompt = (
        "You are given an abstract reasoning plan. Follow it step-by-step to "
        "answer the question below.\n\n"
        f"REASONING PLAN:\n{plan_text}\n\n"
        f"QUESTION: {task.query}\n\n"
        "Work through the plan step-by-step and provide the final answer."
    )
    return prompt, len(steps)


async def execute_category_chains(
    config: EvalConfig,
    tasks: list[BenchmarkTask],
    artifacts_dir: Path,
    models: list[str] | None = None,
) -> list[ChainExecution]:
    """Execute category-level chains on benchmark tasks.

    For each task, loads the category chain matching ``task.benchmark`` (for
    each generation model found), builds a combined prompt, and sends it to
    each execution model.

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    tasks : list[BenchmarkTask]
        Benchmark tasks to execute.
    artifacts_dir : Path
        Root directory containing saved artifacts.
    models : list[str] | None
        Override execution models.

    Returns
    -------
    list[ChainExecution]
        One result per ``(task, generation_model, execution_model)`` combination.
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

    category_chains_dir = artifacts_dir / "category_chains"
    if not category_chains_dir.exists():
        logger.warning("No category_chains directory found at %s", category_chains_dir)
        return []

    # Load category chains: {gen_model: {benchmark: chain_dict}}
    loaded_chains: dict[str, dict[str, dict[str, Any]]] = {}
    for gen_dir in category_chains_dir.iterdir():
        if not gen_dir.is_dir():
            continue
        gen_model = gen_dir.name.replace("--", "/")
        loaded_chains[gen_model] = {}
        for chain_file in gen_dir.glob("*.json"):
            if chain_file.name.endswith("_meta.json"):
                continue
            benchmark = chain_file.stem
            try:
                chain_dict = json.loads(chain_file.read_text(encoding="utf-8"))
                loaded_chains[gen_model][benchmark] = chain_dict
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load category chain %s: %s", chain_file, exc)

    if not loaded_chains:
        logger.warning("No category chain artifacts found to execute")
        return []

    async def _run_one(
        task: BenchmarkTask,
        gen_model: str,
        exec_model: str,
        chain_dict: dict[str, Any],
        chain_json_path: str,
    ) -> ChainExecution:
        prompt, steps_count = build_execution_prompt_from_category(chain_dict, task)
        async with semaphore:
            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=exec_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.baseline_max_tokens,
                    temperature=config.temperature,
                )
                content = response.choices[0].message.content or ""
                elapsed = time.perf_counter() - t0
                return ChainExecution(
                    task=task,
                    generation_model=gen_model,
                    execution_model=exec_model,
                    chain_json_path=chain_json_path,
                    response=content,
                    steps_completed=steps_count,
                    execution_time_seconds=round(elapsed, 3),
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.warning(
                    "Category chain exec failed (%s→%s/%s): %s",
                    gen_model,
                    exec_model,
                    task.task_id,
                    exc,
                )
                return ChainExecution(
                    task=task,
                    generation_model=gen_model,
                    execution_model=exec_model,
                    chain_json_path=chain_json_path,
                    response="",
                    steps_completed=0,
                    execution_time_seconds=round(elapsed, 3),
                    success=False,
                    error=str(exc),
                )

    coros: list[Any] = []

    for gen_model, benchmarks in loaded_chains.items():
        for task in tasks:
            chain_dict = benchmarks.get(task.benchmark)
            if chain_dict is None:
                logger.debug(
                    "No category chain for %s/%s (gen_model=%s)",
                    task.benchmark,
                    task.task_id,
                    gen_model,
                )
                continue

            slug = _model_slug(gen_model)
            chain_json_path = str(artifacts_dir / "category_chains" / slug / f"{task.benchmark}.json")

            for exec_model in target_models:
                coros.append(
                    _run_one(task, gen_model, exec_model, chain_dict, chain_json_path),
                )

    if not coros:
        logger.warning("No category chain artifacts matched any tasks")
        return []

    results = await asyncio.gather(*coros)
    logger.info("Executed %d category chain runs", len(results))
    return list(results)


def save_chain_executions(
    results: list[ChainExecution],
    output_dir: Path,
) -> None:
    """Save chain execution results to disk.

    Layout::

        output_dir/chain_executions/<gen_model_slug>/<exec_model_slug>/<benchmark>/
            <task_id>.txt           — the LLM response
            <task_id>_prompt.txt    — the flattened prompt sent
            <task_id>_meta.json     — metadata (time, steps, success, error)
    """
    base = Path(output_dir) / "chain_executions"

    for result in results:
        gen_slug = _model_slug(result.generation_model)
        exec_slug = _model_slug(result.execution_model)
        task_dir = base / gen_slug / exec_slug / result.task.benchmark
        task_dir.mkdir(parents=True, exist_ok=True)

        # Response text
        (task_dir / f"{result.task.task_id}.txt").write_text(result.response, encoding="utf-8")

        # Reconstruct the prompt for reference
        try:
            if result.chain_json_path and Path(result.chain_json_path).exists():
                chain_dict = json.loads(Path(result.chain_json_path).read_text(encoding="utf-8"))
                prompt, _ = _build_chain_prompt(result.task, chain_dict)
            else:
                prompt = "(chain file not available)"
        except Exception:
            prompt = "(could not reconstruct prompt)"
        (task_dir / f"{result.task.task_id}_prompt.txt").write_text(prompt, encoding="utf-8")

        # Metadata
        meta = {
            "generation_model": result.generation_model,
            "execution_model": result.execution_model,
            "benchmark": result.task.benchmark,
            "task_id": result.task.task_id,
            "chain_json_path": result.chain_json_path,
            "steps_completed": result.steps_completed,
            "execution_time_seconds": result.execution_time_seconds,
            "success": result.success,
            "error": result.error,
        }
        (task_dir / f"{result.task.task_id}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info("Saved %d chain executions to %s", len(results), base)
