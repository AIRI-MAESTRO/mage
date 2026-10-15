"""
CARL chain generation for evaluation — generates chains for each task × model.

Uses :class:`~mmar_mage.generator.MAGEGenerator` with OpenRouter-configured
:class:`~mmar_mage.schemas.MAGEConfig` to produce CARL chains for every
combination of benchmark task and generation model.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

from .config import EvalConfig
from .tasks import BenchmarkTask

logger = logging.getLogger(__name__)


class ChainGenerationResult(BaseModel):
    """Result of generating a single CARL chain.

    Parameters
    ----------
    task : BenchmarkTask
        The benchmark task that was processed.
    generation_model : str
        Model identifier used for generation.
    chain_json : str
        Serialized CARL chain as a JSON string.
    chain_dict : dict[str, Any]
        The chain as a Python dictionary.
    code : str | None
        Generated Python code, or ``None``.
    metadata : dict[str, Any]
        Generation metadata from MAGE.
    generation_time_seconds : float
        Wall-clock generation time.
    success : bool
        Whether generation succeeded.
    error : str | None
        Error message if generation failed.
    """

    task: BenchmarkTask
    generation_model: str
    chain_json: str = ""
    chain_dict: dict[str, Any] = Field(default_factory=dict)
    code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    generation_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None


async def generate_chains(
    config: EvalConfig,
    tasks: list[BenchmarkTask],
) -> list[ChainGenerationResult]:
    """Generate CARL chains for each task × generation_model.

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    tasks : list[BenchmarkTask]
        Benchmark tasks to process.

    Returns
    -------
    list[ChainGenerationResult]
        One result per (model, task) combination.
    """
    results: list[ChainGenerationResult] = []

    for model in config.generation_models:
        logger.info("Generating chains with model: %s", model)
        mage_kwargs: dict[str, Any] = {
            "mode": config.mage_mode,
            "temperature": config.temperature,
            "enable_memory": False,
            "output_format": "both",
        }
        if config.allowed_step_types is not None:
            mage_kwargs["allowed_step_types"] = config.allowed_step_types
        mage_config = MAGEConfig.for_openrouter(
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY", ""),
            model=model,
            **mage_kwargs,
        )
        generator = MAGEGenerator(config=mage_config)

        for task in tasks:
            t0 = time.monotonic()
            try:
                result = await generator.generate(task.query)
                elapsed = time.monotonic() - t0
                results.append(
                    ChainGenerationResult(
                        task=task,
                        generation_model=model,
                        chain_json=result.chain_json,
                        chain_dict=result.chain_dict,
                        code=result.code,
                        metadata=result.metadata.model_dump(),
                        generation_time_seconds=round(elapsed, 6),
                    )
                )
                logger.info("  ✓ %s (%.1fs)", task.task_id, elapsed)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning("  ✗ %s: %s (%.1fs)", task.task_id, exc, elapsed)
                results.append(
                    ChainGenerationResult(
                        task=task,
                        generation_model=model,
                        chain_json="",
                        chain_dict={},
                        success=False,
                        error=str(exc),
                        generation_time_seconds=round(elapsed, 6),
                    )
                )

    return results


def _model_slug(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug.

    Parameters
    ----------
    model : str
        Model identifier (e.g. ``"deepseek/deepseek-v3.2"``).

    Returns
    -------
    str
        Slug like ``"deepseek--deepseek-v3.2"``.
    """
    return model.replace("/", "--")


def save_chains(results: list[ChainGenerationResult], output_dir: str | Path) -> None:
    """Save generated chains to disk.

    Parameters
    ----------
    results : list[ChainGenerationResult]
        Chain generation results.
    output_dir : str | Path
        Root output directory.

    Notes
    -----
    Directory structure::

        output_dir/
        └── chains/
            └── <model_slug>/
                └── <benchmark>/
                    ├── <task_id>.json       # chain JSON
                    ├── <task_id>.py         # Python code (if available)
                    └── <task_id>_meta.json  # metadata
    """
    base = Path(output_dir) / "chains"

    for result in results:
        if not result.success:
            continue

        slug = _model_slug(result.generation_model)
        task_dir = base / slug / result.task.benchmark
        task_dir.mkdir(parents=True, exist_ok=True)

        # Chain JSON
        chain_path = task_dir / f"{result.task.task_id}.json"
        chain_path.write_text(result.chain_json, encoding="utf-8")

        # Python code
        if result.code:
            code_path = task_dir / f"{result.task.task_id}.py"
            code_path.write_text(result.code, encoding="utf-8")

        # Metadata
        meta = {
            "generation_model": result.generation_model,
            "benchmark": result.task.benchmark,
            "task_id": result.task.task_id,
            "generation_time_seconds": result.generation_time_seconds,
            "metadata": result.metadata,
        }
        meta_path = task_dir / f"{result.task.task_id}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Saved %d chain results to %s", len([r for r in results if r.success]), base)
