"""
Category-level chain generation — one abstract chain per benchmark.

Instead of generating one chain per question (which leaks answer-specific
entities into the chain), this module produces ONE abstract reasoning template
per ``(benchmark, generation_model)`` pair.  The template uses placeholders
like ``{question}``, ``{entity}``, ``{topic}`` and describes the *process*
rather than specific facts.

A contamination checker rejects chains that contain specific entities or
numbers from the example questions used during generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig

from .config import EvalConfig
from .tasks import BenchmarkTask, get_tasks

logger = logging.getLogger(__name__)

BENCHMARK_DESCRIPTIONS: dict[str, str] = {
    "hover": "multi-hop fact verification — verifying claims by combining information across multiple sources",
    "ifbench": "instruction following — responses must satisfy multiple specific constraints about format, content, length",
    "hotpotqa": "multi-hop question answering — answers require chaining information across multiple entities",
    "livebench_math": "mathematical reasoning — requires applying mathematical techniques to solve problems step-by-step",
}


class CategoryChainResult(BaseModel):
    """Result of generating a single category-level abstract chain.

    Parameters
    ----------
    benchmark : str
        Benchmark name (e.g. ``"hover"``).
    generation_model : str
        Model identifier used for generation.
    num_examples_used : int
        How many example questions were shown to the generation model.
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
    contamination_check : dict[str, Any]
        Result of the contamination check.
    """

    benchmark: str
    generation_model: str
    num_examples_used: int = 0
    chain_json: str = ""
    chain_dict: dict[str, Any] = Field(default_factory=dict)
    code: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    generation_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None
    contamination_check: dict[str, Any] = Field(default_factory=dict)


def build_category_query(benchmark: str, examples: list[BenchmarkTask]) -> str:
    """Construct the meta-query asked to the generation model.

    The prompt instructs the model to produce an *abstract* reasoning chain
    that generalises across all questions of the given benchmark type.  It
    must use placeholders (``{question}``, ``{entity}``, ``{topic}``) and
    describe the *process*, not specific facts.

    Parameters
    ----------
    benchmark : str
        Benchmark name.
    examples : list[BenchmarkTask]
        Example questions shown as few-shot context.

    Returns
    -------
    str
        The meta-query prompt.
    """
    description = BENCHMARK_DESCRIPTIONS.get(benchmark, f"{benchmark} benchmark tasks")

    examples_text = "\n".join(f"  {i + 1}. {ex.query}" for i, ex in enumerate(examples))

    return (
        f"You are designing an ABSTRACT reasoning chain for the following "
        f"benchmark type:\n\n"
        f"BENCHMARK: {benchmark}\n"
        f"DESCRIPTION: {description}\n\n"
        f"Here are example questions from this benchmark (for reference only — "
        f"your chain must NOT contain any specific entities, facts, or numbers "
        f"from these examples):\n\n"
        f"{examples_text}\n\n"
        f"Create a GENERAL-PURPOSE reasoning chain that would work for ANY "
        f"question of this type.  Requirements:\n"
        f"- Each step must describe a PROCESS, not specific facts\n"
        f"- Use placeholders like {{question}}, {{entity}}, {{topic}} where "
        f"appropriate\n"
        f"- Do NOT include specific entities, names, numbers, dates, or facts "
        f"from the example questions\n"
        f"- The chain should be a reusable reasoning template\n"
        f"- Focus on the reasoning STRATEGY, not the content\n\n"
        f"Generate the abstract reasoning chain."
    )


def _extract_entities(examples: list[BenchmarkTask]) -> set[str]:
    """Extract capitalized multi-word phrases and 4+ digit numbers from example queries.

    Parameters
    ----------
    examples : list[BenchmarkTask]
        Example questions to extract entities from.

    Returns
    -------
    set[str]
        Detected entity strings (lowercased for comparison).
    """
    entities: set[str] = set()
    for ex in examples:
        # Capitalized multi-word phrases (e.g. "Stephen King", "Portland Maine")
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", ex.query):
            entities.add(match.group(0).lower())

        # Individual capitalized words (proper nouns) that are 3+ chars
        for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", ex.query):
            word = match.group(0).lower()
            # Skip common words that happen to start sentences
            common_words = {
                "the",
                "this",
                "that",
                "what",
                "which",
                "where",
                "when",
                "who",
                "how",
                "does",
                "verify",
                "confirm",
                "determine",
                "check",
                "find",
                "list",
                "write",
                "explain",
                "describe",
                "compute",
                "calculate",
                "each",
                "use",
                "include",
                "express",
                "follow",
                "three",
                "also",
                "same",
                "given",
                "below",
            }
            if word not in common_words:
                entities.add(word)

        # 4+ digit numbers
        for match in re.finditer(r"\b(\d{4,})\b", ex.query):
            entities.add(match.group(0))

    return entities


def check_contamination(
    chain_dict: dict[str, Any],
    examples: list[BenchmarkTask],
) -> dict[str, Any]:
    """Detect leaked entities from example questions in chain steps.

    Rule-based:
    - Extract capitalized multi-word phrases and 4+ digit numbers from examples
    - Check if any appear in step fields (aim, reasoning_questions,
      step_context_queries, example_reasoning, stage_action, title)

    Parameters
    ----------
    chain_dict : dict[str, Any]
        The generated chain dictionary.
    examples : list[BenchmarkTask]
        Example questions used during generation.

    Returns
    -------
    dict[str, Any]
        ``{contaminated: bool, detected_entities: [...], entities_checked: int}``
    """
    entities = _extract_entities(examples)
    if not entities:
        return {"contaminated": False, "detected_entities": [], "entities_checked": 0}

    check_fields = [
        "aim",
        "reasoning_questions",
        "step_context_queries",
        "example_reasoning",
        "stage_action",
        "title",
    ]

    chain_text_parts: list[str] = []
    for step in chain_dict.get("steps", []):
        for field in check_fields:
            value = step.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                chain_text_parts.extend(str(v) for v in value)
            else:
                chain_text_parts.append(str(value))

    chain_text = " ".join(chain_text_parts).lower()

    detected: list[str] = []
    for entity in sorted(entities):
        if entity in chain_text:
            detected.append(entity)

    return {
        "contaminated": len(detected) > 0,
        "detected_entities": detected,
        "entities_checked": len(entities),
    }


async def generate_category_chains(
    config: EvalConfig,
    tasks_by_benchmark: dict[str, list[BenchmarkTask]] | None = None,
) -> list[CategoryChainResult]:
    """Generate one abstract chain per (benchmark x generation_model).

    Parameters
    ----------
    config : EvalConfig
        Evaluation configuration.
    tasks_by_benchmark : dict[str, list[BenchmarkTask]] | None
        Pre-grouped tasks.  If ``None``, tasks are loaded from the registry
        using ``config.benchmarks`` and ``config.category_example_count``.

    Returns
    -------
    list[CategoryChainResult]
        One result per ``(benchmark, generation_model)`` combination.
    """
    if tasks_by_benchmark is None:
        all_tasks = get_tasks(config.benchmarks)
        tasks_by_benchmark = {}
        for task in all_tasks:
            tasks_by_benchmark.setdefault(task.benchmark, []).append(task)

    results: list[CategoryChainResult] = []

    for model in config.generation_models:
        logger.info("Generating category chains with model: %s", model)

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

        for benchmark, bench_tasks in tasks_by_benchmark.items():
            examples = bench_tasks[: config.category_example_count]
            query = build_category_query(benchmark, examples)

            t0 = time.monotonic()
            try:
                result = await generator.generate(query)
                elapsed = time.monotonic() - t0

                chain_dict = result.chain_dict
                contamination = (
                    check_contamination(chain_dict, examples)
                    if config.contamination_check_enabled
                    else {"contaminated": False, "detected_entities": [], "entities_checked": 0}
                )

                results.append(
                    CategoryChainResult(
                        benchmark=benchmark,
                        generation_model=model,
                        num_examples_used=len(examples),
                        chain_json=result.chain_json,
                        chain_dict=chain_dict,
                        code=result.code,
                        metadata=result.metadata.model_dump(),
                        generation_time_seconds=round(elapsed, 3),
                        contamination_check=contamination,
                    )
                )
                logger.info(
                    "  ✓ %s (%.1fs, contaminated=%s)",
                    benchmark,
                    elapsed,
                    contamination.get("contaminated", False),
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning("  ✗ %s: %s (%.1fs)", benchmark, exc, elapsed)
                results.append(
                    CategoryChainResult(
                        benchmark=benchmark,
                        generation_model=model,
                        num_examples_used=len(examples),
                        success=False,
                        error=str(exc),
                        generation_time_seconds=round(elapsed, 3),
                    )
                )

    return results


def _model_slug(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return model.replace("/", "--")


def save_category_chains(
    results: list[CategoryChainResult],
    output_dir: str | Path,
) -> None:
    """Save category chain results to disk.

    Layout::

        output_dir/category_chains/<model_slug>/<benchmark>.json
        output_dir/category_chains/<model_slug>/<benchmark>.py
        output_dir/category_chains/<model_slug>/<benchmark>_meta.json

    Parameters
    ----------
    results : list[CategoryChainResult]
        Category chain generation results.
    output_dir : str | Path
        Root output directory.
    """
    base = Path(output_dir) / "category_chains"

    for result in results:
        if not result.success:
            continue

        slug = _model_slug(result.generation_model)
        chain_dir = base / slug
        chain_dir.mkdir(parents=True, exist_ok=True)

        # Chain JSON
        chain_path = chain_dir / f"{result.benchmark}.json"
        chain_path.write_text(result.chain_json, encoding="utf-8")

        # Python code
        if result.code:
            code_path = chain_dir / f"{result.benchmark}.py"
            code_path.write_text(result.code, encoding="utf-8")

        # Metadata
        meta = {
            "generation_model": result.generation_model,
            "benchmark": result.benchmark,
            "num_examples_used": result.num_examples_used,
            "generation_time_seconds": result.generation_time_seconds,
            "contamination_check": result.contamination_check,
            "metadata": result.metadata,
        }
        meta_path = chain_dir / f"{result.benchmark}_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    saved_count = len([r for r in results if r.success])
    logger.info("Saved %d category chain results to %s", saved_count, base)
