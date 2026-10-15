"""
CLI entrypoint for evaluation — two-stage pipeline for MAGE experiments.

Stage A: ``generate-artifacts`` — produces MAGE CARL chains + baseline prompt
text using generation models only (no execution).

Stage B: ``run-benchmarks`` — loads saved artifacts from disk and executes them
on execution models with benchmark metrics.

``run-all`` chains both stages end-to-end.

Usage::

    python -m evaluation.runner generate-artifacts
    python -m evaluation.runner run-benchmarks
    python -m evaluation.runner run-all
    python -m evaluation.runner list-tasks --benchmarks hover hotpotqa
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .baseline_generator import (
    execute_baselines,
    generate_baselines,
    save_baseline_executions,
    save_baselines,
)
from .category_chain_generator import (
    generate_category_chains,
    save_category_chains,
)
from .chain_executor import (
    execute_category_chains,
    execute_chains,
    save_chain_executions,
)
from .chain_generator import generate_chains, save_chains
from .config import EvalConfig
from .tasks import BENCHMARK_TASKS, get_tasks

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="MAGE evaluation — generate artifacts and run benchmarks")


# ---------------------------------------------------------------------------
# Stage A: Generate artifacts
# ---------------------------------------------------------------------------


@app.command("generate-category-chains")
def generate_category_chains_cmd(
    api_key: str | None = typer.Option(
        None,
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    ),
    config_path: Path = typer.Option(
        "evaluation/eval_config.toml",
        help="Path to TOML config",
    ),
    quick: bool = typer.Option(
        False,
        help="Quick mode: 2 tasks per benchmark, 1 generation model",
    ),
) -> None:
    """Generate ONE abstract chain per (benchmark x generation_model).

    Correct setup for benchmark experiments — each benchmark gets one
    reasoning template that generalises across all questions.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = EvalConfig.from_toml(config_path)

    if api_key:
        config.api_key = api_key
    if not config.api_key and not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]ERROR: OpenRouter API key required for chain generation[/red]")
        console.print("Set OPENROUTER_API_KEY env var or pass --api-key")
        raise typer.Exit(code=1)

    if quick:
        config.max_tasks_per_benchmark = 2
        config.generation_models = config.generation_models[:1]

    console.print("[bold]═══ Generate Category Chains ═══[/bold]")
    console.print(f"Generation models: {len(config.generation_models)}")
    console.print(f"Benchmarks: {config.benchmarks}")
    console.print(f"Examples per benchmark: {config.category_example_count}")
    console.print(f"Contamination check: {config.contamination_check_enabled}")
    console.print(f"Output: {config.output_dir}")
    console.print()

    cat_results = asyncio.run(generate_category_chains(config))
    save_category_chains(cat_results, config.output_dir)
    success_count = sum(1 for r in cat_results if r.success)
    console.print(f"  ✓ {success_count}/{len(cat_results)} category chains generated")

    console.print()
    console.print(f"[bold green]✓ Category chains saved to {config.output_dir}[/bold green]")


@app.command("generate-artifacts")
def generate_artifacts(
    api_key: str | None = typer.Option(
        None,
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    ),
    config_path: Path = typer.Option(
        "evaluation/eval_config.toml",
        help="Path to TOML config",
    ),
    quick: bool = typer.Option(
        False,
        help="Quick mode: 2 tasks per benchmark, 1 generation model",
    ),
    mode: str = typer.Option(
        None,
        help='Chain mode: "category" (default) or "per-task" (legacy)',
    ),
) -> None:
    """Stage A: Generate MAGE chains and baseline prompts (no LLM execution).

    Uses generation models only. In *category* mode (default), generates one
    abstract chain per benchmark.  In *per-task* mode (legacy), generates one
    chain per question.

    After this, run ``run-benchmarks`` to execute artifacts on execution models.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = EvalConfig.from_toml(config_path)

    if api_key:
        config.api_key = api_key
    if not config.api_key and not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]ERROR: OpenRouter API key required for chain generation[/red]")
        console.print("Set OPENROUTER_API_KEY env var or pass --api-key")
        raise typer.Exit(code=1)

    if mode is not None:
        config.chain_mode = mode.replace("-", "_")  # type: ignore[assignment]

    if quick:
        config.max_tasks_per_benchmark = 2
        config.generation_models = config.generation_models[:1]

    tasks = get_tasks(config.benchmarks, config.max_tasks_per_benchmark)

    console.print("[bold]═══ Stage A: Generate Artifacts ═══[/bold]")
    console.print(f"Chain mode: {config.chain_mode}")
    console.print(f"Generation models: {len(config.generation_models)}")
    console.print(f"Benchmarks: {config.benchmarks}")
    console.print(f"Tasks: {len(tasks)}")
    console.print(f"Baselines: {config.baseline_types}")
    console.print(f"LLM-only mode: {config.allowed_step_types == ['llm']}")
    console.print(f"Output: {config.output_dir}")
    console.print()

    # Generate MAGE chains
    if config.chain_mode == "category":
        console.print("[bold blue]▶ Generating category chains (one per benchmark)...[/bold blue]")
        cat_results = asyncio.run(generate_category_chains(config))
        save_category_chains(cat_results, config.output_dir)
        success_count = sum(1 for r in cat_results if r.success)
        console.print(f"  ✓ {success_count}/{len(cat_results)} category chains generated")
    else:
        console.print("[bold blue]▶ Generating MAGE chains (per-task, legacy)...[/bold blue]")
        chain_results = asyncio.run(generate_chains(config, tasks))
        save_chains(chain_results, config.output_dir)
        success_count = sum(1 for r in chain_results if r.success)
        console.print(f"  ✓ {success_count}/{len(chain_results)} chains generated")

    # Generate baseline prompts (deterministic, no LLM)
    console.print("[bold blue]▶ Generating baseline prompts...[/bold blue]")
    baseline_results = generate_baselines(config, tasks)
    save_baselines(baseline_results, config.output_dir)
    console.print(f"  ✓ {len(baseline_results)} baseline prompts generated")

    console.print()
    console.print(f"[bold green]✓ Artifacts saved to {config.output_dir}[/bold green]")
    console.print("Run [bold]run-benchmarks[/bold] to execute artifacts on execution models.")


# ---------------------------------------------------------------------------
# Stage B: Run benchmarks
# ---------------------------------------------------------------------------


@app.command("run-benchmarks")
def run_benchmarks(
    api_key: str | None = typer.Option(
        None,
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    ),
    config_path: Path = typer.Option(
        "evaluation/eval_config.toml",
        help="Path to TOML config",
    ),
    artifacts_dir: Path | None = typer.Option(
        None,
        help="Override artifacts location (default: config.output_dir)",
    ),
    skip_chains: bool = typer.Option(False, help="Skip MAGE chain execution"),
    skip_baselines: bool = typer.Option(False, help="Skip baseline execution"),
    quick: bool = typer.Option(
        False,
        help="Quick mode: 2 tasks per benchmark, 1 execution model",
    ),
    chain_mode: str | None = typer.Option(
        None,
        help='Chain mode: "category" (default) or "per-task" (legacy)',
    ),
) -> None:
    """Stage B: Execute saved artifacts on execution models with benchmark metrics.

    Loads chains and baseline prompts from disk, then runs them on
    execution_models.  In *category* mode (default), uses category-level
    abstract chains.  In *per-task* mode, uses legacy per-question chains.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = EvalConfig.from_toml(config_path)

    if api_key:
        config.api_key = api_key
    if not config.api_key and not os.getenv("OPENROUTER_API_KEY"):
        console.print("[red]ERROR: OpenRouter API key required for benchmark execution[/red]")
        console.print("Set OPENROUTER_API_KEY env var or pass --api-key")
        raise typer.Exit(code=1)

    if chain_mode is not None:
        config.chain_mode = chain_mode.replace("-", "_")  # type: ignore[assignment]

    if quick:
        config.max_tasks_per_benchmark = 2
        config.execution_models = config.execution_models[:1]

    resolved_artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path(config.output_dir)

    tasks = get_tasks(config.benchmarks, config.max_tasks_per_benchmark)

    console.print("[bold]═══ Stage B: Run Benchmarks ═══[/bold]")
    console.print(f"Chain mode: {config.chain_mode}")
    console.print(f"Execution models: {len(config.execution_models)}")
    console.print(f"Benchmarks: {config.benchmarks}")
    console.print(f"Tasks: {len(tasks)}")
    console.print(f"Artifacts: {resolved_artifacts_dir}")
    console.print(f"Skip chains: {skip_chains}")
    console.print(f"Skip baselines: {skip_baselines}")
    console.print()

    # Execute chains on execution models
    if not skip_chains:
        if config.chain_mode == "category":
            console.print("[bold blue]▶ Executing category chains on execution models...[/bold blue]")
            chain_exec_results = asyncio.run(execute_category_chains(config, tasks, resolved_artifacts_dir))
        else:
            console.print("[bold blue]▶ Executing MAGE chains on execution models...[/bold blue]")
            chain_exec_results = asyncio.run(execute_chains(config, tasks, resolved_artifacts_dir))
        save_chain_executions(chain_exec_results, resolved_artifacts_dir)
        exec_success = sum(1 for r in chain_exec_results if r.success)
        console.print(f"  ✓ {exec_success}/{len(chain_exec_results)} chains executed")
    else:
        console.print("[yellow]Chain execution skipped (--skip-chains)[/yellow]")

    # Execute baselines on execution models
    if not skip_baselines:
        console.print("[bold blue]▶ Executing baselines on execution models...[/bold blue]")
        baseline_exec_results = asyncio.run(execute_baselines(config, tasks))
        save_baseline_executions(baseline_exec_results, str(resolved_artifacts_dir))
        bl_success = sum(1 for r in baseline_exec_results if r.success)
        console.print(f"  ✓ {bl_success}/{len(baseline_exec_results)} baselines executed")
    else:
        console.print("[yellow]Baseline execution skipped (--skip-baselines)[/yellow]")

    console.print()
    console.print(f"[bold green]✓ Benchmarks complete. Results in {resolved_artifacts_dir}[/bold green]")


# ---------------------------------------------------------------------------
# Combined: run-all
# ---------------------------------------------------------------------------


@app.command("run-all")
def run_all(
    api_key: str | None = typer.Option(
        None,
        envvar="OPENROUTER_API_KEY",
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)",
    ),
    config_path: Path = typer.Option(
        "evaluation/eval_config.toml",
        help="Path to TOML config",
    ),
    quick: bool = typer.Option(
        False,
        help="Quick mode: 2 tasks per benchmark, 1 model",
    ),
    mode: str | None = typer.Option(
        None,
        help='Chain mode: "category" (default) or "per-task" (legacy)',
    ),
) -> None:
    """Run both stages end-to-end: generate artifacts, then benchmark."""
    generate_artifacts(api_key=api_key, config_path=config_path, quick=quick, mode=mode)
    run_benchmarks(
        api_key=api_key,
        config_path=config_path,
        artifacts_dir=None,
        skip_chains=False,
        skip_baselines=False,
        quick=quick,
        chain_mode=mode,
    )


# ---------------------------------------------------------------------------
# Legacy: generate (back-compat alias)
# ---------------------------------------------------------------------------


@app.command()
def generate(
    config_path: Path = typer.Option("evaluation/eval_config.toml", help="Path to TOML config file"),
    chains: bool = typer.Option(True, help="Generate CARL chains"),
    baselines: bool = typer.Option(True, help="Generate baseline prompts"),
) -> None:
    """Generate CARL chains and baseline prompts for evaluation.

    .. deprecated::
        Use ``generate-artifacts`` instead. This command is kept for
        backwards compatibility.
    """
    warnings.warn(
        "The 'generate' command is deprecated. Use 'generate-artifacts' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = EvalConfig.from_toml(config_path)
    tasks = get_tasks(config.benchmarks, config.max_tasks_per_benchmark)

    console.print(f"[bold]Loaded {len(tasks)} tasks across {len(config.benchmarks)} benchmarks[/bold]")

    if chains:
        console.print("[bold blue]Generating CARL chains...[/bold blue]")
        chain_results = asyncio.run(generate_chains(config, tasks))
        save_chains(chain_results, config.output_dir)

        success_count = sum(1 for r in chain_results if r.success)
        console.print(f"  Chains: {success_count}/{len(chain_results)} succeeded")

    if baselines:
        console.print("[bold blue]Generating baseline prompts...[/bold blue]")
        baseline_results = generate_baselines(config, tasks)
        save_baselines(baseline_results, config.output_dir)
        console.print(f"  Baselines: {len(baseline_results)} generated")

    console.print("[bold green]Done.[/bold green]")


# ---------------------------------------------------------------------------
# list-tasks
# ---------------------------------------------------------------------------


@app.command("list-tasks")
def list_tasks(
    benchmarks: list[str] | None = typer.Argument(default=None, help="Benchmarks to list (default: all)"),
) -> None:
    """List available benchmark tasks."""
    table = Table(title="Benchmark Tasks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Task ID", style="green")
    table.add_column("Category")
    table.add_column("Query", max_width=60)

    names = benchmarks or list(BENCHMARK_TASKS.keys())

    for name in names:
        tasks = BENCHMARK_TASKS.get(name, [])
        for task in tasks:
            table.add_row(task.benchmark, task.task_id, task.category, task.query[:60])

    console.print(table)


if __name__ == "__main__":
    app()
