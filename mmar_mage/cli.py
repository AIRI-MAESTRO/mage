"""
CLI interface for MAGE.

Usage::

    mage generate "Analyze a financial report" --mode deep
    mage generate "Classify customer feedback" --mode fast --output chain.json
    mage version
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .evolve import ChainEvolver, EvolveConfig
from .generator import MAGEGenerator
from .schemas import MAGEConfig

app = typer.Typer(
    name="mage",
    help="MAGE — Mesh Architecture Generation Engine. Generate CARL reasoning chains from text queries.",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Rich progress callback
# ---------------------------------------------------------------------------


class RichProgressCallback:
    """Displays pipeline progress via Rich console."""

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task_id = progress.add_task("Generating chain...", total=None)

    def on_stage_start(self, stage: str) -> None:
        self._progress.update(self._task_id, description=f"[bold blue]{stage}[/]...")

    def on_stage_complete(self, stage: str, result: object) -> None:
        self._progress.update(self._task_id, description=f"[green]{stage}[/] ✓")

    def on_error(self, stage: str, error: Exception) -> None:
        self._progress.update(self._task_id, description=f"[red]{stage} failed[/]")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def generate(
    query: str = typer.Argument(..., help="Text query describing the desired reasoning chain"),
    mode: str = typer.Option("deep", "--mode", "-m", help="Generation mode: fast or deep"),
    model: str = typer.Option("gpt-4o", "--model", help="LLM model to use"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to write the CARL JSON file"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable saving to gigaevo-memory"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="MAGE_API_KEY", help="LLM API key"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    temperature: float = typer.Option(0.2, "--temperature", "-t", help="Sampling temperature"),
    skills: bool = typer.Option(
        False,
        "--skills",
        help="Enable SkillDiscoveryAgent — surface packaged AgentSkills (PDF/DOCX/etc.) for the planner.",
    ),
    skills_web: bool = typer.Option(
        False,
        "--skills-web",
        help="Allow skill discovery to fall back to web search (requires --web-search-key).",
    ),
    benchmark_profile: str | None = typer.Option(
        None,
        "--benchmark-profile",
        help="Authoring preset for a benchmark: bfcl | seal_tools | rotbench | toolalpaca | custom (P3.5).",
    ),
) -> None:
    """Generate a CARL reasoning chain from a text query."""
    if mode not in ("fast", "deep"):
        console.print("[red]Error:[/] --mode must be 'fast' or 'deep'")
        raise typer.Exit(1)

    from .profiles import list_profiles

    if benchmark_profile is not None and benchmark_profile not in list_profiles():
        console.print(f"[red]Error:[/] --benchmark-profile must be one of {list_profiles()}")
        raise typer.Exit(1)

    config = MAGEConfig(
        mode=mode,  # type: ignore[arg-type]
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        enable_memory=not no_memory,
        enable_skill_discovery=skills,
        enable_skill_web_discovery=skills_web,
        benchmark_profile=benchmark_profile,  # type: ignore[arg-type]
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        callback = RichProgressCallback(progress)
        # See note at the `mage evolve` call site: partial Protocol match is intentional.
        gen = MAGEGenerator(config=config, progress=callback)  # type: ignore[arg-type]

        try:
            result = asyncio.run(gen.generate(query))
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    # Output
    if output:
        path = Path(output)
        path.write_text(result.chain_json, encoding="utf-8")
        console.print(f"Chain saved to [bold]{path}[/]")
    else:
        console.print(Panel(result.chain_json, title="CARL Chain", border_style="green"))

    console.print(
        f"[dim]Mode: {result.mode} | Steps: {result.metadata.num_steps} | "
        f"Domain: {result.metadata.domain} | Time: {result.metadata.generation_time_seconds}s[/]"
    )

    if result.memory_key:
        console.print(f"[dim]Memory key: {result.memory_key}[/]")


def _print_step_diff(before: dict, after: dict) -> None:
    """Print a compact summary of an edit: name change, step count, after-steps."""
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    if before.get("name") != after.get("name"):
        console.print(f"[dim]name:[/] {before.get('name')!r} → {after.get('name')!r}")
    bsteps = [s for s in before.get("steps", []) if isinstance(s, dict)]
    asteps = [s for s in after.get("steps", []) if isinstance(s, dict)]
    console.print(f"[dim]steps:[/] {len(bsteps)} → {len(asteps)}")
    for s in asteps:
        console.print(
            f"  {s.get('number')}. {s.get('title') or s.get('aim') or ''} "
            f"[dim]({s.get('step_type')})[/]"
        )


@app.command()
def edit(
    instruction: str = typer.Argument(..., help="What to change, in natural language"),
    entity_id: str | None = typer.Option(None, "--id", help="entity_id of the saved chain to edit"),
    chain_file: str | None = typer.Option(
        None, "--chain", help="Path to a local chain JSON to edit (no memory needed)"
    ),
    channel: str = typer.Option("latest", "--channel", help="Version channel to load/save"),
    save: bool = typer.Option(False, "--save", help="Save the edited chain as a new version"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write the edited CARL JSON here"),
    model: str = typer.Option("gpt-4o", "--model", help="LLM model to use"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="MAGE_API_KEY", help="LLM API key"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    temperature: float = typer.Option(0.2, "--temperature", "-t", help="Sampling temperature"),
) -> None:
    """Edit an EXISTING CARL chain from a natural-language instruction.

    Resolve the chain by ``--id``, by a local ``--chain`` file, or — when
    neither is given — from the instruction itself via memory search.
    """
    loaded_chain: dict | None = None
    if chain_file:
        path = Path(chain_file)
        if not path.exists():
            console.print(f"[red]Error:[/] chain file not found: {path}")
            raise typer.Exit(1)
        try:
            loaded_chain = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            console.print(f"[red]Error:[/] could not parse {path}: {exc}")
            raise typer.Exit(1) from exc
    elif not entity_id:
        console.print(
            "[yellow]No --id or --chain:[/] resolving the chain from your "
            "instruction via memory search."
        )

    config = MAGEConfig(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        # A local --chain edit must not try to read/write gigaevo-memory.
        enable_memory=not bool(chain_file),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        callback = RichProgressCallback(progress)
        # See note at the `mage evolve` call site: partial Protocol match is intentional.
        gen = MAGEGenerator(config=config, progress=callback)  # type: ignore[arg-type]
        try:
            result = asyncio.run(
                gen.edit(
                    instruction,
                    chain=loaded_chain,
                    entity_id=entity_id,
                    channel=channel,
                    save=save and not chain_file,
                )
            )
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    if result.needs_disambiguation:
        console.print("[yellow]Multiple chains match — re-run with --id <entity_id>:[/]")
        for c in result.candidates:
            console.print(f"  • [bold]{c.entity_id}[/]  {c.name}  [dim](score {c.score:.2f})[/]")
        raise typer.Exit(0)

    console.print(Panel(result.summary or "No change.", title="Edit summary", border_style="cyan"))
    if result.edits:
        for e in result.edits:
            tgt = f" step {e.target_step_number}" if e.target_step_number is not None else ""
            console.print(f"  • [bold]{e.op}[/]{tgt} — {e.rationale}")
    else:
        console.print("[dim]No edits were applied.[/]")

    _print_step_diff(result.before_chain_dict, result.chain_dict)

    if output:
        Path(output).write_text(result.chain_json, encoding="utf-8")
        console.print(f"Edited chain saved to [bold]{output}[/]")
    elif not save:
        console.print(Panel(result.chain_json, title="Edited CARL Chain", border_style="green"))

    if result.saved and result.memory_key:
        console.print(f"[dim]Saved new version — memory key: {result.memory_key}[/]")
    elif save and not result.saved:
        console.print("[yellow]Not saved[/] (no entity_id resolved or memory disabled).")


@app.command()
def evolve(
    query: str = typer.Argument(..., help="Text query — the task to optimize a chain for"),
    population: int = typer.Option(5, "--population", "-p", help="Population size per generation"),
    generations: int = typer.Option(4, "--generations", "-g", help="Maximum number of generations"),
    benchmark_tasks: int = typer.Option(8, "--benchmark-tasks", "-b", help="Number of synthesized benchmark tasks"),
    enable_web: bool = typer.Option(False, "--web", help="Enable web-based benchmark discovery"),
    enable_memory: bool = typer.Option(True, "--memory/--no-memory", help="Use memory for benchmark recall + saving"),
    score_threshold: float = typer.Option(0.95, "--threshold", help="Early-stop when best score >= threshold"),
    plateau: int = typer.Option(2, "--plateau", help="Stop after N generations without improvement"),
    seed: int | None = typer.Option(None, "--seed", help="RNG seed for reproducibility"),
    model: str = typer.Option("gpt-4o", "--model", help="LLM model to use"),
    execution_model: str | None = typer.Option(
        None,
        "--execution-model",
        help="If set, run each chain on this model (stub v1 flatten) and judge the actual response. "
        "Without it, the judge predicts what the chain would say from its summary.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", envvar="MAGE_API_KEY", help="LLM API key"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    output: str | None = typer.Option(None, "--output", "-o", help="Path to write the best chain JSON"),
    benchmark_output: str | None = typer.Option(
        None, "--benchmark-output", help="Path to write the synthesized benchmark JSON"
    ),
) -> None:
    """Evolve a CARL chain by synthesizing a benchmark and iterating mutations.

    This is the slow-but-better mode: instead of one chain, evolve generates
    a population, scores each candidate against synthesized test cases, and
    applies mutation / crossover / LLM-driven repair until plateau.
    """
    import json as _json

    mage_config = MAGEConfig(
        mode="deep",
        model=model,
        api_key=api_key,
        base_url=base_url,
        enable_memory=enable_memory,
    )
    evolve_config = EvolveConfig(
        num_benchmark_tasks=benchmark_tasks,
        population_size=population,
        num_generations=generations,
        plateau_patience=plateau,
        score_threshold=score_threshold,
        enable_web_synthesis=enable_web,
        enable_memory_synthesis=enable_memory,
        save_benchmark_to_memory=enable_memory,
        execution_model=execution_model,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Evolving...", total=None)
        callback = RichProgressCallback(progress)
        # RichProgressCallback implements the required progress hooks; the
        # optional streaming hooks are probed via hasattr at runtime, so the
        # Protocol's structural match is intentionally partial.
        gen = MAGEGenerator(config=mage_config, progress=callback)  # type: ignore[arg-type]
        evolver = ChainEvolver(generator=gen, evolve_config=evolve_config, rng_seed=seed)

        try:
            result = asyncio.run(evolver.evolve(query))
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, description="[green]Evolution complete[/]")

    best_chain_json = _json.dumps(result.best_individual.chain_dict, ensure_ascii=False, indent=2)

    if output:
        Path(output).write_text(best_chain_json, encoding="utf-8")
        console.print(f"Best chain saved to [bold]{output}[/]")
    else:
        console.print(Panel(best_chain_json, title="Best evolved chain", border_style="green"))

    if benchmark_output:
        bench_dump = _json.dumps(result.benchmark.model_dump(), ensure_ascii=False, indent=2)
        Path(benchmark_output).write_text(bench_dump, encoding="utf-8")
        console.print(f"Benchmark saved to [bold]{benchmark_output}[/]")

    console.print(
        f"[dim]Generations: {result.total_generations_run} | "
        f"Evaluated: {result.total_individuals_evaluated} | "
        f"Best score: {result.best_individual.score.overall:.3f} | "
        f"Tasks: {len(result.benchmark.tasks)}"
        + (f" | Early-stopped: {result.early_stop_reason}" if result.early_stopped else "")
        + "[/]"
    )

    # Per-generation score trace
    if result.history:
        trace = " → ".join(f"{g.best_score:.2f}" for g in result.history)
        console.print(f"[dim]Score trace: {trace}[/]")


@app.command()
def validate(
    chain_path: Path = typer.Argument(..., help="Path to a CARL chain JSON file."),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Run CARL-strict validation (rejects MAGE-internal eval/aggregator types).",
    ),
    check_carl: bool = typer.Option(
        True,
        "--check-carl/--no-check-carl",
        help="If mmar_carl is installed, also run ReasoningChain.from_json.",
    ),
    check_capabilities: bool = typer.Option(
        False,
        "--check-capabilities",
        help=(
            "Soft-check that every agent_skill step's URI exists in the configured"
            " memory environment. Requires --memory-base-url + gigaevo-memory."
        ),
    ),
    memory_base_url: str = typer.Option(
        "http://localhost:8002",
        "--memory-base-url",
        help="gigaevo-memory base URL (only used when --check-capabilities is on).",
    ),
    environment_id: str = typer.Option(
        "default",
        "--environment-id",
        help="capability_registry environment_id (only used when --check-capabilities is on).",
    ),
) -> None:
    """Validate a generated CARL chain JSON file (§8 P3).

    Runs three layers of validation, each independently toggleable:

    1. ``validate_carl_json`` (always) — structural integrity, step
       numbering, dependency targets, cycle detection, valid step types.
    2. ``ReasoningChain.from_json`` (``--check-carl``) — full CARL
       library validation when ``mmar_carl`` is installed.
    3. Soft capability-existence check (``--check-capabilities``) —
       every ``agent_skill`` step's URI is looked up in the configured
       memory environment. Misses are warnings, not errors.

    Exits 0 on success, 1 on validation failure. Designed for use in
    CI / pre-commit hooks.
    """
    import json

    from .agents.capability_lookup_agent import CapabilityLookupAgent
    from .exceptions import MAGEValidationError
    from .generator import _try_load_reasoning_chain, validate_carl_json
    from .memory import MemoryManager

    if not chain_path.is_file():
        console.print(f"[bold red]✗ File not found:[/bold red] {chain_path}")
        raise typer.Exit(1)

    try:
        chain_text = chain_path.read_text(encoding="utf-8")
        chain_dict = json.loads(chain_text)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]✗ Not valid JSON:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(f"[bold red]✗ Cannot read file:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    # 1. Internal validation (always runs).
    try:
        validate_carl_json(chain_dict, strict_carl=False)
        notes.append("[green]✓[/green] validate_carl_json (internal types) passed")
    except MAGEValidationError as exc:
        errors.append(f"validate_carl_json (internal): {exc}")

    # 2. Strict-CARL validation (default on).
    if strict and not errors:
        try:
            validate_carl_json(chain_dict, strict_carl=True)
            notes.append("[green]✓[/green] validate_carl_json (strict CARL) passed")
        except MAGEValidationError as exc:
            errors.append(f"validate_carl_json (strict): {exc}")

    # 3. mmar_carl round-trip (default on, no-ops if mmar_carl missing).
    if check_carl and not errors:
        try:
            loaded = _try_load_reasoning_chain(chain_text)
            if loaded is None:
                notes.append("[dim]·[/dim] mmar_carl not installed — skipped CARL round-trip")
            else:
                notes.append("[green]✓[/green] ReasoningChain.from_json accepted the chain")
        except MAGEValidationError as exc:
            errors.append(f"ReasoningChain.from_json: {exc}")

    # 4. Soft capability-existence check.
    if check_capabilities and not errors:
        skill_refs = _collect_agent_skill_uris(chain_dict)
        if not skill_refs:
            notes.append("[dim]·[/dim] no agent_skill steps — nothing to check")
        else:
            try:
                from .schemas import DomainAnalysis

                cfg = MAGEConfig(
                    enable_memory=True,
                    enable_capability_lookup=True,
                    enable_memory_skill_lookup=True,
                    memory_base_url=memory_base_url,
                    capability_environment_id=environment_id,
                )
                mgr = MemoryManager(cfg)
                agent = CapabilityLookupAgent(mgr, cfg)
                # `find_capability_matches` needs a query — synthesise one
                # from the chain's name so the lookup has something to
                # rank against. The fake-memory test harness ignores it.
                chain_name_part = str(chain_dict.get("name") or chain_path.stem)
                ctx = asyncio.run(
                    agent.lookup(
                        query=f"chain audit: {chain_name_part}",
                        domain_analysis=DomainAnalysis(),
                    )
                )
                known_uris = {entry.uri for entry in ctx.agent_skills}
                missing = sorted(skill_refs - known_uris)
                if missing:
                    for uri in missing:
                        warnings.append(f"agent_skill URI not found in memory env {environment_id!r}: {uri}")
                else:
                    notes.append(f"[green]✓[/green] every agent_skill URI exists in memory env {environment_id!r}")
            except Exception as exc:  # noqa: BLE001 — soft check
                warnings.append(f"capability check could not run: {exc}")

    # ---- Report ----
    title = chain_path.name
    if errors:
        body = "\n".join(notes + [f"[bold red]✗[/bold red] {e}" for e in errors])
        console.print(Panel(body, title=f"FAIL — {title}", border_style="red"))
        raise typer.Exit(1)

    body_lines = list(notes)
    for w in warnings:
        body_lines.append(f"[yellow]⚠[/yellow] {w}")
    if not body_lines:
        body_lines.append("[green]✓[/green] all checks passed")
    console.print(Panel("\n".join(body_lines), title=f"OK — {title}", border_style="green"))


def _collect_agent_skill_uris(chain_dict: dict[str, object]) -> set[str]:
    """Pull every agent_skill step's `skill` / `skill_uri` URI from a chain.

    Reads both the flat shape (``skill_name`` / ``skill``) and the
    nested ``step_config`` shape produced by ``to_carl_compat``.
    """
    out: set[str] = set()
    steps = chain_dict.get("steps") or []
    if not isinstance(steps, list):
        return out
    for step in steps:
        if not isinstance(step, dict):
            continue
        if (step.get("step_type") or "").lower() != "agent_skill":
            continue
        uri = step.get("skill") or step.get("skill_uri") or ""
        sc = step.get("step_config")
        if isinstance(sc, dict):
            uri = uri or sc.get("skill") or sc.get("skill_uri") or ""
        if isinstance(uri, str) and uri:
            out.add(uri)
    return out


@app.command()
def version() -> None:
    """Show MAGE version."""
    from . import __version__

    console.print(f"mmar-mage {__version__}")


if __name__ == "__main__":
    app()
