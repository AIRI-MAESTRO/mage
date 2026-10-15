"""Pydantic schemas for the evolve module.

The benchmark/test-case schemas (``SynthesizedTask``, ``BenchmarkSynthesisResult``,
``ChainScore``) now live in :mod:`mmar_mage.benchmark.schemas` (P1.16) and are
re-exported here for backward-compatibility — ``evolve`` is a *consumer* of the
benchmark module's contract. Only evolution-specific schemas are defined below.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Re-exported from the benchmark module, which now owns these (P1.16).
from ..benchmark.schemas import BenchmarkSynthesisResult, ChainScore, SynthesizedTask
from ..schemas import StageUsage

__all__ = [
    "BenchmarkSynthesisResult",
    "ChainScore",
    "SynthesizedTask",
    "Individual",
    "EvolveGeneration",
    "EvolveConfig",
    "EvolveResult",
]


class Individual(BaseModel):
    """One chain in the evolutionary population."""

    individual_id: str
    chain_dict: dict[str, Any]
    score: ChainScore = Field(default_factory=ChainScore)
    parents: list[str] = Field(default_factory=list)
    operator: str = "initial"  # how this individual was produced
    generation: int = 0
    # WP-5: actual token usage spent producing this individual (when known).
    usage: StageUsage | None = None


class EvolveGeneration(BaseModel):
    """Snapshot of one evolutionary generation."""

    generation: int
    population_size: int
    best_score: float
    median_score: float
    best_individual_id: str
    operators_used: dict[str, int] = Field(default_factory=dict)
    # WP-3: children rejected (invalid chain / over budget) before a valid one
    # filled the slot — surfaces operator flakiness instead of silently shrinking.
    invalid_children: int = 0


class EvolveConfig(BaseModel):
    """Configuration for an evolve run.

    Reasonable defaults for a small dev run. Production runs probably want
    larger ``population_size`` and ``num_generations``.
    """

    # Benchmark synthesis
    num_benchmark_tasks: int = 8
    enable_web_synthesis: bool = False
    enable_memory_synthesis: bool = True
    user_provided_tasks: list[SynthesizedTask] = Field(default_factory=list)

    # Population
    population_size: int = 5
    elitism: int = 2  # keep top-K unchanged each generation
    # P1.8 (GEPA Pareto filter): when on, survivor selection prefers the
    # (quality ↑, cost ↓) Pareto front before filling remaining slots by quality —
    # so a cheaper chain of equal quality is never crowded out by a pricier one.
    # OFF by default (pure-quality selection); cost axis = estimated total tokens.
    enable_pareto_selection: bool = False

    # Evolution loop
    num_generations: int = 4
    plateau_patience: int = 3  # stop if best score doesn't improve for N gens
    # WP-3: an improvement must exceed this to reset the plateau counter — guards
    # a noisy judge signal from triggering a premature stop on ~1e-6 wiggles.
    plateau_min_delta: float = 0.01
    score_threshold: float = 0.95  # early-stop if reached

    # Operator weights (normalized internally)
    weight_mutate_aim: float = 1.0
    weight_change_step_type: float = 0.5
    weight_insert_step: float = 0.7
    weight_remove_step: float = 0.5
    weight_swap_steps: float = 0.5
    weight_crossover: float = 1.0
    weight_llm_repair: float = 1.5  # LLM-driven targeted improvement
    weight_reflective_mutate: float = 1.2  # GEPA-style diagnose-then-edit

    # Grading
    grader_temperature: float = 0.0
    use_llm_judge: bool = True
    use_structural_rubric: bool = True
    # T26: max benchmark tasks graded concurrently (each runs independent judge /
    # execution LLM calls). Bounds provider load; 1 = sequential.
    grading_concurrency: int = 4

    # Real-execution grading (stub v1: chain → flattened prompt → LLM call →
    # judge real output). When ``execution_model`` is set, the grader runs the
    # chain on the configured model BEFORE judging — this turns the LLM-judge
    # from a "predict what the chain would say" into "score what it actually
    # did say". When None, the grader falls back to chain-summary judging.
    execution_model: str | None = None
    execution_max_tokens: int = 1024
    execution_temperature: float = 0.2

    # Diversity
    initial_temperatures: list[float] = Field(default_factory=lambda: [0.2, 0.5, 0.7, 0.9, 1.0])

    # Persistence
    save_benchmark_to_memory: bool = True
    save_history: bool = True

    # Cost budget guard (Phase 3). When max_cost_usd or max_tokens_per_chain
    # is set, candidates exceeding either limit are filtered out before
    # grading — saves compute on obviously-too-expensive mutants.
    max_cost_usd: float | None = None
    max_tokens_per_chain: int | None = None
    pricing: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Optional {model: (in_per_1k_usd, out_per_1k_usd)} for cost guard.",
    )


class EvolveResult(BaseModel):
    """Final output of an evolve run."""

    query: str
    best_individual: Individual
    benchmark: BenchmarkSynthesisResult
    history: list[EvolveGeneration] = Field(default_factory=list)
    top_k: list[Individual] = Field(default_factory=list)
    total_generations_run: int = 0
    total_individuals_evaluated: int = 0
    early_stopped: bool = False
    early_stop_reason: str = ""
