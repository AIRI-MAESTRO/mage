"""Schemas for synthetic-benchmark self-verification (plan item P1.16).

``benchmark`` owns the test-case / score schemas (``SynthesizedTask``,
``BenchmarkSynthesisResult``, ``ChainScore``) that were originally defined in
``evolve``; ``evolve.schemas`` now re-exports them so the evolutionary loop is a
*consumer* of this module. ``SyntheticBenchmark`` and ``QualityReport`` are the
new contract for the generator's self-verification stage.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SynthesizedTask(BaseModel):
    """A single synthesized benchmark task / test case.

    Parameters
    ----------
    task_id : str
        Stable identifier (``"synth_001"`` etc.).
    input : str
        Input the chain should process — a concrete instance, not a template.
    expected_output : str | None
        Reference answer or expected output. May be ``None`` for open-ended
        tasks where only rubric checks apply.
    rubric : list[str]
        Verifiable properties the output must satisfy. Used by structural
        grading and as anchors for the LLM-judge.
    difficulty : Literal["easy", "medium", "hard"]
        Difficulty hint, mainly for diagnostic reporting.
    source : Literal["llm", "web", "memory", "user"]
        Where this task came from. Useful for debugging which source produces
        which kind of failure.
    metadata : dict[str, Any]
        Anything else (citation URL, source dataset name, raw input fields).
    """

    task_id: str
    input: str
    expected_output: str | None = None
    rubric: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source: Literal["llm", "web", "memory", "user"] = "llm"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSynthesisResult(BaseModel):
    """Output of a benchmark synthesizer."""

    tasks: list[SynthesizedTask] = Field(default_factory=list)
    rubric_summary: str = ""
    sources_used: list[str] = Field(default_factory=list)


class ChainScore(BaseModel):
    """Score of a chain on a synthesized benchmark.

    ``per_task_scores`` is keyed by ``task_id`` and stores the raw 0-1 score
    for that task (any aggregation is up to the caller).
    """

    overall: float = 0.0
    per_task_scores: dict[str, float] = Field(default_factory=dict)
    per_task_failures: dict[str, str] = Field(default_factory=dict)
    judge_notes: str = ""
    # P-5: set when the chain shows reward-hacking signs (abnormally high
    # rubric-keyword density across the step JSON). Monitoring only — it does
    # not change `overall`.
    suspect_gaming: bool = False


class SyntheticBenchmark(BaseModel):
    """A mini-benchmark synthesized on the fly for one query (P1.16).

    Parameters
    ----------
    query : str
        The original user query the benchmark was built for.
    domain : str
        Domain label (inherited from the DomainAnalysis).
    tasks : list[SynthesizedTask]
        The synthesized test cases.
    metric_kind : Literal["rule", "llm_judge", "structural_rubric", "hybrid"]
        How the chain's output should be scored against the tasks.
    provenance : dict[str, Any]
        Which prompt / model / seed produced this benchmark (for reproducibility).
    """

    query: str
    domain: str = "general"
    tasks: list[SynthesizedTask] = Field(default_factory=list)
    metric_kind: Literal["rule", "llm_judge", "structural_rubric", "hybrid"] = "hybrid"
    provenance: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """Result of running a generated chain against its synthetic benchmark (P1.16).

    Parameters
    ----------
    n_cases : int
        Number of test cases evaluated.
    pass_rate : float
        Fraction of cases that passed (0-1).
    per_case_results : list[dict[str, Any]]
        Per-case outcome ({task_id, score, passed, failure?}).
    failure_modes : list[str]
        Diagnosed clusters of failures.
    suggested_fixes : list[str]
        Targeted fixes for the next iteration.
    """

    n_cases: int = 0
    pass_rate: float = 0.0
    per_case_results: list[dict[str, Any]] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
