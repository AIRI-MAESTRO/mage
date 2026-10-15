"""Evolve mode — benchmark-driven evolutionary chain optimization.

The evolve module turns MAGE from a single-shot generator into an optimizer
over chain space. Pipeline:

1. **Benchmark synthesis** — given a user query, synthesize a set of
   verifiable test cases (LLM-generated + web-recalled + memory-recalled).
2. **Initial population** — generate K diverse chains via :class:`MAGEGenerator`.
3. **Evaluation** — run every chain on every benchmark task, collect scores.
4. **Evolution** — apply mutation / crossover operators, keep top-K, regenerate.
5. Loop until plateau or ``max_generations`` reached.

The result is the best chain found *along with* the synthesized benchmark
(stored in memory for future runs in the same domain).
"""

from __future__ import annotations

from .engine import ChainEvolver
from .schemas import (
    BenchmarkSynthesisResult,
    ChainScore,
    EvolveConfig,
    EvolveGeneration,
    EvolveResult,
    Individual,
    SynthesizedTask,
)

__all__ = [
    "BenchmarkSynthesisResult",
    "ChainEvolver",
    "ChainScore",
    "EvolveConfig",
    "EvolveGeneration",
    "EvolveResult",
    "Individual",
    "SynthesizedTask",
]
