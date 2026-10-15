"""Synthetic-benchmark self-verification (plan item P1.16).

Owns the test-case / score schemas (consumed by ``evolve``) and the
``SyntheticBenchmark`` / ``QualityReport`` contract for the generator's
self-verification stage.
"""

from __future__ import annotations

from .reporter import QualityReporter
from .runner import ChainRunner
from .schemas import (
    BenchmarkSynthesisResult,
    ChainScore,
    QualityReport,
    SynthesizedTask,
    SyntheticBenchmark,
)
from .synthesizer import BenchmarkSynthesizer

__all__ = [
    "BenchmarkSynthesisResult",
    "BenchmarkSynthesizer",
    "ChainRunner",
    "ChainScore",
    "QualityReport",
    "QualityReporter",
    "SynthesizedTask",
    "SyntheticBenchmark",
]
