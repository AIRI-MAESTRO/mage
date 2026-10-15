"""
Evaluation module for MAGE — batch chain and baseline generation for experiments.

Generates CARL chains using generation models via OpenRouter and constructs
deterministic baseline prompts (zero-shot, CoT, plan-and-solve) for the same
benchmark tasks.  Output artifacts are saved as JSON / text for later benchmark
execution.
"""

from __future__ import annotations

from .baseline_generator import (
    BaselineExecution,
    BaselineResult,
    execute_baselines,
    generate_baselines,
    save_baseline_executions,
    save_baselines,
)
from .category_chain_generator import (
    CategoryChainResult,
    generate_category_chains,
    save_category_chains,
)
from .chain_executor import ChainExecution, execute_category_chains, save_chain_executions
from .chain_executor import execute_chains as execute_chains_on_models
from .chain_generator import ChainGenerationResult, generate_chains, save_chains
from .config import EvalConfig
from .tasks import BENCHMARK_TASKS, BenchmarkTask, get_tasks

__all__ = [
    # Config
    "EvalConfig",
    # Tasks
    "BenchmarkTask",
    "BENCHMARK_TASKS",
    "get_tasks",
    # Chain generation
    "ChainGenerationResult",
    "generate_chains",
    "save_chains",
    # Category chain generation
    "CategoryChainResult",
    "generate_category_chains",
    "save_category_chains",
    # Chain execution
    "ChainExecution",
    "execute_chains_on_models",
    "execute_category_chains",
    "save_chain_executions",
    # Baseline generation
    "BaselineResult",
    "generate_baselines",
    "save_baselines",
    # Baseline execution
    "BaselineExecution",
    "execute_baselines",
    "save_baseline_executions",
]
