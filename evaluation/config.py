"""
Evaluation configuration — Pydantic v2 model loaded from TOML.

Reads from the ``[evaluation]`` section of a TOML file and provides sensible
defaults for OpenRouter-based chain generation experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EvalConfig(BaseModel):
    """Configuration for evaluation experiments.

    Parameters
    ----------
    api_key : str | None
        OpenRouter API key.  Falls back to ``OPENROUTER_API_KEY`` env var.
    base_url : str
        OpenRouter-compatible API base URL.
    generation_models : list[str]
        Models used to generate CARL chains.
    execution_models : list[str]
        Models used for later chain execution (metadata only).
    mage_mode : Literal["fast", "deep"]
        MAGE generation mode.
    temperature : float
        Sampling temperature for LLM calls.
    baseline_types : list[str]
        Baseline prompt strategies to generate.
    benchmarks : list[str]
        Benchmarks to include in evaluation.
    output_dir : str
        Directory for generated artifacts.
    max_tasks_per_benchmark : int | None
        Cap on tasks per benchmark (``None`` = use all seed tasks).
    """

    # OpenRouter settings
    api_key: str | None = Field(default=None, description="OpenRouter API key (fallback: OPENROUTER_API_KEY env)")
    base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter-compatible API base URL")

    # Models
    generation_models: list[str] = Field(
        default_factory=lambda: [
            "deepseek/deepseek-v3.2",
            "minimax/minimax-m2.7",
            "z-ai/glm-5.1",
            "moonshotai/kimi-k2.5",
        ],
        description="Large models used to GENERATE MAGE CARL chains and baseline prompts.",
    )
    execution_models: list[str] = Field(
        default_factory=lambda: [
            "qwen/qwen3.5-9b",
            "meta-llama/llama-3.3-8b-instruct",
            "microsoft/phi-4",
            "mistralai/ministral-3b-2512",
        ],
        description="Small models (≤15B) that EXECUTE chains and baselines on benchmarks.",
    )
    use_carl_runtime: bool = Field(
        default=True,
        description=(
            "Execute chains through the REAL mmar-carl DAG runtime (T35): tool "
            "steps actually run (deterministic mock-HTTP tool set). False falls "
            "back to the legacy flattened-prompt stub, which never executed "
            "tool steps."
        ),
    )

    # MAGE generation settings
    mage_mode: Literal["fast", "deep"] = Field(default="deep", description="MAGE generation mode")
    temperature: float = Field(default=0.2, description="Sampling temperature")

    # Baselines
    baseline_types: list[str] = Field(
        default_factory=lambda: ["zero_shot", "cot", "plan_and_solve"],
        description="Baseline prompt strategies to generate",
    )

    # Benchmarks
    benchmarks: list[str] = Field(
        default_factory=lambda: ["hover", "ifbench", "hotpotqa", "livebench_math"],
        description="Benchmarks to include",
    )

    # Output
    output_dir: str = Field(default="evaluation/results", description="Directory for generated artifacts")

    # Limits
    max_tasks_per_benchmark: int | None = Field(default=None, description="Cap on tasks per benchmark (None = all)")

    # Step type restriction
    allowed_step_types: list[str] | None = Field(
        default=None,
        description="Restrict generated chains to these step types (None = all allowed)",
    )

    # Chain generation mode
    chain_mode: Literal["category", "per_task"] = Field(
        default="category",
        description=(
            'Chain generation mode: "category" produces one abstract chain per benchmark '
            '(correct for experiments), "per_task" produces one chain per question (legacy)'
        ),
    )
    category_example_count: int = Field(
        default=5, description="Number of example questions shown when building a category chain"
    )
    contamination_check_enabled: bool = Field(
        default=True, description="Run contamination check on generated category chains"
    )

    # Baseline execution
    execute_baselines: bool = Field(
        default=True, description="Execute baseline prompts on LLMs (not just generate text)"
    )
    baseline_max_tokens: int = Field(default=2048, description="Max tokens for baseline LLM responses")
    baseline_concurrency: int = Field(default=5, description="Max concurrent baseline execution requests")

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a TOML file's ``[evaluation]`` section.

        Parameters
        ----------
        path : str | Path
            Path to the TOML config file.

        Returns
        -------
        EvalConfig
            Configuration populated from the TOML ``[evaluation]`` section.
        """
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        eval_data = data.get("evaluation", {})

        # Fallback for API key
        if not eval_data.get("api_key"):
            env_key = os.getenv("OPENROUTER_API_KEY")
            if env_key:
                eval_data["api_key"] = env_key

        return cls.model_validate(eval_data)
