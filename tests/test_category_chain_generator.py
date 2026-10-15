"""Tests for category-level chain generation — one abstract chain per benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.category_chain_generator import (
    BENCHMARK_DESCRIPTIONS,
    CategoryChainResult,
    build_category_query,
    check_contamination,
    generate_category_chains,
    save_category_chains,
)
from evaluation.config import EvalConfig
from evaluation.tasks import BenchmarkTask

from .conftest import SAMPLE_CARL_CHAIN, _make_openai_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_EXAMPLES = [
    BenchmarkTask(
        benchmark="hotpotqa",
        task_id="hotpotqa_001",
        query=(
            "What is the birthplace of the author who wrote the novel that "
            "inspired the 1994 film The Shawshank Redemption?"
        ),
    ),
    BenchmarkTask(
        benchmark="hotpotqa",
        task_id="hotpotqa_002",
        query=(
            "Which university did the physicist who formulated the general "
            "theory of relativity attend for his undergraduate studies?"
        ),
    ),
    BenchmarkTask(
        benchmark="hotpotqa",
        task_id="hotpotqa_003",
        query="What is the capital city of the country where the inventor of the World Wide Web was born?",
    ),
]


def _minimal_eval_config(**overrides: Any) -> EvalConfig:
    """Build a minimal EvalConfig for testing."""
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "generation_models": ["test/model-1"],
        "execution_models": ["test/exec-1"],
        "benchmarks": ["hotpotqa"],
        "baseline_types": ["zero_shot"],
        "chain_mode": "category",
        "category_example_count": 5,
        "contamination_check_enabled": True,
    }
    defaults.update(overrides)
    return EvalConfig(**defaults)


# ---------------------------------------------------------------------------
# build_category_query tests
# ---------------------------------------------------------------------------


class TestBuildCategoryQuery:
    """Tests for the category query builder."""

    def test_contains_benchmark_description(self) -> None:
        """Query includes the benchmark description from BENCHMARK_DESCRIPTIONS."""
        examples = SAMPLE_EXAMPLES[:2]
        query = build_category_query("hotpotqa", examples)

        assert "hotpotqa" in query
        desc = BENCHMARK_DESCRIPTIONS["hotpotqa"]
        assert desc in query

    def test_includes_all_examples(self) -> None:
        """Query includes all provided example questions."""
        examples = SAMPLE_EXAMPLES[:3]
        query = build_category_query("hotpotqa", examples)

        for ex in examples:
            assert ex.query in query

    def test_asks_for_abstract_chain(self) -> None:
        """Query instructs the model to create an abstract, general-purpose chain."""
        examples = SAMPLE_EXAMPLES[:1]
        query = build_category_query("hotpotqa", examples)

        assert "ABSTRACT" in query or "abstract" in query.lower()
        assert "GENERAL-PURPOSE" in query or "general" in query.lower()
        assert "placeholder" in query.lower() or "{question}" in query
        assert "PROCESS" in query or "process" in query.lower()

    def test_unknown_benchmark_uses_fallback(self) -> None:
        """Unknown benchmark name gets a sensible fallback description."""
        examples = [BenchmarkTask(benchmark="custom_bench", task_id="c_001", query="Test")]
        query = build_category_query("custom_bench", examples)

        assert "custom_bench" in query
        assert "benchmark tasks" in query


# ---------------------------------------------------------------------------
# check_contamination tests
# ---------------------------------------------------------------------------


class TestCheckContamination:
    """Tests for the contamination detector."""

    def test_detects_capitalized_entities(self) -> None:
        """Capitalized multi-word phrases from examples trigger contamination."""
        examples = [
            BenchmarkTask(
                benchmark="hotpotqa",
                task_id="hp_001",
                query="What is the birthplace of Stephen King?",
            ),
        ]
        chain_dict = {
            "steps": [
                {
                    "aim": "Look up Stephen King's biography",
                    "reasoning_questions": "Where was Stephen King born?",
                    "stage_action": "Search for Stephen King",
                },
            ],
        }

        result = check_contamination(chain_dict, examples)
        assert result["contaminated"] is True
        assert "stephen king" in result["detected_entities"]

    def test_detects_specific_numbers(self) -> None:
        """4+ digit numbers from examples trigger contamination."""
        examples = [
            BenchmarkTask(
                benchmark="hotpotqa",
                task_id="hp_001",
                query="What happened in the 1994 film that was a classic?",
            ),
        ]
        chain_dict = {
            "steps": [
                {
                    "aim": "Look up the 1994 film",
                    "stage_action": "Search for 1994 events",
                },
            ],
        }

        result = check_contamination(chain_dict, examples)
        assert result["contaminated"] is True
        assert "1994" in result["detected_entities"]

    def test_clean_chain_no_false_positives(self) -> None:
        """Abstract chain with no leaked entities passes contamination check."""
        examples = [
            BenchmarkTask(
                benchmark="hotpotqa",
                task_id="hp_001",
                query="What is the birthplace of Stephen King?",
            ),
        ]
        chain_dict = {
            "steps": [
                {
                    "aim": "Identify the key entity mentioned in the question",
                    "reasoning_questions": "What is the main subject of the question?",
                    "step_context_queries": ["entity", "subject", "person"],
                    "stage_action": "Extract the primary entity from {question}",
                    "example_reasoning": "The question asks about a property of some entity",
                    "title": "Extract entity",
                },
                {
                    "aim": "Find the requested property of the entity",
                    "reasoning_questions": "What attribute is being asked about?",
                    "step_context_queries": ["attribute", "property"],
                    "stage_action": "Look up the property for {entity}",
                    "example_reasoning": "Chain the entity to its attribute",
                    "title": "Lookup property",
                },
            ],
        }

        result = check_contamination(chain_dict, examples)
        assert result["contaminated"] is False
        assert result["detected_entities"] == []
        assert result["entities_checked"] > 0

    def test_empty_examples(self) -> None:
        """No examples means nothing to check — always clean."""
        result = check_contamination({"steps": []}, [])
        assert result["contaminated"] is False
        assert result["entities_checked"] == 0

    def test_list_field_checked(self) -> None:
        """Contamination is detected in list-type step fields."""
        examples = [
            BenchmarkTask(
                benchmark="hover",
                task_id="h_001",
                query="Verify that Leonardo DiCaprio won an Oscar.",
            ),
        ]
        chain_dict = {
            "steps": [
                {
                    "aim": "Verify the claim",
                    "step_context_queries": ["Leonardo DiCaprio", "Oscar"],
                },
            ],
        }

        result = check_contamination(chain_dict, examples)
        assert result["contaminated"] is True


# ---------------------------------------------------------------------------
# generate_category_chains tests
# ---------------------------------------------------------------------------


class TestGenerateCategoryChains:
    """Tests for async category chain generation."""

    @pytest.fixture
    def mock_openai_cat(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for category chain generation."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    async def test_one_per_benchmark_per_model(self, mock_openai_cat: MagicMock) -> None:
        """Generates exactly one chain per (benchmark x model)."""
        config = _minimal_eval_config(
            generation_models=["test/model-1"],
            benchmarks=["hotpotqa", "hover"],
        )
        tasks_by_benchmark = {
            "hotpotqa": SAMPLE_EXAMPLES,
            "hover": [
                BenchmarkTask(benchmark="hover", task_id="hover_001", query="Verify X."),
            ],
        }

        results = await generate_category_chains(config, tasks_by_benchmark)

        assert len(results) == 2
        benchmarks = {r.benchmark for r in results}
        assert benchmarks == {"hotpotqa", "hover"}
        assert all(r.generation_model == "test/model-1" for r in results)
        assert all(r.success for r in results)

    async def test_multiple_models(self, mock_openai_cat: MagicMock) -> None:
        """Multiple generation models produce separate chains per benchmark."""
        config = _minimal_eval_config(
            generation_models=["test/model-1", "test/model-2"],
            benchmarks=["hotpotqa"],
        )
        tasks_by_benchmark = {"hotpotqa": SAMPLE_EXAMPLES}

        results = await generate_category_chains(config, tasks_by_benchmark)

        assert len(results) == 2
        models = {r.generation_model for r in results}
        assert models == {"test/model-1", "test/model-2"}

    async def test_error_handling(self, mocker: Any) -> None:
        """LLM failure is captured gracefully."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM error"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        config = _minimal_eval_config(
            generation_models=["test/model-1"],
            benchmarks=["hotpotqa"],
        )
        tasks_by_benchmark = {"hotpotqa": SAMPLE_EXAMPLES}

        results = await generate_category_chains(config, tasks_by_benchmark)

        assert len(results) == 1
        assert results[0].success is False
        assert "LLM error" in (results[0].error or "")

    async def test_contamination_check_runs(self, mock_openai_cat: MagicMock) -> None:
        """Contamination check is performed when enabled."""
        config = _minimal_eval_config(
            generation_models=["test/model-1"],
            benchmarks=["hotpotqa"],
            contamination_check_enabled=True,
        )
        tasks_by_benchmark = {"hotpotqa": SAMPLE_EXAMPLES}

        results = await generate_category_chains(config, tasks_by_benchmark)

        assert len(results) == 1
        assert "contaminated" in results[0].contamination_check

    async def test_contamination_check_disabled(self, mock_openai_cat: MagicMock) -> None:
        """Contamination check is skipped when disabled."""
        config = _minimal_eval_config(
            generation_models=["test/model-1"],
            benchmarks=["hotpotqa"],
            contamination_check_enabled=False,
        )
        tasks_by_benchmark = {"hotpotqa": SAMPLE_EXAMPLES}

        results = await generate_category_chains(config, tasks_by_benchmark)

        assert len(results) == 1
        assert results[0].contamination_check.get("contaminated") is False


# ---------------------------------------------------------------------------
# save_category_chains tests
# ---------------------------------------------------------------------------


class TestSaveCategoryChains:
    """Tests for saving category chain artifacts."""

    def test_save_layout(self, tmp_path: Path) -> None:
        """Verify output directory structure: category_chains/<model>/<benchmark>.json."""
        results = [
            CategoryChainResult(
                benchmark="hotpotqa",
                generation_model="deepseek/deepseek-v3.2",
                num_examples_used=5,
                chain_json=json.dumps(SAMPLE_CARL_CHAIN),
                chain_dict=SAMPLE_CARL_CHAIN,
                code="# generated code",
                metadata={"model": "test"},
                generation_time_seconds=2.0,
                contamination_check={"contaminated": False, "detected_entities": []},
            ),
        ]

        save_category_chains(results, tmp_path)

        chain_dir = tmp_path / "category_chains" / "deepseek--deepseek-v3.2"
        assert chain_dir.exists()
        assert (chain_dir / "hotpotqa.json").exists()
        assert (chain_dir / "hotpotqa.py").exists()
        assert (chain_dir / "hotpotqa_meta.json").exists()

        # Verify chain content
        chain_data = json.loads((chain_dir / "hotpotqa.json").read_text())
        assert "steps" in chain_data

        # Verify metadata
        meta = json.loads((chain_dir / "hotpotqa_meta.json").read_text())
        assert meta["generation_model"] == "deepseek/deepseek-v3.2"
        assert meta["benchmark"] == "hotpotqa"
        assert meta["num_examples_used"] == 5

    def test_save_skips_failed(self, tmp_path: Path) -> None:
        """Failed results are not saved to disk."""
        results = [
            CategoryChainResult(
                benchmark="hotpotqa",
                generation_model="test/model-1",
                success=False,
                error="generation failed",
            ),
        ]

        save_category_chains(results, tmp_path)
        assert not (tmp_path / "category_chains").exists()

    def test_save_no_code(self, tmp_path: Path) -> None:
        """No .py file is created when code is None."""
        results = [
            CategoryChainResult(
                benchmark="hotpotqa",
                generation_model="test/model-1",
                chain_json=json.dumps(SAMPLE_CARL_CHAIN),
                chain_dict=SAMPLE_CARL_CHAIN,
                code=None,
            ),
        ]

        save_category_chains(results, tmp_path)
        chain_dir = tmp_path / "category_chains" / "test--model-1"
        assert (chain_dir / "hotpotqa.json").exists()
        assert not (chain_dir / "hotpotqa.py").exists()
