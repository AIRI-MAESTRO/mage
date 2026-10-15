"""Tests for the chain executor module — load, execute, and save chain artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluation.chain_executor import (
    ChainExecution,
    _build_chain_prompt,
    build_execution_prompt_from_category,
    execute_category_chains,
    execute_chains,
    save_chain_executions,
)
from evaluation.config import EvalConfig
from evaluation.tasks import BenchmarkTask

from .conftest import SAMPLE_CARL_CHAIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_eval_config(**overrides: Any) -> EvalConfig:
    """Build a minimal EvalConfig for testing.

    ``use_carl_runtime=False``: this module tests the LEGACY flattened-prompt
    path (single mocked chat-completions call). The real-runtime path (T35,
    the default in production) is covered by ``test_eval_carl_executor.py``.
    """
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "generation_models": ["test/model-1"],
        "execution_models": ["test/exec-1"],
        "benchmarks": ["hover"],
        "baseline_types": ["zero_shot"],
        "use_carl_runtime": False,
    }
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _write_chain_artifact(
    output_dir: Path,
    gen_model: str,
    benchmark: str,
    task_id: str,
    chain_dict: dict[str, Any] | None = None,
) -> Path:
    """Write a chain JSON artifact to disk and return the path."""
    slug = gen_model.replace("/", "--")
    chain_dir = output_dir / "chains" / slug / benchmark
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_path = chain_dir / f"{task_id}.json"
    chain_path.write_text(
        json.dumps(chain_dict or SAMPLE_CARL_CHAIN),
        encoding="utf-8",
    )
    return chain_path


def _write_category_chain_artifact(
    output_dir: Path,
    gen_model: str,
    benchmark: str,
    chain_dict: dict[str, Any] | None = None,
) -> Path:
    """Write a category chain JSON artifact to disk and return the path."""
    slug = gen_model.replace("/", "--")
    chain_dir = output_dir / "category_chains" / slug
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_path = chain_dir / f"{benchmark}.json"
    chain_path.write_text(
        json.dumps(chain_dict or SAMPLE_CARL_CHAIN),
        encoding="utf-8",
    )
    return chain_path


# ---------------------------------------------------------------------------
# _build_chain_prompt tests
# ---------------------------------------------------------------------------


class TestBuildChainPrompt:
    """Tests for the stub chain prompt builder."""

    def test_builds_prompt_from_aims(self) -> None:
        """Prompt includes QUESTION, numbered step aims, and instructions."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query?")
        prompt, count = _build_chain_prompt(task, SAMPLE_CARL_CHAIN)

        assert "QUESTION: Test query?" in prompt
        assert "REASONING PLAN:" in prompt
        assert "1. Determine the type of document" in prompt
        assert "2. Extract all key facts and data" in prompt
        assert "3. Form a structured summary" in prompt
        assert "step by step" in prompt
        assert count == 3

    def test_empty_steps(self) -> None:
        """Empty chain falls back to a default step."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        prompt, count = _build_chain_prompt(task, {"steps": []})

        assert "QUESTION: Q" in prompt
        assert "Analyze the question" in prompt
        assert count == 0

    def test_falls_back_to_title(self) -> None:
        """When aim is missing, falls back to title."""
        chain = {"steps": [{"title": "My Step Title", "number": 1}]}
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        prompt, count = _build_chain_prompt(task, chain)

        assert "1. My Step Title" in prompt
        assert count == 1


# ---------------------------------------------------------------------------
# execute_chains tests
# ---------------------------------------------------------------------------


class TestExecuteChains:
    """Tests for chain execution via the stub executor."""

    @pytest.fixture
    def mock_openai_chain(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for chain executor."""
        mock_cls = mocker.patch("evaluation.chain_executor.AsyncOpenAI")
        instance = mock_cls.return_value

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Chain execution answer."
        response.usage = MagicMock(total_tokens=30)

        instance.chat.completions.create = AsyncMock(return_value=response)
        return instance

    async def test_execute_chains_loads_from_disk(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """execute_chains reads chain JSON files from artifacts directory."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")
        _write_chain_artifact(tmp_path, "test/model-1", "hover", "hover_001")

        config = _minimal_eval_config(execution_models=["test/exec-1"])
        results = await execute_chains(config, [task], tmp_path)

        assert len(results) == 1
        r = results[0]
        assert r.success is True
        assert r.generation_model == "test/model-1"
        assert r.execution_model == "test/exec-1"
        assert r.task.task_id == "hover_001"
        assert "hover_001.json" in r.chain_json_path

    async def test_execute_chains_calls_execution_model(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """Chain execution sends flattened prompt to the LLM."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")
        _write_chain_artifact(tmp_path, "test/model-1", "hover", "hover_001")

        config = _minimal_eval_config(execution_models=["test/exec-1"])
        results = await execute_chains(config, [task], tmp_path)

        assert len(results) == 1
        assert results[0].response == "Chain execution answer."
        assert results[0].steps_completed == 3  # SAMPLE_CARL_CHAIN has 3 steps
        assert results[0].execution_time_seconds >= 0

        # Verify the LLM was called with the right model
        call_kwargs = mock_openai_chain.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test/exec-1"
        # Verify the prompt contains the reasoning plan
        msg_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "REASONING PLAN:" in msg_content
        assert "QUESTION: Test query" in msg_content

    async def test_execute_chains_error_handling(self, mocker: Any, tmp_path: Path) -> None:
        """LLM failure is captured in result without crashing."""
        mock_cls = mocker.patch("evaluation.chain_executor.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        _write_chain_artifact(tmp_path, "test/model-1", "hover", "hover_001")

        config = _minimal_eval_config(execution_models=["test/exec-1"])
        results = await execute_chains(config, [task], tmp_path)

        assert len(results) == 1
        r = results[0]
        assert r.success is False
        assert "LLM timeout" in (r.error or "")
        assert r.response == ""

    async def test_execute_chains_no_artifacts(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """Returns empty list when no chain artifacts exist."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        config = _minimal_eval_config(execution_models=["test/exec-1"])

        results = await execute_chains(config, [task], tmp_path)
        assert results == []

    async def test_execute_chains_multiple_models(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """Chains are executed on each (gen_model x exec_model) combination."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        _write_chain_artifact(tmp_path, "test/model-1", "hover", "hover_001")

        config = _minimal_eval_config(execution_models=["test/exec-1", "test/exec-2"])
        results = await execute_chains(config, [task], tmp_path)

        assert len(results) == 2
        exec_models = {r.execution_model for r in results}
        assert exec_models == {"test/exec-1", "test/exec-2"}

    async def test_execute_chains_skips_missing_tasks(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """Tasks without chain artifacts on disk are skipped."""
        task1 = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q1")
        task2 = BenchmarkTask(benchmark="hover", task_id="hover_002", query="Q2")
        # Only write artifact for task1
        _write_chain_artifact(tmp_path, "test/model-1", "hover", "hover_001")

        config = _minimal_eval_config(execution_models=["test/exec-1"])
        results = await execute_chains(config, [task1, task2], tmp_path)

        assert len(results) == 1
        assert results[0].task.task_id == "hover_001"

    async def test_execute_chains_invalid_json(self, mock_openai_chain: MagicMock, tmp_path: Path) -> None:
        """Invalid JSON in chain file produces error results."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")

        # Write invalid JSON
        slug = "test--model-1"
        chain_dir = tmp_path / "chains" / slug / "hover"
        chain_dir.mkdir(parents=True, exist_ok=True)
        (chain_dir / "hover_001.json").write_text("not valid json", encoding="utf-8")

        config = _minimal_eval_config(execution_models=["test/exec-1"])
        results = await execute_chains(config, [task], tmp_path)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None


# ---------------------------------------------------------------------------
# save_chain_executions tests
# ---------------------------------------------------------------------------


class TestSaveChainExecutions:
    """Tests for saving chain execution results."""

    def test_save_chain_executions_directory_structure(self, tmp_path: Path) -> None:
        """Verify file layout: response, prompt, meta per gen/exec/benchmark."""
        # Write the chain artifact so prompt reconstruction works
        chain_path = _write_chain_artifact(tmp_path, "test/gen-1", "hover", "hover_001")

        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")
        results = [
            ChainExecution(
                task=task,
                generation_model="test/gen-1",
                execution_model="test/exec-1",
                chain_json_path=str(chain_path),
                response="The answer is 42.",
                steps_completed=3,
                execution_time_seconds=2.5,
            ),
        ]

        save_chain_executions(results, tmp_path)

        exec_dir = tmp_path / "chain_executions" / "test--gen-1" / "test--exec-1" / "hover"
        assert exec_dir.exists()

        # Response file
        response_text = (exec_dir / "hover_001.txt").read_text()
        assert response_text == "The answer is 42."

        # Prompt file
        prompt_text = (exec_dir / "hover_001_prompt.txt").read_text()
        assert "REASONING PLAN:" in prompt_text
        assert "QUESTION: Test query" in prompt_text

        # Metadata
        meta = json.loads((exec_dir / "hover_001_meta.json").read_text())
        assert meta["generation_model"] == "test/gen-1"
        assert meta["execution_model"] == "test/exec-1"
        assert meta["steps_completed"] == 3
        assert meta["execution_time_seconds"] == 2.5
        assert meta["success"] is True
        assert meta["error"] is None

    def test_save_chain_executions_failed_result(self, tmp_path: Path) -> None:
        """Failed results are saved with error metadata."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        results = [
            ChainExecution(
                task=task,
                generation_model="test/gen-1",
                execution_model="test/exec-1",
                chain_json_path="nonexistent.json",
                response="",
                steps_completed=0,
                execution_time_seconds=0.1,
                success=False,
                error="LLM timeout",
            ),
        ]

        save_chain_executions(results, tmp_path)

        exec_dir = tmp_path / "chain_executions" / "test--gen-1" / "test--exec-1" / "hover"
        meta = json.loads((exec_dir / "hover_001_meta.json").read_text())
        assert meta["success"] is False
        assert meta["error"] == "LLM timeout"

    def test_save_chain_executions_multiple_combos(self, tmp_path: Path) -> None:
        """Multiple gen/exec combinations create separate directories."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        results = [
            ChainExecution(
                task=task,
                generation_model="gen/a",
                execution_model="exec/x",
                chain_json_path="",
                response="resp-ax",
                steps_completed=2,
                execution_time_seconds=1.0,
            ),
            ChainExecution(
                task=task,
                generation_model="gen/b",
                execution_model="exec/y",
                chain_json_path="",
                response="resp-by",
                steps_completed=3,
                execution_time_seconds=1.5,
            ),
        ]

        save_chain_executions(results, tmp_path)

        assert (tmp_path / "chain_executions" / "gen--a" / "exec--x" / "hover" / "hover_001.txt").exists()
        assert (tmp_path / "chain_executions" / "gen--b" / "exec--y" / "hover" / "hover_001.txt").exists()


# ---------------------------------------------------------------------------
# build_execution_prompt_from_category tests
# ---------------------------------------------------------------------------


class TestBuildExecutionPromptFromCategory:
    """Tests for category-level prompt building."""

    def test_contains_plan_and_question(self) -> None:
        """Prompt contains the reasoning plan and the task question."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test category query?")
        prompt, count = build_execution_prompt_from_category(SAMPLE_CARL_CHAIN, task)

        assert "REASONING PLAN:" in prompt
        assert "QUESTION: Test category query?" in prompt
        assert "Step 1:" in prompt
        assert "Step 2:" in prompt
        assert "Step 3:" in prompt
        assert count == 3

    def test_includes_reasoning_questions_and_action(self) -> None:
        """Prompt includes reasoning questions and process from steps."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        prompt, _ = build_execution_prompt_from_category(SAMPLE_CARL_CHAIN, task)

        assert "Key questions to think about:" in prompt
        assert "Process:" in prompt

    def test_empty_steps(self) -> None:
        """Empty chain falls back to default step."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        prompt, count = build_execution_prompt_from_category({"steps": []}, task)

        assert "QUESTION: Q" in prompt
        assert "Analyze the question" in prompt
        assert count == 0


# ---------------------------------------------------------------------------
# execute_category_chains tests
# ---------------------------------------------------------------------------


class TestExecuteCategoryChains:
    """Tests for category chain execution on benchmark tasks."""

    @pytest.fixture
    def mock_openai_cat_exec(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for category chain execution."""
        mock_cls = mocker.patch("evaluation.chain_executor.AsyncOpenAI")
        instance = mock_cls.return_value

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Category chain answer."
        response.usage = MagicMock(total_tokens=30)

        instance.chat.completions.create = AsyncMock(return_value=response)
        return instance

    async def test_loads_correct_benchmark_chain(self, mock_openai_cat_exec: MagicMock, tmp_path: Path) -> None:
        """execute_category_chains loads the correct benchmark chain file."""
        _write_category_chain_artifact(tmp_path, "test/model-1", "hover")

        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Verify claim")
        config = _minimal_eval_config(execution_models=["test/exec-1"])

        results = await execute_category_chains(config, [task], tmp_path)

        assert len(results) == 1
        r = results[0]
        assert r.success is True
        assert r.generation_model == "test/model-1"
        assert r.execution_model == "test/exec-1"
        assert r.response == "Category chain answer."
        assert "hover.json" in r.chain_json_path

    async def test_all_tasks_share_chain(self, mock_openai_cat_exec: MagicMock, tmp_path: Path) -> None:
        """All tasks from the same benchmark use the same category chain."""
        _write_category_chain_artifact(tmp_path, "test/model-1", "hover")

        tasks = [
            BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q1"),
            BenchmarkTask(benchmark="hover", task_id="hover_002", query="Q2"),
            BenchmarkTask(benchmark="hover", task_id="hover_003", query="Q3"),
        ]
        config = _minimal_eval_config(execution_models=["test/exec-1"])

        results = await execute_category_chains(config, tasks, tmp_path)

        assert len(results) == 3
        # All should reference the same chain file
        paths = {r.chain_json_path for r in results}
        assert len(paths) == 1
        assert all(r.success for r in results)

    async def test_skips_tasks_without_chain(self, mock_openai_cat_exec: MagicMock, tmp_path: Path) -> None:
        """Tasks with no matching category chain are skipped."""
        _write_category_chain_artifact(tmp_path, "test/model-1", "hover")

        tasks = [
            BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q1"),
            BenchmarkTask(benchmark="hotpotqa", task_id="hotpotqa_001", query="Q2"),
        ]
        config = _minimal_eval_config(execution_models=["test/exec-1"])

        results = await execute_category_chains(config, tasks, tmp_path)

        assert len(results) == 1
        assert results[0].task.benchmark == "hover"

    async def test_no_artifacts_returns_empty(self, mock_openai_cat_exec: MagicMock, tmp_path: Path) -> None:
        """Returns empty list when no category chain directory exists."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        config = _minimal_eval_config(execution_models=["test/exec-1"])

        results = await execute_category_chains(config, [task], tmp_path)
        assert results == []
