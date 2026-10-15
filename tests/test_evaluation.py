"""Tests for the evaluation module — config, tasks, chain/baseline generation, runner CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from evaluation.baseline_generator import (
    BASELINE_TEMPLATES,
    BaselineExecution,
    BaselineResult,
    execute_baselines,
    generate_baselines,
    save_baseline_executions,
    save_baselines,
)
from evaluation.chain_generator import (
    ChainGenerationResult,
    generate_chains,
    save_chains,
)
from evaluation.config import EvalConfig
from evaluation.runner import app
from evaluation.tasks import BENCHMARK_TASKS, BenchmarkTask, get_tasks
from mmar_mage.schemas import MAGEConfig

from .conftest import SAMPLE_CARL_CHAIN, _make_openai_response

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_eval_config(**overrides: Any) -> EvalConfig:
    """Build a minimal EvalConfig for testing."""
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "generation_models": ["test/model-1"],
        "execution_models": ["test/exec-1"],
        "benchmarks": ["hover"],
        "baseline_types": ["zero_shot", "cot", "plan_and_solve"],
    }
    defaults.update(overrides)
    return EvalConfig(**defaults)


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_chain_artifact(output_dir: Path, gen_model: str, benchmark: str, task_id: str) -> None:
    """Write a minimal chain JSON artifact to disk for run-benchmarks tests."""
    slug = gen_model.replace("/", "--")
    chain_dir = output_dir / "chains" / slug / benchmark
    chain_dir.mkdir(parents=True, exist_ok=True)
    chain_path = chain_dir / f"{task_id}.json"
    chain_path.write_text(json.dumps(SAMPLE_CARL_CHAIN), encoding="utf-8")


# ---------------------------------------------------------------------------
# EvalConfig tests
# ---------------------------------------------------------------------------


class TestEvalConfig:
    """Tests for EvalConfig loading and defaults."""

    def test_defaults(self) -> None:
        """Default values are populated correctly."""
        cfg = EvalConfig()
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.mage_mode == "deep"
        assert cfg.temperature == 0.2
        assert len(cfg.generation_models) == 4
        assert len(cfg.execution_models) == 4
        assert cfg.baseline_types == ["zero_shot", "cot", "plan_and_solve"]
        assert cfg.benchmarks == ["hover", "ifbench", "hotpotqa", "livebench_math"]
        assert cfg.output_dir == "evaluation/results"
        assert cfg.max_tasks_per_benchmark is None
        assert cfg.allowed_step_types is None

    def test_from_toml(self, tmp_path: Path) -> None:
        """Load config from a TOML file."""
        toml_content = """\
[evaluation]
api_key = "test-key-toml"
mage_mode = "fast"
temperature = 0.5
generation_models = ["model/a", "model/b"]
execution_models = ["exec/a"]
benchmarks = ["hotpotqa"]
output_dir = "custom/output"
max_tasks_per_benchmark = 3
"""
        toml_path = tmp_path / "test.toml"
        _write_toml(toml_path, toml_content)

        cfg = EvalConfig.from_toml(toml_path)
        assert cfg.api_key == "test-key-toml"
        assert cfg.mage_mode == "fast"
        assert cfg.temperature == 0.5
        assert cfg.generation_models == ["model/a", "model/b"]
        assert cfg.execution_models == ["exec/a"]
        assert cfg.benchmarks == ["hotpotqa"]
        assert cfg.output_dir == "custom/output"
        assert cfg.max_tasks_per_benchmark == 3

    def test_from_toml_env_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """API key falls back to OPENROUTER_API_KEY env var."""
        toml_content = """\
[evaluation]
benchmarks = ["hover"]
"""
        toml_path = tmp_path / "test.toml"
        _write_toml(toml_path, toml_content)

        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key-123")
        cfg = EvalConfig.from_toml(toml_path)
        assert cfg.api_key == "env-key-123"

    def test_from_toml_empty_section(self, tmp_path: Path) -> None:
        """Missing [evaluation] section returns defaults."""
        toml_path = tmp_path / "empty.toml"
        _write_toml(toml_path, "[other]\nfoo = 1\n")

        cfg = EvalConfig.from_toml(toml_path)
        assert cfg.mage_mode == "deep"
        assert cfg.benchmarks == ["hover", "ifbench", "hotpotqa", "livebench_math"]

    def test_chain_mode_default_category(self) -> None:
        """chain_mode defaults to 'category'."""
        cfg = EvalConfig()
        assert cfg.chain_mode == "category"

    def test_category_example_count_default(self) -> None:
        """category_example_count defaults to 5."""
        cfg = EvalConfig()
        assert cfg.category_example_count == 5

    def test_contamination_check_enabled_default(self) -> None:
        """contamination_check_enabled defaults to True."""
        cfg = EvalConfig()
        assert cfg.contamination_check_enabled is True

    def test_chain_mode_from_toml(self, tmp_path: Path) -> None:
        """chain_mode is loaded from TOML."""
        toml_content = """\
[evaluation]
api_key = "test-key"
chain_mode = "per_task"
"""
        toml_path = tmp_path / "test.toml"
        _write_toml(toml_path, toml_content)

        cfg = EvalConfig.from_toml(toml_path)
        assert cfg.chain_mode == "per_task"

    def test_allowed_step_types_default_none(self) -> None:
        """allowed_step_types defaults to None."""
        cfg = EvalConfig()
        assert cfg.allowed_step_types is None

    def test_allowed_step_types_from_toml(self, tmp_path: Path) -> None:
        """allowed_step_types is loaded from TOML."""
        toml_content = """\
[evaluation]
api_key = "test-key"
allowed_step_types = ["llm"]
"""
        toml_path = tmp_path / "test.toml"
        _write_toml(toml_path, toml_content)

        cfg = EvalConfig.from_toml(toml_path)
        assert cfg.allowed_step_types == ["llm"]

    def test_allowed_step_types_passed_to_mage_config(self) -> None:
        """allowed_step_types is settable on EvalConfig."""
        cfg = _minimal_eval_config(allowed_step_types=["llm", "eval"])
        assert cfg.allowed_step_types == ["llm", "eval"]


# ---------------------------------------------------------------------------
# BenchmarkTask / get_tasks tests
# ---------------------------------------------------------------------------


class TestBenchmarkTasks:
    """Tests for benchmark task registry and filtering."""

    def test_registry_completeness(self) -> None:
        """All expected benchmarks have tasks."""
        expected = {"hover", "ifbench", "hotpotqa", "pupa", "livebench_math", "simpleqa", "live_composite"}
        assert set(BENCHMARK_TASKS.keys()) == expected

    def test_registry_counts(self) -> None:
        """Each benchmark has the expected number of seed tasks."""
        assert len(BENCHMARK_TASKS["hover"]) == 5
        assert len(BENCHMARK_TASKS["ifbench"]) == 5
        assert len(BENCHMARK_TASKS["hotpotqa"]) == 5
        assert len(BENCHMARK_TASKS["pupa"]) == 3
        assert len(BENCHMARK_TASKS["livebench_math"]) == 5
        assert len(BENCHMARK_TASKS["simpleqa"]) == 3
        assert len(BENCHMARK_TASKS["live_composite"]) == 2  # T12 regression cases

    def test_task_ids_unique(self) -> None:
        """All task IDs across all benchmarks are unique."""
        all_ids: list[str] = []
        for tasks in BENCHMARK_TASKS.values():
            all_ids.extend(t.task_id for t in tasks)
        assert len(all_ids) == len(set(all_ids))

    def test_get_tasks_single_benchmark(self) -> None:
        """get_tasks returns tasks for a single benchmark."""
        tasks = get_tasks(["hover"])
        assert len(tasks) == 5
        assert all(t.benchmark == "hover" for t in tasks)

    def test_get_tasks_multiple_benchmarks(self) -> None:
        """get_tasks returns tasks for multiple benchmarks."""
        tasks = get_tasks(["hover", "simpleqa"])
        assert len(tasks) == 8  # 5 + 3

    def test_get_tasks_max_per_benchmark(self) -> None:
        """get_tasks respects max_per_benchmark."""
        tasks = get_tasks(["hover", "hotpotqa"], max_per_benchmark=2)
        assert len(tasks) == 4  # 2 + 2

    def test_get_tasks_unknown_benchmark(self) -> None:
        """get_tasks returns empty for unknown benchmarks."""
        tasks = get_tasks(["nonexistent"])
        assert tasks == []

    def test_benchmark_task_model(self) -> None:
        """BenchmarkTask model validates correctly."""
        task = BenchmarkTask(benchmark="test", task_id="test_001", query="What?")
        assert task.category == ""
        assert task.metadata == {}


# ---------------------------------------------------------------------------
# Chain generation tests
# ---------------------------------------------------------------------------


class TestChainGeneration:
    """Tests for CARL chain generation."""

    @pytest.fixture
    def mock_openai_eval(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for evaluation chain generation."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    async def test_generate_chains_success(self, mock_openai_eval: MagicMock) -> None:
        """Successful chain generation produces valid results."""
        config = _minimal_eval_config(generation_models=["test/model-1"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        results = await generate_chains(config, tasks)

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.error is None
        assert result.generation_model == "test/model-1"
        assert result.task.task_id == "hover_001"
        assert result.chain_json != ""
        assert "steps" in result.chain_dict
        assert result.generation_time_seconds > 0

    async def test_generate_chains_passes_allowed_step_types(self, mock_openai_eval: MagicMock, mocker: Any) -> None:
        """Chain generator passes allowed_step_types to MAGEConfig."""
        spy = mocker.spy(MAGEConfig, "for_openrouter")
        config = _minimal_eval_config(generation_models=["test/model-1"], allowed_step_types=["llm"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        await generate_chains(config, tasks)

        # Verify for_openrouter was called with allowed_step_types in kwargs
        spy.assert_called_once()
        call_kwargs = spy.call_args
        assert call_kwargs.kwargs.get("allowed_step_types") == ["llm"]

    async def test_generate_chains_multiple_models(self, mock_openai_eval: MagicMock) -> None:
        """Chain generation runs for each model × task combination."""
        config = _minimal_eval_config(generation_models=["test/model-1", "test/model-2"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")]

        results = await generate_chains(config, tasks)
        assert len(results) == 2
        assert results[0].generation_model == "test/model-1"
        assert results[1].generation_model == "test/model-2"

    async def test_generate_chains_error(self, mocker: Any) -> None:
        """Chain generation captures errors gracefully."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM error"))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        config = _minimal_eval_config(generation_models=["test/model-1"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")]

        results = await generate_chains(config, tasks)
        assert len(results) == 1
        assert results[0].success is False
        assert "LLM error" in (results[0].error or "")

    def test_save_chains_directory_structure(self, tmp_path: Path) -> None:
        """save_chains creates the expected directory structure."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        results = [
            ChainGenerationResult(
                task=task,
                generation_model="deepseek/deepseek-v3.2",
                chain_json='{"steps": []}',
                chain_dict={"steps": []},
                code="# generated code",
                metadata={"model": "test"},
                generation_time_seconds=1.5,
            ),
        ]

        save_chains(results, tmp_path)

        chain_dir = tmp_path / "chains" / "deepseek--deepseek-v3.2" / "hover"
        assert chain_dir.exists()
        assert (chain_dir / "hover_001.json").exists()
        assert (chain_dir / "hover_001.py").exists()
        assert (chain_dir / "hover_001_meta.json").exists()

        # Verify content
        chain_data = json.loads((chain_dir / "hover_001.json").read_text())
        assert chain_data == {"steps": []}

        meta_data = json.loads((chain_dir / "hover_001_meta.json").read_text())
        assert meta_data["generation_model"] == "deepseek/deepseek-v3.2"

    def test_save_chains_skips_failed(self, tmp_path: Path) -> None:
        """save_chains skips failed results."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        results = [
            ChainGenerationResult(
                task=task,
                generation_model="test/model-1",
                success=False,
                error="failed",
            ),
        ]

        save_chains(results, tmp_path)
        assert not (tmp_path / "chains").exists()

    def test_save_chains_no_code(self, tmp_path: Path) -> None:
        """save_chains omits .py file when code is None."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")
        results = [
            ChainGenerationResult(
                task=task,
                generation_model="test/model-1",
                chain_json='{"steps": []}',
                chain_dict={"steps": []},
                code=None,
            ),
        ]

        save_chains(results, tmp_path)
        chain_dir = tmp_path / "chains" / "test--model-1" / "hover"
        assert (chain_dir / "hover_001.json").exists()
        assert not (chain_dir / "hover_001.py").exists()


# ---------------------------------------------------------------------------
# Baseline generation tests
# ---------------------------------------------------------------------------


class TestBaselineGeneration:
    """Tests for baseline prompt generation."""

    def test_zero_shot(self) -> None:
        """Zero-shot baseline is the raw query."""
        config = _minimal_eval_config(baseline_types=["zero_shot"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        results = generate_baselines(config, tasks)
        assert len(results) == 1
        assert results[0].baseline_type == "zero_shot"
        assert results[0].prompt == "Test query"

    def test_cot(self) -> None:
        """CoT baseline appends step-by-step instruction."""
        config = _minimal_eval_config(baseline_types=["cot"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        results = generate_baselines(config, tasks)
        assert len(results) == 1
        assert results[0].baseline_type == "cot"
        assert "Let's think step by step" in results[0].prompt
        assert results[0].prompt.startswith("Test query")

    def test_plan_and_solve(self) -> None:
        """Plan-and-solve baseline appends planning instruction."""
        config = _minimal_eval_config(baseline_types=["plan_and_solve"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        results = generate_baselines(config, tasks)
        assert len(results) == 1
        assert results[0].baseline_type == "plan_and_solve"
        assert "devise a plan" in results[0].prompt
        assert results[0].prompt.startswith("Test query")

    def test_all_baselines(self) -> None:
        """All three baseline types are generated for each task."""
        config = _minimal_eval_config()
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q1")]

        results = generate_baselines(config, tasks)
        assert len(results) == 3
        types = {r.baseline_type for r in results}
        assert types == {"zero_shot", "cot", "plan_and_solve"}

    def test_multiple_tasks(self) -> None:
        """Baselines are generated for each task × baseline_type."""
        config = _minimal_eval_config(baseline_types=["zero_shot"])
        tasks = [
            BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q1"),
            BenchmarkTask(benchmark="hover", task_id="hover_002", query="Q2"),
        ]

        results = generate_baselines(config, tasks)
        assert len(results) == 2

    def test_unknown_baseline_type(self) -> None:
        """Unknown baseline types are skipped."""
        config = _minimal_eval_config(baseline_types=["unknown_type"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")]

        results = generate_baselines(config, tasks)
        assert len(results) == 0

    def test_templates_registered(self) -> None:
        """All expected templates are registered."""
        assert "zero_shot" in BASELINE_TEMPLATES
        assert "cot" in BASELINE_TEMPLATES
        assert "plan_and_solve" in BASELINE_TEMPLATES

    def test_save_baselines_directory_structure(self, tmp_path: Path) -> None:
        """save_baselines creates the expected directory structure."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        results = [
            BaselineResult(task=task, baseline_type="cot", prompt="Q\n\nLet's think step by step."),
        ]

        save_baselines(results, tmp_path)

        baseline_dir = tmp_path / "baselines" / "cot" / "hover"
        assert baseline_dir.exists()
        assert (baseline_dir / "hover_001.txt").exists()
        assert (baseline_dir / "hover_001_meta.json").exists()

        prompt_text = (baseline_dir / "hover_001.txt").read_text()
        assert "Let's think step by step" in prompt_text

        meta = json.loads((baseline_dir / "hover_001_meta.json").read_text())
        assert meta["baseline_type"] == "cot"

    def test_save_baselines_multiple_types(self, tmp_path: Path) -> None:
        """save_baselines creates directories for multiple baseline types."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Q")
        results = [
            BaselineResult(task=task, baseline_type="zero_shot", prompt="Q"),
            BaselineResult(task=task, baseline_type="cot", prompt="Q\n\nLet's think step by step."),
        ]

        save_baselines(results, tmp_path)

        assert (tmp_path / "baselines" / "zero_shot" / "hover" / "hover_001.txt").exists()
        assert (tmp_path / "baselines" / "cot" / "hover" / "hover_001.txt").exists()


# ---------------------------------------------------------------------------
# Runner CLI tests (legacy generate + list-tasks)
# ---------------------------------------------------------------------------


class TestRunnerCLI:
    """Tests for the evaluation runner CLI."""

    @pytest.fixture
    def mock_openai_runner(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for runner CLI tests."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    def test_list_tasks_command(self) -> None:
        """list-tasks command runs and displays tasks."""
        result = runner.invoke(app, ["list-tasks"])
        assert result.exit_code == 0
        assert "hover" in result.stdout

    def test_list_tasks_filtered(self) -> None:
        """list-tasks with specific benchmarks filters output."""
        result = runner.invoke(app, ["list-tasks", "simpleqa"])
        assert result.exit_code == 0
        assert "simpleqa" in result.stdout

    def test_generate_command(self, mock_openai_runner: MagicMock, tmp_path: Path) -> None:
        """generate command runs end-to-end with mocked LLM."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate", "--config-path", str(toml_path)])
        assert result.exit_code == 0
        assert "Done" in result.stdout

    def test_generate_chains_only(self, mock_openai_runner: MagicMock, tmp_path: Path) -> None:
        """generate with --no-baselines skips baselines."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate", "--config-path", str(toml_path), "--no-baselines"])
        assert result.exit_code == 0

    def test_generate_baselines_only(self, mock_openai_runner: MagicMock, tmp_path: Path) -> None:
        """generate with --no-chains skips chain generation."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate", "--config-path", str(toml_path), "--no-chains"])
        assert result.exit_code == 0
        # Baselines should be generated
        assert (tmp_path / "baselines").exists()


# ---------------------------------------------------------------------------
# Baseline execution tests
# ---------------------------------------------------------------------------


class TestBaselineExecution:
    """Tests for real LLM baseline execution."""

    @pytest.fixture
    def mock_openai_baseline(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for baseline execution tests."""
        mock_cls = mocker.patch("evaluation.baseline_generator.AsyncOpenAI")
        instance = mock_cls.return_value

        usage = MagicMock()
        usage.total_tokens = 42

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "This is the LLM response."
        response.usage = usage

        instance.chat.completions.create = AsyncMock(return_value=response)
        return instance

    async def test_execute_baselines_success(self, mock_openai_baseline: MagicMock) -> None:
        """Verify LLM is called on execution_models, response captured."""
        config = _minimal_eval_config(execution_models=["test/exec-1"], baseline_types=["zero_shot"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")]

        results = await execute_baselines(config, tasks)

        assert len(results) == 1
        r = results[0]
        assert r.success is True
        assert r.response == "This is the LLM response."
        assert r.execution_model == "test/exec-1"
        assert r.baseline_type == "zero_shot"
        assert r.tokens_used == 42
        assert r.execution_time_seconds > 0
        assert r.error is None

    async def test_execute_baselines_error_handling(self, mocker: Any) -> None:
        """LLM failure is captured in result without crashing."""
        mock_cls = mocker.patch("evaluation.baseline_generator.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=RuntimeError("API timeout"))

        config = _minimal_eval_config(execution_models=["test/exec-1"], baseline_types=["cot"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")]

        results = await execute_baselines(config, tasks)

        assert len(results) == 1
        r = results[0]
        assert r.success is False
        assert "API timeout" in (r.error or "")
        assert r.response == ""

    async def test_execute_baselines_concurrency_limit(self, mock_openai_baseline: MagicMock) -> None:
        """Verify semaphore caps concurrency."""
        config = _minimal_eval_config(
            execution_models=["test/exec-1"],
            baseline_types=["zero_shot"],
            baseline_concurrency=2,
        )
        tasks = [BenchmarkTask(benchmark="hover", task_id=f"hover_{i:03d}", query=f"Q{i}") for i in range(5)]

        results = await execute_baselines(config, tasks)

        assert len(results) == 5
        assert all(r.success for r in results)

    async def test_execute_baselines_multiple_models_and_types(self, mock_openai_baseline: MagicMock) -> None:
        """Verify all model x baseline_type x task triples are executed."""
        config = _minimal_eval_config(
            execution_models=["test/exec-1", "test/exec-2"],
            baseline_types=["zero_shot", "cot"],
        )
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")]

        results = await execute_baselines(config, tasks)

        assert len(results) == 4  # 2 models x 2 types x 1 task

    async def test_execute_baselines_override_models(self, mock_openai_baseline: MagicMock) -> None:
        """Passing models= overrides config.execution_models."""
        config = _minimal_eval_config(execution_models=["test/exec-1"], baseline_types=["zero_shot"])
        tasks = [BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test")]

        results = await execute_baselines(config, tasks, models=["custom/model"])

        assert len(results) == 1
        assert results[0].execution_model == "custom/model"

    def test_save_baseline_executions_directory_structure(self, tmp_path: Path) -> None:
        """Verify file layout: response, prompt, meta — uses execution_model_slug."""
        task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="Test query")
        results = [
            BaselineExecution(
                task=task,
                execution_model="test/exec-1",
                baseline_type="cot",
                prompt="Test query\n\nLet's think step by step.",
                response="Here is my answer.",
                execution_time_seconds=1.23,
                tokens_used=50,
            ),
        ]

        save_baseline_executions(results, tmp_path)

        exec_dir = tmp_path / "baseline_executions" / "cot" / "test--exec-1" / "hover"
        assert exec_dir.exists()

        # Response file
        response_text = (exec_dir / "hover_001.txt").read_text()
        assert response_text == "Here is my answer."

        # Prompt file
        prompt_text = (exec_dir / "hover_001_prompt.txt").read_text()
        assert "Let's think step by step" in prompt_text

        # Metadata
        meta = json.loads((exec_dir / "hover_001_meta.json").read_text())
        assert meta["execution_model"] == "test/exec-1"
        assert meta["baseline_type"] == "cot"
        assert meta["execution_time_seconds"] == 1.23
        assert meta["tokens_used"] == 50
        assert meta["success"] is True
        assert meta["error"] is None


# ---------------------------------------------------------------------------
# generate-artifacts CLI tests
# ---------------------------------------------------------------------------


class TestGenerateArtifactsCLI:
    """Tests for the generate-artifacts command (Stage A)."""

    @pytest.fixture
    def mock_openai_gen(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for chain generation only."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    def test_generate_artifacts_command(self, mock_openai_gen: MagicMock, tmp_path: Path) -> None:
        """generate-artifacts calls generate_chains + generate_baselines only (NO execute_baselines)."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
chain_mode = "per_task"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-artifacts", "--config-path", str(toml_path)])
        assert result.exit_code == 0, result.stdout
        assert "chains generated" in result.stdout
        assert "baseline prompts generated" in result.stdout
        assert "Artifacts saved" in result.stdout

        # Chains and baselines should exist
        assert (tmp_path / "chains").exists()
        assert (tmp_path / "baselines").exists()
        # No execution directories should be created
        assert not (tmp_path / "baseline_executions").exists()
        assert not (tmp_path / "chain_executions").exists()

    def test_generate_artifacts_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate-artifacts exits with error when no API key provided."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        toml_content = f"""\
[evaluation]
api_key = ""
output_dir = "{tmp_path}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-artifacts", "--config-path", str(toml_path)])
        assert result.exit_code == 1

    def test_generate_artifacts_quick_mode(self, mock_openai_gen: MagicMock, tmp_path: Path) -> None:
        """--quick limits to 2 tasks and 1 generation model."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1", "test/model-2", "test/model-3"]
execution_models = ["test/exec-1", "test/exec-2"]
benchmarks = ["hover", "simpleqa"]
output_dir = "{tmp_path}"
mage_mode = "fast"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-artifacts", "--config-path", str(toml_path), "--quick"])
        assert result.exit_code == 0, result.stdout
        assert "Generation models: 1" in result.stdout


# ---------------------------------------------------------------------------
# generate-category-chains CLI tests
# ---------------------------------------------------------------------------


class TestGenerateCategoryChainsCLI:
    """Tests for the generate-category-chains command."""

    @pytest.fixture
    def mock_openai_cat_gen(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for category chain generation."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    def test_generate_category_chains_cli(self, mock_openai_cat_gen: MagicMock, tmp_path: Path) -> None:
        """generate-category-chains produces category chain artifacts."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
output_dir = "{tmp_path}"
mage_mode = "fast"
category_example_count = 2
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-category-chains", "--config-path", str(toml_path)])
        assert result.exit_code == 0, result.stdout
        assert "category chains generated" in result.stdout
        assert (tmp_path / "category_chains").exists()

    def test_generate_category_chains_missing_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate-category-chains exits with error when no API key provided."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        toml_content = f"""\
[evaluation]
api_key = ""
output_dir = "{tmp_path}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-category-chains", "--config-path", str(toml_path)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# generate-artifacts category mode CLI tests
# ---------------------------------------------------------------------------


class TestGenerateArtifactsCategoryModeCLI:
    """Tests for the generate-artifacts command in category mode."""

    @pytest.fixture
    def mock_openai_cat_artifacts(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for category chain generation via generate-artifacts."""
        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN)))
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
        return instance

    def test_generate_artifacts_category_mode(self, mock_openai_cat_artifacts: MagicMock, tmp_path: Path) -> None:
        """generate-artifacts with --mode=category produces category chains."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
chain_mode = "category"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-artifacts", "--config-path", str(toml_path)])
        assert result.exit_code == 0, result.stdout
        assert "category chains generated" in result.stdout
        assert "baseline prompts generated" in result.stdout
        assert (tmp_path / "category_chains").exists()
        assert (tmp_path / "baselines").exists()

    def test_generate_artifacts_per_task_mode(self, mock_openai_cat_artifacts: MagicMock, tmp_path: Path) -> None:
        """generate-artifacts with --mode=per-task uses legacy per-task chains."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["generate-artifacts", "--config-path", str(toml_path), "--mode", "per-task"])
        assert result.exit_code == 0, result.stdout
        assert "chains generated" in result.stdout
        assert (tmp_path / "chains").exists()


# ---------------------------------------------------------------------------
# run-benchmarks CLI tests
# ---------------------------------------------------------------------------


class TestRunBenchmarksCLI:
    """Tests for the run-benchmarks command (Stage B)."""

    @pytest.fixture
    def mock_openai_benchmarks(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for both chain and baseline execution."""
        # Mock chain executor's AsyncOpenAI
        mock_chain_cls = mocker.patch("evaluation.chain_executor.AsyncOpenAI")
        chain_instance = mock_chain_cls.return_value
        chain_resp = MagicMock()
        chain_resp.choices = [MagicMock()]
        chain_resp.choices[0].message.content = "chain execution response"
        chain_resp.usage = MagicMock(total_tokens=20)
        chain_instance.chat.completions.create = AsyncMock(return_value=chain_resp)

        # Mock baseline executor's AsyncOpenAI
        mock_baseline_cls = mocker.patch("evaluation.baseline_generator.AsyncOpenAI")
        baseline_instance = mock_baseline_cls.return_value
        baseline_resp = MagicMock()
        baseline_resp.choices = [MagicMock()]
        baseline_resp.choices[0].message.content = "baseline response"
        baseline_resp.usage = MagicMock(total_tokens=10)
        baseline_instance.chat.completions.create = AsyncMock(return_value=baseline_resp)

        return chain_instance

    def test_run_benchmarks_command(self, mock_openai_benchmarks: MagicMock, tmp_path: Path) -> None:
        """run-benchmarks calls execute_baselines + execute_chains."""
        # Write chain artifacts for the task
        _write_chain_artifact(tmp_path, "test/model-1", "simpleqa", "simpleqa_001")

        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
baseline_types = ["zero_shot"]
chain_mode = "per_task"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-benchmarks", "--config-path", str(toml_path)])
        assert result.exit_code == 0, result.stdout
        assert "chains executed" in result.stdout
        assert "baselines executed" in result.stdout
        assert "Benchmarks complete" in result.stdout

        # Verify execution directories exist
        assert (tmp_path / "chain_executions").exists()
        assert (tmp_path / "baseline_executions").exists()

    def test_run_benchmarks_skip_chains(self, mock_openai_benchmarks: MagicMock, tmp_path: Path) -> None:
        """--skip-chains flag skips chain execution."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
baseline_types = ["zero_shot"]
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-benchmarks", "--config-path", str(toml_path), "--skip-chains"])
        assert result.exit_code == 0, result.stdout
        assert "Chain execution skipped" in result.stdout
        assert "baselines executed" in result.stdout
        assert not (tmp_path / "chain_executions").exists()

    def test_run_benchmarks_skip_baselines(self, mock_openai_benchmarks: MagicMock, tmp_path: Path) -> None:
        """--skip-baselines flag skips baseline execution."""
        _write_chain_artifact(tmp_path, "test/model-1", "simpleqa", "simpleqa_001")

        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
baseline_types = ["zero_shot"]
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-benchmarks", "--config-path", str(toml_path), "--skip-baselines"])
        assert result.exit_code == 0, result.stdout
        assert "Baseline execution skipped" in result.stdout
        assert "chains executed" in result.stdout
        assert not (tmp_path / "baseline_executions").exists()

    def test_run_benchmarks_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """run-benchmarks exits with error when no API key provided."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        toml_content = f"""\
[evaluation]
api_key = ""
output_dir = "{tmp_path}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-benchmarks", "--config-path", str(toml_path)])
        assert result.exit_code == 1

    def test_run_benchmarks_quick_mode(self, mock_openai_benchmarks: MagicMock, tmp_path: Path) -> None:
        """--quick limits to 2 tasks and 1 execution model."""
        _write_chain_artifact(tmp_path, "test/model-1", "simpleqa", "simpleqa_001")

        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1", "test/exec-2"]
benchmarks = ["simpleqa"]
output_dir = "{tmp_path}"
baseline_types = ["zero_shot"]
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-benchmarks", "--config-path", str(toml_path), "--quick"])
        assert result.exit_code == 0, result.stdout
        assert "Execution models: 1" in result.stdout

    def test_run_benchmarks_category_mode_flag(self, mock_openai_benchmarks: MagicMock, tmp_path: Path) -> None:
        """--chain-mode category uses category chain execution."""
        # Write a category chain artifact
        slug = "test--model-1"
        cat_dir = tmp_path / "category_chains" / slug
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "simpleqa.json").write_text(json.dumps(SAMPLE_CARL_CHAIN), encoding="utf-8")

        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
baseline_types = ["zero_shot"]
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(
            app,
            ["run-benchmarks", "--config-path", str(toml_path), "--chain-mode", "category"],
        )
        assert result.exit_code == 0, result.stdout
        assert "chains executed" in result.stdout
        assert "Chain mode: category" in result.stdout


# ---------------------------------------------------------------------------
# run-all CLI tests
# ---------------------------------------------------------------------------


class TestRunAllCLI:
    """Tests for the run-all CLI command."""

    @pytest.fixture
    def mock_openai_run_all(self, mocker: Any) -> MagicMock:
        """Mock AsyncOpenAI for run-all CLI tests (covers chain gen, chain exec, and baseline exec)."""
        # Mock MAGE chain generation
        mock_mage_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        mage_instance = mock_mage_cls.return_value
        mage_instance.chat.completions.create = AsyncMock(
            return_value=_make_openai_response(json.dumps(SAMPLE_CARL_CHAIN))
        )
        mage_instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        # Mock chain executor's AsyncOpenAI
        mock_chain_exec_cls = mocker.patch("evaluation.chain_executor.AsyncOpenAI")
        chain_exec_instance = mock_chain_exec_cls.return_value
        chain_exec_resp = MagicMock()
        chain_exec_resp.choices = [MagicMock()]
        chain_exec_resp.choices[0].message.content = "chain execution response"
        chain_exec_resp.usage = MagicMock(total_tokens=20)
        chain_exec_instance.chat.completions.create = AsyncMock(return_value=chain_exec_resp)

        # Mock baseline executor's AsyncOpenAI
        mock_baseline_cls = mocker.patch("evaluation.baseline_generator.AsyncOpenAI")
        baseline_instance = mock_baseline_cls.return_value
        usage = MagicMock()
        usage.total_tokens = 10
        baseline_resp = MagicMock()
        baseline_resp.choices = [MagicMock()]
        baseline_resp.choices[0].message.content = "baseline response"
        baseline_resp.usage = usage
        baseline_instance.chat.completions.create = AsyncMock(return_value=baseline_resp)

        return mage_instance

    def test_run_all_calls_both_stages(self, mock_openai_run_all: MagicMock, tmp_path: Path) -> None:
        """run-all invokes both generate-artifacts and run-benchmarks."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1"]
execution_models = ["test/exec-1"]
benchmarks = ["simpleqa"]
max_tasks_per_benchmark = 1
output_dir = "{tmp_path}"
mage_mode = "fast"
baseline_types = ["zero_shot"]
chain_mode = "per_task"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-all", "--config-path", str(toml_path)])
        assert result.exit_code == 0, result.stdout

        # Stage A outputs
        assert "chains generated" in result.stdout
        assert "baseline prompts generated" in result.stdout
        # Stage B outputs
        assert "chains executed" in result.stdout
        assert "baselines executed" in result.stdout

        # Verify all output directories exist
        assert (tmp_path / "chains").exists()
        assert (tmp_path / "baselines").exists()
        assert (tmp_path / "chain_executions").exists()
        assert (tmp_path / "baseline_executions").exists()

    def test_run_all_quick_mode(self, mock_openai_run_all: MagicMock, tmp_path: Path) -> None:
        """--quick flag limits tasks and models (both generation and execution)."""
        toml_content = f"""\
[evaluation]
api_key = "test-key"
generation_models = ["test/model-1", "test/model-2", "test/model-3"]
execution_models = ["test/exec-1", "test/exec-2"]
benchmarks = ["hover", "simpleqa"]
output_dir = "{tmp_path}"
mage_mode = "fast"
baseline_types = ["zero_shot"]
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-all", "--config-path", str(toml_path), "--quick"])
        assert result.exit_code == 0, result.stdout
        # Quick mode: 1 generation model, 1 execution model, 2 tasks per benchmark
        assert "Generation models: 1" in result.stdout
        assert "Execution models: 1" in result.stdout

    def test_run_all_missing_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """run-all exits with error when no API key provided."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        toml_content = f"""\
[evaluation]
api_key = ""
output_dir = "{tmp_path}"
"""
        toml_path = tmp_path / "test.toml"
        toml_path.write_text(toml_content, encoding="utf-8")

        result = runner.invoke(app, ["run-all", "--config-path", str(toml_path)])
        assert result.exit_code == 1
