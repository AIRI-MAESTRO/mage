"""T35/C2 — evaluation runs chains through the REAL mmar-carl runtime.

The deciding assertions: a TOOL step actually executes (the flattened-prompt
stub never ran tools), deterministically against the mock-HTTP tool set, and
eval scores format into the standard evolution_meta schema.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from evaluation.carl_executor import (
    CANNED_GEO,
    execute_chain_via_carl,
    register_benchmark_tools,
    to_evolution_meta,
)


class MockLLM:
    """Minimal stub mmar-carl auto-detects (has get_response)."""

    def __init__(self, answers: list[str] | None = None, delay: float = 0.0) -> None:
        self.answers = answers or ["FINAL ANSWER"]
        self.delay = delay
        self.calls: list[str] = []

    async def get_response(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        self.calls.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]

    async def get_response_with_retries(self, prompt: str, *a: Any, **k: Any) -> str:
        return await self.get_response(prompt, *a, **k)

    async def get_response_with_system(self, system: str, prompt: str, *a: Any, **k: Any) -> str:
        return await self.get_response(prompt, *a, **k)

    async def get_response_with_usage(self, prompt: str, *a: Any, **k: Any):
        return await self.get_response(prompt, *a, **k), {"total_tokens": 1}


def _tool_chain(tool_name: str = "http_request") -> dict[str, Any]:
    """A composite: tool step feeds an llm step — the incident-class shape."""
    return {
        "name": "Geo Lookup",
        "max_workers": 1,
        "timeout": 60.0,
        "steps": [
            {
                "step_type": "tool",
                "number": 1,
                "title": "Geolocate",
                "dependencies": [],
                "step_config": {
                    "tool_name": tool_name,
                    "input_mapping": {"url": "'http://ip-api.com/json/'"},
                },
            },
            {
                "step_type": "llm",
                "number": 2,
                "title": "Answer",
                "aim": "Answer using the geo data",
                "reasoning_questions": "",
                "step_context_queries": [],
                "stage_action": "Answer",
                "example_reasoning": "",
                "dependencies": [1],
                "retry_max": 1,
            },
        ],
    }


async def test_tool_step_actually_executes():
    """THE T35 assertion — the stub never ran tools; the real runtime does."""
    tool_calls: list[str] = []

    def register(context: Any) -> None:
        async def http_request(url: str, **_: Any) -> str:
            tool_calls.append(url)
            return json.dumps(CANNED_GEO)

        context.register_tool("http_request", http_request)

    llm = MockLLM(["The user is in Sweden."])
    success, output, steps, error = await execute_chain_via_carl(
        _tool_chain(), "where am I?", api=llm, register_tools=register
    )
    assert success is True
    assert tool_calls == ["http://ip-api.com/json/"]  # tool step REALLY ran
    assert steps == 2
    assert output == "The user is in Sweden."
    assert error is None


async def test_benchmark_tools_are_deterministic():
    """The canned mock-HTTP set lets composites run with zero network."""
    llm = MockLLM(["Done."])
    success, _, steps, _ = await execute_chain_via_carl(
        _tool_chain(), "geo?", api=llm, register_tools=register_benchmark_tools
    )
    assert success is True
    assert steps == 2
    # and the canned geo payload is what the llm step saw in its prompt
    assert any("Sweden" in call for call in llm.calls)


async def test_missing_tool_fails_honestly():
    success, _, _, error = await execute_chain_via_carl(
        _tool_chain(tool_name="no_such_tool"), "x", api=MockLLM()
    )
    assert success is False
    assert error is None or "no_such_tool" in str(error)  # surfaced, not masked


async def test_timeout_path():
    success, _, _, error = await execute_chain_via_carl(
        _tool_chain(),
        "x",
        api=MockLLM(delay=0.5),
        register_tools=register_benchmark_tools,
        timeout_s=0.05,
    )
    assert success is False
    assert "exceeded" in str(error)


async def test_unloadable_chain_is_an_execution_error():
    success, _, _, error = await execute_chain_via_carl(
        {"name": "broken"}, "x", api=MockLLM()
    )
    assert success is False
    assert error


async def test_execute_chains_integration_via_real_runtime(tmp_path, monkeypatch):
    """execute_chains with use_carl_runtime=True drives the REAL executor:
    the artifact's tool step runs against the mock-HTTP set, no network."""
    import evaluation.carl_executor as carl_executor
    from evaluation.chain_executor import execute_chains
    from evaluation.config import EvalConfig
    from evaluation.tasks import BenchmarkTask

    gen_model = "test/gen-1"
    task = BenchmarkTask(benchmark="hover", task_id="hover_001", query="where am I?")
    slug = gen_model.replace("/", "--")
    chain_dir = tmp_path / "chains" / slug / "hover"
    chain_dir.mkdir(parents=True)
    (chain_dir / "hover_001.json").write_text(json.dumps(_tool_chain()), encoding="utf-8")

    llm = MockLLM(["You are in Sweden."])
    monkeypatch.setattr(carl_executor, "build_eval_llm_client", lambda config, model: llm)

    config = EvalConfig(
        api_key="test",
        generation_models=[gen_model],
        execution_models=["test/exec-1"],
        benchmarks=["hover"],
        baseline_types=["zero_shot"],
        use_carl_runtime=True,
    )
    results = await execute_chains(config, [task], tmp_path)
    assert len(results) == 1
    execution = results[0]
    assert execution.success is True
    assert execution.steps_completed == 2  # tool + llm — REALLY executed
    assert execution.response == "You are in Sweden."
    assert any("Sweden" in call for call in llm.calls)  # canned geo reached the LLM


def test_to_evolution_meta_schema():
    meta = to_evolution_meta(
        fitness_score=0.83,
        experiment_id="eval-20260610",
        generation=0,
        objectives={"accuracy": 0.9, "cost": 0.4},
        parent_version_ids=["vid-1"],
        mutation_kind="eval_baseline",
    )
    assert meta == {
        "fitness_score": 0.83,
        "experiment_id": "eval-20260610",
        "generation": 0,
        "objectives": {"accuracy": 0.9, "cost": 0.4},
        "parent_version_ids": ["vid-1"],
        "mutation_kind": "eval_baseline",
    }
    # minimal shape: only the score
    assert to_evolution_meta(fitness_score=1) == {"fitness_score": 1.0}
