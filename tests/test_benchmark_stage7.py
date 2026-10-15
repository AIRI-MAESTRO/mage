"""Generator stage 7 — benchmark-mode self-verification (plan item P1.16).

Drives ``MAGEGenerator._run_benchmark_mode`` (synthesize → run → report) with a
directly-mocked LLM, and checks the config + result plumbing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig, MAGEResult

CHAIN = {"name": "c", "steps": [{"number": 1, "step_type": "llm", "title": "Analyze", "aim": "do x"}]}


def _gen(**cfg) -> MAGEGenerator:
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", **cfg))
    gen._llm = MagicMock()
    return gen


async def test_disabled_returns_none_without_llm_calls():
    gen = _gen(enable_benchmark_mode=False)
    gen._llm.generate_json = AsyncMock()
    assert await gen._run_benchmark_mode("q", CHAIN, "finance") is None
    gen._llm.generate_json.assert_not_called()


async def test_produces_quality_report():
    gen = _gen(enable_benchmark_mode=True, benchmark_n_cases=2)
    gen._llm.generate_json = AsyncMock(
        side_effect=[
            # synthesize → 2 tasks (renumbered synth_001/002)
            {"tasks": [{"input": "case 1", "rubric": ["r"]}, {"input": "case 2", "rubric": ["r"]}]},
            # run → one pass, one fail
            {
                "results": [
                    {"task_id": "synth_001", "passed": True, "score": 0.9},
                    {"task_id": "synth_002", "passed": False, "score": 0.2, "reason": "missing step"},
                ]
            },
        ]
    )

    report = await gen._run_benchmark_mode("q", CHAIN, "finance")

    assert report is not None
    assert report["n_cases"] == 2
    assert report["pass_rate"] == 0.5
    assert "missing step" in report["failure_modes"]


async def test_no_synthesized_tasks_returns_none():
    gen = _gen(enable_benchmark_mode=True)
    gen._llm.generate_json = AsyncMock(return_value={"tasks": []})  # synth yields nothing
    assert await gen._run_benchmark_mode("q", CHAIN, "finance") is None


async def test_failure_is_nonfatal_returns_none():
    gen = _gen(enable_benchmark_mode=True)
    gen._llm.generate_json = AsyncMock(side_effect=RuntimeError("llm down"))
    assert await gen._run_benchmark_mode("q", CHAIN, "finance") is None


def test_config_and_result_fields_exist():
    assert MAGEConfig().enable_benchmark_mode is False  # off by default
    assert MAGEConfig().benchmark_n_cases == 5
    # MAGEResult carries the optional report
    r = MAGEResult(chain_json="{}", quality_report={"n_cases": 1, "pass_rate": 1.0})
    assert r.quality_report["pass_rate"] == 1.0
    assert MAGEResult(chain_json="{}").quality_report is None
