"""T45 — per-stage timing telemetry surfaced for the replay bundle."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from mmar_mage.generator import MAGEGenerator, _StageTimingRecorder
from mmar_mage.schemas import MAGEConfig, MAGEMetadata


def test_recorder_times_stages_and_delegates() -> None:
    inner = MagicMock()
    rec = _StageTimingRecorder(inner)
    rec.on_stage_start("domain")
    rec.on_stage_complete("domain", {"x": 1})
    assert "domain" in rec.timings and rec.timings["domain"] >= 0.0
    inner.on_stage_start.assert_called_once_with("domain")
    inner.on_stage_complete.assert_called_once_with("domain", {"x": 1})


def test_recorder_records_on_error() -> None:
    inner = MagicMock()
    rec = _StageTimingRecorder(inner)
    rec.on_stage_start("planning")
    rec.on_error("planning", ValueError("boom"))
    assert "planning" in rec.timings
    inner.on_error.assert_called_once()


def test_snapshot_and_reset_clears() -> None:
    rec = _StageTimingRecorder(MagicMock())
    rec.on_stage_start("a")
    rec.on_stage_complete("a", None)
    snap = rec.snapshot_and_reset()
    assert "a" in snap
    assert rec.timings == {} and rec.snapshot_and_reset() == {}  # second run starts clean


def test_getattr_delegates_only_present_hooks() -> None:
    class MinimalCb:  # only the three required methods — no optional streaming hooks
        def on_stage_start(self, stage: str) -> None: ...
        def on_stage_complete(self, stage: str, result: Any) -> None: ...
        def on_error(self, stage: str, error: Exception) -> None: ...

    rec = _StageTimingRecorder(MinimalCb())
    assert getattr(rec, "on_llm_chunk", None) is None  # not delegated → probe stays None


def test_stage_telemetry_merges_timings_and_usage() -> None:
    md = MAGEMetadata(
        stage_timings={"domain": 1.2, "planning": 3.4},
        usage_by_stage={"domain": {"total_tokens": 100}},
    )
    tel = md.stage_telemetry()
    assert tel["domain"] == {"time_seconds": 1.2, "usage": {"total_tokens": 100}}
    assert tel["planning"] == {"time_seconds": 3.4}  # timed but no usage recorded


async def test_deep_generate_populates_stage_timings(
    deep_config: MAGEConfig,
    mock_openai_deep: MagicMock,
) -> None:
    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Analyze a financial report")
    timings = result.metadata.stage_timings
    assert timings and "domain_analysis" in timings  # stages were timed
    assert all(v >= 0.0 for v in timings.values())
    assert result.metadata.stage_telemetry()  # merged replay-bundle view is available
