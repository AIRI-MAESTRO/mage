"""Chain-level metrics emission — plan item P3.4.

CARL's runtime ``metrics`` field is ``exclude=True`` and holds live ``MetricBase``
ABC instances (no JSON spec), so it can't carry persisted metric specs. MAGE
instead carries declared ``step_metrics`` (``{name, type, threshold}``) as
analytics metadata — validated, summarizable, and ignored by CARL on load (so it
never breaks execution).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.carl_export import (
    VALID_METRIC_TYPES,
    chain_metrics_summary,
    to_carl_compat,
    validate_step_metrics,
)


def _step(number: int, metrics: list[dict] | None = None, **f: Any) -> dict[str, Any]:
    s = {"step_type": "llm", "number": number, "title": f"S{number}", "aim": "do", "dependencies": [], **f}
    if metrics is not None:
        s["step_metrics"] = metrics
    return s


def _chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"name": "c", "max_workers": 1, "steps": list(steps)}


_GOOD = [{"name": "answer_present", "type": "contains", "threshold": 1.0}]


# --------------------------------------------------------------------------- #
# validate_step_metrics
# --------------------------------------------------------------------------- #
def test_valid_metrics_pass():
    assert validate_step_metrics(_chain(_step(1, _GOOD))) == {}


def test_no_metrics_pass():
    assert validate_step_metrics(_chain(_step(1))) == {}


def test_missing_name_flagged():
    out = validate_step_metrics(_chain(_step(1, [{"type": "contains"}])))
    assert 1 in out and any("name" in p for p in out[1])


def test_invalid_type_flagged():
    out = validate_step_metrics(_chain(_step(1, [{"name": "x", "type": "bogus"}])))
    assert 1 in out and any("type" in p for p in out[1])


def test_non_numeric_threshold_flagged():
    out = validate_step_metrics(_chain(_step(1, [{"name": "x", "type": "numeric", "threshold": "high"}])))
    assert 1 in out and any("threshold" in p for p in out[1])


def test_all_valid_types_accepted():
    for t in VALID_METRIC_TYPES:
        assert validate_step_metrics(_chain(_step(1, [{"name": "m", "type": t}]))) == {}


# --------------------------------------------------------------------------- #
# chain_metrics_summary
# --------------------------------------------------------------------------- #
def test_summary_aggregates_by_type_and_step():
    chain = _chain(
        _step(1, [{"name": "a", "type": "contains"}, {"name": "b", "type": "numeric", "threshold": 0.5}]),
        _step(2, [{"name": "c", "type": "contains"}]),
        _step(3),
    )
    summary = chain_metrics_summary(chain)
    assert summary["total"] == 3
    assert summary["by_type"] == {"contains": 2, "numeric": 1}
    assert summary["per_step"] == {1: ["a", "b"], 2: ["c"]}


def test_summary_empty_when_no_metrics():
    assert chain_metrics_summary(_chain(_step(1), _step(2))) == {"total": 0, "by_type": {}, "per_step": {}}


# --------------------------------------------------------------------------- #
# Export safety — metrics ride along but never reach CARL's runtime field
# --------------------------------------------------------------------------- #
def test_metrics_survive_to_carl_compat():
    out = to_carl_compat(_chain(_step(1, _GOOD)))
    assert out["steps"][0]["step_metrics"] == _GOOD  # kept as MAGE analytics metadata


def test_metrics_chain_still_loads_in_carl():
    pytest.importorskip("mmar_carl")
    from mmar_carl import ReasoningChain

    out = to_carl_compat(_chain(_step(1, _GOOD)))
    rc = ReasoningChain.from_json(json.dumps(out))  # CARL ignores step_metrics, must not raise
    assert rc is not None


# --------------------------------------------------------------------------- #
# Real-execution scenario — generate() carries + flags metrics
# --------------------------------------------------------------------------- #
async def test_generate_carries_metrics_and_flags_malformed(caplog: Any):
    import logging

    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    # fast-mode chain: step 1 has a valid metric, step 2 a malformed one
    chain = _chain(
        _step(1, _GOOD),
        _step(2, [{"type": "contains"}], dependencies=[1]),  # missing name
    )
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", mode="fast"))
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    gen._llm.generate = AsyncMock(return_value=json.dumps(chain))

    with caplog.at_level(logging.WARNING):
        result = await gen.generate("answer the question")

    # valid metric is carried in the returned chain for analytics
    assert result.chain_dict["steps"][0]["step_metrics"] == _GOOD
    # malformed metric is surfaced (non-fatal)
    assert any("Malformed step_metrics" in r.message for r in caplog.records)
    assert chain_metrics_summary(result.chain_dict)["total"] >= 1
