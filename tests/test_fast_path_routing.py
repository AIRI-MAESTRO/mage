"""T13 — fast-path routing: a simple query skips the deep pipeline for one fast call."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.generator import MAGEGenerator, _should_fast_path
from mmar_mage.schemas import DomainAnalysis, MAGEConfig


def _da(complexity: str, steps: int) -> DomainAnalysis:
    return DomainAnalysis(complexity=complexity, suggested_step_count=steps)


def _cfg(**kw: Any) -> MAGEConfig:
    return MAGEConfig(api_key="test", **kw)


def test_should_fast_path_decision_matrix() -> None:
    on = _cfg(enable_fast_path_routing=True)
    assert _should_fast_path(on, _da("simple", 1)) is True
    assert _should_fast_path(on, _da("simple", 2)) is True
    assert _should_fast_path(on, _da("simple", 3)) is False  # too many steps for fast mode
    assert _should_fast_path(on, _da("medium", 1)) is False  # not simple (a composite, post-T2)
    assert _should_fast_path(on, _da("complex", 1)) is False
    assert _should_fast_path(_cfg(), _da("simple", 1)) is False  # off by default


def _resp(content: dict[str, Any]) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps(content)
    r = MagicMock()
    r.choices = [choice]
    return r


_DOMAIN_SIMPLE = {"domain": "general", "task_type": "qa", "complexity": "simple",
                  "key_concepts": [], "suggested_step_count": 1, "preferred_step_types": ["llm"]}
_FAST_CHAIN = {"name": "IP", "max_workers": 1, "steps": [
    {"step_type": "llm", "number": 1, "title": "Answer", "aim": "answer the question", "dependencies": []}]}


async def test_simple_query_routes_to_fast(mocker: Any) -> None:
    instance = mocker.patch("mmar_mage.llm.AsyncOpenAI").return_value
    instance.chat.completions.create = AsyncMock(side_effect=[_resp(_DOMAIN_SIMPLE), _resp(_FAST_CHAIN)])
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))
    # naming off so the LLM call count is exactly domain + fast
    gen = MAGEGenerator(config=_cfg(mode="deep", enable_fast_path_routing=True, enable_suggested_naming_llm=False))
    result = await gen.generate("what is my ip")
    assert result.chain_dict["steps"]  # a chain was produced
    assert instance.chat.completions.create.await_count == 2  # domain + fast, NOT the 4-stage deep
    assert "fast_path" in (result.metadata.deep_stages_completed or [])


def test_care_default_enables_fast_path() -> None:
    mage = tomllib.loads(Path("configs/care_default.toml").read_text(encoding="utf-8"))["mage"]
    assert mage["enable_fast_path_routing"] is True
