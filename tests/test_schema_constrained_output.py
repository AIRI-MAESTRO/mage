"""P-7 — schema-constrained output via response_format json_schema (opt-in)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel

from mmar_mage.llm import LLMClient
from mmar_mage.schemas import MAGEConfig


class _Model(BaseModel):
    name: str
    count: int = 0


def _mk_client(mocker: Any, *, schema_constrained: bool) -> tuple[LLMClient, AsyncMock]:
    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    inst = mock_cls.return_value
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({"name": "x", "count": 1})
    resp.usage = None
    create = AsyncMock(return_value=resp)
    inst.chat.completions.create = create
    cfg = MAGEConfig(api_key="test", use_schema_constrained_output=schema_constrained)
    return LLMClient(cfg), create


async def test_default_uses_json_object(mocker: Any) -> None:
    client, create = _mk_client(mocker, schema_constrained=False)
    out = await client.generate_structured("sys", "user", _Model, stage="s")
    assert out.name == "x"
    assert create.call_args.kwargs["response_format"] == {"type": "json_object"}


async def test_flag_on_passes_json_schema(mocker: Any) -> None:
    client, create = _mk_client(mocker, schema_constrained=True)
    out = await client.generate_structured("sys", "user", _Model, stage="s")
    assert out.name == "x"
    rf = create.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Model"
    assert rf["json_schema"]["schema"] == _Model.model_json_schema()
    assert rf["json_schema"]["strict"] is True


async def test_generate_json_without_schema_stays_json_object(mocker: Any) -> None:
    # generate_json with no schema → json_object even when the flag is on
    client, create = _mk_client(mocker, schema_constrained=True)
    await client.generate_json("sys", "user", stage="s")
    assert create.call_args.kwargs["response_format"] == {"type": "json_object"}


async def test_json_schema_downgrades_to_json_object_on_failure(mocker: Any) -> None:
    # T9: a provider that rejects json_schema must degrade to json_object, not hard-fail
    inst = mocker.patch("mmar_mage.llm.AsyncOpenAI").return_value
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = '{"name": "y", "count": 2}'
    ok.usage = None
    create = AsyncMock(side_effect=[RuntimeError("response_format json_schema not supported"), ok])
    inst.chat.completions.create = create
    client = LLMClient(MAGEConfig(api_key="test", use_schema_constrained_output=True, max_retries=1))
    out = await client.generate_structured("sys", "user", _Model, stage="s")
    assert out.name == "y"
    assert create.call_count == 2
    assert create.call_args_list[0].kwargs["response_format"]["type"] == "json_schema"
    assert create.call_args_list[1].kwargs["response_format"]["type"] == "json_object"


def test_care_profiles_enable_schema_constrained_output() -> None:
    # T9: the OpenAI-provider care profiles opt into schema-guided decoding
    for name in ("care_default", "care_research"):
        cfg = MAGEConfig.from_toml(f"configs/{name}.toml")
        assert cfg.use_schema_constrained_output is True, name
