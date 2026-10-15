"""T44 — step_describe_mode='auto' picks iterative for weak models on long chains."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from mmar_mage.generator import _has_structured_output, _resolve_describe_mode
from mmar_mage.schemas import MAGEConfig


def _cfg(**kw: Any) -> MAGEConfig:
    kw.setdefault("api_key", "test")
    if kw.get("provider") in ("openrouter", "local", "custom"):
        kw.setdefault("base_url", "http://localhost:8000/v1")
    return MAGEConfig(**kw)


def test_explicit_modes_pass_through() -> None:
    for mode in ("batch", "iterative", "parallel"):
        assert _resolve_describe_mode(_cfg(step_describe_mode=mode), num_steps=10) == mode


def test_auto_openai_always_batch() -> None:
    cfg = _cfg(provider="openai", step_describe_mode="auto")
    assert _has_structured_output(cfg)
    assert _resolve_describe_mode(cfg, num_steps=10) == "batch"  # reliable JSON even on a long chain


def test_auto_openrouter_long_chain_is_iterative() -> None:
    # OpenRouter routes to flash models that break long batch JSON, even with the flag set
    cfg = _cfg(provider="openrouter", use_schema_constrained_output=True, step_describe_mode="auto")
    assert not _has_structured_output(cfg)
    assert _resolve_describe_mode(cfg, num_steps=4) == "iterative"  # long chain → one step per call
    assert _resolve_describe_mode(cfg, num_steps=3) == "batch"  # short chain stays batch (cheaper)


def test_auto_local_depends_on_guided_decoding() -> None:
    vllm = _cfg(provider="local", use_schema_constrained_output=True, json_mode_fallback=False, step_describe_mode="auto")
    assert _has_structured_output(vllm)
    assert _resolve_describe_mode(vllm, num_steps=10) == "batch"  # guided decoding → trusted
    ollama = _cfg(provider="local", json_mode_fallback=True, step_describe_mode="auto")
    assert not _has_structured_output(ollama)
    assert _resolve_describe_mode(ollama, num_steps=5) == "iterative"


def test_care_flash_profiles_use_auto() -> None:
    for name in ("care_default", "care_local"):
        mage = tomllib.loads(Path(f"configs/{name}.toml").read_text(encoding="utf-8"))["mage"]
        assert mage["step_describe_mode"] == "auto"
