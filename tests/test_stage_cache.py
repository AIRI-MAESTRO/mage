"""T42 — content-addressable stage cache + the generator's _cached_stage helper."""

from __future__ import annotations

from typing import Any

from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig
from mmar_mage.stage_cache import StageCache


def test_key_is_deterministic_and_input_sensitive() -> None:
    assert StageCache.key("s", {"a": 1, "b": 2}) == StageCache.key("s", {"b": 2, "a": 1})  # sort_keys
    assert StageCache.key("s", "x") != StageCache.key("s", "y")  # payload-sensitive
    assert StageCache.key("a", "x") != StageCache.key("b", "x")  # stage-sensitive


def test_get_put_miss_and_counters() -> None:
    c = StageCache()
    assert c.get("s", "k") is None and c.misses == 1 and c.hits == 0
    c.put("s", "k", 42)
    assert c.get("s", "k") == 42 and c.hits == 1
    c.put("s", "k2", None)  # a cached None is a hit, not a miss
    assert c.get("s", "k2") is None and c.hits == 2
    c.clear()
    assert c.get("s", "k") is None and c.hits == 0


def _gen(mocker: Any, *, cache: bool) -> MAGEGenerator:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    return MAGEGenerator(config=MAGEConfig(api_key="test", enable_stage_cache=cache))


async def test_cached_stage_computes_once(mocker: Any) -> None:
    gen = _gen(mocker, cache=True)
    calls = 0

    async def factory() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"v": calls}

    r1 = await gen._cached_stage("domain", "q1", factory)
    r2 = await gen._cached_stage("domain", "q1", factory)
    assert calls == 1 and r1 == r2 == {"v": 1}
    r3 = await gen._cached_stage("domain", "q2", factory)  # different key → recompute
    assert calls == 2 and r3 == {"v": 2}


async def test_cached_stage_disabled_recomputes(mocker: Any) -> None:
    gen = _gen(mocker, cache=False)  # default off
    calls = 0

    async def factory() -> int:
        nonlocal calls
        calls += 1
        return calls

    await gen._cached_stage("s", "k", factory)
    await gen._cached_stage("s", "k", factory)
    assert calls == 2  # no caching when disabled


async def test_cached_stage_returns_isolated_copy(mocker: Any) -> None:
    gen = _gen(mocker, cache=True)

    async def factory() -> dict[str, list[int]]:
        return {"steps": [1, 2]}

    r1 = await gen._cached_stage("s", "k", factory)
    r1["steps"].append(99)  # mutate the returned value
    r2 = await gen._cached_stage("s", "k", factory)
    assert r2 == {"steps": [1, 2]}  # the cache wasn't tainted by the mutation
