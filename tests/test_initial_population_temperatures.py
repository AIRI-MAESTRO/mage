"""WP-1 / WP-6 — initial-population generation.

WP-1: the per-temperature *top-up* generations must use distinct per-call
temperatures without mutating the evolver's shared MAGE config. The pre-WP-1
code wrote ``self._mage_config.temperature = temp`` inside coroutines run under
``asyncio.gather`` on a single shared generator/LLMClient, so the
``initial_temperatures`` diversification silently did nothing.

WP-6: the population is now seeded by ONE ``generate(query,
return_candidates=True)`` call (sharing the research/domain pass across every
topology-sampled candidate) and only the *remaining* slots are filled by the
per-temperature top-up path. In fast mode that shared call yields exactly one
candidate at the generator's base temperature, so the top-up fills
``population_size - 1`` slots, iterating ``initial_temperatures``. This test
pins that contract: the base temperature plus the first ``population_size - 1``
distinct top-up temperatures are all exercised, the population reaches
``population_size``, and the shared config/generator are never mutated.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.evolve.engine import ChainEvolver
from mmar_mage.evolve.schemas import EvolveConfig
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import DomainAnalysis, MAGEConfig

_CHAIN = {
    "name": "T",
    "max_workers": 2,
    "timeout": 300.0,
    "search_config": {
        "strategy": "substring",
        "substring_config": {"case_sensitive": False, "min_word_length": 3, "max_matches_per_query": 5},
    },
    "steps": [
        {
            "step_type": "llm", "number": 1, "title": "A", "aim": "do a",
            "reasoning_questions": "q", "step_context_queries": ["x"], "stage_action": "do",
            "example_reasoning": "e", "dependencies": [], "retry_max": 3, "llm_config": None,
        },
        {
            "step_type": "llm", "number": 2, "title": "B", "aim": "do b",
            "reasoning_questions": "q", "step_context_queries": ["x"], "stage_action": "do",
            "example_reasoning": "e", "dependencies": [1], "retry_max": 3, "llm_config": None,
        },
    ],
}


def _resp(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    r = MagicMock()
    r.choices = [choice]
    return r


async def test_initial_population_uses_distinct_temperatures(mocker: Any) -> None:
    recorded: list[float] = []

    async def _create(**kwargs: Any) -> MagicMock:
        if kwargs.get("temperature") is not None:
            recorded.append(kwargs["temperature"])
        return _resp(json.dumps(_CHAIN))

    mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
    instance = mock_cls.return_value
    instance.chat.completions.create = AsyncMock(side_effect=_create)
    instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

    temps = [0.1, 0.4, 0.7, 0.95]
    base_temp = 0.2
    pop_size = 4
    config = MAGEConfig(mode="fast", api_key="test", temperature=base_temp, enable_memory=False)
    gen = MAGEGenerator(config=config)
    evolver = ChainEvolver(
        generator=gen,
        evolve_config=EvolveConfig(population_size=pop_size, initial_temperatures=temps),
    )

    pop = await evolver._build_initial_population("analyze X", DomainAnalysis(domain="test"))

    # WP-6: full population — 1 shared (research-once) member + (pop_size - 1) top-ups.
    assert len(pop) == pop_size
    # The shared call ran at the generator's base temperature.
    assert base_temp in recorded, f"base temp {base_temp} missing from {recorded}"
    # WP-1: the per-temperature top-up actually used distinct temperatures (the
    # first pop_size-1 of `initial_temperatures`). Under the old shared-config
    # code these collapsed to the base temperature.
    expected_topup = set(temps[: pop_size - 1])
    assert expected_topup <= set(recorded), f"recorded temps {recorded} missing {expected_topup - set(recorded)}"
    # The shared config / generator were never mutated.
    assert config.temperature == base_temp
    assert gen._config.temperature == base_temp  # noqa: SLF001
