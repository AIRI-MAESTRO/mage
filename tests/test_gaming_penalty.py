"""T25 — benchmark-gaming-suspected chains are penalized in selection."""

from __future__ import annotations

from typing import Any

import pytest

from mmar_mage.benchmark.schemas import ChainScore
from mmar_mage.evolve.engine import ChainEvolver
from mmar_mage.evolve.schemas import EvolveConfig, Individual
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


def _ind(iid: str, overall: float, *, gaming: bool = False) -> Individual:
    return Individual(
        individual_id=iid,
        chain_dict={"steps": []},
        score=ChainScore(overall=overall, suspect_gaming=gaming),
    )


def _evolver(mocker: Any, **cfg: Any) -> ChainEvolver:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    gen = MAGEGenerator(config=MAGEConfig(api_key="test"))
    return ChainEvolver(gen, EvolveConfig(**cfg))


def test_rank_key_penalizes_gaming() -> None:
    assert ChainEvolver._rank_key(_ind("h", 0.6)) == 0.6  # honest: exactly overall
    gamed = ChainEvolver._rank_key(_ind("g", 0.9, gaming=True))
    assert gamed == pytest.approx(0.4)  # 0.9 - _GAMING_PENALTY(0.5)
    assert gamed < ChainEvolver._rank_key(_ind("h", 0.6))  # demoted below an honest 0.6


def test_ranking_demotes_but_does_not_exclude_gamed() -> None:
    pop = [_ind("g", 0.9, gaming=True), _ind("h", 0.6), _ind("low", 0.2)]
    ranked = sorted(pop, key=ChainEvolver._rank_key, reverse=True)
    # gamed (raw 0.9) sinks below the honest 0.6 but stays above the honest 0.2 — penalty, not exclusion
    assert [i.individual_id for i in ranked] == ["h", "g", "low"]


def test_select_keeps_honest_over_higher_raw_gamed(mocker: Any) -> None:
    evolver = _evolver(mocker, population_size=1, elitism=0, enable_pareto_selection=False)
    survivors = evolver._select([_ind("h", 0.6)], [_ind("g", 0.9, gaming=True)])
    assert [i.individual_id for i in survivors] == ["h"]


def test_select_pareto_also_penalizes_gaming(mocker: Any) -> None:
    evolver = _evolver(mocker, population_size=1, elitism=0, enable_pareto_selection=True)
    survivors = evolver._select([_ind("h", 0.6)], [_ind("g", 0.9, gaming=True)])
    assert survivors[0].individual_id == "h"
