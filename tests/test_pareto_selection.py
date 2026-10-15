"""GEPA-style Pareto selection in evolve — plan item P1.8 (deferred follow-up).

Survivor selection over (quality ↑, cost ↓): a cheaper chain of equal quality is
never crowded out by a pricier one. Pure evolve logic (no CARL round-trip). OFF by
default — pure-quality selection is unchanged.
"""

from __future__ import annotations

from mmar_mage.evolve.engine import ChainEvolver, _pareto_front
from mmar_mage.evolve.schemas import ChainScore, EvolveConfig, Individual
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


# --------------------------------------------------------------------------- #
# _pareto_front (pure function)
# --------------------------------------------------------------------------- #
def _pt(q: float, c: float) -> dict[str, float]:
    return {"q": q, "c": c}


def test_pareto_front_drops_dominated():
    a = _pt(0.9, 100)  # dominated by b (equal quality, cheaper)
    b = _pt(0.9, 10)
    c = _pt(0.5, 5)  # cheapest — not dominated (lower quality)
    front = _pareto_front([a, b, c], quality=lambda x: x["q"], cost=lambda x: x["c"])
    assert b in front and c in front
    assert a not in front


def test_pareto_front_all_nondominated_when_tradeoffs():
    # strictly increasing quality with increasing cost → all on the front
    items = [_pt(0.3, 1), _pt(0.6, 5), _pt(0.9, 20)]
    front = _pareto_front(items, quality=lambda x: x["q"], cost=lambda x: x["c"])
    assert len(front) == 3


def test_pareto_front_single_global_best_dominates_pricier_ties():
    items = [_pt(1.0, 50), _pt(1.0, 10), _pt(1.0, 90)]  # all top quality → cheapest dominates
    front = _pareto_front(items, quality=lambda x: x["q"], cost=lambda x: x["c"])
    assert front == [_pt(1.0, 10)]


# --------------------------------------------------------------------------- #
# _select integration (real ChainEvolver + real cost estimation)
# --------------------------------------------------------------------------- #
def _evolver(pareto: bool, population_size: int = 2) -> ChainEvolver:
    gen = MAGEGenerator(config=MAGEConfig(api_key="t", enable_memory=False))
    cfg = EvolveConfig(population_size=population_size, enable_pareto_selection=pareto)
    return ChainEvolver(generator=gen, evolve_config=cfg)


def _ind(iid: str, quality: float, aim_len: int) -> Individual:
    # aim length controls estimated token cost (chars/4); same step count.
    return Individual(
        individual_id=iid,
        chain_dict={
            "name": "c",
            "steps": [{"step_type": "llm", "number": 1, "title": "S", "aim": "x" * aim_len, "dependencies": []}],
        },
        score=ChainScore(overall=quality),
    )


def test_individual_cost_tracks_chain_size():
    ev = _evolver(pareto=True)
    cheap = ev._individual_cost(_ind("a", 0.9, 8))
    pricey = ev._individual_cost(_ind("b", 0.9, 800))
    assert pricey > cheap > 0


def test_pareto_select_keeps_cost_efficient_over_redundant_pricey():
    ev = _evolver(pareto=True, population_size=2)
    pricey = _ind("pricey", 0.9, 800)  # dominated by `cheap`
    cheap = _ind("cheap", 0.9, 8)
    frugal = _ind("frugal", 0.5, 4)  # lower quality but cheapest → on the front
    out = {i.individual_id for i in ev._select([pricey], [cheap, frugal])}
    assert out == {"cheap", "frugal"}  # the redundant pricey 0.9 is dropped


def test_pure_quality_select_keeps_both_top_quality():
    ev = _evolver(pareto=False, population_size=2)
    pricey = _ind("pricey", 0.9, 800)
    cheap = _ind("cheap", 0.9, 8)
    frugal = _ind("frugal", 0.5, 4)
    out = {i.individual_id for i in ev._select([pricey], [cheap, frugal])}
    assert out == {"pricey", "cheap"}  # both 0.9s kept; frugal (0.5) dropped — contrast w/ Pareto


def test_pareto_select_preserves_global_best_and_fills_by_quality():
    ev = _evolver(pareto=True, population_size=3)
    best = _ind("best", 0.95, 50)
    mid = _ind("mid", 0.7, 50)
    low = _ind("low", 0.4, 50)  # same cost → strictly quality-dominated, off the front
    out = ev._select([best, mid], [low])
    ids = [i.individual_id for i in out]
    assert ids[0] == "best"  # global best preserved + ranked first
    assert len(out) == 3  # fills to population_size


def test_pareto_select_caps_at_population_size():
    ev = _evolver(pareto=True, population_size=2)
    # three mutually-non-dominated (quality up, cost up) → front bigger than size
    a = _ind("a", 0.3, 4)
    b = _ind("b", 0.6, 40)
    c = _ind("c", 0.9, 400)
    out = ev._select([a, b], [c])
    assert len(out) == 2
    assert out[0].individual_id == "c"  # best-quality first within the capped front
