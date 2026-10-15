"""WP-3 — honest accounting of invalid children + plateau tuning.

* A slot whose operator yields an invalid child is retried with a *different*
  operator (up to twice), so one flaky operator no longer silently shrinks the
  generation; the count rides into the snapshot.
* `plateau_patience` defaults to 3 and a plateau needs improvement ≤
  `plateau_min_delta` (not ~1e-6), so a noisy judge can't trigger a premature stop.
"""

from __future__ import annotations

from typing import Any

from mmar_mage.evolve.engine import ChainEvolver
from mmar_mage.evolve.schemas import EvolveConfig, Individual
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import MAGEConfig


def _ind(op: str, idx: int = 0) -> Individual:
    return Individual(individual_id=f"{op}-{idx}", chain_dict={"name": "c", "steps": []}, operator=op)


def test_evolve_config_plateau_defaults() -> None:
    cfg = EvolveConfig()
    assert cfg.plateau_patience == 3
    assert cfg.plateau_min_delta == 0.01


def _make_evolver(mocker: Any, **cfg: Any) -> ChainEvolver:
    mocker.patch("mmar_mage.llm.AsyncOpenAI")
    gen = MAGEGenerator(config=MAGEConfig(mode="fast", api_key="test"))
    return ChainEvolver(generator=gen, evolve_config=EvolveConfig(**cfg))


async def test_invalid_operator_does_not_shrink_generation(mocker: Any) -> None:
    evolver = _make_evolver(mocker, population_size=4, elitism=0)

    async def fake_apply(op: str, ranked: Any, query: str) -> Individual | None:
        return None if op == "remove_step" else _ind(op)  # one operator is always invalid

    # First pick the bad operator, then a good one on retry — exercises the retry path.
    seq = iter(["remove_step", "mutate_aim"] * 50)
    mocker.patch.object(evolver, "_apply_operator", new=fake_apply)
    mocker.patch.object(evolver, "_pick_operator", new=lambda exclude=None: next(seq))

    parents = [_ind("initial", i) for i in range(4)]
    children, invalid = await evolver._next_generation(parents, None, None, "q", 1)

    assert len(children) == 4  # no shrink — retries filled every slot
    assert invalid == 4  # exactly one rejected (remove_step) attempt per slot


def test_snapshot_records_invalid_children(mocker: Any) -> None:
    evolver = _make_evolver(mocker, population_size=3, elitism=0)
    pop = [_ind("initial", i) for i in range(3)]
    snap = evolver._snapshot(pop, generation=2, invalid_children=7)
    assert snap.invalid_children == 7
    assert snap.population_size == 3
