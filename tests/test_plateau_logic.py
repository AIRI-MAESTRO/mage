"""T24 — evolve plateau early-stop logic on synthetic best-score sequences."""

from __future__ import annotations

from mmar_mage.evolve.engine import ChainEvolver
from mmar_mage.evolve.schemas import EvolveConfig

_step = ChainEvolver._plateau_step


def _run(scores: list[float], *, min_delta: float, patience: int) -> tuple[int, bool, int]:
    """Replay a best-score sequence (``scores[0]`` is the gen-0 baseline) through
    the plateau step; return ``(generations_run, stopped, final_counter)``."""
    last_best = scores[0]
    counter = 0
    for i, best in enumerate(scores[1:], start=1):
        counter, stop = _step(best, last_best, counter, min_delta=min_delta, patience=patience)
        last_best = best
        if stop:
            return i, True, counter
    return len(scores) - 1, False, counter


def test_stops_after_exactly_patience_flat_gens() -> None:
    # baseline + 3 flat gens → stop on the 3rd (patience=3), not a generation early
    gens, stopped, counter = _run([0.5, 0.5, 0.5, 0.5], min_delta=0.01, patience=3)
    assert stopped and gens == 3 and counter == 3


def test_significant_improvement_resets_counter() -> None:
    # flat, flat, big jump, flat → the jump resets, so no early stop across 4 gens
    gens, stopped, _ = _run([0.5, 0.5, 0.5, 0.7, 0.7], min_delta=0.01, patience=3)
    assert not stopped and gens == 4


def test_subthreshold_improvement_counts_as_plateau() -> None:
    # an improvement must EXCEED min_delta to reset; +0.005 and exactly +0.01 do not
    c1, s1 = _step(0.505, 0.5, 0, min_delta=0.01, patience=3)
    assert c1 == 1 and s1 is False
    c2, s2 = _step(0.51, 0.5, c1, min_delta=0.01, patience=3)  # +0.01 == delta, still <=
    assert c2 == 2 and s2 is False


def test_steady_improvement_never_plateaus() -> None:
    gens, stopped, counter = _run([0.1, 0.3, 0.5, 0.7, 0.9], min_delta=0.01, patience=3)
    assert not stopped and counter == 0 and gens == 4


def test_wp3_defaults() -> None:
    cfg = EvolveConfig()
    assert cfg.plateau_patience == 3
    assert cfg.plateau_min_delta == 0.01
