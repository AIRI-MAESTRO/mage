"""WP-2 — crossover must not silently orphan suffix steps.

The pre-fix `crossover` bridged only the FIRST suffix step to the prefix; deeper
suffix steps whose dependencies pointed into B's cut-off prefix had those deps
dropped by `_renumber` and survived validation as orphans.
"""

from __future__ import annotations

from typing import Any

from mmar_mage.evolve.operators import crossover


class _SeqRng:
    """Deterministic stand-in: returns the queued cut points in order."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def randint(self, _lo: int, _hi: int) -> int:
        return self._values.pop(0)


def _step(number: int, deps: list[int]) -> dict[str, Any]:
    return {"step_type": "llm", "number": number, "title": f"s{number}", "aim": f"do {number}", "dependencies": deps}


def _chain(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": "C", "max_workers": 2, "timeout": 300.0, "steps": steps}


def test_crossover_rebridges_orphaned_diamond_suffix() -> None:
    # A: linear 1 -> 2 -> 3
    a = _chain([_step(1, []), _step(2, [1]), _step(3, [2])])
    # B: diamond 1 -> (2, 3) -> 4
    b = _chain([_step(1, []), _step(2, [1]), _step(3, [1]), _step(4, [2, 3])])

    # cut_a=2 (prefix = A1,A2), cut_b=2 (suffix = B3, B4) — a mid cut of the diamond.
    child = crossover(a, b, _SeqRng([2, 2]))
    steps = child["steps"]
    nums = [s["number"] for s in steps]

    assert nums == list(range(1, len(steps) + 1)), "steps must be contiguously renumbered"
    for s in steps:
        deps = s.get("dependencies", [])
        for d in deps:  # every dep is valid: existing + strictly earlier
            assert d in nums and d < s["number"], f"step {s['number']} has bad dep {d}"
        if s["number"] != 1:  # only the level-1 entry may have no deps
            assert deps, f"step {s['number']} was orphaned (no dependencies)"


def test_crossover_aborts_when_too_many_orphans() -> None:
    # A: 1 -> 2
    a = _chain([_step(1, []), _step(2, [1])])
    # B: star — 2,3,4,5 all depend ONLY on B1
    b = _chain([_step(1, []), _step(2, [1]), _step(3, [1]), _step(4, [1]), _step(5, [1])])

    # cut_a=1 (prefix = A1), cut_b=1 (suffix = B2..B5, all dep on the cut-off B1).
    # 4 of 5 child steps would be orphaned (0.8 > 0.30) → abort, keep parent A.
    child = crossover(a, b, _SeqRng([1, 1]))
    assert [s["title"] for s in child["steps"]] == ["s1", "s2"]  # == parent_a, untouched
