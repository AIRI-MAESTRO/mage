"""T27 — evolve operators stay robust on degenerate / boundary chains."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pytest

from mmar_mage.evolve import operators as ops


def _chain(*specs: tuple[int, str, list[int]]) -> dict[str, Any]:
    return {
        "name": "t", "max_workers": 1, "timeout": 300.0,
        "steps": [
            {"step_type": st, "number": n, "title": f"S{n}", "aim": f"aim {n}", "dependencies": list(deps)}
            for (n, st, deps) in specs
        ],
    }


# The audit's named boundary cases (2-step chain, single synthesis step) + extras.
_CASES: dict[str, dict[str, Any]] = {
    "empty": {"steps": []},
    "one_step": _chain((1, "llm", [])),
    "two_step": _chain((1, "llm", []), (2, "llm", [1])),
    "three_step": _chain((1, "llm", []), (2, "tool", [1]), (3, "llm", [2])),
    "mixed_types": _chain((1, "tool", []), (2, "memory", [1]), (3, "transform", [2]), (4, "llm", [3])),
    "no_deps_key": {"steps": [{"step_type": "llm", "number": 1, "title": "A"},
                              {"step_type": "llm", "number": 2, "title": "B"}]},
}

_MUTATORS: list[Callable[[dict[str, Any], random.Random], dict[str, Any]]] = [
    ops.mutate_aim, ops.change_step_type, ops.insert_step, ops.remove_step, ops.swap_steps,
]


def _assert_valid(chain: dict[str, Any]) -> None:
    steps = chain.get("steps", [])
    assert isinstance(steps, list)
    numbers = [s["number"] for s in steps]
    assert numbers == list(range(1, len(steps) + 1))  # contiguous 1..N, in order
    valid = set(numbers)
    for s in steps:
        for d in s.get("dependencies", []) or []:
            assert d in valid and d != s["number"]  # no dangling / self deps


@pytest.mark.parametrize("op", _MUTATORS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_mutators_never_crash_or_corrupt(op: Any, case: str) -> None:
    out = op(_CASES[case], random.Random(1234))  # must not raise on any boundary chain
    _assert_valid(out)


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_crossover_never_crashes_or_corrupts(case: str) -> None:
    child = ops.crossover(_CASES[case], _CASES["three_step"], random.Random(7))  # must not raise
    _assert_valid(child)


def test_renumber_tolerates_missing_number() -> None:
    chain = {"steps": [{"step_type": "llm", "title": "A", "dependencies": []},
                       {"step_type": "llm", "number": 5, "title": "B", "dependencies": [5]}]}
    out = ops._renumber(chain)
    assert [s["number"] for s in out["steps"]] == [1, 2]  # positional, no crash
    assert out["steps"][1]["dependencies"] == []  # the self-dep on old 5 is dropped


def test_renumber_drops_deps_to_duplicate_numbers() -> None:
    chain = _chain((1, "llm", []), (1, "tool", []), (3, "llm", [1]))  # number 1 duplicated
    out = ops._renumber(chain)
    assert [s["number"] for s in out["steps"]] == [1, 2, 3]
    assert out["steps"][2]["dependencies"] == []  # dep on the ambiguous duplicate '1' dropped


def test_renumber_unique_chain_remaps_normally() -> None:
    out = ops._renumber(_chain((10, "llm", []), (20, "tool", [10]), (30, "llm", [20])))
    assert [s["number"] for s in out["steps"]] == [1, 2, 3]
    assert out["steps"][1]["dependencies"] == [1]
    assert out["steps"][2]["dependencies"] == [2]
