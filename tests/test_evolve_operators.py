"""Tests for evolutionary operators on chain_dict."""

from __future__ import annotations

import random
from typing import Any

import pytest

from mmar_mage.evolve.operators import (
    change_step_type,
    crossover,
    insert_step,
    mutate_aim,
    remove_step,
    swap_steps,
)
from mmar_mage.generator import validate_carl_json


def _llm_chain(n: int = 3) -> dict[str, Any]:
    return {
        "name": "demo",
        "max_workers": 2,
        "steps": [
            {
                "step_type": "llm",
                "number": i + 1,
                "title": f"Step {i + 1}",
                "aim": f"Determine X{i + 1}",
                "reasoning_questions": "what?",
                "step_context_queries": ["foo", "bar"],
                "stage_action": "Process",
                "example_reasoning": "",
                "dependencies": [i] if i > 0 else [],
                "retry_max": 3,
            }
            for i in range(n)
        ],
    }


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


class TestMutateAim:
    def test_changes_one_aim(self) -> None:
        rng = random.Random(0)
        original = _llm_chain(3)
        mutated = mutate_aim(original, rng)
        assert mutated is not original
        # exactly one step's aim changed
        diffs = [i for i in range(3) if mutated["steps"][i]["aim"] != original["steps"][i]["aim"]]
        assert len(diffs) == 1

    def test_does_not_break_validation(self) -> None:
        rng = random.Random(1)
        chain = mutate_aim(_llm_chain(3), rng)
        validate_carl_json(chain)


class TestChangeStepType:
    def test_llm_to_structured_output(self) -> None:
        rng = random.Random(0)
        chain = change_step_type(_llm_chain(2), rng)
        types = [s["step_type"] for s in chain["steps"]]
        assert "structured_output" in types
        # the SO step has the schema fields
        so_step = next(s for s in chain["steps"] if s["step_type"] == "structured_output")
        assert "structured_output_schema" in so_step
        validate_carl_json(chain)

    def test_round_trip(self) -> None:
        rng = random.Random(0)
        chain = _llm_chain(2)
        chain = change_step_type(chain, rng)
        chain = change_step_type(chain, rng)
        # back to llm — schema fields should be cleaned up on at least one step
        # (deterministic, since random.Random(0) picks the same step twice for size=2)
        validate_carl_json(chain)


class TestInsertStep:
    def test_insert_increases_step_count(self) -> None:
        rng = random.Random(0)
        original = _llm_chain(3)
        mutated = insert_step(original, rng)
        assert len(mutated["steps"]) == 4
        # final step now depends on inserted step
        validate_carl_json(mutated)

    def test_renumbering_contiguous(self) -> None:
        rng = random.Random(0)
        chain = insert_step(_llm_chain(3), rng)
        nums = [s["number"] for s in chain["steps"]]
        assert nums == [1, 2, 3, 4]


class TestRemoveStep:
    def test_remove_middle_step(self) -> None:
        rng = random.Random(0)
        original = _llm_chain(4)
        mutated = remove_step(original, rng)
        assert len(mutated["steps"]) == 3
        validate_carl_json(mutated)

    def test_does_not_remove_when_too_short(self) -> None:
        rng = random.Random(0)
        chain = _llm_chain(2)
        out = remove_step(chain, rng)
        assert len(out["steps"]) == 2  # untouched


class TestSwapSteps:
    def test_swap_when_independent(self) -> None:
        rng = random.Random(0)
        chain = _llm_chain(5)
        # break the linear dependency between steps 2 and 3 so a swap is legal
        chain["steps"][2]["dependencies"] = [1]
        chain["steps"][3]["dependencies"] = [2, 3]
        mutated = swap_steps(chain, rng)
        validate_carl_json(mutated)

    def test_no_swap_when_dependent(self) -> None:
        rng = random.Random(0)
        chain = _llm_chain(5)  # strict pipeline, every step depends on previous
        out = swap_steps(chain, rng)
        # Either nothing changed (pipeline) or it produced something valid.
        validate_carl_json(out)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


class TestCrossover:
    def test_crossover_produces_valid_chain(self) -> None:
        rng = random.Random(0)
        a = _llm_chain(4)
        b = _llm_chain(4)
        # tag steps so we can see which parent contributed
        for s in a["steps"]:
            s["title"] = f"A-{s['number']}"
        for s in b["steps"]:
            s["title"] = f"B-{s['number']}"
        child = crossover(a, b, rng)
        validate_carl_json(child)
        titles = [s["title"] for s in child["steps"]]
        # at least one A and one B
        assert any(t.startswith("A-") for t in titles)
        assert any(t.startswith("B-") for t in titles)

    def test_short_parent_falls_back(self) -> None:
        rng = random.Random(0)
        a = _llm_chain(1)  # too short
        b = _llm_chain(4)
        child = crossover(a, b, rng)
        # should be a copy of a, not blow up
        assert len(child["steps"]) == 1


# ---------------------------------------------------------------------------
# Sanity: every operator produces a CARL-loadable chain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name",
    ["mutate_aim", "change_step_type", "insert_step", "remove_step", "swap_steps"],
)
def test_operator_output_passes_strict_carl(op_name: str) -> None:
    from mmar_mage.carl_export import to_carl_compat

    rng = random.Random(0)
    op_map = {
        "mutate_aim": mutate_aim,
        "change_step_type": change_step_type,
        "insert_step": insert_step,
        "remove_step": remove_step,
        "swap_steps": swap_steps,
    }
    chain = _llm_chain(4)
    out = op_map[op_name](chain, rng)
    out = to_carl_compat(out)
    validate_carl_json(out, strict_carl=True)
