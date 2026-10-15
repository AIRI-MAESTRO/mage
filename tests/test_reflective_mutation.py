"""Reflective (GEPA-style) mutation operator — plan item P1.8.

Drives ``mmar_mage.evolve.operators.reflective_mutate`` and the shared
``_apply_edit`` helper with a directly-mocked LLM. Reflective mutation diagnoses
the systematic weakness behind the per-task failures, then applies ONE structural
edit as a "gradient" toward a better chain. Also includes a regression test that
the refactored ``llm_repair`` still applies edits via the shared helper.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mmar_mage.benchmark.schemas import ChainScore
from mmar_mage.evolve.operators import _apply_edit, llm_repair, reflective_mutate
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


def _mock_llm(data: dict[str, Any] | Exception) -> MagicMock:
    llm = MagicMock()
    if isinstance(data, Exception):
        llm.generate_json = AsyncMock(side_effect=data)
    else:
        llm.generate_json = AsyncMock(return_value=data)
    return llm


def _score(**failures: str) -> ChainScore:
    return ChainScore(overall=0.4, per_task_failures=dict(failures))


# --------------------------------------------------------------------------- #
# _apply_edit — shared structural-edit helper
# --------------------------------------------------------------------------- #
def test_apply_edit_remove_step_rewires_deps():
    out = _apply_edit(_llm_chain(3), "remove_step", 2, None)
    assert len(out["steps"]) == 2
    # renumbered contiguously; no dependency points at the removed step
    nums = [s["number"] for s in out["steps"]]
    assert nums == [1, 2]
    for s in out["steps"]:
        assert all(d in nums for d in s["dependencies"])
    validate_carl_json(out, strict_carl=True)


def test_apply_edit_insert_step_after_target():
    new = {"step_type": "llm", "title": "Injected", "aim": "Validate intermediate result"}
    out = _apply_edit(_llm_chain(2), "insert_step", 1, new)
    assert len(out["steps"]) == 3
    titles = [s["title"] for s in out["steps"]]
    assert "Injected" in titles
    validate_carl_json(out, strict_carl=True)


def test_apply_edit_rewrite_preserves_number_and_deps():
    new = {"step_type": "llm", "title": "Rewritten", "aim": "New aim", "stage_action": "Do"}
    out = _apply_edit(_llm_chain(3), "rewrite_step", 3, new)
    target = next(s for s in out["steps"] if s["number"] == 3)
    assert target["title"] == "Rewritten"
    assert target["dependencies"] == [2]  # preserved
    validate_carl_json(out, strict_carl=True)


def test_apply_edit_unknown_type_returns_unchanged():
    chain = _llm_chain(2)
    out = _apply_edit(chain, "frobnicate", 1, None)
    assert out == chain


def test_apply_edit_does_not_mutate_input():
    chain = _llm_chain(3)
    before = len(chain["steps"])
    _apply_edit(chain, "remove_step", 2, None)
    assert len(chain["steps"]) == before  # deepcopy isolation


# --------------------------------------------------------------------------- #
# reflective_mutate
# --------------------------------------------------------------------------- #
async def test_reflective_mutate_applies_proposed_edit():
    llm = _mock_llm(
        {
            "diagnosis": "No validation of the intermediate computation.",
            "edit_type": "insert_step",
            "target_step_number": 2,
            "new_step": {"step_type": "llm", "title": "Verify", "aim": "Sanity-check the result"},
            "rationale": "Adds the missing verification the failures point to.",
        }
    )
    out = await reflective_mutate(_llm_chain(3), _score(synth_001="wrong total"), "q", llm)
    assert len(out["steps"]) == 4
    assert any(s["title"] == "Verify" for s in out["steps"])
    validate_carl_json(out, strict_carl=True)
    # diagnosis-first prompt was used
    assert llm.generate_json.await_args.kwargs["stage"] == "evolve_reflective"


async def test_reflective_mutate_no_failures_returns_unchanged():
    llm = _mock_llm({"edit_type": "remove_step", "target_step_number": 1})
    chain = _llm_chain(2)
    out = await reflective_mutate(chain, ChainScore(overall=1.0), "q", llm)
    assert out == chain
    llm.generate_json.assert_not_called()  # short-circuits before any LLM call


async def test_reflective_mutate_llm_failure_returns_unchanged():
    llm = _mock_llm(RuntimeError("boom"))
    chain = _llm_chain(2)
    out = await reflective_mutate(chain, _score(synth_001="bad"), "q", llm)
    assert out == chain


async def test_reflective_mutate_unknown_edit_returns_unchanged():
    llm = _mock_llm({"diagnosis": "x", "edit_type": "noop", "target_step_number": None})
    chain = _llm_chain(2)
    out = await reflective_mutate(chain, _score(synth_001="bad"), "q", llm)
    assert out == chain


# --------------------------------------------------------------------------- #
# llm_repair regression — still applies edits via the shared helper
# --------------------------------------------------------------------------- #
async def test_llm_repair_still_applies_edit_after_refactor():
    llm = _mock_llm({"edit_type": "remove_step", "target_step_number": 2})
    out = await llm_repair(_llm_chain(3), _score(synth_001="too slow"), "q", llm)
    assert len(out["steps"]) == 2
    validate_carl_json(out, strict_carl=True)


async def test_llm_repair_no_failures_short_circuits():
    llm = _mock_llm({"edit_type": "remove_step", "target_step_number": 1})
    chain = _llm_chain(2)
    out = await llm_repair(chain, ChainScore(overall=1.0), "q", llm)
    assert out == chain
    llm.generate_json.assert_not_called()


# --------------------------------------------------------------------------- #
# Real-execution scenario — dispatch through the ChainEvolver engine path
# --------------------------------------------------------------------------- #
async def test_engine_dispatches_reflective_mutate_to_valid_child():
    """Exercise the full engine path: _pick_operator weight → _apply_operator
    dispatch → reflective_mutate → _apply_edit → to_carl_compat → validate."""
    from mmar_mage.evolve.engine import ChainEvolver
    from mmar_mage.evolve.schemas import EvolveConfig, Individual
    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    gen = MAGEGenerator(config=MAGEConfig(api_key="t", enable_memory=False))
    gen._llm = _mock_llm(
        {
            "diagnosis": "Missing a synthesis step that combines the prior outputs.",
            "edit_type": "insert_step",
            "target_step_number": 2,
            "new_step": {"step_type": "llm", "title": "Synthesize", "aim": "Combine prior results"},
            "rationale": "Adds the aggregation the failures require.",
        }
    )
    # Zero every other operator weight so the weighted picker is DETERMINISTIC
    # (reflective_mutate is the only candidate) — a high-but-finite weight left a
    # ~6% chance of another op, which made this assertion flaky under some RNG seeds.
    evolver = ChainEvolver(
        generator=gen,
        evolve_config=EvolveConfig(
            weight_reflective_mutate=1.0,
            weight_mutate_aim=0.0,
            weight_change_step_type=0.0,
            weight_insert_step=0.0,
            weight_remove_step=0.0,
            weight_swap_steps=0.0,
            weight_crossover=0.0,
            weight_llm_repair=0.0,
        ),
    )

    # reflective_mutate is the only non-zero-weight operator → deterministic pick.
    assert evolver._pick_operator() == "reflective_mutate"

    parent = Individual(
        individual_id="seed",
        chain_dict=_llm_chain(3),
        score=_score(synth_001="no final synthesis"),
    )
    child = await evolver._apply_operator("reflective_mutate", [parent], "q")

    assert child is not None
    assert child.operator == "reflective_mutate"
    assert child.parents == ["seed"]
    assert any(s["title"] == "Synthesize" for s in child.chain_dict["steps"])
    validate_carl_json(child.chain_dict, strict_carl=True)
