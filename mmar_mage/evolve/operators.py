"""Evolutionary operators on ``chain_dict``.

Each operator takes one or two parent chain dicts and returns a new chain
dict that:

* preserves the top-level fields (``name``, ``max_workers``, ``timeout``,
  ``search_config``),
* updates ``steps`` with structural edits,
* renumbers steps contiguously and rewires ``dependencies`` so the result
  passes :func:`mmar_mage.generator.validate_carl_json`.

LLM-driven operators (``llm_repair``) require an :class:`LLMClient` plus the
parent's score so the prompt can target specific failures.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from typing import Any

from ..llm import LLMClient
from ..prompts import LLM_REPAIR_PROMPT, REFLECTIVE_MUTATION_PROMPT
from ..schemas import CARL_VALID_STEP_TYPES
from .schemas import ChainScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _renumber(chain: dict[str, Any]) -> dict[str, Any]:
    """Renumber steps 1..N contiguously and remap dependencies accordingly.

    Preserves order. Drops dependencies that point to removed steps, to a
    self-reference, or to an ambiguous duplicate number; a surviving dependency
    is remapped from its old number to the new one. Steps missing or sharing a
    ``number`` (a malformed chain) are renumbered positionally instead of
    crashing (T27) — for unique-numbered chains the behaviour is unchanged.
    """
    steps = chain.get("steps", []) or []
    # A number that repeats (or is missing) is an ambiguous remap target, so deps
    # pointing at it are dropped — we can't know which copy was meant. Only
    # unique, present old numbers map to a new position.
    counts: dict[Any, int] = {}
    for s in steps:
        counts[s.get("number")] = counts.get(s.get("number"), 0) + 1
    old_to_new: dict[Any, int] = {
        s.get("number"): i + 1
        for i, s in enumerate(steps)
        if s.get("number") is not None and counts.get(s.get("number")) == 1
    }
    for i, s in enumerate(steps):
        old_num = s.get("number")
        # Keep deps that point to a surviving, non-self step — checked in the
        # OLD numbering — then remap to the new numbers. The self-reference
        # check MUST use ``old_num`` (the dep's own old number), not the new
        # number: doing it after reassigning ``s["number"]`` wrongly dropped a
        # valid dep whose old number happened to equal this step's new number.
        s["dependencies"] = [
            old_to_new[d]
            for d in (s.get("dependencies") or [])
            if d in old_to_new and d != old_num
        ]
        s["number"] = i + 1
    return chain


def _make_step_template(number: int, step_type: str = "llm") -> dict[str, Any]:
    """Generate a placeholder step skeleton — operators flesh it out."""
    return {
        "step_type": step_type,
        "number": number,
        "title": "New step",
        "aim": "Refine intermediate result",
        "reasoning_questions": "What needs to be checked or extracted at this stage?",
        "step_context_queries": [],
        "stage_action": "Refine and validate the prior output",
        "example_reasoning": "",
        "dependencies": [],
        "retry_max": 3,
    }


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def mutate_aim(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Tweak the ``aim`` of a randomly selected LLM step (small perturbation)."""
    chain = copy.deepcopy(chain)
    candidates = [s for s in chain.get("steps", []) if s.get("step_type") in {"llm", "structured_output"}]
    if not candidates:
        return chain
    s = rng.choice(candidates)
    aim = s.get("aim") or "Process the input"
    # Heuristic perturbation — flip verb to a related one or add a refining clause.
    perturbations = [
        f"{aim} — and explicitly state assumptions",
        f"{aim} — emphasizing edge cases",
        aim.replace("Determine", "Verify"),
        aim.replace("Analyze", "Decompose and analyze"),
        aim.replace("Synthesize", "Synthesize and cross-check"),
        f"{aim}, validating against the rubric",
    ]
    s["aim"] = rng.choice([p for p in perturbations if p != aim] or [aim])
    return chain


def change_step_type(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Promote one ``llm`` step to ``structured_output`` (or vice versa).

    Useful when the task wants machine-readable JSON but the chain emits prose.
    """
    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])
    candidates = [s for s in steps if s.get("step_type") in {"llm", "structured_output"}]
    if not candidates:
        return chain
    s = rng.choice(candidates)
    if s["step_type"] == "llm":
        s["step_type"] = "structured_output"
        s.setdefault("structured_output_schema_name", "Output")
        s.setdefault(
            "structured_output_schema",
            {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        )
        s.setdefault("structured_output_strict_json", True)
    else:
        s["step_type"] = "llm"
        for k in [
            "structured_output_schema",
            "structured_output_schema_name",
            "structured_output_input_source",
            "structured_output_instruction",
            "structured_output_strict_json",
        ]:
            s.pop(k, None)
    return chain


def insert_step(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Insert a new validation/refinement step before the final synthesis."""
    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])
    if not steps:
        return chain
    # Insert before the last step — that's typically synthesis.
    insert_at = max(0, len(steps) - 1)

    # The new step depends on the step that immediately preceded the insert point.
    pred_number = steps[insert_at - 1]["number"] if insert_at > 0 else None
    new_step = _make_step_template(number=999, step_type="llm")  # number is fixed by _renumber
    new_step["title"] = "Sanity-check intermediate result"
    new_step["aim"] = "Verify intermediate output for completeness and consistency"
    new_step["reasoning_questions"] = (
        "Does the previous step's output cover all required rubric items? Are there contradictions or gaps?"
    )
    if pred_number is not None:
        new_step["dependencies"] = [pred_number]

    # Final step must now also depend on the new step.
    final = steps[-1]
    final.setdefault("dependencies", [])
    if pred_number is not None and pred_number in final["dependencies"]:
        final["dependencies"].remove(pred_number)
    # We use the placeholder 999 here; _renumber rewires it.
    final["dependencies"].append(999)

    steps.insert(insert_at, new_step)
    chain["steps"] = steps
    return _renumber(chain)


def remove_step(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Remove a non-terminal, non-source step. Re-wires dependencies through it."""
    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])
    if len(steps) <= 2:
        return chain  # nothing safe to remove

    middle = [i for i in range(1, len(steps) - 1)]
    if not middle:
        return chain
    idx = rng.choice(middle)
    removed = steps[idx]
    removed_num = removed["number"]
    removed_deps = removed.get("dependencies", []) or []

    # Any step that depended on the removed one inherits its dependencies.
    for s in steps:
        if removed_num in (s.get("dependencies") or []):
            s["dependencies"] = [d for d in s["dependencies"] if d != removed_num]
            for d in removed_deps:
                if d not in s["dependencies"]:
                    s["dependencies"].append(d)

    steps.pop(idx)
    chain["steps"] = steps
    return _renumber(chain)


def swap_steps(chain: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Swap two adjacent middle steps if it doesn't break dependencies."""
    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])
    if len(steps) < 4:
        return chain
    candidates = list(range(1, len(steps) - 2))
    if not candidates:
        return chain
    i = rng.choice(candidates)
    a, b = steps[i], steps[i + 1]
    # Only swap if neither depends on the other.
    if a["number"] in (b.get("dependencies") or []) or b["number"] in (a.get("dependencies") or []):
        return chain
    steps[i], steps[i + 1] = b, a
    chain["steps"] = steps
    return _renumber(chain)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------


def crossover(parent_a: dict[str, Any], parent_b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Take prefix from A and suffix from B, stitch with a fresh dependency.

    Cut points are chosen so each parent contributes at least one step.
    """
    a_steps = parent_a.get("steps", []) or []
    b_steps = parent_b.get("steps", []) or []
    if len(a_steps) < 2 or len(b_steps) < 2:
        return copy.deepcopy(parent_a)

    cut_a = rng.randint(1, len(a_steps) - 1)
    cut_b = rng.randint(1, len(b_steps) - 1)

    prefix = copy.deepcopy(a_steps[:cut_a])
    suffix = copy.deepcopy(b_steps[cut_b:])

    # Renumber the suffix to continue after the prefix so B's numbers can't
    # collide with A's (a collision corrupts dependency remapping in _renumber).
    offset = max((s["number"] for s in prefix), default=0)
    b_old_to_new = {s["number"]: offset + i + 1 for i, s in enumerate(suffix)}
    bridge_dep = prefix[-1]["number"]  # last prefix step — B's part hangs off it

    # Re-wire every suffix step. Deps that pointed to OTHER suffix steps are
    # remapped; deps that pointed into B's cut-off prefix are gone. A step thereby
    # left with no dependency (but which HAD one in B) is re-bridged to the last
    # prefix step instead of being silently orphaned — the WP-2 bug: only the
    # first suffix step got the bridge, so deeper suffix steps lost their deps in
    # `_renumber` and survived validation as orphans.
    orphaned = 0
    for s in suffix:
        original_deps = s.get("dependencies", []) or []
        remapped = [b_old_to_new[d] for d in original_deps if d in b_old_to_new]
        if not remapped and original_deps:
            remapped = [bridge_dep]
            orphaned += 1
        s["number"] = b_old_to_new[s["number"]]
        s["dependencies"] = remapped

    # The first suffix step is B's entry point into the stitched chain — always
    # connect it to the prefix tail.
    if suffix:
        first = suffix[0]
        if bridge_dep not in first["dependencies"]:
            first["dependencies"] = sorted({*first["dependencies"], bridge_dep})

    # If the splice orphaned too much of B's structure it's more damage than
    # signal — abort the recombination and keep parent A unchanged.
    child_len = len(prefix) + len(suffix)
    if child_len and orphaned / child_len > 0.30:
        return copy.deepcopy(parent_a)

    child = copy.deepcopy(parent_a)
    child["steps"] = prefix + suffix
    return _renumber(child)


# ---------------------------------------------------------------------------
# LLM-driven targeted repair
# ---------------------------------------------------------------------------


async def llm_repair(
    chain: dict[str, Any],
    score: ChainScore,
    query: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """Ask an LLM to propose ONE structural edit that addresses observed failures."""
    if not score.per_task_failures:
        return chain  # nothing concrete to fix

    failure_list = "\n".join(f"- {tid}: {reason}" for tid, reason in list(score.per_task_failures.items())[:6])
    user_prompt = (
        f"User query: {query}\n\n"
        f"Current chain JSON:\n{json.dumps(chain, ensure_ascii=False, indent=2)}\n\n"
        f"Score: overall={score.overall}\n"
        f"Per-task failures:\n{failure_list}\n\n"
        "Propose ONE structural edit."
    )

    try:
        data = await llm.generate_json(
            system_prompt=LLM_REPAIR_PROMPT,
            user_prompt=user_prompt,
            stage="evolve_repair",
        )
    except Exception as exc:
        logger.debug("llm_repair JSON failure: %s", exc)
        return chain

    return _apply_edit(
        chain,
        data.get("edit_type"),
        data.get("target_step_number"),
        data.get("new_step"),
    )


def _apply_edit(
    chain: dict[str, Any],
    edit_type: Any,
    target: Any,
    new_step: Any,
) -> dict[str, Any]:
    """Apply one structural edit (from LLM repair / reflective mutation) to a chain.

    Shared by :func:`llm_repair` and :func:`reflective_mutate`. Returns a new
    (renumbered) chain; unknown ``edit_type`` returns the chain unchanged.
    """
    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])

    if edit_type == "remove_step" and isinstance(target, int):
        steps[:] = [s for s in steps if s.get("number") != target]
        for s in steps:
            s["dependencies"] = [d for d in s.get("dependencies", []) if d != target]
        return _renumber(chain)

    if edit_type == "insert_step" and isinstance(new_step, dict):
        new_step.setdefault("step_type", "llm")
        if new_step["step_type"] not in CARL_VALID_STEP_TYPES and new_step["step_type"] not in {"eval", "aggregator"}:
            new_step["step_type"] = "llm"
        new_step.setdefault("title", "Inserted refinement")
        new_step.setdefault("aim", "Refine intermediate output")
        new_step.setdefault("dependencies", [])
        new_step["number"] = max((s["number"] for s in steps), default=0) + 1000
        # Insert right after target, or at the end if target absent.
        if isinstance(target, int):
            for i, s in enumerate(steps):
                if s.get("number") == target:
                    steps.insert(i + 1, new_step)
                    break
            else:
                steps.append(new_step)
        else:
            steps.append(new_step)
        return _renumber(chain)

    if edit_type == "rewrite_step" and isinstance(target, int) and isinstance(new_step, dict):
        for s in steps:
            if s.get("number") == target:
                # Preserve number + dependencies unless explicitly rewritten.
                preserved = {"number": s["number"], "dependencies": s.get("dependencies", [])}
                s.clear()
                s.update(new_step)
                s.update(preserved)
                break
        return _renumber(chain)

    if edit_type == "change_step_type" and isinstance(target, int):
        for s in steps:
            if s.get("number") == target and isinstance(new_step, dict):
                s["step_type"] = new_step.get("step_type", s.get("step_type", "llm"))
                # Carry over any type-specific fields the LLM provided.
                for k, v in new_step.items():
                    if k not in {"number", "dependencies"}:
                        s[k] = v
                break
        return _renumber(chain)

    if edit_type == "swap_steps" and isinstance(target, int):
        for i, s in enumerate(steps):
            if s.get("number") == target and i + 1 < len(steps):
                steps[i], steps[i + 1] = steps[i + 1], steps[i]
                break
        return _renumber(chain)

    # Unknown edit_type — return unchanged.
    return chain


async def reflective_mutate(
    chain: dict[str, Any],
    score: ChainScore,
    query: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """Reflective (GEPA-style) mutation: diagnose the systematic weakness, then edit.

    Unlike :func:`llm_repair`, the LLM first produces an explicit ``diagnosis`` of
    the root weakness behind the failures and proposes the edit as a "gradient"
    from it. Returns the edited (renumbered) chain, or the chain unchanged when
    there's nothing concrete to fix or the call fails.
    """
    if not score.per_task_failures:
        return chain

    failure_list = "\n".join(f"- {tid}: {reason}" for tid, reason in list(score.per_task_failures.items())[:6])
    user_prompt = (
        f"User query: {query}\n\n"
        f"Current chain JSON:\n{json.dumps(chain, ensure_ascii=False, indent=2)}\n\n"
        f"Score: overall={score.overall}\n"
        f"Per-task failures:\n{failure_list}\n\n"
        "Diagnose the systematic weakness, then propose ONE edit."
    )

    try:
        data = await llm.generate_json(
            system_prompt=REFLECTIVE_MUTATION_PROMPT,
            user_prompt=user_prompt,
            stage="evolve_reflective",
        )
    except Exception as exc:
        logger.debug("reflective_mutate JSON failure: %s", exc)
        return chain

    if data.get("diagnosis"):
        logger.debug("reflective_mutate diagnosis: %s", data.get("diagnosis"))
    return _apply_edit(
        chain,
        data.get("edit_type"),
        data.get("target_step_number"),
        data.get("new_step"),
    )
