"""Apply a natural-language-planned edit to an existing CARL chain.

:class:`mmar_mage.agents.chain_editor.ChainEditPlanner` turns a user's
instruction into a :class:`~mmar_mage.schemas.ChainEditPlan` — an ordered list
of :class:`~mmar_mage.schemas.ChainEdit` ops. This module *applies* that plan.

Five of the eight ops (``remove_step`` / ``insert_step`` / ``rewrite_step`` /
``change_step_type`` / ``swap_steps``) reuse the evolutionary edit primitive
:func:`mmar_mage.evolve.operators._apply_edit`, so there is a single source of
truth for structural mutation + renumbering. Three lighter ops are added here
for natural-language editing:

* ``edit_field`` — patch specific keys on a step (``fields``) without
  re-emitting the whole step.
* ``set_dependencies`` — replace a step's ``dependencies`` (``value``: list[int]).
* ``set_chain_field`` — set a chain-level field (``name`` / ``max_workers`` /
  ``timeout``) via ``fields``.

All functions are pure: they deep-copy and never mutate their input.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from .schemas import ChainEdit

logger = logging.getLogger(__name__)

# Ops delegated wholesale to the evolve primitive (it deep-copies + renumbers).
_STRUCTURAL_OPS = frozenset(
    {"remove_step", "insert_step", "rewrite_step", "change_step_type", "swap_steps"}
)

# Chain-level fields a ``set_chain_field`` op may touch. Anything else is
# ignored so the planner can't rewrite the step list or search config wholesale.
_CHAIN_FIELDS = frozenset({"name", "max_workers", "timeout"})


def _merge_field(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """Apply an ``edit_field`` patch in place, DEEP-merging nested dicts.

    A flat ``target[k] = v`` would let a patch to a sub-key (e.g.
    ``step_config.input_mapping.query``) replace the WHOLE ``step_config`` and
    silently drop siblings like ``tool_name`` — which makes a ``tool`` step
    fail ``ToolStepConfig`` validation. Deep-merging preserves the siblings.
    Scalars / lists still replace. ``number`` is owned by the renumberer.
    """
    for k, v in patch.items():
        if k == "number":
            continue
        existing = target.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            _merge_field(existing, v)
        else:
            target[k] = v


def apply_edit(chain: dict[str, Any], edit: ChainEdit) -> dict[str, Any]:
    """Apply ONE :class:`ChainEdit` to ``chain``; return a new (renumbered) chain.

    Unknown or inapplicable ops return the chain unchanged (logged at debug).
    """
    # Lazy import: ``evolve.operators`` lives under the ``evolve`` package whose
    # __init__ pulls ``engine`` → ``generator``, and ``generator`` imports this
    # module — importing it at module load would form a cycle. Deferring to call
    # time (everything initialised by then) mirrors ``chain_repair``'s lazy
    # ``from .generator import validate_carl_json``.
    from .evolve.operators import _apply_edit, _renumber

    op = edit.op

    if op in _STRUCTURAL_OPS:
        # Reuse the evolve primitive — it deep-copies + renumbers + remaps deps.
        return _apply_edit(chain, op, edit.target_step_number, edit.new_step)

    chain = copy.deepcopy(chain)
    steps = chain.get("steps", [])

    if op == "edit_field":
        patch = edit.fields or {}
        if isinstance(edit.target_step_number, int) and isinstance(patch, dict):
            for s in steps:
                if s.get("number") == edit.target_step_number:
                    # Deep-merge so a patch to step_config.input_mapping keeps
                    # sibling keys (tool_name, etc.) instead of replacing the
                    # whole step_config.
                    _merge_field(s, patch)
                    break
        return _renumber(chain)

    if op == "set_dependencies":
        deps = edit.value
        if deps is None and isinstance(edit.fields, dict):
            deps = edit.fields.get("dependencies")
        if isinstance(edit.target_step_number, int) and isinstance(deps, list):
            clean = [d for d in deps if isinstance(d, int)]
            for s in steps:
                if s.get("number") == edit.target_step_number:
                    s["dependencies"] = clean
                    break
        return _renumber(chain)

    if op == "set_chain_field":
        patch = edit.fields or {}
        if isinstance(patch, dict):
            for k, v in patch.items():
                if k in _CHAIN_FIELDS:
                    chain[k] = v
        return chain  # chain-level only; numbering untouched

    logger.debug("apply_edit: unknown op %r — chain unchanged", op)
    return chain


def apply_edit_plan(chain: dict[str, Any], edits: list[ChainEdit]) -> dict[str, Any]:
    """Fold a list of edits over ``chain`` and return a new chain.

    Non-structural edits (``edit_field`` / ``set_dependencies`` /
    ``set_chain_field``) are applied first: they reference the original
    numbering and never shift it, so their ``target_step_number`` stays valid.
    Structural edits (which may renumber) run afterwards. This keeps the common
    case — any number of field edits plus a single structural change, each
    targeting the numbering the planner was shown — exactly correct.

    A plan with multiple *structural* edits is applied sequentially (each sees
    the result of the previous); the planner is instructed to prefer one
    structural change per request, so this is rare.
    """
    result = copy.deepcopy(chain)
    nonstructural = [e for e in edits if e.op not in _STRUCTURAL_OPS]
    structural = [e for e in edits if e.op in _STRUCTURAL_OPS]
    for edit in nonstructural:
        result = apply_edit(result, edit)
    for edit in structural:
        result = apply_edit(result, edit)
    return result


__all__ = ["apply_edit", "apply_edit_plan"]
