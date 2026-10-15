"""ChainEditPlanner — turn a natural-language instruction into a ChainEditPlan.

Given an existing CARL chain and a user instruction ("make step 2 cheaper and
add a validation step"), the planner asks an LLM for a minimal, targeted edit
plan: a list of :class:`~mmar_mage.schemas.ChainEdit` ops plus a one-line
summary. It mirrors the LLM-repair operator
(:func:`mmar_mage.evolve.operators.llm_repair`) but is driven by an instruction
rather than benchmark failures.

The actual mutation lives in :mod:`mmar_mage.chain_edit`; this agent only
decides *what* to change.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..llm import LLMClient
from ..prompts import CHAIN_EDIT_PROMPT
from ..schemas import ChainEdit, ChainEditPlan

logger = logging.getLogger(__name__)

# Ops that legitimately carry no (or a not-yet-existing) target step.
_TARGETLESS_OPS = frozenset({"insert_step", "set_chain_field"})


class ChainEditPlanner:
    """Plan a targeted edit to an existing chain from an NL instruction."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def plan(
        self,
        chain: dict[str, Any],
        instruction: str,
        *,
        domain: str | None = None,
    ) -> ChainEditPlan:
        """Return a :class:`ChainEditPlan` for ``instruction`` against ``chain``.

        Best-effort: on an LLM/JSON failure returns an empty plan (no-op) so the
        caller can surface "nothing changed" rather than crash. Edits whose
        ``target_step_number`` does not exist in ``chain`` are dropped (except
        the target-less ops ``insert_step`` / ``set_chain_field``).
        """
        valid_numbers = {
            s.get("number") for s in chain.get("steps", []) if isinstance(s, dict)
        }

        parts = [f"Instruction: {instruction}", ""]
        if domain:
            parts += [f"Domain: {domain}", ""]
        parts += [
            f"Current chain JSON:\n{json.dumps(chain, ensure_ascii=False, indent=2)}",
            "",
            "Produce the minimal edit plan that satisfies the instruction.",
        ]
        user_prompt = "\n".join(parts)

        try:
            data = await self._llm.generate_json(
                system_prompt=CHAIN_EDIT_PROMPT,
                user_prompt=user_prompt,
                stage="chain_edit",
            )
        except Exception as exc:  # noqa: BLE001 — best-effort; never crash the edit
            logger.warning("ChainEditPlanner: planning failed (%s); empty plan", exc)
            return ChainEditPlan()

        if not isinstance(data, dict):
            return ChainEditPlan()

        raw_edits = data.get("edits")
        if not isinstance(raw_edits, list):
            raw_edits = []

        edits: list[ChainEdit] = []
        for item in raw_edits:
            if not isinstance(item, dict):
                continue
            try:
                edit = ChainEdit.model_validate(item)
            except Exception as exc:  # noqa: BLE001 — drop malformed edits
                logger.debug("ChainEditPlanner: dropping invalid edit %r (%s)", item, exc)
                continue
            if (
                edit.target_step_number is not None
                and edit.op not in _TARGETLESS_OPS
                and edit.target_step_number not in valid_numbers
            ):
                logger.debug(
                    "ChainEditPlanner: dropping %s for missing step %s",
                    edit.op,
                    edit.target_step_number,
                )
                continue
            edits.append(edit)

        return ChainEditPlan(summary=str(data.get("summary") or ""), edits=edits)


__all__ = ["ChainEditPlanner"]
