"""Validate→repair retry for generated chains (TODO G1.4).

Open models on OpenRouter lack native constrained decoding, so a generated chain
can be schema-invalid (a missing title, a dependency to a non-existent step, a
duplicate number, a cycle, …). Rather than fail the whole generation, feed the
exact validation error back to the LLM for ONE corrective pass and re-validate.
If it's still invalid afterwards, raise a CLEAR error — never a silent pass.

Disproportionately helps weaker open models (deepseek/qwen-class) that can't be
held to a schema by the API alone.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .carl_export import to_carl_compat
from .exceptions import MAGEValidationError

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)

STAGE = "chain_repair"

_REPAIR_SYSTEM = (
    "You repair an INVALID CARL reasoning-chain JSON. You are given the chain "
    "and the exact validation error. Return ONLY the corrected chain as a single "
    "JSON object with the SAME intent — fix exactly what the error names (e.g. a "
    "missing title, a bad dependency, a duplicate step number, an invalid "
    "step_type / step_config field) and change nothing else. Keep every valid "
    "step. Output JSON only, no prose, no code fences."
)


def _validate_compat(compat: dict[str, Any]) -> None:
    """Strict CARL validation on an already-compat dict. Raises
    MAGEValidationError on failure. Lazy import avoids a generator↔chain_repair
    import cycle."""
    from .generator import validate_carl_json

    validate_carl_json(compat, strict_carl=True)


# A url-shaped field must hold a real URL or a $-reference — never the raw query.
_URL_FIELD_RE = re.compile(r"(?:^|_)(url|uri|endpoint)$", re.IGNORECASE)
_URLISH_RE = re.compile(r"^https?://", re.IGNORECASE)

# A `transform` substitutes WHOLE $history[N]/$outer_context values into a template —
# it can't extract, slice, or call a model. These two patterns mean the planner wrote
# LLM-style extraction logic a transform silently ships as garbage (→ structured_output).
_TRANSFORM_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)  # "{{extracted from …}}"
_HISTORY_SLICE_RE = re.compile(r"\$history\[-?\d+\]\s*\[")  # "$history[0][:10]"

# A $-reference resolves ONLY as the WHOLE value — spliced into a longer string it is
# silently dropped (history matches the prefix and returns that step, losing the rest)
# or left literal. Flag any value that embeds a ref but isn't exactly one.
_REF_PARTS = r"history\[-?\d+\]|outer_context|memory\.[\w.]+|ltm\.[\w.]+|steps\.[\w.]+|event\.[\w.]+"
_REF_TOKEN_RE = re.compile(r"\$(?:" + _REF_PARTS + r")")
_WHOLE_REF_RE = re.compile(r"^\s*\$(?:" + _REF_PARTS + r")\s*$")


def _tool_fields(step: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read a tool step's name + input mapping from either the flat MAGE form or
    the CARL-compat nested ``step_config`` form."""
    raw_cfg = step.get("step_config")
    cfg: dict[str, Any] = raw_cfg if isinstance(raw_cfg, dict) else {}
    name = step.get("tool_name") or cfg.get("tool_name") or ""
    mapping = step.get("tool_input_mapping")
    if not isinstance(mapping, dict):
        raw_map = cfg.get("input_mapping")
        mapping = raw_map if isinstance(raw_map, dict) else {}
    return str(name or ""), mapping


_PREDETERMINE_AIM_MARKERS: tuple[str, ...] = (
    "since the user is in",
    "since you are in",
    "assuming the user is in",
    "assuming you are in",
    "the user is located in",
    "you are located in",
    "the user is based in",
    "user's country is",
    "user's city is",
    "user's location is",
    "your country is",
    "your city is",
    "your location is",
)


def _lint_transform_step(step: dict[str, Any], num: Any) -> list[str]:
    """Flag a `transform` whose expression fakes LLM-style extraction.

    A transform only substitutes WHOLE ``$history[N]`` / ``$outer_context`` values into a
    template; it cannot extract a field, slice a string, or call a model. ``{{…}}``
    placeholders and ``$history[i][slice]`` mean the planner wrote logic a transform
    can't run — re-route it to a ``structured_output`` step (an LLM fills the schema)."""
    raw_cfg = step.get("step_config")
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    expr = str(step.get("transform_expression") or cfg.get("expression") or "")
    if expr and (_TRANSFORM_PLACEHOLDER_RE.search(expr) or _HISTORY_SLICE_RE.search(expr)):
        return [
            f"step {num}: transform expression uses non-executable syntax "
            '("{{…}}" placeholders or "$history[i][slice]") — a transform only '
            "substitutes WHOLE $history[N]/$outer_context values; it cannot extract, "
            'slice, or call a model. Use a "structured_output" step (an LLM fills the '
            "schema from the source) to derive these fields."
        ]
    return []


def lint_chain(chain_dict: dict[str, Any], *, known_tools: set[str] | None = None) -> list[str]:
    """Deterministic pre-export lint of tool steps (T5) — no LLM.

    Catches the failure class behind the 09.06 incident before a chain ships: a
    tool step with an empty ``tool_name``, a url-shaped field holding
    ``$outer_context`` / free text (→ ``GET https://<the question text>``) or
    left empty, or (when ``known_tools`` is given) a reference to an unknown
    tool. Reads both the flat MAGE form and the CARL-compat nested form, so it
    works before or after ``to_carl_compat``. Returns one message per violation;
    an empty list means clean.
    """
    issues: list[str] = []
    known = {t.lower() for t in known_tools} if known_tools else None
    for step in chain_dict.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        num = step.get("number", "?")
        stype = step.get("step_type")
        if stype == "transform":
            issues.extend(_lint_transform_step(step, num))
            continue
        if stype != "tool":
            continue
        name, mapping = _tool_fields(step)
        if not name.strip():
            issues.append(f"step {num}: tool step has an empty tool_name")
        elif known is not None and name.lower() not in known:
            valid = ", ".join(sorted(known_tools)) if known_tools else ""
            issues.append(
                f"step {num}: tool_name '{name}' is not a registered tool"
                + (
                    f" — rebind this step to ONE of the registered tools ({valid}) "
                    "and fix its input_mapping to match that tool's parameters"
                    if valid
                    else ""
                )
            )
        # T18 — a live-fetch step's aim must not presume the personal fact it should
        # discover (run-096: aim hardcoded "user in Russia"; geolocation gave Sweden).
        aim = str(step.get("aim") or step.get("rough_aim") or "").lower()
        for marker in _PREDETERMINE_AIM_MARKERS:
            if marker in aim:
                issues.append(
                    f"step {num}: aim presumes a personal fact ('{marker} …') the step should "
                    "fetch live — let the step DETERMINE it and reference it via $history, "
                    "don't bake a remembered location into a fetch step's aim"
                )
                break
        for field, value in mapping.items():
            sval = value if isinstance(value, str) else ""
            is_url_field = bool(_URL_FIELD_RE.search(str(field)))
            # Whole-value rule (free-text fields): a $-ref must BE the whole value,
            # never spliced into a longer string (e.g. "$history[0] погода сейчас" →
            # CARL returns just history[0], dropping " погода сейчас"). url-shaped
            # fields are judged by the shape check below instead — a URL the planner
            # composed ("…/$history[1]") is url-field business, not a splice error.
            if (
                not is_url_field
                and _REF_TOKEN_RE.search(sval)
                and not _WHOLE_REF_RE.match(sval)
            ):
                issues.append(
                    f"step {num}: input '{field}'={value!r} splices a $-reference into a "
                    "longer string — $history[N]/$outer_context resolve ONLY as the WHOLE "
                    "value (the rest is dropped or left literal). Map the whole reference, "
                    'or build the combined string in a `transform` "format" step first and '
                    "map this field to that step's output."
                )
            if not is_url_field:
                continue
            if not sval.strip():
                issues.append(f"step {num}: url-field '{field}' is empty")
            elif sval == "$outer_context":
                issues.append(
                    f"step {num}: url-field '{field}' is $outer_context (the raw user query) — "
                    "use a literal https?:// URL or a $history[N] reference"
                )
            elif not sval.startswith("$") and not _URLISH_RE.match(sval):
                issues.append(
                    f"step {num}: url-field '{field}'={value!r} is free text, not a URL — "
                    "use a literal https?:// URL or a $history[N] reference"
                )
    return issues


async def repair_chain(
    chain_dict: dict[str, Any], *, query: str, error: str, llm: LLMClient
) -> dict[str, Any]:
    """One LLM pass that returns a corrected chain, given the chain + the error."""
    user_prompt = (
        f"User task: {query}\n\n"
        f"Validation error:\n{error}\n\n"
        f"Invalid chain JSON:\n{json.dumps(chain_dict, ensure_ascii=False)}\n\n"
        "Return the corrected chain JSON."
    )
    return await llm.generate_json(
        system_prompt=_REPAIR_SYSTEM, user_prompt=user_prompt, stage=STAGE
    )


async def validate_or_repair(
    chain_dict: dict[str, Any], *, query: str, llm: LLMClient
) -> dict[str, Any]:
    """Return a strict-valid, CARL-compat chain — repairing once if needed.

    Validates the CARL-compat form; on failure does ONE LLM repair pass and
    re-validates. Raises :class:`MAGEValidationError` if the chain is still
    invalid after the repair (or if the repair call itself fails) — a clear
    error, never a silent pass. Returns the compat dict ready to serialize.
    """
    compat = to_carl_compat(chain_dict)
    lint_issues = lint_chain(chain_dict)
    try:
        _validate_compat(compat)
        if not lint_issues:
            return compat
        # Schema-valid but the deterministic tool-step lint failed (T5) — route
        # through the same one-shot repair pass instead of shipping the chain.
        raise MAGEValidationError("Tool-step lint failures: " + "; ".join(lint_issues))
    except MAGEValidationError as first:
        logger.info("chain failed validation/lint (%s) — one repair pass", first)
        try:
            repaired = to_carl_compat(
                await repair_chain(compat, query=query, error=str(first), llm=llm)
            )
        except Exception as exc:  # noqa: BLE001 — repair call failed → surface the clear original error
            logger.warning("chain repair call failed (%s); giving up", exc)
            raise first from exc
        _validate_compat(repaired)  # raises clearly if STILL schema-invalid
        remaining = lint_chain(repaired)
        if remaining:
            raise MAGEValidationError(
                "Tool-step lint still failing after repair: " + "; ".join(remaining)
            ) from None
        logger.info("chain repaired + re-validated OK")
        return repaired
