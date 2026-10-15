"""Single source of truth for the CARL step types the planner may emit (P-1).

The registry is kept in sync with :data:`mmar_mage.schemas.CARL_VALID_STEP_TYPES`
(asserted by ``tests/test_prompt_loader.py``). The MAGE-internal ``eval`` /
``aggregator`` labels are intentionally NOT here — they're translated to CARL in
``carl_export.to_carl_compat`` for backward compatibility but must never be
offered to the planner (offering two names for the same thing, one deprecated,
was a direct source of confusion + validation errors).
"""

from __future__ import annotations

# {step_type: one-line "when to use" description}.
STEP_TYPE_DESCRIPTIONS: dict[str, str] = {
    "llm": "reason / write / summarise with the model",
    "tool": "call a registered tool (web_search, rest_call, read_file, run_python, …)",
    "mcp": "call a tool on a connected MCP server (an external service)",
    "mcp_resource": "read a resource exposed by an MCP server",
    "memory": "recall from long-term memory",
    "transform": "deterministic data reshape (split / map / filter / format)",
    "conditional": "branch on a condition (if / else)",
    "structured_output": "emit JSON validated against a schema",
    "agent_skill": "run a packaged AgentSkill (pdf / docx / pptx / xlsx / webdev / …)",
    "evaluation": "check or grade an answer against criteria / gathered evidence",
    "parallel_sampling": "fan out N samples of the same step, then reduce",
    "agent_handoff": "delegate a sub-task to another chain / agent",
    "supervisor": "route between specialist sub-agents",
    "debate": "opposing agents argue, a judge decides",
    "human_input": "PAUSE and ASK the user for data / a choice / approval",
    "tool_discovery": "discover or request a tool not yet available",
}


def step_types_for(allowed: list[str] | None = None) -> list[str]:
    """The registry step types, optionally filtered to ``allowed``.

    ``allowed`` entries not in the registry (e.g. MAGE-internal eval/aggregator,
    or typos) are ignored. An empty / all-invalid ``allowed`` falls back to the
    full menu so the planner is never left with zero options.
    """
    if not allowed:
        return list(STEP_TYPE_DESCRIPTIONS)
    filtered = [t for t in STEP_TYPE_DESCRIPTIONS if t in allowed]
    return filtered or list(STEP_TYPE_DESCRIPTIONS)


def render_step_type_menu(allowed: list[str] | None = None) -> str:
    """Render the (optionally filtered) step-type menu as ``- `name` — desc`` lines."""
    return "\n".join(f"- `{t}` — {STEP_TYPE_DESCRIPTIONS[t]}" for t in step_types_for(allowed))


__all__ = ["STEP_TYPE_DESCRIPTIONS", "step_types_for", "render_step_type_menu"]
