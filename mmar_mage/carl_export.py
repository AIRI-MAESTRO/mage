"""CARL-compatibility exporter.

MAGE generates chains using a flat schema (``CARLStepSchema``) with extra
MAGE-internal step types (``eval``, ``aggregator``).  The ``mmar_carl``
library (``agent-features`` branch) expects:

* ``step_type`` from ``CARL_VALID_STEP_TYPES``
* per-type config nested under ``step_config`` (not flat top-level fields),
  with field names matching the ``*StepConfig`` Pydantic models in
  ``mmar_carl.models.config`` / ``mmar_carl.models.agent_skill``.

This module bridges the two: :func:`to_carl_compat` returns a deep copy of
the chain dict that:

1. translates step types CARL cannot load from JSON to ``llm`` with a
   concrete ``aim``:

   * ``eval`` / ``aggregator`` — MAGE-internal labels, no CARL equivalent;
   * ``agent_handoff`` / ``supervisor`` / ``parallel_sampling`` — these
     require a runtime-only object (``sub_chain`` / ``agents`` /
     ``base_step``, all ``Field(..., exclude=True)``) that a serialized
     chain cannot carry, so ``ReasoningChain.from_json`` cannot reconstruct
     them.  Translating to ``llm`` keeps the chain loadable and approximates
     the intent.  This is the interim policy of plan item P0.3 (option A);
     once CARL gains JSON sub-chain loading (P0.3 option B) these can become
     first-class steps.

2. moves flat MAGE fields into nested ``step_config`` using
   :data:`CARL_FIELD_MAP` (one entry per loadable step type), with dedicated
   handlers for the types whose CARL shape differs from MAGE's flat form
   (``mcp``, ``mcp_resource``, ``debate``, ``tool_discovery``);

3. strips MAGE-internal scratch fields, known orphan flat fields, and
   ``None`` values.

The MAGE *flat* field names (left-hand side of every mapping) are part of
MAGE's own contract — they are declared on ``CARLStepSchema``
(``schemas.py``), instructed in ``STEP_DESCRIBER_PROMPT`` (``prompts.py``),
and read by ``cost.py``.  Only the CARL *nested* names (right-hand side) are
the target of this module, so they are the only thing that changes when the
CARL schema moves.

The result is loadable by ``ReasoningChain.from_json``.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any

from .schemas import CARL_VALID_STEP_TYPES

# Conservative set of step types that *every* released CARL can load from
# JSON. Used as the fallback when the installed ``mmar_carl`` cannot be
# introspected (or is absent).
_BASE_CARL_LOADABLE_STEP_TYPES: frozenset[str] = frozenset(
    {"llm", "tool", "mcp", "memory", "transform", "conditional", "structured_output"}
)


@lru_cache(maxsize=1)
def carl_loadable_step_types() -> frozenset[str]:
    """Step types the *installed* ``mmar_carl`` can load from JSON.

    MAGE's internal ``CARL_VALID_STEP_TYPES`` is forward-looking (it lists
    every agent-features type MAGE can author). The actually-installed CARL
    may be older and load far fewer types — e.g. released ``mmar-carl`` only
    accepts the seven base types. ``to_carl_compat`` keeps emitting MAGE's
    canonical nested form for every type; this helper lets callers and tests
    detect which of those the *installed* CARL will actually accept at
    ``from_json`` time (agent-features types only load on a CARL that supports
    them).

    The set is read once from the installed ``mmar_carl.StepDescription``
    ``step_type`` enum and cached. Falls back to
    :data:`_BASE_CARL_LOADABLE_STEP_TYPES` if ``mmar_carl`` is missing or its
    shape can't be introspected.
    """
    try:
        import mmar_carl

        step_type_enum = mmar_carl.StepDescription.model_fields["step_type"].annotation
        values = {getattr(m, "value", m) for m in step_type_enum}
        found = frozenset(v for v in values if isinstance(v, str))
        if found:
            return found
    except Exception:
        pass
    return _BASE_CARL_LOADABLE_STEP_TYPES


@lru_cache(maxsize=1)
def carl_mcp_transports() -> frozenset[str]:
    """MCP server transports the installed ``mmar_carl`` accepts.

    Introspected from ``mmar_carl.MCPServerConfig.transport``. Falls back to
    the conservative ``{stdio, http, websocket}`` set if it can't be read.
    """
    try:
        import typing

        import mmar_carl

        ann = mmar_carl.MCPServerConfig.model_fields["transport"].annotation
        args = typing.get_args(ann)
        found = frozenset(a for a in args if isinstance(a, str))
        if found:
            return found
        # enum fallback
        found = frozenset(getattr(m, "value", m) for m in ann if isinstance(getattr(m, "value", m), str))
        if found:
            return found
    except Exception:
        pass
    return frozenset({"stdio", "http", "websocket"})


def _normalize_mcp_transport(server: dict[str, Any]) -> dict[str, Any]:
    """Map an MCP transport the installed CARL can't load to a supported one.

    ``sse`` (Server-Sent Events) is an HTTP streaming transport, so it degrades
    to ``http`` when the installed CARL lacks an explicit ``sse`` value. Keeps
    the server config loadable by ``MCPServerConfig`` (P0.3 — "work with CARE").
    """
    transport = server.get("transport")
    if not transport:
        return server
    allowed = carl_mcp_transports()
    if transport not in allowed:
        server["transport"] = "http" if "http" in allowed else next(iter(sorted(allowed)), "stdio")
    return server


# ---------------------------------------------------------------------------
# Flat (MAGE) -> nested (CARL) field maps, one per *loadable* step type.
#
# Verified against mmar_carl @ agent-features:
#   - stable types        : models/config.py  (Tool/Memory/Transform/...)
#   - agent_skill         : models/agent_skill.py::AgentSkillStepConfig
#   - evaluation          : models/config.py::EvaluationStepConfig
#   - human_input         : models/config.py::HumanInputStepConfig
#
# Types with a CARL shape that differs from MAGE's flat representation
# (mcp, mcp_resource, debate, tool_discovery) are handled by dedicated
# helpers below, NOT this table.
#
# Types translated to ``llm`` during export (agent_handoff / supervisor /
# parallel_sampling) keep no entry here — their config is unused once the
# step is translated.  See module docstring (P0.3).
# ---------------------------------------------------------------------------
CARL_FIELD_MAP: dict[str, dict[str, str]] = {
    # --- stable types (unchanged; verified against config.py) ---
    "tool": {
        "tool_name": "tool_name",
        "tool_description": "tool_description",
        "tool_input_mapping": "input_mapping",
        "tool_output_key": "output_key",
        "tool_timeout": "timeout",
    },
    "memory": {
        "memory_operation": "operation",
        "memory_key": "memory_key",
        "memory_value_source": "value_source",
        "memory_namespace": "namespace",
    },
    "transform": {
        "transform_type": "transform_type",
        "transform_input_key": "input_key",
        "transform_output_format": "output_format",
        "transform_expression": "expression",
    },
    "structured_output": {
        "structured_output_schema": "output_schema",
        "structured_output_schema_name": "schema_name",
        "structured_output_input_source": "input_source",
        "structured_output_instruction": "instruction",
        "structured_output_strict_json": "strict_json",
    },
    "conditional": {
        "condition_branches": "branches",
        "condition_default_step": "default_step",
        "condition_context_key": "condition_context_key",
    },
    # --- agent-features: AgentSkillStepConfig (already correct) ---
    "agent_skill": {
        "skill": "skill",
        "skill_task": "task",
        "skill_execution_mode": "execution_mode",
        "skill_input_mapping": "input_mapping",
        "skill_output_key": "output_key",
        "skill_output_memory_key": "output_memory_key",
        "skill_output_memory_namespace": "output_memory_namespace",
        "skill_system_prompt_prefix": "system_prompt_prefix",
        "skill_script_name": "script_name",
        "skill_script_args": "script_args",
        "skill_timeout": "timeout",
        "skill_llm_max_iterations": "llm_max_iterations",
        "skill_runtime": "runtime",
        "skill_runtime_config": "runtime_config",
        "skill_output_schema": "output_schema",
        "skill_trust_policy": "trust_policy",
        "skill_sha256": "skill_sha256",
        "skill_extra_pip": "extra_pip",
    },
    # --- agent-features: EvaluationStepConfig (FIXED) ---
    # CARL: evaluates_step / criteria / on_fail / evaluation_method /
    #       input_source / max_retries.  ``evaluation_threshold`` and the
    #       "python" method value have no CARL home — see _strip + P2.1.
    "evaluation": {
        "evaluation_target_step": "evaluates_step",
        "evaluation_criteria": "criteria",
        "evaluation_fail_action": "on_fail",
        "evaluation_method": "evaluation_method",
    },
    # --- agent-features: HumanInputStepConfig (FIXED: default -> fallback_value) ---
    "human_input": {
        "human_prompt": "prompt",
        "human_input_timeout": "timeout",
        "human_default": "fallback_value",
    },
}

# MCP server sub-object field map — shared by ``mcp`` and ``mcp_resource``.
# ``headers`` (P2.2) carries auth etc. for the http/sse transports.
_MCP_SERVER_FIELDS = {
    "mcp_server_name": "server_name",
    "mcp_server_transport": "transport",
    "mcp_server_command": "command",
    "mcp_server_args": "args",
    "mcp_server_url": "url",
    "mcp_server_headers": "headers",
}

# Step types whose results are side-effect-free and therefore safe to auto-cache
# (plan item P2.2). MCP resources are read-only fetches by definition.
_CACHEABLE_STEP_TYPES = ("mcp_resource",)


def mark_cacheable_steps(chain_dict: dict[str, Any], default_ttl: int = 300) -> dict[str, Any]:
    """Set ``step_cache_ttl`` on read-only steps that don't already have it (P2.2).

    Conservative — only auto-marks :data:`_CACHEABLE_STEP_TYPES` (MCP resource
    reads), never side-effecting ``tool``/``mcp`` steps (those opt in explicitly,
    guided by the describer prompt). Mutates and returns ``chain_dict``.
    """
    for step in chain_dict.get("steps", []) or []:
        if step.get("step_type") in _CACHEABLE_STEP_TYPES and step.get("step_cache_ttl") is None:
            step["step_cache_ttl"] = default_ttl
    return chain_dict


def _apply_cache(step: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat ``step_cache_ttl`` into CARL's top-level ``cache: {ttl}`` (P2.2).

    ``StepCache.key_fn`` is a runtime-only callable, so only ``ttl`` is serialized.
    """
    ttl = step.get("step_cache_ttl")
    if ttl is None:
        return step
    step = dict(step)
    step["cache"] = {"ttl": int(ttl)}
    step.pop("step_cache_ttl", None)
    return step


# Flat fields that have no CARL home and must not leak into the chain JSON.
_DROPPED_FLAT_FIELDS = frozenset(
    {
        "evaluation_threshold",  # no field on EvaluationStepConfig
        "evaluation_rule",  # no field on EvaluationStepConfig
        "sampling_temperature",  # belongs in base_step.llm_config (P0.3 / P2.1)
    }
)

# Fields that are MAGE-internal and should not appear in CARL JSON.
_MAGE_INTERNAL_FIELDS = frozenset(
    {
        "eval_target_steps",
        "eval_criteria",
        "eval_on_fail",
        "aggregate_from_steps",
        "aggregation_strategy",
    }
)

# Per-type *essential* MAGE-flat fields (plan item P2.1). A typed step missing
# these is semantically empty — it would export to a no-op (e.g. an ``evaluation``
# with no target, a ``parallel_sampling`` with no sample count). Field names are
# the MAGE flat names consumed by ``to_carl_compat`` above (NOT the CARL nested
# names). Kept conservative — only the field(s) without which the step is
# meaningless — to avoid false positives.
REQUIRED_STEP_CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "tool": ("tool_name",),
    "mcp": ("mcp_server_name",),
    "mcp_resource": ("mcp_server_name", "mcp_resource_uri"),
    "agent_skill": ("skill",),
    "evaluation": ("evaluation_target_step", "evaluation_criteria"),
    "parallel_sampling": ("sampling_n",),
    "supervisor": ("supervisor_routes",),
    "debate": ("debate_agents",),
    "agent_handoff": ("handoff_chain_path",),
    "human_input": ("human_prompt",),
    "tool_discovery": ("discovery_sources",),
}


def _is_blank(value: Any) -> bool:
    """True for ``None``, empty string/whitespace, or empty collection."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


# Declared metric types MAGE understands (plan item P3.4), mirroring CARL's
# concrete MetricBase subclasses (ContainsMetric / ExactMatchMetric / …) plus a
# generic numeric-threshold kind for analytics. Used to validate ``step_metrics``.
VALID_METRIC_TYPES: frozenset[str] = frozenset(
    {"contains", "exact_match", "case_insensitive_match", "regex_match", "numeric", "llm_judge"}
)


def validate_step_metrics(chain_dict: dict[str, Any]) -> dict[int, list[str]]:
    """Report malformed declared ``step_metrics`` per step (plan item P3.4).

    A metric spec must be a dict with a non-empty ``name`` and a ``type`` in
    :data:`VALID_METRIC_TYPES`; ``threshold`` (if present) must be a number.

    Returns
    -------
    dict[int, list[str]]
        ``{step_number: [problem descriptions]}`` — empty when all specs are valid.
    """
    out: dict[int, list[str]] = {}
    for step in chain_dict.get("steps", []):
        if not isinstance(step, dict):
            continue
        metrics = step.get("step_metrics")
        if not metrics:
            continue
        problems: list[str] = []
        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                problems.append(f"metric {i} is not an object")
                continue
            if not str(m.get("name", "")).strip():
                problems.append(f"metric {i} missing 'name'")
            mtype = m.get("type")
            if mtype not in VALID_METRIC_TYPES:
                problems.append(f"metric {i} has invalid type {mtype!r}")
            thr = m.get("threshold")
            if thr is not None and not isinstance(thr, (int, float)):
                problems.append(f"metric {i} threshold must be numeric")
        if problems:
            out[int(step.get("number", 0))] = problems
    return out


def chain_metrics_summary(chain_dict: dict[str, Any]) -> dict[str, Any]:
    """Aggregate declared ``step_metrics`` across a chain for downstream analytics (P3.4).

    Returns ``{total, by_type, per_step}`` where ``per_step`` maps step number →
    list of metric names. Empty/structured-zero when no metrics are declared.
    """
    by_type: dict[str, int] = {}
    per_step: dict[int, list[str]] = {}
    total = 0
    for step in chain_dict.get("steps", []):
        if not isinstance(step, dict):
            continue
        metrics = step.get("step_metrics") or []
        names: list[str] = []
        for m in metrics:
            if not isinstance(m, dict):
                continue
            total += 1
            mtype = str(m.get("type", "unknown"))
            by_type[mtype] = by_type.get(mtype, 0) + 1
            names.append(str(m.get("name", "")))
        if names:
            per_step[int(step.get("number", 0))] = names
    return {"total": total, "by_type": by_type, "per_step": per_step}


def missing_step_config_fields(chain_dict: dict[str, Any]) -> dict[int, list[str]]:
    """Report typed steps missing their essential config (plan item P2.1).

    Operates on the **MAGE flat** chain form (before :func:`to_carl_compat`). A
    required field counts as present when it's set either as a flat field on the
    step or inside an existing ``step_config`` dict.

    Returns
    -------
    dict[int, list[str]]
        ``{step_number: [missing flat field names]}`` — empty when every typed
        step carries its essentials.
    """
    out: dict[int, list[str]] = {}
    for step in chain_dict.get("steps", []):
        if not isinstance(step, dict):
            continue
        required = REQUIRED_STEP_CONFIG_FIELDS.get(step.get("step_type", "llm"))
        if not required:
            continue
        cfg = step.get("step_config") or {}
        missing = [f for f in required if _is_blank(step.get(f)) and _is_blank(cfg.get(f))]
        if missing:
            out[int(step.get("number", 0))] = missing
    return out


def to_carl_compat(chain_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert MAGE chain dict to a CARL-loadable chain dict.

    Parameters
    ----------
    chain_dict : dict[str, Any]
        Chain in MAGE flat format.

    Returns
    -------
    dict[str, Any]
        Chain in CARL nested format. Loadable by
        ``mmar_carl.ReasoningChain.from_json``.
    """
    out = copy.deepcopy(chain_dict)
    out["steps"] = [_convert_step(s) for s in out.get("steps", [])]
    return out


# ---------------------------------------------------------------------------
# $history reference re-indexing (step NUMBER -> history POSITION)
# ---------------------------------------------------------------------------

_HISTORY_REF_RE = re.compile(r"\$history\[(\d+)\]")


def _history_position_map(steps: list[dict[str, Any]]) -> dict[int, int]:
    """Map each step NUMBER -> its 0-indexed position in CARL's flat history.

    ``mmar_carl`` executes topologically and appends every result to one flat
    ``context.history`` list, so ``$history[N]`` resolves POSITIONALLY. We
    reproduce CARL's batch order — Kahn levels, lowest step number first within
    a level — to translate a step number into the position its result lands at.
    """
    by_num: dict[int, dict[str, Any]] = {}
    deps: dict[int, list[int]] = {}
    for s in steps:
        n = s.get("number")
        if isinstance(n, int):
            by_num[n] = s
            deps[n] = [d for d in (s.get("dependencies") or []) if isinstance(d, int)]
    order: list[int] = []
    placed: set[int] = set()
    remaining = set(by_num)
    while remaining:
        ready = sorted(n for n in remaining if all(d in placed for d in deps.get(n, [])))
        if not ready:  # cycle / dangling dependency — flush the rest by number
            ready = sorted(remaining)
        for n in ready:
            order.append(n)
            placed.add(n)
            remaining.discard(n)
    return {num: pos for pos, num in enumerate(order)}


def _rewrite_history_refs(value: Any, pos_map: dict[int, int]) -> Any:
    """Recursively rewrite positive ``$history[N]`` (step number) substrings to
    ``$history[<position>]``. Leaves ``$history[-1]`` and unknown N untouched."""
    if isinstance(value, str):
        return _HISTORY_REF_RE.sub(
            lambda m: (
                f"$history[{pos_map[int(m.group(1))]}]"
                if int(m.group(1)) in pos_map
                else m.group(0)
            ),
            value,
        )
    if isinstance(value, dict):
        return {k: _rewrite_history_refs(v, pos_map) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite_history_refs(v, pos_map) for v in value]
    return value


def reindex_history_refs(chain_dict: dict[str, Any]) -> dict[str, Any]:
    """Translate MAGE's step-NUMBER ``$history[N]`` refs into the 0-indexed
    history POSITION the CARL runtime actually resolves.

    MAGE plans references by step number ("$history[2]" = step 2's output), but
    ``mmar_carl`` indexes ``context.history`` positionally — so a reference to a
    step whose number exceeds the run's history depth raised *"list index out of
    range"* (and lower ones silently pointed one step late). This rewrites every
    POSITIVE ``$history[N]`` to its history position; ``dependencies`` (kept as
    step numbers for the DAG), ``$history[-1]``, and unknown N are untouched.
    Mutates and returns ``chain_dict``. NOT idempotent — call once on the final
    CARL-compat chain, just before serialization."""
    steps = chain_dict.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return chain_dict
    pos_map = _history_position_map(steps)
    if not pos_map:
        return chain_dict
    for s in steps:
        if not isinstance(s, dict):
            continue
        for key, val in list(s.items()):
            if key in ("number", "dependencies"):
                continue  # DAG wiring stays by step number
            s[key] = _rewrite_history_refs(val, pos_map)
    return chain_dict


def _convert_step(step: dict[str, Any]) -> dict[str, Any]:
    step_type = step.get("step_type", "llm")

    # 1. Types CARL cannot load from JSON -> translate to llm with a real aim.
    if step_type == "eval":
        step = _translate_eval(step)
        step_type = "llm"
    elif step_type == "aggregator":
        step = _translate_aggregator(step)
        step_type = "llm"
    elif step_type == "agent_handoff":
        step = _translate_agent_handoff(step)
        step_type = "llm"
    elif step_type == "supervisor":
        step = _translate_supervisor(step)
        step_type = "llm"
    elif step_type == "parallel_sampling":
        step = _translate_parallel_sampling(step)
        step_type = "llm"

    if step_type not in CARL_VALID_STEP_TYPES:
        # Unknown type — fall back to llm so the chain stays loadable.
        step["step_type"] = "llm"
        step_type = "llm"

    # 2. Build nested step_config for non-LLM types.
    if step_type == "mcp":
        step = _nest_mcp_fields(step)
    elif step_type == "mcp_resource":
        step = _nest_mcp_resource_fields(step)
    elif step_type == "debate":
        step = _transform_debate(step)
    elif step_type == "tool_discovery":
        step = _transform_tool_discovery(step)
    elif step_type in CARL_FIELD_MAP:
        if step_type == "agent_skill":
            step = refine_agent_skill_step(step)  # P2.4: fill mode/runtime/trust defaults
        elif step_type == "evaluation":
            step = refine_evaluation_step(step)  # coerce criteria/on_fail/method to CARL's vocab
        step = _nest_fields(step, CARL_FIELD_MAP[step_type])

    # P2.2: lift a flat step_cache_ttl into the top-level CARL ``cache`` object.
    step = _apply_cache(step)

    return _strip_mage_internal(step)


# ---------------------------------------------------------------------------
# Translate-to-llm helpers (types CARL cannot load from JSON)
# ---------------------------------------------------------------------------
def _translate_eval(step: dict[str, Any]) -> dict[str, Any]:
    """Translate eval step to llm step with a concrete aim."""
    targets = step.get("eval_target_steps") or []
    criteria = (step.get("eval_criteria") or "quality").strip()
    on_fail = step.get("eval_on_fail")

    target_str = ", ".join(str(n) for n in targets) if targets else "previous step output"
    aim = step.get("aim") or f"Critique step(s) {target_str} against criteria: {criteria}"
    if on_fail:
        aim = f"{aim}. On failure: {on_fail}"

    step = dict(step)
    step["step_type"] = "llm"
    step["aim"] = aim
    if not step.get("reasoning_questions"):
        step["reasoning_questions"] = (
            f"Does the output of step(s) {target_str} satisfy: {criteria}? What are concrete deficiencies, if any?"
        )
    return step


def _translate_aggregator(step: dict[str, Any]) -> dict[str, Any]:
    """Translate aggregator step to llm step with a concrete aim."""
    sources = step.get("aggregate_from_steps") or step.get("dependencies") or []
    strategy = (step.get("aggregation_strategy") or "merge").strip()
    src_str = ", ".join(str(n) for n in sources) if sources else "preceding parallel steps"
    aim = step.get("aim") or f"Aggregate results from step(s) {src_str} using strategy: {strategy}"

    step = dict(step)
    step["step_type"] = "llm"
    step["aim"] = aim
    if not step.get("reasoning_questions"):
        step["reasoning_questions"] = f"What is the unified output combining step(s) {src_str} via {strategy}?"
    return step


def _translate_agent_handoff(step: dict[str, Any]) -> dict[str, Any]:
    """Translate agent_handoff to llm (CARL needs a runtime ``sub_chain``)."""
    step = dict(step)
    target = step.get("handoff_chain_path") or "a specialised sub-task"
    aim = step.get("aim") or f"Carry out the delegated sub-task ({target}) and return its result."
    step["step_type"] = "llm"
    step["aim"] = aim
    return _drop_keys(step, ("handoff_chain_path", "handoff_input_mapping", "handoff_output_key"))


def _translate_supervisor(step: dict[str, Any]) -> dict[str, Any]:
    """Translate supervisor to llm (CARL needs runtime ``agents``)."""
    step = dict(step)
    routing = step.get("supervisor_routing_prompt") or ""
    routes = step.get("supervisor_routes") or {}
    options = ", ".join(routes.keys()) if isinstance(routes, dict) and routes else "the available approaches"
    aim = step.get("aim") or (
        f"Select the most appropriate approach among {options} and carry it out. {routing}".strip()
    )
    step["step_type"] = "llm"
    step["aim"] = aim
    return _drop_keys(step, ("supervisor_routes", "supervisor_routing_prompt"))


def _translate_parallel_sampling(step: dict[str, Any]) -> dict[str, Any]:
    """Translate parallel_sampling to llm (CARL needs a runtime ``base_step``)."""
    step = dict(step)
    n = step.get("sampling_n") or 3
    aim = step.get("aim") or (
        f"Reason about this task {n} independent ways and return the most consistent answer (self-consistency)."
    )
    step["step_type"] = "llm"
    step["aim"] = aim
    return _drop_keys(
        step,
        ("sampling_n", "sampling_temperature", "sampling_aggregation", "sampling_judge_prompt"),
    )


# ---------------------------------------------------------------------------
# Nesting / transform helpers (loadable step types)
# ---------------------------------------------------------------------------
def _nest_fields(step: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Move flat MAGE fields into nested step_config per the given mapping."""
    cfg: dict[str, Any] = dict(step.get("step_config") or {})
    for flat_key, nested_key in mapping.items():
        if flat_key in step and step[flat_key] is not None:
            cfg[nested_key] = step.pop(flat_key)
    if cfg:
        step["step_config"] = cfg
    return step


def _nest_mcp_fields(step: dict[str, Any]) -> dict[str, Any]:
    """MCP requires a nested ``server`` object inside step_config."""
    cfg: dict[str, Any] = dict(step.get("step_config") or {})
    server: dict[str, Any] = dict(cfg.get("server") or {})

    tool_field_map = {
        "mcp_tool_name": "tool_name",
        "mcp_arguments": "arguments",
        "mcp_argument_mapping": "argument_mapping",
        "mcp_timeout": "timeout",
    }

    for flat_key, nested_key in _MCP_SERVER_FIELDS.items():
        if flat_key in step and step[flat_key] is not None:
            server[nested_key] = step.pop(flat_key)
    for flat_key, nested_key in tool_field_map.items():
        if flat_key in step and step[flat_key] is not None:
            cfg[nested_key] = step.pop(flat_key)

    if server:
        cfg["server"] = _normalize_mcp_transport(server)
    if cfg:
        step["step_config"] = cfg
    return step


def _nest_mcp_resource_fields(step: dict[str, Any]) -> dict[str, Any]:
    """MCPResourceStepConfig: nested ``server`` + resource fields (FIXED).

    CARL: server / resource_uri / output_memory_key / output_namespace / timeout.
    """
    cfg: dict[str, Any] = dict(step.get("step_config") or {})
    server: dict[str, Any] = dict(cfg.get("server") or {})

    resource_field_map = {
        "mcp_resource_uri": "resource_uri",
        "mcp_resource_memory_key": "output_memory_key",
    }

    for flat_key, nested_key in _MCP_SERVER_FIELDS.items():
        if flat_key in step and step[flat_key] is not None:
            server[nested_key] = step.pop(flat_key)
    for flat_key, nested_key in resource_field_map.items():
        if flat_key in step and step[flat_key] is not None:
            cfg[nested_key] = step.pop(flat_key)

    if server:
        cfg["server"] = _normalize_mcp_transport(server)
    if cfg:
        step["step_config"] = cfg
    return step


def _transform_debate(step: dict[str, Any]) -> dict[str, Any]:
    """DebateStepConfig: reshape MAGE's ``debate_agents`` list into CARL's
    ``roles`` (list[str]) + ``role_prompts`` (dict) (FIXED).

    CARL: roles / role_prompts / rounds / judge_prompt / task_source / output_*.
    """
    cfg: dict[str, Any] = dict(step.get("step_config") or {})

    agents = step.pop("debate_agents", None)
    if isinstance(agents, list) and agents:
        roles = [a.get("name") for a in agents if isinstance(a, dict) and a.get("name")]
        role_prompts = {
            a["name"]: a.get("role") for a in agents if isinstance(a, dict) and a.get("name") and a.get("role")
        }
        if roles:
            cfg["roles"] = roles
        if role_prompts:
            cfg["role_prompts"] = role_prompts

    scalar_map = {
        "debate_rounds": "rounds",
        "debate_judge_prompt": "judge_prompt",
    }
    for flat_key, nested_key in scalar_map.items():
        if flat_key in step and step[flat_key] is not None:
            cfg[nested_key] = step.pop(flat_key)

    if cfg:
        step["step_config"] = cfg
    return step


def _transform_tool_discovery(step: dict[str, Any]) -> dict[str, Any]:
    """ToolDiscoveryStepConfig: reshape MAGE's ``discovery_sources`` list into
    CARL's single ``source`` (FIXED).

    Only ``module`` sources are JSON-serializable; ``callable`` / ``dict``
    sources carry live Python objects and cannot round-trip (P2.1).
    """
    cfg: dict[str, Any] = dict(step.get("step_config") or {})

    sources = step.pop("discovery_sources", None)
    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, dict) and (first.get("kind", "module") == "module") and first.get("module"):
            src: dict[str, Any] = {"module": first["module"]}
            for opt in ("name_prefix", "tag", "strip_prefix"):
                if first.get(opt) is not None:
                    src[opt] = first[opt]
            cfg["source"] = src

    if cfg:
        step["step_config"] = cfg
    return step


# Remote (untrusted) skill URI schemes — fetched from the network, so they
# warrant SHA pinning + runtime isolation (plan item P2.4).
_REMOTE_SKILL_PREFIXES = ("github://", "https://", "http://")


def _skill_uri(step: dict[str, Any]) -> str:
    """Best-effort skill URI string from the flat ``skill`` field."""
    skill = step.get("skill")
    if isinstance(skill, str):
        return skill
    if isinstance(skill, dict):  # an AgentSkillSource-shaped dict
        return str(skill.get("git_url") or skill.get("path") or skill.get("name") or "")
    return ""


# Keyword → registered skill name, for correcting hallucinated / stale skill
# URIs. Weaker LLMs invent github paths (e.g. "presentation-skills/powerpoint")
# instead of the canonical registry URI; map well-known document skills back.
_SKILL_NAME_ALIASES: dict[str, str] = {
    "pptx": "pptx", "powerpoint": "pptx", "presentation": "pptx",
    "slides": "pptx", "slide": "pptx", "deck": "pptx", "keynote": "pptx",
    "docx": "docx", "msword": "docx", "word document": "docx",
    "pdf": "pdf",
    "xlsx": "xlsx", "excel": "xlsx", "spreadsheet": "xlsx", "csv": "xlsx",
}


def _normalize_skill_uri(step: dict[str, Any]) -> dict[str, Any]:
    """Correct an ``agent_skill`` step's ``skill`` URI against the registry.

    Only touches ``github://anthropics/skills/...`` refs that DON'T already match
    a registered URI (i.e. the planner invented or used a stale path). Maps the
    well-known document skills (pptx / docx / pdf / xlsx) back to their canonical
    registry URI by keyword — so a hallucinated ``presentation-skills/powerpoint``
    resolves to the real ``skills/pptx``. Custom / third-party URIs are left as-is.
    """
    uri = _skill_uri(step)
    if not uri:
        return step
    try:
        from mmar_mage.skills import default_registry
    except Exception:  # noqa: BLE001
        return step
    # BARE registry name (e.g. "pptx" / "powerpoint") → canonical URI, so CARL
    # resolves it instead of searching local skill dirs and failing with
    # "Skill 'pptx' not found in any search path". Planners often name the skill
    # rather than spell the full github:// URI.
    if "://" not in uri and "/" not in uri:
        bare = _SKILL_NAME_ALIASES.get(uri.strip().lower(), uri.strip().lower())
        known = default_registry.get(bare)
        if known is not None:
            step = dict(step)
            step["skill"] = known.uri
            return step
    if not uri.startswith("github://anthropics/skills/"):
        return step
    if uri in {s.uri for s in default_registry.skills}:
        return step  # already canonical
    blob = " ".join(
        str(step.get(k, "")) for k in ("skill", "skill_task", "title", "aim")
    ).lower()
    for keyword, name in _SKILL_NAME_ALIASES.items():
        if keyword in blob:
            target = default_registry.get(name)
            if target is not None:
                step = dict(step)
                step["skill"] = target.uri
                return step
    return step


# CARL's EvalFailAction enum is {continue, abort, retry_with_feedback}. The
# planner (and older prompts) emit a looser vocabulary — map every plausible
# synonym onto a valid value so the step doesn't fail CARL validation.
_EVAL_ON_FAIL_MAP: dict[str, str] = {
    "continue": "continue", "warn": "continue", "log": "continue",
    "ignore": "continue", "pass": "continue", "skip": "continue",
    "abort": "abort", "fail": "abort", "halt": "abort", "stop": "abort",
    "error": "abort", "raise": "abort", "block": "abort",
    "retry_with_feedback": "retry_with_feedback", "retry": "retry_with_feedback",
    "replan": "retry_with_feedback", "feedback": "retry_with_feedback",
    "revise": "retry_with_feedback", "redo": "retry_with_feedback",
}
# CARL's evaluation_method is Literal["rule", "llm"]. Only an explicit "rule"
# (simpleeval boolean expressions) stays rule; everything else — including the
# planner's "python" / "llm_judge" / "judge" — maps to the LLM judge, which
# evaluates plain-English criteria (what the planner almost always emits).
_EVAL_METHOD_MAP: dict[str, str] = {
    "rule": "rule", "rules": "rule", "simpleeval": "rule",
    "expression": "rule", "expr": "rule", "boolean": "rule",
}


def refine_evaluation_step(step: dict[str, Any]) -> dict[str, Any]:
    """Coerce an ``evaluation`` step's fields to CARL's ``EvaluationStepConfig``
    vocabulary so a planner that uses looser values doesn't fail validation:

    * ``evaluation_criteria`` — a bare string is wrapped in a one-element list
      (CARL requires ``list[str]``).
    * ``evaluation_fail_action`` → one of {continue, abort, retry_with_feedback}
      (e.g. ``fail`` → ``abort``, ``retry`` / ``replan`` → ``retry_with_feedback``).
    * ``evaluation_method`` → ``rule`` only when explicitly rule-like, else
      ``llm`` (handles the plain-English criteria the planner emits; CARL has no
      ``python`` method, and ``llm_judge`` isn't a valid value).

    Operates on the FLAT step (before ``_nest_fields`` renames the keys). Never
    raises; unknown values fall back to the safe default.
    """
    if step.get("step_type") != "evaluation":
        return step
    step = dict(step)

    crit = step.get("evaluation_criteria")
    if isinstance(crit, str) and crit.strip():
        step["evaluation_criteria"] = [crit.strip()]

    fa = step.get("evaluation_fail_action")
    if isinstance(fa, str):
        step["evaluation_fail_action"] = _EVAL_ON_FAIL_MAP.get(fa.strip().lower(), "continue")

    method = step.get("evaluation_method")
    if isinstance(method, str):
        step["evaluation_method"] = _EVAL_METHOD_MAP.get(method.strip().lower(), "llm")

    return step


def refine_agent_skill_step(step: dict[str, Any]) -> dict[str, Any]:
    """Fill sensible execution-mode / runtime / trust defaults for an agent_skill
    step (plan item P2.4), based on its signals — WITHOUT overriding explicit choices.

    Heuristics (each fires only when the field is unset and a clear signal exists):

    * ``skill_output_schema`` present → ``execution_mode = "llm_agent"`` (a
      structured-output skill needs the tool-call agent loop, not single-shot).
    * remote skill (``github://`` / ``https://``) that runs a bundled script →
      ``runtime = "docker"`` (isolate untrusted remote code).
    * remote skill with a known ``skill_sha256`` → ``trust_policy = "sha_pinned"``
      (verify the digest). Without a digest we leave ``trust_policy`` alone —
      ``sha_pinned`` *requires* a sha, so forcing it would make the step invalid.

    Also corrects a hallucinated / stale skill URI to the canonical registry one
    first (:func:`_normalize_skill_uri`).
    """
    step = _normalize_skill_uri(step)
    step = dict(step)
    uri = _skill_uri(step)
    remote = uri.startswith(_REMOTE_SKILL_PREFIXES)

    if not step.get("skill_execution_mode") and step.get("skill_output_schema"):
        step["skill_execution_mode"] = "llm_agent"

    if not step.get("skill_runtime") and remote and step.get("skill_script_name"):
        step["skill_runtime"] = "docker"

    if not step.get("skill_trust_policy") and remote and step.get("skill_sha256"):
        step["skill_trust_policy"] = "sha_pinned"

    # CARL's AgentSkillStepConfig requires a non-empty `task` (it becomes the user
    # turn in LLM mode). Default it from the step's aim/title so every exported
    # agent_skill step loads via ReasoningChain.from_json instead of raising
    # "task: Field required".
    if not step.get("skill_task"):
        step["skill_task"] = step.get("aim") or step.get("title") or "Perform the task using the skill."

    return step


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _drop_keys(step: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Remove the given keys from a step dict if present."""
    for k in keys:
        step.pop(k, None)
    return step


def _strip_mage_internal(step: dict[str, Any]) -> dict[str, Any]:
    """Remove MAGE-internal scratch fields, known orphans, and ``None`` values."""
    for k in list(step.keys()):
        if k in _MAGE_INTERNAL_FIELDS or k in _DROPPED_FLAT_FIELDS or step[k] is None:
            step.pop(k)
    return step
