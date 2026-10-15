"""Pre-execution cost estimation for MAGE-generated CARL chains.

Mirrors :mod:`mmar_carl.cost` but works on the dict-form chains that MAGE
emits (no ``ReasoningChain`` object needed). Useful as a pre-flight check
before evolve / production runs to spot chains that will burn money.

Two entry points:

* :func:`estimate_chain_cost` — walk a chain dict and return a
  :class:`CostEstimate` with per-step + chain-level token/cost rows.
* :func:`fits_in_budget` — fast bool check used by the evolve loop's
  optional budget guard.

Tokens are estimated via a coarse char-per-token heuristic (default 4),
which is intentionally rough — the goal is "spot a chain about to burn
$50", not exact billing. Pricing is opt-in: pass a ``pricing`` dict of
``{model: (in_per_1k_usd, out_per_1k_usd)}``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Step types whose executors invoke an LLM at least once per call.
# MAGE-internal labels (eval/aggregator) are treated as LLM steps.
_LLM_CALLING_TYPES: frozenset[str] = frozenset(
    [
        "llm",
        "structured_output",
        "eval",
        "aggregator",
        "evaluation",  # only when method == "llm" (handled below)
        "parallel_sampling",  # n_samples (+1 if best_of judge)
        "agent_skill",  # at least 1 LLM call for llm/hybrid/subagent/llm_agent modes
        "supervisor",  # 1 routing call
        "debate",  # rounds × agents (+1 judge)
    ]
)


class StepCostEstimate(BaseModel):
    """Per-step row in a :class:`CostEstimate`."""

    step_number: int
    step_title: str
    step_type: str
    calls_llm: bool = Field(description="Whether this step invokes an LLM at runtime")
    model: str | None = Field(default=None, description="Resolved model name (None when unresolved)")
    estimated_calls: int = Field(default=0, ge=0, description="LLM calls this step will make")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_cost_usd: float = Field(default=0.0, ge=0.0)
    output_cost_usd: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    pricing_missing: bool = Field(
        default=False,
        description="True when this step calls an LLM but no pricing entry was found for its model.",
    )
    note: str = Field(default="", description="Free-text caveat (e.g. 'LLM_AGENT iterations not modelled').")


class CostEstimate(BaseModel):
    """Aggregated dry-run cost estimate for an entire chain."""

    steps: list[StepCostEstimate]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    char_per_token: int = 4
    currency: str = "USD"
    pricing_complete: bool = True

    @property
    def llm_call_steps(self) -> list[StepCostEstimate]:
        return [s for s in self.steps if s.calls_llm]

    def format_table(self) -> str:
        """Human-readable one-line-per-step summary suitable for stdout."""
        if not self.steps:
            return "(empty chain)"
        header = f"{'#':>3}  {'title':<28}  {'type':<18}  {'calls':>5}  {'in':>6}  {'out':>6}  {'cost':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for row in self.steps:
            cost_str = f"${row.total_cost_usd:.4f}" if row.calls_llm else "-"
            if row.pricing_missing:
                cost_str = "(no price)"
            lines.append(
                f"{row.step_number:>3}  {row.step_title[:28]:<28}  {row.step_type:<18}  "
                f"{row.estimated_calls:>5}  {row.input_tokens:>6}  {row.output_tokens:>6}  {cost_str:>10}"
            )
        lines.append(sep)
        lines.append(
            f"TOTAL  {'':<28}  {'':<18}  {sum(s.estimated_calls for s in self.steps):>5}  "
            f"{self.total_input_tokens:>6}  {self.total_output_tokens:>6}  "
            f"${self.total_cost_usd:>9.4f}"
        )
        return "\n".join(lines)


def _calls_llm_for_step(step: dict[str, Any]) -> tuple[bool, int, str]:
    """Return ``(calls_llm, estimated_calls, note)`` for one step dict."""
    step_type = step.get("step_type", "llm")

    if step_type in ("llm", "structured_output", "eval", "aggregator"):
        return True, 1, ""

    if step_type == "evaluation":
        method = step.get("evaluation_method", "rule")
        if method == "llm":
            return True, 1, "LLM judge call"
        return False, 0, f"{method}-based evaluation — no LLM call"

    if step_type == "parallel_sampling":
        n = int(step.get("sampling_n") or 1)
        agg = step.get("sampling_aggregation")
        if agg == "best_of":
            return True, n + 1, "n samples + 1 judge call"
        return True, n, "n parallel samples"

    if step_type == "agent_skill":
        mode = step.get("skill_execution_mode", "llm")
        if mode == "script":
            return False, 0, "script-only skill — no LLM call"
        if mode == "llm_agent":
            max_iters = int(step.get("skill_llm_max_iterations") or 8)
            return True, max_iters, f"llm_agent tool-loop up to {max_iters} iterations"
        # llm | hybrid | subagent — single-shot LLM call
        return True, 1, f"{mode} mode — 1 LLM call"

    if step_type == "supervisor":
        return True, 1, "1 routing call (sub-chain calls not counted)"

    if step_type == "debate":
        agents = step.get("debate_agents") or []
        rounds = int(step.get("debate_rounds") or 1)
        n_agents = len(agents) if agents else 2
        return True, rounds * n_agents + 1, f"{rounds} rounds × {n_agents} agents + 1 judge"

    if step_type == "agent_handoff":
        return False, 0, "sub-chain not recursively estimated — call estimate on it separately"

    # tool, mcp, memory, transform, conditional, human_input, mcp_resource,
    # tool_discovery — no LLM call.
    return False, 0, ""


def cost_usd_for_usage(
    prompt_tokens: int, completion_tokens: int, in_per_1k: float, out_per_1k: float
) -> float:
    """Actual USD cost from ACTUAL token usage (T14), using the same per-1k pricing
    as :func:`estimate_chain_cost`. The estimate functions price a chain *before*
    execution; this prices what actually ran. CARE writes the result into the
    ledger / replay (``cost_usd``) from a run's ``MAGEMetadata`` token counts, with
    its own per-model pricing. Returns ``0.0`` when the model's price is unknown."""
    return round(
        (prompt_tokens / 1000.0) * max(0.0, in_per_1k)
        + (completion_tokens / 1000.0) * max(0.0, out_per_1k),
        6,
    )


def estimate_chain_cost(
    chain_dict: dict[str, Any],
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
    default_model: str | None = None,
    default_output_tokens: int = 512,
    char_per_token: int = 4,
    outer_context_chars: int = 0,
) -> CostEstimate:
    """Walk a CARL chain dict and produce a :class:`CostEstimate`.

    Parameters
    ----------
    chain_dict : dict
        CARL chain in either MAGE-flat or CARL-nested form (we read only
        ``steps`` and per-step ``aim``/``title``/``llm_config``/type-specific
        config — both forms expose these).
    pricing : dict[str, tuple[float, float]] | None
        Mapping ``model_name → (input_per_1k_usd, output_per_1k_usd)``.
        If None, all cost fields are 0 and ``pricing_missing=True`` on each
        LLM-calling row.
    default_model : str | None
        Fallback model if step has no explicit override. Use the configured
        chain model.
    default_output_tokens : int
        Output budget per call when ``llm_config.max_tokens`` is unset.
    char_per_token : int
        Rough conversion factor. 4 is reasonable for English.
    outer_context_chars : int
        Approximate size of input data context (passed to every LLM step
        as part of its prompt). Set to len(input_data) if known.

    Returns
    -------
    CostEstimate
    """
    if char_per_token <= 0:
        raise ValueError("char_per_token must be positive")

    rows: list[StepCostEstimate] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    pricing_complete = pricing is not None
    history_chars = 0  # accumulated estimate of $history payload size

    for step in chain_dict.get("steps", []) or []:
        calls_llm, est_calls, note = _calls_llm_for_step(step)

        if not calls_llm:
            rows.append(
                StepCostEstimate(
                    step_number=int(step.get("number", 0)),
                    step_title=str(step.get("title", "")),
                    step_type=str(step.get("step_type", "llm")),
                    calls_llm=False,
                    note=note,
                )
            )
            continue

        # Per-step model override → chain default
        llm_cfg = step.get("llm_config") or {}
        if isinstance(llm_cfg, dict):
            model = llm_cfg.get("model") or default_model
            max_tokens = llm_cfg.get("max_tokens") or default_output_tokens
        else:
            model = default_model
            max_tokens = default_output_tokens

        aim_chars = len(str(step.get("aim", "")))
        title_chars = len(str(step.get("title", "")))
        per_call_input_chars = outer_context_chars + aim_chars + title_chars + history_chars
        per_call_input_tokens = max(1, per_call_input_chars // char_per_token)
        input_tokens = per_call_input_tokens * est_calls
        output_tokens = int(max_tokens) * est_calls

        input_cost = 0.0
        output_cost = 0.0
        pricing_missing = False
        if pricing is not None:
            entry = pricing.get(model) if model else None
            if entry is None:
                pricing_missing = True
                pricing_complete = False
            else:
                in_per_1k, out_per_1k = entry
                input_cost = (input_tokens / 1000.0) * in_per_1k
                output_cost = (output_tokens / 1000.0) * out_per_1k

        step_total = input_cost + output_cost
        rows.append(
            StepCostEstimate(
                step_number=int(step.get("number", 0)),
                step_title=str(step.get("title", "")),
                step_type=str(step.get("step_type", "llm")),
                calls_llm=True,
                model=model,
                estimated_calls=est_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=round(input_cost, 6),
                output_cost_usd=round(output_cost, 6),
                total_cost_usd=round(step_total, 6),
                pricing_missing=pricing_missing,
                note=note,
            )
        )
        total_input += input_tokens
        total_output += output_tokens
        total_cost += step_total

        # Advance history proxy. Cap at 4000 chars so estimates don't run
        # away on deep chains.
        history_chars = min(history_chars + int(max_tokens) * char_per_token, 4000)

    return CostEstimate(
        steps=rows,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_usd=round(total_cost, 6),
        char_per_token=char_per_token,
        pricing_complete=pricing_complete,
    )


def estimate_pipeline_cost(
    config: Any,
    *,
    query: str = "",
    pricing: dict[str, tuple[float, float]] | None = None,
    default_output_tokens: int = 512,
    char_per_token: int = 4,
) -> CostEstimate:
    """Pre-flight cost estimate for the MAGE pipeline itself (§6 P3).

    Walks the deep-mode pipeline stages that are enabled on ``config``
    and produces a :class:`CostEstimate` that approximates the cost
    of running MAGE end-to-end on ``query``. Each stage shows up as a
    pseudo-step in the result so CARE's TUI can show a
    per-stage breakdown before generation actually starts.

    Used by ``MAGEGenerator`` (when
    ``MAGEConfig.enable_preflight_cost_estimate=True``) to fire the
    optional ``MAGEProgressCallback.on_cost_estimate`` hook so users
    can confirm/deny before the first LLM call.

    Parameters
    ----------
    config : MAGEConfig
        The generator config — read for `mode`, `model`, the various
        `enable_*` deep-mode toggles, and per-stage overrides.
    query : str
        User query. Adds its character count to every stage's
        per-call input budget so the estimate reflects real-world
        prompt sizes.
    pricing : dict[str, tuple[float, float]] | None
        ``model_name → (input_per_1k_usd, output_per_1k_usd)``. When
        ``None``, cost fields stay at 0 and rows are flagged with
        ``pricing_missing=True``.
    default_output_tokens : int
        Per-LLM-call output budget. Defaults to 512 (matches the
        in-chain cost estimator).
    char_per_token : int
        Conversion factor; 4 is the typical English ratio.

    Returns
    -------
    CostEstimate
        Per-stage rows + totals. Fast mode produces a single row;
        deep mode produces one row per enabled stage.
    """
    # Import locally to break the circular dep with prompts.py (which
    # loads YAML at import time, taking a tiny dependency on
    # mmar_mage.prompt_loader).
    from . import prompts as _prompts

    mode = getattr(config, "mode", "deep")
    default_model = getattr(config, "model", None)
    overrides = getattr(config, "stage_llm_overrides", {}) or {}

    def _stage_model(stage: str) -> str | None:
        ov = overrides.get(stage)
        if ov is None:
            return default_model
        # Accept both typed StageLLMOverride and bare dicts (§6 P2 backwards compat)
        ov_model = getattr(ov, "model", None) if not isinstance(ov, dict) else ov.get("model")
        return ov_model or default_model

    # Each stage descriptor: (stage_name, prompt_constant_name, enabled).
    # Fast mode is special — only one virtual step.
    pipeline_stages: list[tuple[str, str, bool, str]] = []
    if mode == "fast":
        pipeline_stages.append(("fast_generation", "FAST_MODE_SYSTEM_PROMPT", True, "Single-shot chain generation"))
    else:
        pipeline_stages = [
            (
                "domain_analysis",
                "DOMAIN_ANALYZER_PROMPT",
                bool(getattr(config, "enable_domain_analysis", True)),
                "Domain analysis (DomainAnalyzer)",
            ),
            (
                "step_planning",
                "STEP_PLANNER_PROMPT",
                bool(getattr(config, "enable_step_planning", True)),
                "Step planning (StepPlanner)",
            ),
            (
                "dag_building",
                "DAG_BUILDER_PROMPT",
                bool(getattr(config, "enable_dag_optimization", True)),
                "DAG building (DAGBuilder)",
            ),
            (
                "step_describing",
                "STEP_DESCRIBER_PROMPT",
                bool(getattr(config, "enable_step_descriptions", True)),
                "Step descriptions (StepDescriber)",
            ),
            (
                "step_critique",
                "STEP_CRITIC_PROMPT",
                bool(getattr(config, "enable_step_critique", False)),
                "Step quality critique",
            ),
            (
                "chain_verification",
                "CHAIN_VERIFICATION_PROMPT",
                bool(getattr(config, "enable_chain_verification", False)),
                "Chain-of-verification",
            ),
            (
                "self_refine",
                "SELF_REFINE_CRITIQUE_PROMPT",
                bool(getattr(config, "enable_self_refine", False)),
                "Self-refinement (uses 2 LLM calls per iteration)",
            ),
            (
                "suggested_naming",
                "SUGGESTED_NAMING_PROMPT",
                bool(getattr(config, "enable_suggested_naming_llm", True)),
                "Library-ready naming LLM call",
            ),
        ]

    # Treat self_refine as iterations × 2 calls per iteration (critique + revise).
    max_refine_iters = int(getattr(config, "max_refine_iterations", 3))

    query_chars = len(query)
    rows: list[StepCostEstimate] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    pricing_complete = pricing is not None

    for idx, (stage_name, prompt_const, enabled, note_text) in enumerate(pipeline_stages, start=1):
        if not enabled:
            rows.append(
                StepCostEstimate(
                    step_number=idx,
                    step_title=stage_name,
                    step_type="pipeline_stage",
                    calls_llm=False,
                    note=f"{note_text} — disabled",
                )
            )
            continue

        prompt_body = getattr(_prompts, prompt_const, "")
        prompt_chars = len(prompt_body)
        per_call_input_chars = prompt_chars + query_chars
        per_call_input_tokens = max(1, per_call_input_chars // char_per_token)

        # self_refine runs critique + revise × max_iters
        if stage_name == "self_refine":
            est_calls = max(1, max_refine_iters) * 2
        else:
            est_calls = 1

        input_tokens = per_call_input_tokens * est_calls
        output_tokens = default_output_tokens * est_calls

        model = _stage_model(stage_name)
        input_cost = 0.0
        output_cost = 0.0
        pricing_missing = False
        if pricing is not None and model is not None and model in pricing:
            in_per_1k, out_per_1k = pricing[model]
            input_cost = (input_tokens / 1000.0) * in_per_1k
            output_cost = (output_tokens / 1000.0) * out_per_1k
        elif pricing is not None:
            pricing_missing = True
            pricing_complete = False

        step_total = input_cost + output_cost
        total_input += input_tokens
        total_output += output_tokens
        total_cost += step_total

        rows.append(
            StepCostEstimate(
                step_number=idx,
                step_title=stage_name,
                step_type="pipeline_stage",
                calls_llm=True,
                model=model,
                estimated_calls=est_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=step_total,
                pricing_missing=pricing_missing,
                note=note_text,
            )
        )

    return CostEstimate(
        steps=rows,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_usd=total_cost,
        char_per_token=char_per_token,
        pricing_complete=pricing_complete,
    )


def fits_in_budget(
    chain_dict: dict[str, Any],
    *,
    max_cost_usd: float | None = None,
    max_total_tokens: int | None = None,
    pricing: dict[str, tuple[float, float]] | None = None,
    default_model: str | None = None,
) -> tuple[bool, CostEstimate]:
    """Quick budget check for the evolve loop's optional cost guard.

    Returns ``(fits, estimate)``. ``fits`` is False iff either limit is
    exceeded by the estimate. Both limits are optional — when both are None
    the function always returns True.
    """
    estimate = estimate_chain_cost(chain_dict, pricing=pricing, default_model=default_model)
    over_cost = max_cost_usd is not None and estimate.total_cost_usd > max_cost_usd
    over_tokens = max_total_tokens is not None and estimate.total_tokens > max_total_tokens
    return (not (over_cost or over_tokens)), estimate
