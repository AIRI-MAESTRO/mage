"""
MAGEGenerator — the main orchestrator for CARL chain generation.

Supports two modes:
- **fast**: single LLM call, returns CARL JSON directly.
- **deep**: agentic pipeline (DomainAnalyzer → StepPlanner → DAGBuilder →
  StepDescriber → serialization), each stage a separate LLM call with
  individually toggleable sub-stages.  Optionally preceded by memory
  research (stage 0a) and web research (stage 0b).

Inspired by the Claude Code *tool use loop* pattern: the generator calls
agents iteratively, validates intermediate outputs, and retries on failure.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, NamedTuple, Protocol, cast, overload, runtime_checkable

from .agents import (
    DAGBuilder,
    DomainAnalyzer,
    MemoryResearchAgent,
    StepDescriber,
    StepPlanner,
    WebResearchAgent,
)
from .agents.capability_lookup_agent import CapabilityContext, CapabilityLookupAgent
from .agents.chain_editor import ChainEditPlanner
from .agents.chain_verifier import ChainVerifier, VerificationResult
from .agents.feedback_recall_agent import FeedbackDigest, FeedbackRecallAgent
from .agents.mcts_constructor import MCTSConstructor
from .agents.self_refiner import RefinementResult, SelfRefiner
from .agents.step_critic import StepCriticAgent, StepCritiqueResult
from .agents.template_assembler_agent import TemplateAssemblerAgent, TemplateSkeleton
from .agents.tool_spec_generator import ToolSpecGenerator, format_tool_spec_for_planner
from .agents.topology_selector import TopologySelectorAgent
from .agents.tot_planner import ToTPlanner
from .carl_export import (
    mark_cacheable_steps,
    missing_step_config_fields,
    reindex_history_refs,
    to_carl_compat,
    validate_step_metrics,
)
from .chain_edit import apply_edit_plan
from .chain_repair import lint_chain, repair_chain, validate_or_repair
from .exceptions import MAGECancelledError, MAGEValidationError
from .llm import LLMClient
from .memory import MemoryManager, is_reusable_step
from .prompts import (
    EXECUTION_REPAIR_PROMPT,
    FAST_MODE_SYSTEM_PROMPT,
    PLAN_PAIRWISE_PROMPT,
    SUGGESTED_NAMING_PROMPT,
    build_planning_guidance,
)
from .schemas import (
    CARL_VALID_STEP_TYPES,
    VALID_STEP_TYPES,
    CARLChainSchema,
    CARLStepSchema,
    ChainEditCandidate,
    DAGStep,
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    MAGEEditResult,
    MAGEMetadata,
    MAGEResult,
    MemoryResearchResult,
    PlannedStep,
    StepPlan,
    WebResearchResult,
)
from .stage_cache import StageCache

# SkillDiscoveryAgent is now invoked INSIDE CapabilityLookupAgent (§2 P0);
# skills flow to the planner via capability_ctx.skills_digest() — see _generate_deep.

logger = logging.getLogger(__name__)


class _ResearchOutcome(NamedTuple):
    """Bundled results of the concurrent research stages 0a–0e (WP-4)."""

    memory_result: MemoryResearchResult
    web_result: WebResearchResult
    capability_ctx: CapabilityContext
    template_skel: TemplateSkeleton | None
    feedback_digest: FeedbackDigest
    memory_hits_used: int
    web_results_used: int
    was_cold_start: bool
    feedback_recalled: int | None


# ---------------------------------------------------------------------------
# Progress callback protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MAGEProgressCallback(Protocol):
    """Protocol for receiving progress updates during deep-mode generation.

    Implementations may provide any subset of the methods below. The
    generator probes via ``hasattr`` before invoking the optional
    streaming / fine-grained hooks, so older callbacks remain compatible.
    """

    def on_stage_start(self, stage: str) -> None:
        """Called when a pipeline stage begins."""
        ...

    def on_stage_complete(self, stage: str, result: Any) -> None:
        """Called when a pipeline stage completes successfully."""
        ...

    def on_error(self, stage: str, error: Exception) -> None:
        """Called when a pipeline stage fails."""
        ...

    # --- Optional streaming / fine-grained hooks (CARE TUI) ---------------

    def on_llm_chunk(self, stage: str, delta: str) -> None:
        """Called for each partial token / chunk while the LLM is streaming.

        Optional. Only invoked when the caller passes a streaming-capable
        callback into ``LLMClient.generate(stream_callback=...)`` and the
        underlying API supports streaming.
        """
        ...

    def on_stage_progress(self, stage: str, artifact: Any) -> None:
        """Called when a stage emits an incremental artifact.

        Optional. Examples: ``StepDescriber`` fires this per described
        step, ``ToTPlanner`` per branch, ``MCTSConstructor`` per
        simulation. Lets CARE render in-flight progress without waiting
        for the whole stage to finish.
        """
        ...

    def on_retry(self, stage: str, attempt: int, error: Exception) -> None:
        """Called when the LLM client retries after a transient failure.

        Optional. Fires when an attempt fails AND there is at least one
        more attempt remaining. ``attempt`` is 1-indexed (so a failed
        first attempt fires ``on_retry(..., attempt=1, ...)``).
        """
        ...

    def on_cost_estimate(self, estimate: Any) -> None:
        """Called once before any LLM calls when pre-flight is enabled.

        Optional. Fires when
        ``MAGEConfig.enable_preflight_cost_estimate=True`` so CARE can
        show a "this will cost ~$X" confirmation panel. ``estimate`` is
        a :class:`CostEstimate` from :mod:`mmar_mage.cost`.
        """
        ...

    def on_chain_preview(self, preview: Any) -> None:
        """Called once the chain skeleton / DAG is assembled — BEFORE the
        steps are described + serialized — with a lightweight preview so a UI
        can render the chain forming.

        Optional (P5.8). ``preview`` is a JSON-friendly dict::

            {"topology": <pattern>, "max_workers": <int>,
             "steps": [{"number", "title", "step_type", "dependencies"}]}
        """
        ...


class _DefaultProgressCallback:
    """No-op progress callback."""

    def on_stage_start(self, stage: str) -> None:
        logger.info("Stage started: %s", stage)

    def on_stage_complete(self, stage: str, result: Any) -> None:
        logger.info("Stage completed: %s", stage)

    def on_error(self, stage: str, error: Exception) -> None:
        logger.error("Stage %s failed: %s", stage, error)

    def on_llm_chunk(self, stage: str, delta: str) -> None:
        # No-op: most consumers (CLI, evaluation runner) don't render
        # streaming tokens. CARE supplies its own callback when needed.
        return

    def on_stage_progress(self, stage: str, artifact: Any) -> None:
        # No-op for the default callback.
        return

    def on_retry(self, stage: str, attempt: int, error: Exception) -> None:
        logger.info("Retrying stage %s (attempt %d): %s", stage, attempt, error)

    def on_cost_estimate(self, estimate: Any) -> None:
        # No-op for the default callback; CARE supplies its own.
        return

    def on_chain_preview(self, preview: Any) -> None:
        # No-op for the default callback; CARE supplies its own.
        return


class _StageTimingRecorder:
    """Wraps a progress callback, timing each pipeline stage between its
    ``on_stage_start`` and ``on_stage_complete`` / ``on_error`` (T45). Every other
    callback method — including the optional streaming hooks — is delegated to the
    wrapped callback via ``__getattr__`` only when it actually defines it, so the
    generator's ``getattr(progress, name, None)`` probes keep working unchanged."""

    def __init__(self, inner: MAGEProgressCallback) -> None:
        self._inner = inner
        self._starts: dict[str, float] = {}
        self.timings: dict[str, float] = {}

    def snapshot_and_reset(self) -> dict[str, float]:
        """Return the timings collected since the last call and clear them, so each
        generation's metadata carries only its own stages."""
        out = dict(self.timings)
        self.timings.clear()
        self._starts.clear()
        return out

    def on_stage_start(self, stage: str) -> None:
        self._starts[stage] = time.monotonic()
        self._inner.on_stage_start(stage)

    def on_stage_complete(self, stage: str, result: Any) -> None:
        self._record(stage)
        self._inner.on_stage_complete(stage, result)

    def on_error(self, stage: str, error: Exception) -> None:
        self._record(stage)
        self._inner.on_error(stage, error)

    def _record(self, stage: str) -> None:
        t0 = self._starts.pop(stage, None)
        if t0 is not None:
            self.timings[stage] = round(time.monotonic() - t0, 3)

    def __getattr__(self, name: str) -> Any:
        # Reached only for names not defined above (the optional hooks). Private /
        # dunder names raise so this can't recurse on ``self._inner`` access.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_carl_json(chain_dict: dict[str, Any], *, strict_carl: bool = False) -> None:
    """Validate that a dict conforms to CARL chain structure.

    Parameters
    ----------
    chain_dict : dict[str, Any]
        Parsed chain dictionary.
    strict_carl : bool
        If True, enforce CARL-only step types (rejects MAGE-internal
        ``eval``/``aggregator``). Use after CARL-compat export to verify
        the dict is loadable by ``ReasoningChain.from_json``.

    Raises
    ------
    MAGEValidationError
        If the structure is invalid.
    """
    if "steps" not in chain_dict:
        raise MAGEValidationError("CARL chain must contain 'steps' array")

    steps = chain_dict["steps"]
    if not isinstance(steps, list) or len(steps) == 0:
        raise MAGEValidationError("CARL chain 'steps' must be a non-empty list")

    valid_types = CARL_VALID_STEP_TYPES if strict_carl else VALID_STEP_TYPES

    # Pass 1 — structural integrity: numbers, titles, dependency targets.
    seen_numbers: set[int] = set()
    for step in steps:
        num = step.get("number")
        if num is None:
            raise MAGEValidationError("Every step must have a 'number' field")
        if num in seen_numbers:
            raise MAGEValidationError(f"Duplicate step number: {num}")
        seen_numbers.add(num)

        if not step.get("title"):
            raise MAGEValidationError(f"Step {num} missing 'title'")

    all_numbers = seen_numbers
    for step in steps:
        num = step["number"]
        for dep in step.get("dependencies", []):
            if dep == num:
                raise MAGEValidationError(f"Step {num} depends on itself")
            if dep not in all_numbers:
                raise MAGEValidationError(f"Step {num} depends on non-existent step {dep}")

    # Pass 2 — cycle detection (independent of step ordering).
    _detect_cycle(steps)

    # Pass 3 — semantic checks: step_type validity, aim non-empty for llm.
    for step in steps:
        num = step["number"]
        step_type = step.get("step_type", "llm")
        if step_type not in valid_types:
            raise MAGEValidationError(
                f"Step {num} has invalid step_type '{step_type}'. Valid types: {sorted(valid_types)}"
            )
        if step_type == "llm" and not (step.get("aim") or "").strip():
            raise MAGEValidationError(f"Step {num} (llm) requires non-empty 'aim'")


def _detect_cycle(steps: list[dict[str, Any]]) -> None:
    """Detect cycles in step dependencies via DFS.

    Raises
    ------
    MAGEValidationError
        If a cycle is found, with the cycle path included in the message.
    """
    graph: dict[int, list[int]] = {s["number"]: list(s.get("dependencies", [])) for s in steps}
    # 0 = unvisited, 1 = on current DFS stack, 2 = finished
    color: dict[int, int] = {n: 0 for n in graph}

    def dfs(node: int, path: list[int]) -> list[int] | None:
        color[node] = 1
        path.append(node)
        for dep in graph.get(node, []):
            if dep not in color:
                continue
            if color[dep] == 1:
                cycle_start = path.index(dep)
                return path[cycle_start:] + [dep]
            if color[dep] == 0:
                cycle = dfs(dep, path)
                if cycle is not None:
                    return cycle
        path.pop()
        color[node] = 2
        return None

    for node in graph:
        if color[node] == 0:
            cycle = dfs(node, [])
            if cycle is not None:
                cycle_str = " -> ".join(str(n) for n in cycle)
                raise MAGEValidationError(f"Cyclic dependency detected: {cycle_str}")


# Risk score at/above which the default policy escalates to multi-checker voting
# (rule checker + LLM judge) instead of a single rule checker (plan item P2.3).
_REPLAN_RISK_THRESHOLD = 3


def _chain_replan_risk(steps: list[dict[str, Any]]) -> int:
    """Score a chain's replan risk: external/side-effecting steps + debate + length.

    Tool/MCP/skill steps can fail in ways worth rolling back; debate is expensive
    and contested; long chains have more failure surface. Higher → deeper policy.
    """
    risky = sum(1 for s in steps if s.get("step_type") in _CHECKPOINT_STEP_TYPES)
    has_debate = any(s.get("step_type") == "debate" for s in steps)
    return risky + (2 if has_debate else 0) + (1 if len(steps) >= 6 else 0)


def _build_default_replan_policy(
    chain_dict: dict[str, Any],
    *,
    max_replans: int = 3,
    action_on_failure: str = "replan_from_checkpoint",
) -> dict[str, Any] | None:
    """Synthesize a default ReplanPolicy from checkpoint hints, scaled by risk (P2.3).

    Emits the **real** ``mmar_carl.ReplanPolicy`` shape: flat
    ``RuleBasedReplanCheckerConfig`` fields (no ``config`` nesting — the old shape
    silently dropped every checker setting on load), a ``budgets`` sub-object
    (where ``max_replans`` actually lives), and a ``default_checkpoint_target``.

    Policy depth scales with :func:`_chain_replan_risk`: low-risk chains get a
    single rule-based checker (``aggregation="any"``); high-risk chains add an LLM
    judge with ``mandatory_plus_k_of_rest`` voting (the rule checker is mandatory,
    plus k of the rest). Budget knobs (``max_replans_per_chain``,
    ``max_visits_per_checkpoint``) scale with chain length.

    Returns ``None`` when no checkpoint-marked step exists — there's nothing
    meaningful to roll back to, so no policy is attached.
    """
    steps = chain_dict.get("steps", []) or []
    if not any(bool(s.get("checkpoint")) for s in steps):
        return None

    n = len(steps)
    rollback = {
        "target_type": "nearest_checkpoint",
        "reason": "Step failed; rolling back to the nearest checkpoint.",
    }
    rule_checker = {
        "type": "rule_based",
        "name": "auto-replan-on-failure",
        "trigger_on_failed_step": True,
        "action_on_failure": action_on_failure,
        "rollback_target_on_failure": rollback,
        "confidence": 1.0,
        "feedback_on_failure": ["Step failed; rolling back to the nearest checkpoint and retrying."],
    }
    checkers: list[dict[str, Any]] = [rule_checker]
    aggregation: dict[str, Any] = {"strategy": "any"}

    # High-risk chains: add an LLM judge and require the rule checker (mandatory)
    # plus at least one other checker to agree before replanning.
    if _chain_replan_risk(steps) >= _REPLAN_RISK_THRESHOLD:
        checkers.append(
            {
                "type": "llm",
                "name": "llm-replan-judge",
                "parse_error_action": "continue",  # conservative when the judge can't be parsed
            }
        )
        aggregation = {
            "strategy": "mandatory_plus_k_of_rest",
            "k": 1,
            "mandatory_checkers": ["auto-replan-on-failure"],
        }

    return {
        "enabled": True,
        "checkers": checkers,
        "aggregation": aggregation,
        "trigger": {"evaluate_after_step": True, "evaluate_after_failure": True},
        "budgets": {
            "max_replans_per_chain": max(max_replans, min(n, 6)),
            "max_visits_per_checkpoint": max(2, min(n // 2, 5)),
        },
        "default_checkpoint_target": rollback,
        "source": "mage_default",  # MAGE-side marker (ignored by CARL on load)
    }


def _resolve_skill_payload(step: dict[str, Any]) -> dict[str, Any]:
    """Extract the AgentSkill descriptor from a CARL ``agent_skill`` step.

    Handles both the flat shape (``skill_name``, ``skill_uri``, ...)
    and the nested ``step_config`` shape produced by ``to_carl_compat``.
    Returns a dict compatible with
    :meth:`MemoryManager.save_agent_skill` — at minimum carrying
    ``name`` and ``uri``.
    """
    name = step.get("skill_name") or ""
    uri = step.get("skill") or step.get("skill_uri") or ""
    sc = step.get("step_config")
    if isinstance(sc, dict):
        name = name or sc.get("skill_name") or ""
        uri = uri or sc.get("skill") or sc.get("skill_uri") or ""

    if not name and isinstance(uri, str) and uri:
        # Derive a friendly name from the trailing path segment of the URI.
        name = uri.rstrip("/").split("/")[-1].split("@")[0]

    return {
        "name": name or "unnamed-skill",
        "uri": uri,
        "description": step.get("aim") or step.get("title") or "",
    }


def _build_evolution_meta(metadata: MAGEMetadata) -> dict[str, Any]:
    """Extract the platform's `evolution_meta` block from MAGEMetadata (§5 P2).

    Only includes fields that carry a non-None signal — keeps the
    persisted block small and lets the evolution engine treat missing
    keys as "unknown" rather than "zero".
    """
    signals = {
        "step_critique_score": metadata.step_critique_score,
        "verification_passed": metadata.verification_passed,
        "refine_iterations": metadata.refine_iterations,
        "refine_quality_delta": metadata.refine_quality_delta,
        "tot_branches_explored": metadata.tot_branches_explored,
        "mcts_simulations_run": metadata.mcts_simulations_run,
        "mcts_best_reward": metadata.mcts_best_reward,
        "feedback_recalled": metadata.feedback_recalled,
    }
    return {k: v for k, v in signals.items() if v is not None}


def _check_cancel(cancel: asyncio.Event | None, stage: str) -> None:
    """Raise :class:`MAGECancelledError` if the caller flipped the event.

    No-op when ``cancel`` is ``None``. Designed to be called liberally
    between stages and before each LLM call — the check is cheap and
    lets CARE's ``Esc`` key abort generation within one stage of
    latency.
    """
    if cancel is not None and cancel.is_set():
        raise MAGECancelledError(f"Generation cancelled before stage '{stage}'", stage=stage)


# Step types worth checkpointing in the absence of a clear synthesizer — external
# / side-effecting calls whose failure is worth rolling back to (P1.14).
_CHECKPOINT_STEP_TYPES = ("tool", "mcp", "agent_skill", "mcp_resource")


def _promote_checkpoints(steps: list[dict[str, Any]]) -> None:
    """Promote one risk step to a checkpoint so a default ReplanPolicy can attach (P1.14).

    ``_build_default_replan_policy`` returns ``None`` unless some step is marked
    ``checkpoint=true``, but the StepDescriber never marks one — so a policy was
    never attached. This heuristic picks, in order, and mutates ``steps`` in place:

    1. the latest *synthesizer* step (>= 2 dependencies — the merge point of a
       funnel / diamond, where a failure should roll the chain back);
    2. otherwise, in a chain of >= 3 steps, the first external call
       (tool / mcp / agent_skill / mcp_resource) — a composite live query
       (IP -> geolocate -> advise) is the run-096 class that needs a rollback point.

    The chosen step gets ``checkpoint=true`` + ``checkpoint_name="primary"`` (a
    stable rollback target). No-op if a checkpoint already exists or nothing fits.
    """
    if not steps or any(s.get("checkpoint") for s in steps):
        return

    merge_steps = [s for s in steps if len(s.get("dependencies") or []) >= 2]
    target: dict[str, Any] | None = None
    if merge_steps:
        target = max(merge_steps, key=lambda s: s.get("number", 0))
    elif len(steps) >= 3:
        for s in sorted(steps, key=lambda s: s.get("number", 0)):
            if s.get("step_type") in _CHECKPOINT_STEP_TYPES:
                target = s
                break

    if target is not None:
        target["checkpoint"] = True
        target.setdefault("checkpoint_name", "primary")


# Floor for the biased step count — below 2, a chain can't meaningfully
# decompose-then-act, so the simplicity bias never collapses past it.
_SIMPLICITY_MIN_STEPS = 2


def _apply_simplicity_bias(count: int, bias: float) -> int:
    """Scale a suggested step count by ``(1 - bias)`` (P1.10, AFlow-style).

    ``bias`` is clamped to ``[0, 1]``; the result is rounded and floored at
    :data:`_SIMPLICITY_MIN_STEPS` so the bias simplifies without collapsing a
    chain to a single step. ``bias <= 0`` is a no-op.
    """
    if bias <= 0 or count <= _SIMPLICITY_MIN_STEPS:
        return count
    bias = min(bias, 1.0)
    return max(_SIMPLICITY_MIN_STEPS, round(count * (1.0 - bias)))


def _has_structured_output(config: MAGEConfig) -> bool:
    """Whether the configured model reliably returns valid JSON for a *batch*
    describe. OpenAI always does; a local server only when guided decoding is on
    (``use_schema_constrained_output`` and no json_mode_fallback — i.e. vLLM).
    OpenRouter / custom route to many models, including flash models that ignore
    ``json_schema`` and break long batch replies (runs 731/773), so they are not
    trusted here."""
    if config.provider == "openai":
        return True
    if config.provider == "local":
        return config.use_schema_constrained_output and not bool(config.json_mode_fallback)
    return False


_ITERATIVE_DESCRIBE_MIN_STEPS = 4  # batch JSON stays small enough below this even on weak models


def _resolve_describe_mode(config: MAGEConfig, num_steps: int) -> str:
    """Resolve ``step_describe_mode`` (T44). ``"auto"`` describes one step per call
    when the model lacks structured output AND the chain is long enough that a single
    batch JSON reply tends to break on flash models (runs 731/773) — a noisy reply
    then costs one step's re-request, not the whole chain. Explicit modes pass through.
    """
    mode = config.step_describe_mode
    if mode != "auto":
        return mode
    if not _has_structured_output(config) and num_steps >= _ITERATIVE_DESCRIBE_MIN_STEPS:
        return "iterative"
    return "batch"


def _select_critique_steps(
    carl_steps: list[CARLStepSchema], step_types: list[str] | None
) -> list[CARLStepSchema]:
    """The steps the step critic should see (T11): all of them, or only the given
    types — the cheap care_default verifier critiques just tool / structured_output
    instead of paying the full critique on every step."""
    if step_types is None:
        return list(carl_steps)
    allowed = set(step_types)
    return [s for s in carl_steps if s.step_type in allowed]


def _should_fast_path(config: MAGEConfig, domain_analysis: DomainAnalysis) -> bool:
    """Route a genuinely simple query straight to a single fast-mode call instead of
    the 7-stage deep pipeline (T13) — deep spent 88-144s / 44k tokens on 'what's my
    IP'. Relies on the post-T2 complexity classification, so a composite query that
    needs several calls is now 'medium' and stays deep. Off by default."""
    return (
        config.enable_fast_path_routing
        and domain_analysis.complexity == "simple"
        and domain_analysis.suggested_step_count <= config.fast_path_max_steps
    )


def _enforce_step_budget(plan: StepPlan, max_steps: int) -> StepPlan:
    """Truncate an over-budget plan back to ``max_steps`` unless it's justified (P1.10).

    A plan with more than ``max_steps`` steps and an *empty* ``rationale`` is the
    unjustified-runaway case AFlow's bounded vocabulary guards against — keep the
    first ``max_steps`` steps (``PlannedStep`` carries no dependencies, so the
    DAGBuilder rewires from the truncated plan) and renumber. A filled rationale
    means the model justified the complexity, so the plan is left intact.
    """
    if max_steps <= 0 or len(plan.steps) <= max_steps or plan.rationale.strip():
        return plan
    logger.warning(
        "StepPlan has %d steps (> cap %d) with no rationale — truncating to the cap (P1.10).",
        len(plan.steps),
        max_steps,
    )
    kept = plan.steps[:max_steps]
    for i, step in enumerate(kept):
        step.number = i + 1
    plan.steps = kept
    return plan


def _try_load_reasoning_chain(chain_json: str, *, raise_on_error: bool = False) -> Any | None:
    """Attempt to load chain via mmar_carl.ReasoningChain.from_json().

    Returns the chain object on success, or ``None`` if mmar_carl is not
    installed or the load fails non-fatally.

    CARL's ``from_json`` currently lags its own runtime: it cannot reconstruct
    several agent-features step types from serialized JSON (``agent_skill``
    raises; ``evaluation`` / ``debate`` / ``mcp_resource`` / ``human_input`` /
    ``tool_discovery`` load with their config dropped — see IMPROVEMENT_PLAN
    P0.2/P0.3). Since the caller has already run
    ``validate_carl_json(strict_carl=True)`` — MAGE's authority for chain
    validity — a ``from_json`` failure here is a best-effort cross-check, not an
    authoring error: by default it is logged as a warning and ``None`` is
    returned so a valid chain is still emitted. Pass ``raise_on_error=True``
    (``MAGEConfig.raise_on_carl_load_error``) to restore hard failure.
    """
    try:
        from mmar_carl import ReasoningChain

        return ReasoningChain.from_json(chain_json)
    except ImportError:
        logger.debug("mmar_carl not installed, skipping ReasoningChain validation")
        return None
    except Exception as exc:
        if raise_on_error:
            raise MAGEValidationError(f"ReasoningChain.from_json() failed: {exc}", raw_output=chain_json) from exc
        logger.warning(
            "ReasoningChain.from_json() failed (non-fatal; chain passed MAGE strict validation): %s",
            exc,
        )
        return None


# Domain-keyed fallback templates (plan item P3.1). Each is a list of
# ``(title, rough_aim)`` pairs, ordered core-first with a synthesis/output step
# LAST — so truncating to a smaller step count still ends on a meaningful result.
# Used when the StepPlanner is disabled, instead of "Step N / Process part N".
_FALLBACK_GENERIC: list[tuple[str, str]] = [
    ("Understand the request", "Interpret the query and identify what is being asked"),
    ("Work through the task", "Reason step by step toward the answer"),
    ("Produce the result", "Synthesize the final answer for the user"),
]
_FALLBACK_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "extraction": [
        ("Parse the input", "Read and structure the raw input"),
        ("Extract the target information", "Pull out the requested entities or fields"),
        ("Validate the extraction", "Check the extracted data for completeness and correctness"),
        ("Emit the results", "Return the extracted data in the requested format"),
    ],
    "analysis": [
        ("Gather the relevant inputs", "Collect the data and context the analysis needs"),
        ("Analyze the key factors", "Examine the inputs for patterns and drivers"),
        ("Synthesize findings", "Combine the analysis into a clear conclusion"),
    ],
    "comparison": [
        ("Define comparison criteria", "Establish the dimensions to compare on"),
        ("Assess each option", "Evaluate every candidate against the criteria"),
        ("Compare and recommend", "Produce a side-by-side verdict with a recommendation"),
    ],
    "qa": [
        ("Interpret the question", "Clarify what specific answer is required"),
        ("Reason toward the answer", "Work through the relevant facts and logic"),
        ("Answer the question", "State the answer clearly with justification"),
    ],
    "generation": [
        ("Plan the output", "Outline the structure and key points"),
        ("Draft the content", "Produce a first complete version"),
        ("Refine and finalize", "Polish the draft into the final output"),
    ],
    "verification": [
        ("Restate what must be checked", "Identify exactly which claim or property to verify"),
        ("Check against the evidence", "Test the claim against the available facts"),
        ("Report the verdict", "State whether it holds, with supporting reasons"),
    ],
}

# (template key, keyword triggers) — first match wins; order matters (specific → broad).
_FALLBACK_TEMPLATE_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("extraction", ("extract", "parse", "scrape")),
    ("comparison", ("compar", "versus", " vs ", "benchmark", "rank")),
    ("verification", ("verif", "validat", "check", "fact")),
    ("generation", ("generat", "write", "draft", "summari", "compose", "author")),
    ("qa", ("question", "answer", "lookup", "retriev")),
    ("analysis", ("analy", "assess", "investigat", "diagnos", "evaluat")),
)


def _select_fallback_template(domain: DomainAnalysis) -> list[tuple[str, str]]:
    """Pick a fallback template from the domain analysis (P3.1)."""
    hay = f" {(domain.task_type or '').lower()} {(domain.domain or '').lower()} "
    for key, triggers in _FALLBACK_TEMPLATE_TRIGGERS:
        if any(trigger in hay for trigger in triggers):
            return _FALLBACK_TEMPLATES[key]
    return _FALLBACK_GENERIC


def _fit_template(template: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
    """Resize ``template`` to exactly ``count`` steps, always keeping the final
    (output/synthesis) step last (P3.1).

    Fewer than the template → keep the first ``count-1`` steps + the output step.
    More → pad with "refine" steps inserted before the output step.
    """
    count = max(1, count)
    if count == 1:
        return [template[-1]]
    if count <= len(template):
        return list(template[: count - 1]) + [template[-1]]
    head = list(template[:-1])
    while len(head) < count - 1:
        head.append(("Refine the working result", "Iterate to improve the intermediate result"))
    return head + [template[-1]]


# ---------------------------------------------------------------------------
# CARE library-ready naming (§7.2)
# ---------------------------------------------------------------------------


_PDF_HINTS = ("pdf",)
_WEB_HINTS = ("web", "search", "tavily", "serpapi", "brave", "browse")
_MEMORY_STEP_TYPES = {"memory"}


def _detect_capability_tags(
    chain_dict: dict[str, Any],
    *,
    web_results_used: int = 0,
    memory_hits_used: int = 0,
) -> list[str]:
    """Detect coarse capability tags from a finalized chain dict.

    Looks at step types and step / skill names to derive ``uses-pdf``,
    ``uses-web``, ``uses-memory``. Used both as standalone metadata tags
    and as input to the deep-mode naming LLM call.
    """
    tags: list[str] = []
    uses_memory = memory_hits_used > 0
    uses_web = web_results_used > 0
    uses_pdf = False

    for step in chain_dict.get("steps", []) or []:
        step_type = (step.get("step_type") or "").lower()
        if step_type in _MEMORY_STEP_TYPES:
            uses_memory = True

        # Pull a freeform "name-ish" string for substring matching. Steps
        # of different types stash their name under different keys; pull
        # them all and lowercase once.
        haystack_parts: list[str] = []
        for key in ("tool_name", "mcp_server_name", "skill_name", "title", "aim"):
            v = step.get(key)
            if isinstance(v, str):
                haystack_parts.append(v.lower())
        # Nested step_config may carry the same fields post-CARL-export.
        sc = step.get("step_config")
        if isinstance(sc, dict):
            for key in ("tool_name", "mcp_server_name", "skill_name"):
                v = sc.get(key)
                if isinstance(v, str):
                    haystack_parts.append(v.lower())
        haystack = " ".join(haystack_parts)

        if any(h in haystack for h in _PDF_HINTS):
            uses_pdf = True
        if any(h in haystack for h in _WEB_HINTS):
            uses_web = True

    if uses_pdf:
        tags.append("uses-pdf")
    if uses_web:
        tags.append("uses-web")
    if uses_memory:
        tags.append("uses-memory")
    return tags


_MEMORY_NAME_MARKERS = (
    "what i remember about you",
    "(long-term memory)",
    "here is what i know about you",
    "here's what i know about you",
)


def _clean_query_for_name(query: str) -> str:
    """Strip a leaked memory / context block from a query before it becomes a chain
    name or description (T16) — a polluted query produced ``General — ## What I
    remember about you (long-term memory)\\n- role: ML r`` as the chain name. Drops
    markdown headers, bullet / blockquote lines, and memory-marker lines, keeping the
    actual task prose. Returns "" when nothing usable remains (callers fall back to
    the domain / task type)."""
    out: list[str] = []
    for raw in (query or "").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "- ", "* ", "• ", ">")):
            continue
        if any(m in line.lower() for m in _MEMORY_NAME_MARKERS):
            continue
        out.append(line)
    return " ".join(out).strip()


def _naming_heuristic(
    query: str,
    *,
    domain: str = "general",
    task_type: str = "analysis",
    key_concepts: list[str] | None = None,
    capability_tags: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Default heuristic for the three ``suggested_*`` fields.

    Used in fast mode (no DomainAnalysis available) and as the fallback
    when the deep-mode LLM naming call fails or is disabled.
    """
    query = _clean_query_for_name(query)  # T16: never let a memory block leak into the name
    domain_title = (domain or "general").title()

    query_slice = query[:50].rstrip()
    display_name = f"{domain_title} — {query_slice}" if query_slice else domain_title
    if len(display_name) > 60:
        display_name = display_name[:57].rstrip() + "..."

    desc_slice = query[:150].rstrip()
    description = f"{task_type}: {desc_slice}" if desc_slice else task_type

    tags: list[str] = []
    if domain and domain.lower() != "general":
        tags.append(domain.lower())
    if task_type and task_type.lower() not in {t.lower() for t in tags}:
        tags.append(task_type.lower())
    for concept in (key_concepts or [])[:3]:
        slug = concept.strip().lower().replace(" ", "-")
        if slug and slug not in tags:
            tags.append(slug)
    for cap in capability_tags or []:
        if cap not in tags:
            tags.append(cap)
    return display_name, description, tags


def _coerce_naming_payload(
    data: dict[str, Any],
    *,
    fallback_display_name: str,
    fallback_description: str,
    fallback_tags: list[str],
) -> tuple[str, str, list[str]]:
    """Validate the LLM response shape and coerce into the trio of fields.

    Any missing / wrong-typed field falls back to the heuristic value so a
    partially-bad LLM response still yields something usable.
    """
    display_name = data.get("display_name")
    if not isinstance(display_name, str) or not display_name.strip():
        display_name = fallback_display_name
    display_name = display_name.strip().rstrip(".")
    if len(display_name) > 60:
        display_name = display_name[:57].rstrip() + "..."

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        description = fallback_description
    description = description.strip()

    raw_tags = data.get("tags")
    if not isinstance(raw_tags, list):
        tags = list(fallback_tags)
    else:
        tags = []
        seen: set[str] = set()
        for t in raw_tags:
            if not isinstance(t, str):
                continue
            slug = t.strip().lower().replace(" ", "-")
            if slug and slug not in seen:
                seen.add(slug)
                tags.append(slug)
        # Make sure detected capability flags are surfaced even if the LLM
        # omitted them — they're literal facts about the chain.
        for cap in fallback_tags:
            if cap.startswith("uses-") and cap not in seen:
                seen.add(cap)
                tags.append(cap)
    return display_name, description, tags


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------


class MAGEGenerator:
    """Orchestrate CARL chain generation in fast or deep mode.

    Parameters
    ----------
    config : MAGEConfig | None
        Generation configuration.  Uses defaults if ``None``.
    progress : MAGEProgressCallback | None
        Optional callback for progress reporting.

    Examples
    --------
    >>> gen = MAGEGenerator(MAGEConfig(mode="fast"))
    >>> result = await gen.generate("Analyze a financial report")
    >>> print(result.chain_json)
    """

    def __init__(
        self,
        config: MAGEConfig | None = None,
        progress: MAGEProgressCallback | None = None,
    ) -> None:
        self._config = config or MAGEConfig()
        self._llm = LLMClient(self._config)
        self._memory = MemoryManager(self._config)
        # T45: wrap the callback so each stage is timed for the replay bundle; the
        # recorder delegates every other call straight through to the user callback.
        self._timing = _StageTimingRecorder(progress or _DefaultProgressCallback())
        self._progress: MAGEProgressCallback = cast(MAGEProgressCallback, self._timing)
        # T42: session-scoped content-addressable cache for deterministic stages.
        self._stage_cache: StageCache | None = StageCache() if self._config.enable_stage_cache else None

    async def _cached_stage(self, stage: str, payload: Any, factory: Callable[[], Awaitable[Any]]) -> Any:
        """Run a deterministic deep-mode stage through the content-addressable
        cache when enabled (T42): the same ``(query, domain)`` isn't recomputed
        across a session or a pipeline retry. Stores + returns deep copies so a
        downstream mutation can't taint the cache. Off by default (no behaviour
        change)."""
        cache = self._stage_cache
        if cache is None:
            return await factory()
        cached = cache.get(stage, payload)
        if cached is not None:
            logger.debug("stage cache hit: %s", stage)
            return copy.deepcopy(cached)
        result = await factory()
        cache.put(stage, payload, copy.deepcopy(result))
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    async def generate(
        self,
        query: str,
        *,
        context_files: list[dict[str, Any]] | None = ...,
        cancel: asyncio.Event | None = ...,
        capabilities: CapabilityContext | None = ...,
        return_candidates: Literal[False] = ...,
    ) -> MAGEResult: ...

    @overload
    async def generate(
        self,
        query: str,
        *,
        context_files: list[dict[str, Any]] | None = ...,
        cancel: asyncio.Event | None = ...,
        capabilities: CapabilityContext | None = ...,
        return_candidates: Literal[True],
    ) -> list[MAGEResult]: ...

    async def generate(
        self,
        query: str,
        *,
        context_files: list[dict[str, Any]] | None = None,
        cancel: asyncio.Event | None = None,
        capabilities: CapabilityContext | None = None,
        return_candidates: bool = False,
    ) -> MAGEResult | list[MAGEResult]:
        """Generate a CARL chain from a text query.

        Parameters
        ----------
        query : str
            The user's text query describing the desired reasoning task.
        context_files : list[dict] | None
            File references the caller wants persisted alongside the
            chain for CARE's "Re-run from library" flow. Each dict must
            contain ``path``, ``sha256``, ``size_bytes``. Optional —
            when ``None`` the metadata block omits ``context_files``.
        cancel : asyncio.Event | None
            Optional cancel signal. Generation checks this event between
            stages and before each LLM call; setting it aborts the run
            with :class:`MAGECancelledError`. CARE wires its ``Esc``
            key here.
        capabilities : CapabilityContext | None
            Caller-supplied capability context (§2 P1). When provided,
            it **completely bypasses** the internal
            :class:`CapabilityLookupAgent` — Memory lookups, the local
            skill catalog, and the folded SkillDiscoveryAgent. Use this
            to inject skills CARE has installed locally but not yet
            promoted to Memory. Precedence rule: caller-supplied >
            Memory + local merge.
        return_candidates : bool
            WP-6. When ``True`` (deep mode only) and the planner sampled
            multiple topology candidates
            (``enable_parallel_topology_sampling`` with ≥2 distinct
            hints), each candidate plan is carried independently through
            DAG → describe → validation — sharing the single
            research/domain/capability pass — and a **list** of
            :class:`MAGEResult` is returned (one per candidate). When no
            candidates were sampled (or in fast mode) a single-element
            list is returned. When ``False`` (default) a single
            :class:`MAGEResult` is returned exactly as before.

        Returns
        -------
        MAGEResult | list[MAGEResult]
            A single result by default; a list of results (one per
            finalized candidate plan) when ``return_candidates=True``.

        Raises
        ------
        MAGEValidationError
            If the generated chain is structurally invalid.
        MAGELLMError
            If LLM calls fail.
        MAGECancelledError
            If ``cancel`` is set during generation.
        """
        # Auto-detect model if needed
        await self._llm.resolve_model()

        # Wire retry visibility for the lifetime of this call. The probe
        # via getattr lets legacy callbacks (without on_retry) keep
        # working — see test_generator_works_with_legacy_callback.
        on_retry = getattr(self._progress, "on_retry", None)
        self._llm.set_retry_callback(on_retry)

        # §6 P3 — pre-flight cost estimate. Surfaces an aggregate
        # per-stage cost preview via `on_cost_estimate` before the
        # first LLM call fires, so CARE can offer a "this will cost
        # ~$X — continue?" panel. Best-effort: a failure in the
        # estimator never blocks generation.
        if self._config.enable_preflight_cost_estimate:
            try:
                from .cost import estimate_pipeline_cost

                cost_estimate = estimate_pipeline_cost(
                    self._config,
                    query=query,
                    pricing=self._config.pricing,
                )
                cb = getattr(self._progress, "on_cost_estimate", None)
                if cb is not None:
                    try:
                        cb(cost_estimate)
                    except Exception as cb_exc:  # noqa: BLE001
                        logger.warning("on_cost_estimate raised %s; continuing", cb_exc)
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning("pre-flight cost estimate failed: %s", exc)

        _check_cancel(cancel, "startup")

        t0 = time.monotonic()

        # Ground tool steps in the advertised catalogue: collect the registered
        # tool names (+ their signatures) so the assembly tail can catch a planner
        # that invented a tool the environment doesn't have (e.g. ``get_weather``)
        # and rebind it onto a real one. Only when the caller advertised tools —
        # with no catalogue we can't tell an invented name from a valid one.
        known_tools: set[str] | None = None
        tool_catalog_hint = ""
        if capabilities is not None:
            cap_tools = [t for t in (getattr(capabilities, "tools", None) or []) if isinstance(t, dict)]
            names = {str(t["name"]) for t in cap_tools if t.get("name")}
            if names:
                known_tools = names
                tool_catalog_hint = "\n".join(
                    f"- {t['name']}: {t.get('description') or ''}".rstrip() for t in cap_tools if t.get("name")
                )

        # WP-6: deep mode with return_candidates carries each topology-sampled
        # plan through the tail independently, yielding one result tuple per
        # candidate. Every other path (fast, or deep without candidates) yields
        # a single tuple; the default return type stays a single MAGEResult.
        if return_candidates and self._config.mode != "fast":
            tuples = await self._generate_deep(
                query, cancel=cancel, capabilities=capabilities, return_candidates=True
            )
            assert isinstance(tuples, list)  # _generate_deep contract under return_candidates
            results: list[MAGEResult] = []
            for tup in tuples:
                results.append(
                    await self._assemble_result(
                        *tup,
                        query=query,
                        context_files=context_files,
                        cancel=cancel,
                        t0=t0,
                        known_tools=known_tools,
                        tool_catalog_hint=tool_catalog_hint,
                    )
                )
            return results

        quality_meta: dict[str, Any]
        if self._config.mode == "fast":
            chain_dict, chain_json = await self._generate_fast(query, cancel=cancel)
            stages_completed = None
            memory_hits_used = 0
            web_results_used = 0
            was_cold_start = False
            quality_meta = {}
        else:
            single = await self._generate_deep(query, cancel=cancel, capabilities=capabilities)
            assert not isinstance(single, list)  # _generate_deep contract without return_candidates
            (
                chain_dict,
                chain_json,
                stages_completed,
                memory_hits_used,
                web_results_used,
                was_cold_start,
                quality_meta,
            ) = single

        result = await self._assemble_result(
            chain_dict,
            chain_json,
            stages_completed,
            memory_hits_used,
            web_results_used,
            was_cold_start,
            quality_meta,
            query=query,
            context_files=context_files,
            cancel=cancel,
            t0=t0,
            known_tools=known_tools,
            tool_catalog_hint=tool_catalog_hint,
        )
        if return_candidates:
            # Fast mode (or any single-tuple path) with return_candidates=True
            # returns a one-element list for a consistent caller contract.
            return [result]
        return result

    async def _assemble_result(
        self,
        chain_dict: dict[str, Any],
        chain_json: str,
        stages_completed: list[str] | None,
        memory_hits_used: int,
        web_results_used: int,
        was_cold_start: bool,
        quality_meta: dict[str, Any],
        *,
        query: str,
        context_files: list[dict[str, Any]] | None,
        cancel: asyncio.Event | None,
        t0: float,
        known_tools: set[str] | None = None,
        tool_catalog_hint: str = "",
    ) -> MAGEResult:
        """Validate, export, name, persist and package one chain into a result.

        This is the post-generation "assembly" tail of :meth:`generate`,
        extracted so each WP-6 candidate tuple can be turned into its own
        :class:`MAGEResult` while sharing the single research pass. The default
        single-generation path calls it exactly once, so behaviour is unchanged.
        """
        domain_analysis_meta: DomainAnalysis | None = quality_meta.pop("_domain_analysis", None)
        failure_meta: list[dict[str, Any]] = list(quality_meta.pop("_failure_meta", None) or [])
        intermediate_artifacts: dict[str, Any] = dict(quality_meta.pop("_intermediate_artifacts", None) or {})

        _check_cancel(cancel, "finalize")

        elapsed = time.monotonic() - t0

        # Validate (MAGE-internal types eval/aggregator allowed here).
        validate_carl_json(chain_dict)

        # T5: deterministic tool-step lint (url shape / $outer_context / empty
        # tool_name). Non-fatal here (like the P2.1 check below) — surfaced in
        # logs + failure_meta so a chain that would GET https://<free text> is
        # visible; the edit path (validate_or_repair) repairs lint failures.
        tool_lint = lint_chain(chain_dict, known_tools=known_tools)
        # Two lint classes are effectively FATAL at run time: an invented tool_name
        # (not in the advertised catalogue → "Tool '<name>' not registered") and a
        # transform that fakes LLM-style extraction ("{{…}}" / "$history[i][slice]" →
        # ships garbage). Repair them once: feed the chain + the registered-tool
        # catalogue back to the LLM to rebind the tool and/or convert the bogus
        # transform into a structured_output step. Best-effort — any failure keeps the
        # original chain (a chain that runs, even if degraded, beats aborting).
        def _is_hard_lint(msg: str) -> bool:
            return (
                "not a registered tool" in msg
                or "non-executable syntax" in msg
                or "splices a $-reference" in msg
            )

        if any(_is_hard_lint(i) for i in tool_lint):
            try:
                error = "; ".join(tool_lint)
                if tool_catalog_hint:
                    error += (
                        "\n\nRegistered tools you MUST choose from "
                        "(name: what it does + call signature):\n" + tool_catalog_hint
                    )
                regrounded = await repair_chain(chain_dict, query=query, error=error, llm=self._llm)
                # Accept only if the repaired chain is structurally valid in BOTH the
                # MAGE-internal and strict-CARL forms and no hard-lint issue remains;
                # otherwise keep the original.
                validate_carl_json(regrounded)
                _probe = json.loads(json.dumps(regrounded, ensure_ascii=False))
                _probe.pop("_domain", None)
                validate_carl_json(to_carl_compat(_probe), strict_carl=True)
                if not any(_is_hard_lint(i) for i in lint_chain(regrounded, known_tools=known_tools)):
                    logger.info("repaired hard lint (invented tool / bogus transform)")
                    chain_dict = regrounded
                    tool_lint = lint_chain(chain_dict, known_tools=known_tools)
            except Exception as exc:  # noqa: BLE001 — never let repair abort assembly
                logger.warning("hard-lint repair failed (%s); keeping original chain", exc)
        if tool_lint:
            logger.warning("Tool-step lint (T5): %s", "; ".join(tool_lint))
            failure_meta.append({"stage": "tool_lint", "reason": "tool-step lint failures", "issues": tool_lint})

        # P2.1: reliability check — flag typed steps missing essential config
        # (e.g. an evaluation step with no target). Non-fatal: these would export
        # to a degraded no-op, so surface them in metadata + logs for visibility.
        incomplete_steps = missing_step_config_fields(chain_dict)
        if incomplete_steps:
            logger.warning(
                "Typed steps missing required config (P2.1): %s",
                "; ".join(f"step {n}: {', '.join(fields)}" for n, fields in incomplete_steps.items()),
            )

        # P3.4: flag malformed declared success metrics (analytics annotations).
        bad_metrics = validate_step_metrics(chain_dict)
        if bad_metrics:
            logger.warning(
                "Malformed step_metrics (P3.4): %s",
                "; ".join(f"step {n}: {', '.join(probs)}" for n, probs in bad_metrics.items()),
            )

        # P2.2: memoize side-effect-free steps (read-only MCP resource reads).
        # Runs for both fast + deep modes, before export builds the cache object.
        if self._config.enable_step_caching:
            mark_cacheable_steps(chain_dict, default_ttl=self._config.step_cache_default_ttl)

        # Determine domain (MAGE-internal scratch field, must come off
        # before CARL-compat export and final serialization).
        domain = chain_dict.get("_domain", "general")
        chain_dict.pop("_domain", None)

        # CARL-compat export: translate eval/aggregator → llm and nest
        # tool/mcp/memory/transform/structured_output configs. The result
        # is what gets serialized as the final chain JSON.
        chain_dict = to_carl_compat(chain_dict)

        # Strict CARL validation on the exported form.
        validate_carl_json(chain_dict, strict_carl=True)

        # Execution-feedback loop (P1.9): verify against a synthetic benchmark,
        # repair the weakest chain, re-verify — may REWRITE chain_dict, so it runs
        # before serialization. Produces the quality_report when enabled.
        quality_report: dict[str, Any] | None = None
        if self._config.enable_execution_feedback:
            chain_dict, quality_report = await self._run_execution_feedback(query, chain_dict, domain)

        # Final pass: translate MAGE's step-NUMBER $history[N] refs into the
        # 0-indexed history POSITION the CARL runtime resolves — without this a
        # multi-step chain dies at execution with "list index out of range".
        # Runs LAST (after any execution-feedback rewrite) and exactly once.
        chain_dict = reindex_history_refs(chain_dict)

        chain_json = json.dumps(chain_dict, ensure_ascii=False, indent=2)
        _try_load_reasoning_chain(chain_json, raise_on_error=self._config.raise_on_carl_load_error)

        # CARE library-ready naming (§7.2). Always compute heuristic first so
        # we have a safe fallback; in deep mode, attempt an LLM call to
        # replace it with a punchier display_name / description.
        capability_tags = _detect_capability_tags(
            chain_dict,
            web_results_used=web_results_used,
            memory_hits_used=memory_hits_used,
        )
        domain_for_naming = domain_analysis_meta.domain if domain_analysis_meta is not None else domain
        task_type_for_naming = domain_analysis_meta.task_type if domain_analysis_meta is not None else "analysis"
        key_concepts_for_naming = list(domain_analysis_meta.key_concepts) if domain_analysis_meta is not None else None
        display_name, description, tags = _naming_heuristic(
            query,
            domain=domain_for_naming,
            task_type=task_type_for_naming,
            key_concepts=key_concepts_for_naming,
            capability_tags=capability_tags,
        )
        if (
            self._config.mode == "deep"
            and self._config.enable_suggested_naming_llm
            and domain_analysis_meta is not None
        ):
            display_name, description, tags = await self._generate_suggested_naming(
                query,
                domain_analysis_meta,
                capability_tags,
                fallback_display_name=display_name,
                fallback_description=description,
                fallback_tags=tags,
            )

        # 7. Benchmark mode (P1.16): single-shot self-verification, when the
        # execution-feedback loop above didn't already produce a report.
        if quality_report is None:
            quality_report = await self._run_benchmark_mode(query, chain_dict, domain)

        # WP-5 — fold actual token usage (response.usage) into metadata.
        usage_total = self._llm.total_usage()
        usage_by_stage_dump = {k: v.model_dump() for k, v in self._llm.usage_snapshot().items()}

        metadata = MAGEMetadata(
            domain=domain,
            num_steps=len(chain_dict.get("steps", [])),
            generation_time_seconds=round(elapsed, 3),
            model=self._config.model,
            deep_stages_completed=stages_completed,
            memory_hits_used=memory_hits_used,
            web_results_used=web_results_used,
            was_cold_start=was_cold_start,
            # Quality enhancement metadata
            step_critique_score=quality_meta.get("step_critique_score"),
            verification_passed=quality_meta.get("verification_passed"),
            refine_iterations=quality_meta.get("refine_iterations"),
            refine_quality_delta=quality_meta.get("refine_quality_delta"),
            tot_branches_explored=quality_meta.get("tot_branches_explored"),
            mcts_simulations_run=quality_meta.get("mcts_simulations_run"),
            mcts_best_reward=quality_meta.get("mcts_best_reward"),
            feedback_recalled=quality_meta.get("feedback_recalled"),
            # CARE library-ready naming (§7.2)
            suggested_display_name=display_name,
            suggested_description=description,
            suggested_tags=tags,
            # WP-5 — actual token usage from response.usage
            prompt_tokens=usage_total.prompt_tokens,
            completion_tokens=usage_total.completion_tokens,
            total_tokens=usage_total.total_tokens,
            reasoning_tokens=usage_total.reasoning_tokens,
            usage_estimated=usage_total.estimated,
            usage_by_stage=usage_by_stage_dump,
            stage_timings=self._timing.snapshot_and_reset(),
        )

        # §5 P2 — evolution_meta block, attached to the chain row when
        # the platform's evolution engine should treat it as a fitness
        # prior. Built from the same `quality_meta` already on
        # `MAGEMetadata`, but kept as a separate kwarg so the memory
        # entity is self-describing without depending on the
        # mage_metadata blob.
        evolution_meta: dict[str, Any] | None = None
        if self._config.enable_evolution_meta_save:
            evolution_meta = _build_evolution_meta(metadata)

        # Memory — persist with CARE metadata block (§7.3) so the chain is
        # re-runnable from CARE's library: task_description re-primes the
        # query, context_files re-hydrate file inputs, suggested_* names
        # surface in the library list.
        memory_key = await self._memory.save_chain(
            chain_dict,
            domain=domain,
            name=chain_dict.get("name"),
            task_description=query,
            context_files=context_files,
            mage_metadata=metadata.model_dump(),
            display_name=metadata.suggested_display_name,
            description=metadata.suggested_description,
            evolution_meta=evolution_meta,
        )

        # §5 P1 + P2 memory writes — best-effort, non-fatal. Run after
        # the chain itself is persisted so failures here don't strand
        # the main entity. Returns the entity ids persisted on this
        # call so we can surface them in `MAGEResult.memory_keys`
        # (§7.1 replay bundle).
        ecosystem_keys = await self._persist_ecosystem_artifacts(
            chain_dict,
            domain=domain,
            query=query,
            failure_meta=failure_meta,
            intermediate_artifacts=intermediate_artifacts,
            chain_memory_key=memory_key,
        )

        # §7.1 — assemble the memory_keys map for the replay bundle.
        memory_keys: dict[str, Any] = {}
        if memory_key:
            memory_keys["chain"] = memory_key
        for k, v in ecosystem_keys.items():
            if v:
                memory_keys[k] = v

        result = MAGEResult(
            chain_json=chain_json,
            chain_dict=chain_dict,
            memory_key=memory_key,
            mode=self._config.mode,
            metadata=metadata,
            intermediate_artifacts=intermediate_artifacts,
            memory_keys=memory_keys,
            quality_report=quality_report,
        )
        result._config = self._config

        if self._config.output_format in ("python", "both"):
            from .code_generator import CodeGenerator

            result.code = CodeGenerator.generate(chain_dict, query, self._config)

        return result

    async def edit(
        self,
        instruction: str,
        *,
        chain: dict[str, Any] | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        save: bool = False,
        cancel: asyncio.Event | None = None,
    ) -> MAGEEditResult:
        """Apply a natural-language edit to an EXISTING CARL chain.

        Resolution precedence:

        1. ``chain`` (a dict) — edit it directly (``resolved_by="provided"``).
        2. ``entity_id`` — load that chain from memory (``resolved_by="id"``).
        3. otherwise resolve from ``instruction`` via memory search
           (``resolved_by="search"``): 0 hits → :class:`MAGEValidationError`,
           exactly 1 → use it, >1 → return ``needs_disambiguation=True`` with
           ``candidates`` and perform NO edit.

        With ``save=True`` and a known ``entity_id`` the edited chain is
        persisted as a NEW VERSION under the same id (history preserved).

        Parameters
        ----------
        instruction : str
            What to change, in natural language.
        chain : dict | None
            An in-memory chain to edit directly (skips resolution).
        entity_id : str | None
            Id of a saved chain to load and edit.
        channel : str
            Version channel to load from / save to (default ``"latest"``).
        save : bool
            Persist the edited chain as a new version when an id is known.
        cancel : asyncio.Event | None
            Cooperative cancel signal, checked between stages.

        Returns
        -------
        MAGEEditResult

        Raises
        ------
        MAGEValidationError
            If no chain can be resolved, or the edited chain is invalid and
            cannot be repaired.
        MAGECancelledError
            If ``cancel`` is set during the run.
        """
        await self._llm.resolve_model()
        self._llm.set_retry_callback(getattr(self._progress, "on_retry", None))

        t0 = time.monotonic()

        # 1. Resolve which chain to edit.
        before, entity_id, resolved_by, disambiguation = await self._resolve_edit_target(
            instruction, chain=chain, entity_id=entity_id, channel=channel
        )
        if disambiguation is not None:
            return disambiguation

        _check_cancel(cancel, "edit_resolve")

        # 2. Plan the edit (LLM).
        self._progress.on_stage_start("chain_edit")
        domain = self._chain_domain(before)
        plan = await ChainEditPlanner(self._llm).plan(before, instruction, domain=domain)
        self._progress.on_stage_complete("chain_edit", plan.model_dump())

        _check_cancel(cancel, "edit_apply")

        # 3. Apply the plan, then validate (one LLM repair pass if it broke the
        #    schema; raises MAGEValidationError if still invalid). Returns the
        #    CARL-compat, strict-valid dict ready to serialize / save.
        after = apply_edit_plan(before, plan.edits)
        after = await validate_or_repair(after, query=instruction, llm=self._llm)
        chain_json = json.dumps(after, ensure_ascii=False, indent=2)

        # 4. Optionally save as a NEW VERSION under the same entity_id.
        memory_key: str | None = None
        if save and entity_id:
            memory_key = await self._memory.save_chain(
                after,
                domain=domain,
                name=after.get("name"),
                entity_id=entity_id,
                channel=channel,
            )

        elapsed = time.monotonic() - t0
        return MAGEEditResult(
            chain_json=chain_json,
            chain_dict=after,
            before_chain_dict=before,
            edits=plan.edits,
            summary=plan.summary,
            entity_id=entity_id,
            memory_key=memory_key,
            saved=bool(memory_key),
            resolved_by=resolved_by,
            metadata=MAGEMetadata(
                domain=domain,
                num_steps=len(after.get("steps", [])),
                generation_time_seconds=round(elapsed, 3),
                model=self._config.model,
            ),
        )

    @staticmethod
    def _chain_domain(chain: dict[str, Any]) -> str:
        """Best-effort domain label from a chain's metadata block."""
        md = chain.get("metadata")
        if isinstance(md, dict) and md.get("domain"):
            return str(md["domain"])
        return "general"

    async def _resolve_edit_target(
        self,
        instruction: str,
        *,
        chain: dict[str, Any] | None,
        entity_id: str | None,
        channel: str,
    ) -> tuple[
        dict[str, Any],
        str | None,
        Literal["id", "search", "provided"],
        MAGEEditResult | None,
    ]:
        """Resolve the chain to edit.

        Returns ``(before, entity_id, resolved_by, disambiguation)``. When the
        instruction matched multiple chains, ``disambiguation`` is a ready-to-
        return :class:`MAGEEditResult` (``needs_disambiguation=True``) and
        ``before`` is ``{}``.
        """
        if chain is not None:
            return chain, entity_id, "provided", None

        if entity_id:
            before = await self._memory.load_chain(entity_id, channel=channel)
            if not before:
                raise MAGEValidationError(f"Chain {entity_id!r} not found in memory.")
            return before, entity_id, "id", None

        hits = [h for h in await self._memory.search_similar(instruction, top_k=5) if h.get("entity_id")]
        if not hits:
            raise MAGEValidationError(
                "Could not find a chain to edit. Pass an entity_id, or describe "
                "the chain to edit more specifically."
            )
        if len(hits) > 1:
            candidates = [
                ChainEditCandidate(
                    entity_id=str(h["entity_id"]),
                    name=str(h.get("name") or ""),
                    score=float(h.get("score") or 0.0),
                )
                for h in hits
            ]
            return (
                {},
                None,
                "search",
                MAGEEditResult(
                    resolved_by="search",
                    needs_disambiguation=True,
                    candidates=candidates,
                    summary="Multiple chains match — pick one to edit.",
                ),
            )

        chosen = str(hits[0]["entity_id"])
        before = await self._memory.load_chain(chosen, channel=channel)
        if not before:
            raise MAGEValidationError(f"Chain {chosen!r} not found in memory.")
        return before, chosen, "search", None

    async def _run_benchmark_mode(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain: str,
    ) -> dict[str, Any] | None:
        """Stage 7 (P1.16): synthesize a mini-benchmark, run the chain, report.

        Returns the ``QualityReport`` as a dict, or ``None`` when disabled, when
        no test cases were synthesized, or on failure. Non-fatal — a
        self-verification hiccup must never block returning the chain.
        """
        if not self._config.enable_benchmark_mode:
            return None
        try:
            # Lazy import: the benchmark package pulls llm/prompts/schemas, so
            # importing it at call time (not module load) avoids an import cycle.
            from .benchmark import BenchmarkSynthesizer, ChainRunner, QualityReporter

            self._progress.on_stage_start("benchmark_mode")
            domain_analysis = DomainAnalysis(domain=domain)
            benchmark = await BenchmarkSynthesizer(self._llm).synthesize(
                query, domain_analysis, n_cases=self._config.benchmark_n_cases
            )
            if not benchmark.tasks:
                self._progress.on_stage_complete("benchmark_mode", None)
                return None
            results = await ChainRunner(self._llm).run(chain_dict, benchmark)
            report = QualityReporter.report(benchmark, results)
            self._progress.on_stage_complete("benchmark_mode", report)
            return report.model_dump()
        except Exception as exc:
            logger.warning("Benchmark mode (stage 7) failed (non-fatal): %s", exc)
            self._progress.on_error("benchmark_mode", exc)
            return None

    async def _run_execution_feedback(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Stage 5.7 (P1.9): verify → repair → re-verify against a synthetic benchmark.

        Synthesizes one benchmark, scores the chain, and while the pass-rate is
        below ``execution_feedback_threshold`` repairs the chain (guided by its
        failing cases) and re-scores — keeping the best chain seen, stopping on no
        improvement or an invalid repair. Returns ``(best_chain, report_dict)``;
        ``report`` is ``None`` only when no benchmark could be synthesized.
        Non-fatal: on error returns the original chain unchanged.
        """
        try:
            from .benchmark import BenchmarkSynthesizer, ChainRunner, QualityReporter

            self._progress.on_stage_start("execution_feedback")
            domain_analysis = DomainAnalysis(domain=domain)
            benchmark = await BenchmarkSynthesizer(self._llm).synthesize(
                query, domain_analysis, n_cases=self._config.benchmark_n_cases
            )
            if not benchmark.tasks:
                self._progress.on_stage_complete("execution_feedback", None)
                return chain_dict, None

            runner = ChainRunner(self._llm)
            best_chain = chain_dict
            best_report = QualityReporter.report(benchmark, await runner.run(chain_dict, benchmark))

            for _ in range(self._config.max_execution_feedback_iterations):
                if best_report.pass_rate >= self._config.execution_feedback_threshold:
                    break
                repaired = await self._repair_chain(query, best_chain, benchmark, best_report)
                if repaired is None:
                    break  # invalid / no-op repair — keep the best chain
                report = QualityReporter.report(benchmark, await runner.run(repaired, benchmark))
                if report.pass_rate > best_report.pass_rate:
                    best_chain, best_report = repaired, report
                else:
                    break  # repair didn't help — stop, keep the best

            self._progress.on_stage_complete("execution_feedback", best_report)
            return best_chain, best_report.model_dump()
        except Exception as exc:
            logger.warning("Execution-feedback loop (P1.9) failed (non-fatal): %s", exc)
            self._progress.on_error("execution_feedback", exc)
            return chain_dict, None

    async def _repair_chain(
        self,
        query: str,
        chain_dict: dict[str, Any],
        benchmark: Any,
        report: Any,
    ) -> dict[str, Any] | None:
        """Repair a chain against its failing test cases (P1.9).

        Returns a strict-CARL-validated revised chain, or ``None`` if there is
        nothing to repair or the LLM's revision is invalid (so the loop keeps the
        last good chain).
        """
        by_id = {t.task_id: t for t in benchmark.tasks}
        failing = [
            {"input": t.input, "rubric": t.rubric, "failure": r.get("failure")}
            for r in report.per_case_results
            if not r.get("passed") and (t := by_id.get(r.get("task_id"))) is not None
        ]
        if not failing:
            return None

        context = (
            f"User query: {query}\n\n"
            f"Current chain:\n{json.dumps(chain_dict, ensure_ascii=False, indent=2)}\n\n"
            f"Failing test cases:\n{json.dumps(failing, ensure_ascii=False, indent=2)}"
        )
        revised = await self._llm.generate_json(
            system_prompt=EXECUTION_REPAIR_PROMPT,
            user_prompt=context,
            stage="execution_feedback",
        )
        if not isinstance(revised, dict) or not revised.get("steps"):
            return None
        try:
            validate_carl_json(revised, strict_carl=True)
        except MAGEValidationError as exc:
            logger.warning("Execution-feedback: repaired chain failed validation, discarding: %s", exc)
            return None
        return revised

    # ------------------------------------------------------------------
    # Memory write-back for ecosystem reuse (§5 P1)
    # ------------------------------------------------------------------

    async def _persist_ecosystem_artifacts(
        self,
        chain_dict: dict[str, Any],
        *,
        domain: str,
        query: str | None = None,
        failure_meta: list[dict[str, Any]] | None = None,
        intermediate_artifacts: dict[str, Any] | None = None,
        chain_memory_key: str | None = None,
    ) -> dict[str, list[str]]:
        """Save reusable artifacts from a finalized chain (§5 P1+P2, §4 P2).

        Best-effort: each helper logs and continues on failure, never
        raises. Saves agent_skill provenance (when
        ``enable_skill_provenance_save=True``), reusable LLM steps
        (when ``enable_reusable_step_save=True``), failure cards
        (when ``enable_failure_card_save=True`` and ``failure_meta``
        carries one or more entries — verifier rejection or
        self-refine cap hit), and intermediate-stage audit cards
        (when ``enable_intermediate_artifact_save=True`` and
        ``intermediate_artifacts`` is non-empty — §4 P2).

        Returns a dict ``{"skills": [...], "reusable_steps": [...],
        "failure_cards": [...], "intermediate_artifacts": [...]}``
        carrying the entity IDs persisted on this call. Lists are
        present even when empty (drops ``None`` ids — meaning the
        save failed or memory was off — so callers can rely on every
        list element being a real key). Used by the outer
        :meth:`generate` to populate :attr:`MAGEResult.memory_keys`
        (§7.1).
        """
        keys: dict[str, list[str]] = {
            "skills": [],
            "reusable_steps": [],
            "failure_cards": [],
            "intermediate_artifacts": [],
        }

        save_skills = self._config.enable_skill_provenance_save
        save_steps = self._config.enable_reusable_step_save
        save_failures = self._config.enable_failure_card_save and bool(failure_meta)
        save_intermediate = self._config.enable_intermediate_artifact_save and bool(intermediate_artifacts)
        if not (save_skills or save_steps or save_failures or save_intermediate):
            return keys

        for step in chain_dict.get("steps") or []:
            if not isinstance(step, dict):
                continue
            step_type = (step.get("step_type") or "llm").lower()

            # AgentSkill provenance — every agent_skill step gets persisted
            # (idempotent via SHA on the SDK side).
            if save_skills and step_type == "agent_skill":
                resolved = _resolve_skill_payload(step)
                if resolved.get("uri"):
                    try:
                        skill_key = await self._memory.save_agent_skill(resolved)
                        if skill_key:
                            keys["skills"].append(skill_key)
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        logger.warning("save_agent_skill failed for step %s: %s", step.get("number"), exc)

            # Reusable LLM step — only when the heuristic passes.
            if save_steps and is_reusable_step(step):
                try:
                    step_key = await self._memory.save_reusable_step(step, domain=domain)
                    if step_key:
                        keys["reusable_steps"].append(step_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("save_reusable_step failed for step %s: %s", step.get("number"), exc)

        # Failure cards — one entity per failure incident.
        if save_failures and failure_meta:
            for incident in failure_meta:
                stage = str(incident.get("stage") or "unknown")
                reason = str(incident.get("reason") or "")
                extra = {k: v for k, v in incident.items() if k not in {"stage", "reason"}}
                try:
                    card_key = await self._memory.save_failure_card(
                        query=query or "",
                        domain=domain,
                        reason=reason,
                        stage=stage,
                        chain_dict=chain_dict,
                        extra=extra or None,
                    )
                    if card_key:
                        keys["failure_cards"].append(card_key)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("save_failure_card failed for stage %s: %s", stage, exc)

        # Intermediate audit cards — one entity per stage entry.
        if save_intermediate and intermediate_artifacts:
            try:
                ids = await self._memory.save_intermediate_artifacts(
                    intermediate_artifacts,
                    domain=domain,
                    query=query,
                    chain_memory_key=chain_memory_key,
                )
                keys["intermediate_artifacts"].extend(ids)
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning("save_intermediate_artifacts top-level failed: %s", exc)

        return keys

    # ------------------------------------------------------------------
    # Per-stage public entrypoints (§4 P1)
    #
    # CARE re-runs individual pipeline stages (e.g. "regenerate DAG only")
    # without re-driving the whole `generate()` flow. Each method is a
    # thin wrapper around the existing agent — same models, same
    # prompts, same configuration. Model auto-detection runs on first
    # call; the per-stage `set_retry_callback` from `generate()` is NOT
    # automatically applied, so callers wanting retry visibility on
    # standalone stage runs should wire `progress.on_retry` themselves
    # via the constructor.
    # ------------------------------------------------------------------

    async def analyze_domain(self, query: str) -> DomainAnalysis:
        """Run only the domain-analysis stage.

        Parameters
        ----------
        query : str
            The user's text query.

        Returns
        -------
        DomainAnalysis
            Domain, task type, complexity, key concepts, suggested step count.
        """
        await self._llm.resolve_model()
        return await DomainAnalyzer(self._llm).analyze(query)

    async def plan_steps(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        *,
        memory_digest: str = "",
        web_digest: str = "",
        capability_context: str = "",
        template_skeleton: str = "",
        skill_digest: str = "",
        allowed_step_types: list[str] | None = None,
    ) -> StepPlan:
        """Run only the step-planning stage.

        Lets CARE regenerate just the plan when the user tweaks the
        domain analysis or the capability context.
        """
        await self._llm.resolve_model()
        return await StepPlanner(self._llm).plan(
            query,
            domain_analysis,
            memory_digest=memory_digest,
            web_digest=web_digest,
            capability_context=capability_context,
            template_skeleton=template_skeleton,
            allowed_step_types=allowed_step_types,
            skill_digest=skill_digest,
        )

    async def build_dag(self, plan: StepPlan) -> DAGStructure:
        """Run only the DAG-building stage.

        Useful when CARE wants to re-run the dependency-graph optimiser
        after editing the planned steps.
        """
        await self._llm.resolve_model()
        return await DAGBuilder(self._llm).build(plan)

    async def describe_steps(
        self,
        query: str,
        dag: DAGStructure,
        *,
        allowed_step_types: list[str] | None = None,
    ) -> list[CARLStepSchema]:
        """Run only the step-description stage.

        Streaming + per-step `on_stage_progress` callbacks (§3) are
        propagated from the generator's progress callback exactly as
        they are inside `generate()` — disabled by default so the
        method behaves identically for callers that don't supply a
        custom callback.
        """
        await self._llm.resolve_model()
        on_llm_chunk = getattr(self._progress, "on_llm_chunk", None) if self._config.enable_streaming else None
        on_stage_progress = getattr(self._progress, "on_stage_progress", None)
        return await StepDescriber(self._llm).describe(
            query,
            dag,
            allowed_step_types=allowed_step_types,
            stream_callback=on_llm_chunk,
            on_step=on_stage_progress,
        )

    async def critique_steps(
        self,
        query: str,
        steps: list[CARLStepSchema],
        domain_analysis: DomainAnalysis,
        *,
        threshold: float | None = None,
    ) -> StepCritiqueResult:
        """Run only the step-critique stage.

        Returns a :class:`StepCritiqueResult` with per-step quality
        scores and the list of step numbers that fall below
        ``threshold`` (defaults to ``MAGEConfig.step_critique_threshold``).
        """
        await self._llm.resolve_model()
        effective_threshold = self._config.step_critique_threshold if threshold is None else threshold
        return await StepCriticAgent(self._llm).critique(query, steps, domain_analysis, threshold=effective_threshold)

    async def verify_chain(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain_analysis: DomainAnalysis,
        *,
        max_questions: int | None = None,
    ) -> VerificationResult:
        """Run only the chain-verification stage (Chain-of-Verification).

        Returns a :class:`VerificationResult` containing the
        verification questions, per-question answers, and whether the
        chain passed overall.
        """
        await self._llm.resolve_model()
        effective_max_q = self._config.max_verification_questions if max_questions is None else max_questions
        return await ChainVerifier(self._llm).verify(query, chain_dict, domain_analysis, max_questions=effective_max_q)

    async def refine(
        self,
        query: str,
        chain_dict: dict[str, Any],
        domain_analysis: DomainAnalysis,
        *,
        max_iterations: int | None = None,
        quality_threshold: float | None = None,
    ) -> RefinementResult:
        """Run only the self-refinement loop.

        Returns the post-refinement :class:`RefinementResult` carrying
        the (possibly identical) chain dict + chain JSON + before/after
        quality deltas.
        """
        await self._llm.resolve_model()
        effective_max_iter = self._config.max_refine_iterations if max_iterations is None else max_iterations
        effective_q_threshold = (
            self._config.refine_quality_threshold if quality_threshold is None else quality_threshold
        )
        return await SelfRefiner(self._llm).refine(
            query,
            chain_dict,
            domain_analysis,
            max_iterations=effective_max_iter,
            quality_threshold=effective_q_threshold,
        )

    # ------------------------------------------------------------------
    # Fast mode
    # ------------------------------------------------------------------

    async def _generate_fast(
        self,
        query: str,
        *,
        cancel: asyncio.Event | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Single-shot CARL chain generation.

        Retries up to ``max_retries`` times if the model returns text that
        is not parseable JSON, with a corrective system message appended.

        Returns
        -------
        tuple[dict, str]
            (chain_dict, chain_json)
        """
        self._progress.on_stage_start("fast_generation")
        attempts = max(1, self._config.max_retries + 1)
        last_raw = ""
        last_exc: json.JSONDecodeError | None = None

        # P6.6: lead with the capabilities/step-type guidance so even the
        # one-shot fast path knows every step type + that producing a FILE needs
        # an `agent_skill` step (not an `llm` step) — same toolbox the planner sees.
        fast_system = build_planning_guidance() + "\n" + FAST_MODE_SYSTEM_PROMPT

        for attempt in range(attempts):
            _check_cancel(cancel, "fast_generation")
            system_prompt = fast_system
            user_prompt = f"Task: {query}"
            if attempt > 0:
                system_prompt = (
                    fast_system
                    + "\n\nIMPORTANT: your previous response could not be parsed as JSON. "
                    + "Return ONLY a single JSON object — no prose, no markdown fences, "
                    + "no comments. The first character must be '{' and the last must be '}'."
                )
            try:
                raw = await self._llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    json_mode=True,
                    stage="fast_generation",
                )
                last_raw = raw
                chain_dict: dict[str, Any] = json.loads(raw)
                chain_json = json.dumps(chain_dict, ensure_ascii=False, indent=2)
                self._progress.on_stage_complete("fast_generation", chain_dict)
                return chain_dict, chain_json
            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning(
                    "Fast mode JSON parse failed on attempt %d/%d: %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                continue
            except Exception as exc:
                self._progress.on_error("fast_generation", exc)
                raise

        err = MAGEValidationError(
            f"Fast mode produced invalid JSON after {attempts} attempts: {last_exc}",
            raw_output=last_raw,
        )
        self._progress.on_error("fast_generation", err)
        raise err from last_exc

    async def _bootstrap_demos(self, query: str, domain_analysis: DomainAnalysis) -> str:
        """MIPRO-style bootstrap (P1.7): synth a benchmark, generate candidate
        chains, score them, return the top-K as few-shot demos for the StepPlanner.

        Non-fatal: any failure returns an empty string, so the deep pipeline
        proceeds without demos. Candidates are generated via *fast* mode (a single
        LLM call each), which never recurses into the deep pipeline / this bootstrap.
        """
        try:
            from .benchmark import BenchmarkSynthesizer
            from .bootstrap import DemoBootstrapper, format_demos_for_planner

            benchmark = await BenchmarkSynthesizer(self._llm).synthesize(
                query, domain_analysis, n_cases=self._config.benchmark_n_cases
            )
            if not benchmark.tasks:
                return ""

            candidates: list[dict[str, Any]] = []
            for _ in range(max(1, self._config.demo_bootstrap_candidates)):
                try:
                    candidate, _json = await self._generate_fast(query)
                    candidates.append(candidate)
                except Exception as exc:
                    logger.debug("Demo bootstrap: candidate generation failed: %s", exc)

            demos = await DemoBootstrapper(self._llm).bootstrap(
                query,
                domain_analysis,
                candidates,
                benchmark,
                top_k=self._config.demo_bootstrap_top_k,
                min_pass_rate=self._config.demo_bootstrap_min_pass_rate,
            )
            text = format_demos_for_planner(demos)
            if text:
                logger.info("Demo bootstrap: kept %d demo(s) for StepPlanner", len(demos))
            return text
        except Exception as exc:
            logger.warning("Demo bootstrap failed (non-fatal): %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Deep mode
    # ------------------------------------------------------------------

    async def _run_research_stages(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        *,
        capabilities: CapabilityContext | None,
        plan_will_run: bool,
        stages_completed: list[str],
    ) -> _ResearchOutcome:
        """Run research stages 0a–0e concurrently (WP-4 — ``LAT↓``).

        Memory research / web research / capability lookup / template assembly /
        feedback recall depend only on ``domain_analysis`` and are mutually
        independent, so they run under a single ``asyncio.gather`` instead of
        five sequential awaits. Each is wrapped in its own non-fatal
        try/except; ``on_stage_start`` fires for all up front,
        ``on_stage_complete`` fires from inside each coroutine as it finishes,
        and ``stages_completed`` is appended in a FIXED order afterwards so the
        recorded order stays deterministic regardless of completion order.
        """
        memory_result = MemoryResearchResult()
        web_result = WebResearchResult()
        capability_ctx = CapabilityContext()
        template_skel: TemplateSkeleton | None = None
        feedback_digest = FeedbackDigest()
        memory_hits_used = 0
        web_results_used = 0
        was_cold_start = False
        feedback_recalled: int | None = None

        run_memory = plan_will_run and self._config.enable_memory_research
        run_web = plan_will_run and self._config.enable_web_research
        run_capability = (
            capabilities is None
            and plan_will_run
            and (self._config.enable_capability_lookup or self._config.enable_skill_discovery)
        )
        run_template = plan_will_run and self._config.enable_template_assembly
        run_feedback = plan_will_run and self._config.enable_feedback_loop

        # §2 P1 — caller-supplied capabilities bypass the agent entirely. This is
        # synchronous, so handle it outside the gather.
        if capabilities is not None:
            self._progress.on_stage_start("capability_lookup")
            capability_ctx = capabilities
            self._progress.on_stage_complete("capability_lookup", capability_ctx)
            stages_completed.append("capability_lookup_injected")

        # on_stage_start for every concurrent stage that will run — before gather.
        for stage, will in (
            ("memory_research", run_memory),
            ("web_research", run_web),
            ("capability_lookup", run_capability),
            ("template_assembly", run_template),
            ("feedback_recall", run_feedback),
        ):
            if will:
                self._progress.on_stage_start(stage)

        async def _memory() -> tuple[bool, Any]:
            try:
                res = await MemoryResearchAgent(self._llm).research(
                    query, domain_analysis, self._memory, self._config
                )
                self._progress.on_stage_complete("memory_research", res)
                return True, res
            except Exception as exc:
                logger.warning("Memory research failed (non-fatal): %s", exc)
                self._progress.on_error("memory_research", exc)
                return False, None

        async def _web() -> tuple[bool, Any]:
            try:
                res = await WebResearchAgent(self._llm).research(query, domain_analysis, self._config)
                self._progress.on_stage_complete("web_research", res)
                return True, res
            except Exception as exc:
                logger.warning("Web research failed (non-fatal): %s", exc)
                self._progress.on_error("web_research", exc)
                return False, None

        async def _capability() -> tuple[bool, Any]:
            try:
                skill_web_agent = WebResearchAgent(self._llm) if self._config.enable_skill_web_discovery else None
                cap_agent = CapabilityLookupAgent(
                    self._memory, self._config, llm=self._llm, web_agent=skill_web_agent
                )
                res = await self._cached_stage(
                    "capability",
                    [query, getattr(domain_analysis, "domain", "")],
                    lambda: cap_agent.lookup(query=query, domain_analysis=domain_analysis),
                )
                self._progress.on_stage_complete("capability_lookup", res)
                return True, res
            except Exception as exc:
                logger.warning("Capability lookup failed (non-fatal): %s", exc)
                self._progress.on_error("capability_lookup", exc)
                return False, None

        async def _template() -> tuple[bool, Any]:
            try:
                res = await TemplateAssemblerAgent(self._memory, self._config).assemble(
                    domain_analysis.domain, domain_analysis.task_type, domain_analysis.complexity
                )
                self._progress.on_stage_complete("template_assembly", res)
                return True, res
            except Exception as exc:
                logger.warning("Template assembly failed (non-fatal): %s", exc)
                self._progress.on_error("template_assembly", exc)
                return False, None

        async def _feedback() -> tuple[bool, Any]:
            try:
                res = await FeedbackRecallAgent(self._llm, self._memory).recall(
                    query, domain_analysis, self._config
                )
                self._progress.on_stage_complete("feedback_recall", res)
                return True, res
            except Exception as exc:
                logger.warning("Feedback recall failed (non-fatal): %s", exc)
                self._progress.on_error("feedback_recall", exc)
                return False, None

        async def _skip() -> tuple[bool, Any]:
            return False, None

        (mem_ok, mem_res), (web_ok, web_res), (cap_ok, cap_res), (tpl_ok, tpl_res), (fb_ok, fb_res) = (
            await asyncio.gather(
                _memory() if run_memory else _skip(),
                _web() if run_web else _skip(),
                _capability() if run_capability else _skip(),
                _template() if run_template else _skip(),
                _feedback() if run_feedback else _skip(),
            )
        )

        # Fixed-order accumulation → deterministic stages_completed.
        if mem_ok:
            memory_result = mem_res
            memory_hits_used = memory_result.hits_count
            was_cold_start = memory_result.was_cold_start
            stages_completed.append("memory_research")
        if web_ok:
            web_result = web_res
            web_results_used = len(web_result.sources)
            stages_completed.append("web_research")
        if cap_ok:
            capability_ctx = cap_res
            stages_completed.append("capability_lookup")
        if tpl_ok:
            template_skel = tpl_res
            stages_completed.append("template_assembly")
        if fb_ok:
            feedback_digest = fb_res
            feedback_recalled = feedback_digest.recalled_count
            stages_completed.append("feedback_recall")

        return _ResearchOutcome(
            memory_result=memory_result,
            web_result=web_result,
            capability_ctx=capability_ctx,
            template_skel=template_skel,
            feedback_digest=feedback_digest,
            memory_hits_used=memory_hits_used,
            web_results_used=web_results_used,
            was_cold_start=was_cold_start,
            feedback_recalled=feedback_recalled,
        )

    async def _generate_deep(
        self,
        query: str,
        *,
        cancel: asyncio.Event | None = None,
        capabilities: CapabilityContext | None = None,
        return_candidates: bool = False,
    ) -> (
        tuple[dict[str, Any], str, list[str], int, int, bool, dict[str, Any]]
        | list[tuple[dict[str, Any], str, list[str], int, int, bool, dict[str, Any]]]
    ):
        """Agentic multi-stage CARL chain generation.

        Parameters
        ----------
        return_candidates : bool
            When ``True`` and the planner sampled multiple topology
            candidates (``enable_parallel_topology_sampling`` with ≥2
            distinct hints), each candidate plan is carried independently
            through DAG → describe → validation (WP-6) instead of being
            collapsed via :meth:`_score_plans`, and a **list** of result
            tuples is returned (one per candidate). Otherwise — and always
            when ``False`` — a single result tuple is returned, identical
            to the legacy behaviour.

        Returns
        -------
        tuple[dict, str, list[str], int, int, bool, dict] | list[tuple]
            ``(chain_dict, chain_json, stages_completed, memory_hits,
            web_results, was_cold_start, quality_meta)`` — a single tuple
            by default, or a list of such tuples when ``return_candidates``
            collapsing was bypassed.
        """
        stages_completed: list[str] = []
        memory_hits_used = 0
        web_results_used = 0
        was_cold_start = False

        # Metadata accumulator set in this prefix (research stage). The other
        # seven `meta_*` accumulators are tail-only and live in
        # `_finalize_chain_from_plan`, initialised per candidate.
        meta_feedback_recalled: int | None = None

        # §7.1 — accumulate per-stage outputs for the replay-artifact
        # bundle. Each value is the agent's `model_dump()` (or a plain
        # list of dumps for collections) so the bundle round-trips
        # through JSON via `MAGEResult.to_care_dict()`.
        intermediate_artifacts: dict[str, Any] = {}

        # 1. Domain analysis
        _check_cancel(cancel, "domain_analysis")
        if self._config.enable_domain_analysis:
            self._progress.on_stage_start("domain_analysis")
            analyzer = DomainAnalyzer(self._llm)
            domain_analysis = await self._cached_stage("domain", query, lambda: analyzer.analyze(query))
            intermediate_artifacts["domain_analysis"] = domain_analysis.model_dump()
            self._progress.on_stage_complete("domain_analysis", domain_analysis)
            stages_completed.append("domain_analysis")
        else:
            domain_analysis = DomainAnalysis()

        # T13 — fast-path routing: a genuinely simple query (post-T2 classification)
        # doesn't need the 7-stage deep pipeline. Route it to a single fast-mode call.
        # Never under return_candidates (parallel sampling wants the full pipeline).
        if not return_candidates and _should_fast_path(self._config, domain_analysis):
            logger.info(
                "Fast-path: simple query (%d steps) → single fast-mode call, skipping deep stages",
                domain_analysis.suggested_step_count,
            )
            self._progress.on_stage_start("fast_path")
            chain_dict, chain_json = await self._generate_fast(query, cancel=cancel)
            self._progress.on_stage_complete("fast_path", chain_dict)
            return chain_dict, chain_json, ["domain_analysis", "fast_path"], 0, 0, False, {}

        # Override preferred_step_types if allowed_step_types is set
        if self._config.allowed_step_types is not None:
            domain_analysis.preferred_step_types = list(self._config.allowed_step_types)

        # P1.10: bias the suggested step count toward simpler chains (AFlow-style)
        # before it reaches the StepPlanner prompt / fallback plan.
        biased_count = _apply_simplicity_bias(domain_analysis.suggested_step_count, self._config.simplicity_bias)
        if biased_count != domain_analysis.suggested_step_count:
            logger.info(
                "Simplicity bias %.2f: suggested_step_count %d → %d",
                self._config.simplicity_bias,
                domain_analysis.suggested_step_count,
                biased_count,
            )
            domain_analysis.suggested_step_count = biased_count

        # P3.2 (flag orthogonality): the research / bootstrap / tool-synthesis
        # stages exist ONLY to feed the StepPlanner; topology also feeds the
        # DAGBuilder. Skip them when nothing downstream will consume their output —
        # otherwise we burn LLM calls whose digests the fallback ignores.
        plan_will_run = self._config.enable_step_planning
        topology_useful = self._config.enable_step_planning or self._config.enable_dag_optimization
        if not plan_will_run:
            logger.info("Step planning disabled — skipping planner-input stages (research/bootstrap/tool-synthesis)")

        # 0.5 Topology selection (P1.15) — pick a structural pattern to constrain
        # planning, so MAGE stops defaulting to linear pipelines. On by default;
        # falls back to an empty hint (free-form planning) on failure.
        topology_hint = ""
        topology_hints: list[str] = []  # P1.15 part 4: distinct hints for parallel sampling

        # T43: the research stages depend only on domain_analysis — the same input
        # as topology selection + tool synthesis below — so kick them off NOW as a
        # task and let them run concurrently with those two (≈20-30% less deep-mode
        # latency). The result is awaited where it's first consumed. There is no
        # cancel point between here and that await, so the task can't be orphaned.
        research_task = asyncio.ensure_future(
            self._run_research_stages(
                query,
                domain_analysis,
                capabilities=capabilities,
                plan_will_run=plan_will_run,
                stages_completed=stages_completed,
            )
        )

        if topology_useful and self._config.enable_topology_selection:
            self._progress.on_stage_start("topology_selection")
            try:
                from .topology import format_topology_hint

                topo_candidates = await self._cached_stage(
                    "topology",
                    [query, getattr(domain_analysis, "domain", ""), self._config.topology_max_candidates],
                    lambda: TopologySelectorAgent(self._llm).select(
                        query, domain_analysis, max_candidates=self._config.topology_max_candidates
                    ),
                )
                best = topo_candidates.best
                if best is not None:
                    topology_hint = format_topology_hint(best)
                    # Distinct formatted hints across candidates (pipeline → "" once),
                    # so parallel sampling explores genuinely different topologies.
                    seen: set[str] = set()
                    for cand in topo_candidates.candidates:
                        h = format_topology_hint(cand)
                        if h not in seen:
                            seen.add(h)
                            topology_hints.append(h)
                    self._progress.on_stage_complete("topology_selection", best)
                    stages_completed.append("topology_selection")
            except Exception as exc:
                logger.warning("Topology selection failed (non-fatal): %s", exc)
                self._progress.on_error("topology_selection", exc)

        # 0.6 Tool synthesis (P1.17 Phase A, optional, OFF by default) — when the
        # query needs a tool that doesn't exist, propose a ToolSpec (text artifact,
        # no code) and surface it to the StepPlanner. Non-fatal.
        proposed_tools_hint = ""
        if plan_will_run and self._config.enable_tool_synthesis:
            self._progress.on_stage_start("tool_synthesis")
            try:
                tool_spec = await ToolSpecGenerator(self._llm).synthesize(query, domain_analysis)
                if tool_spec is not None:
                    proposed_tools_hint = format_tool_spec_for_planner(tool_spec)
                self._progress.on_stage_complete("tool_synthesis", tool_spec)
                if proposed_tools_hint:
                    stages_completed.append("tool_synthesis")
            except Exception as exc:
                logger.warning("Tool synthesis failed (non-fatal): %s", exc)
                self._progress.on_error("tool_synthesis", exc)

        # 0a–0e. Research stages — independent (depend only on domain_analysis), so
        # WP-4 runs them concurrently inside the helper, and T43 overlaps that whole
        # block with topology selection + tool synthesis above by starting it as a
        # task; here we just await the result.
        research = await research_task
        memory_result = research.memory_result
        web_result = research.web_result
        capability_ctx = research.capability_ctx
        template_skel = research.template_skel
        feedback_digest = research.feedback_digest
        memory_hits_used = research.memory_hits_used
        web_results_used = research.web_results_used
        was_cold_start = research.was_cold_start
        meta_feedback_recalled = research.feedback_recalled

        # (0f. Skill Discovery has been folded into 0c. CapabilityLookup
        # — §2 P0 "Pick one source of truth". The planner now reads one
        # unified agent_skills list off `capability_ctx`; the skill digest
        # passed to the planner is `capability_ctx.skills_digest()` below.)

        # Build capability and template context strings for planner
        cap_context_str = ""
        if capability_ctx.tools or capability_ctx.mcp_servers:
            # Synthesized suggestions (P1.3) are labelled so the planner treats
            # them as hints to provide/confirm, not confirmed-available tools.
            synthesized = any(
                isinstance(c, dict) and c.get("source") == "synthesized"
                for c in (*capability_ctx.tools, *capability_ctx.mcp_servers)
            )
            header = (
                "Suggested capabilities (synthesized from the task — provide/confirm as needed):"
                if synthesized
                else "Available capabilities for this environment:"
            )
            def _fmt_caps(items: list) -> str:
                # Readable "- name: description" list (was a raw json blob,
                # hard for the planner to scan). Tolerates name-only dicts.
                lines = []
                for c in items:
                    if isinstance(c, dict):
                        nm = c.get("name") or c.get("tool_name") or "?"
                        ds = (c.get("description") or c.get("summary") or "").strip()
                        lines.append(f"- {nm}: {ds}" if ds else f"- {nm}")
                    else:
                        lines.append(f"- {c}")
                return "\n".join(lines)

            tools_block = _fmt_caps(capability_ctx.tools)
            mcp_block = _fmt_caps(capability_ctx.mcp_servers)
            cap_context_str = f"{header}\n"
            if tools_block:
                cap_context_str += (
                    "Tools (call by EXACT name, via a step with step_type \"tool\" and tool_name=<name>):\n"
                    + tools_block + "\n"
                )
            if mcp_block:
                cap_context_str += "MCP servers:\n" + mcp_block
            cap_context_str = cap_context_str.rstrip()
        skill_digest = capability_ctx.skills_digest()

        tpl_context_str = ""
        if template_skel and template_skel.abstract_dag:
            tpl_context_str = (
                "Template skeleton (suggested DAG structure from past experience):\n"
                "Step types: " + json.dumps(template_skel.step_type_sequence) + "\n"
                "Use as guidance, not as a rigid template."
            )

        # Build feedback context string for planner
        feedback_context_str = ""
        if feedback_digest.digest:
            feedback_context_str = f"\nPast execution feedback:\n{feedback_digest.digest}\n"

        _check_cancel(cancel, "step_planning")
        # 2. Step planning (pass feedback_digest if available)
        # 1.5 Bootstrapped few-shot demos (P1.7, optional) — generate+score
        # candidate chains and keep the best as planner few-shots. Off by default.
        bootstrapped_demos = ""
        if plan_will_run and self._config.enable_demo_bootstrap:
            self._progress.on_stage_start("demo_bootstrap")
            try:
                bootstrapped_demos = await self._bootstrap_demos(query, domain_analysis)
                self._progress.on_stage_complete("demo_bootstrap", bootstrapped_demos)
                if bootstrapped_demos:
                    stages_completed.append("demo_bootstrap")
            except Exception as exc:
                logger.warning("Demo bootstrap stage failed (non-fatal): %s", exc)
                self._progress.on_error("demo_bootstrap", exc)

        # P1.7: curated per-benchmark reference plans from the selected profile —
        # cheap, benchmark-specific few-shots that complement the optional runtime
        # bootstrap. Non-fatal.
        curated_demos = ""
        if plan_will_run and self._config.benchmark_profile:
            try:
                from .profiles import format_profile_demos, get_profile

                curated_demos = format_profile_demos(get_profile(self._config.benchmark_profile))
            except Exception as exc:
                logger.debug("Curated profile demos unavailable (non-fatal): %s", exc)

        if self._config.enable_step_planning:
            self._progress.on_stage_start("step_planning")
            planner = StepPlanner(self._llm)

            # One planning pass with a given topology hint; all other inputs shared.
            def _plan(th: str) -> Any:
                return planner.plan(
                    query,
                    domain_analysis,
                    memory_digest=memory_result.digest + feedback_context_str,
                    web_digest=web_result.digest,
                    capability_context=cap_context_str,
                    template_skeleton=tpl_context_str,
                    allowed_step_types=self._config.allowed_step_types,
                    skill_digest=skill_digest,
                    max_digest_chars=self._config.max_digest_chars,
                    bootstrapped_demos=bootstrapped_demos,
                    topology_hint=th,
                    proposed_tools=proposed_tools_hint,
                    curated_demos=curated_demos,
                )

            # Plan candidates from topology sampling. WP-6: when the caller
            # asked for candidates AND we sampled ≥2 distinct topologies, we
            # carry EACH candidate plan independently through the rest of the
            # pipeline instead of collapsing to one via `_score_plans`.
            plan_candidates: list[StepPlan] | None = None
            if self._config.enable_parallel_topology_sampling and len(topology_hints) > 1:
                # P1.15 part 4: one plan per distinct topology → keep the best
                # (pairwise judge, P1.5). Explores diverse structures, not just the
                # selector's top pick.
                logger.info("Parallel topology sampling over %d topologies", len(topology_hints))
                candidates: list[StepPlan] = await asyncio.gather(*[_plan(h) for h in topology_hints])
                if return_candidates:
                    plan_candidates = candidates
                    step_plan = candidates[0]  # placeholder; each candidate finalized below
                else:
                    step_plan = await self._score_plans(query, domain_analysis, candidates)
            elif memory_result.was_cold_start and self._config.cold_start_candidates > 1:
                # Cold start multi-sampling (same topology hint across candidates).
                candidates = await asyncio.gather(
                    *[_plan(topology_hint) for _ in range(self._config.cold_start_candidates)]
                )
                step_plan = await self._score_plans(query, domain_analysis, candidates)
            else:
                step_plan = await _plan(topology_hint)

            emit_step_planning = True
        else:
            # Fallback: create a minimal plan from domain analysis
            step_plan = self._fallback_step_plan(domain_analysis)
            plan_candidates = None
            emit_step_planning = False

        # WP-6: carry every sampled candidate through DAG → describe →
        # validation in parallel and return one finalized tuple each. The
        # research/domain/topology stages above ran exactly once; only this
        # plan→chain tail fans out. The default (single) path finalizes once,
        # so its behaviour is byte-identical.
        if return_candidates and plan_candidates is not None:
            return list(
                await asyncio.gather(
                    *[
                        self._finalize_chain_from_plan(
                            cand,
                            query=query,
                            domain_analysis=domain_analysis,
                            topology_hint=topology_hint,
                            cancel=cancel,
                            memory_was_cold_start=memory_result.was_cold_start,
                            was_cold_start=was_cold_start,
                            memory_hits_used=memory_hits_used,
                            web_results_used=web_results_used,
                            meta_feedback_recalled=meta_feedback_recalled,
                            emit_step_planning=emit_step_planning,
                            base_artifacts=intermediate_artifacts,
                            base_stages=stages_completed,
                            tools=capability_ctx.tools,
                        )
                        for cand in plan_candidates
                    ]
                )
            )

        single = await self._finalize_chain_from_plan(
            step_plan,
            query=query,
            domain_analysis=domain_analysis,
            topology_hint=topology_hint,
            cancel=cancel,
            memory_was_cold_start=memory_result.was_cold_start,
            was_cold_start=was_cold_start,
            memory_hits_used=memory_hits_used,
            web_results_used=web_results_used,
            meta_feedback_recalled=meta_feedback_recalled,
            emit_step_planning=emit_step_planning,
            base_artifacts=intermediate_artifacts,
            base_stages=stages_completed,
            tools=capability_ctx.tools,
        )
        if return_candidates:
            return [single]
        return single

    async def _finalize_chain_from_plan(
        self,
        step_plan: StepPlan,
        *,
        query: str,
        domain_analysis: DomainAnalysis,
        topology_hint: str,
        cancel: asyncio.Event | None,
        memory_was_cold_start: bool,
        was_cold_start: bool,
        memory_hits_used: int,
        web_results_used: int,
        meta_feedback_recalled: int | None,
        emit_step_planning: bool,
        base_artifacts: dict[str, Any],
        base_stages: list[str],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str, list[str], int, int, bool, dict[str, Any]]:
        """Carry one chosen ``step_plan`` through DAG → describe → validation.

        This is the "tail" of :meth:`_generate_deep`, extracted so each
        topology-sampled candidate plan can be finalized independently and in
        parallel (WP-6). The default single-plan path calls it exactly once, so
        behaviour is unchanged.

        ``base_artifacts`` / ``base_stages`` are the prefix-stage accumulators
        (domain analysis, topology, research, …). They are **copied** per call so
        sibling candidates never share mutable state. The seven tail-only
        ``meta_*`` accumulators are initialised locally here.

        Parameters
        ----------
        step_plan : StepPlan
            The chosen / candidate plan to finalize.
        emit_step_planning : bool
            Whether to record the ``step_planning`` artifact + stage completion
            (true when the StepPlanner ran; false for the fallback plan).
        memory_was_cold_start : bool
            ``memory_result.was_cold_start`` — guards the post-generation
            ``save_ideas`` write.

        Returns
        -------
        tuple
            ``(chain_dict, chain_json, stages_completed, memory_hits,
            web_results, was_cold_start, quality_meta)``.
        """
        # Per-candidate copies so siblings don't share mutable state. Prefix
        # entries (domain_analysis, topology_selection, research, …) are carried
        # in; tail entries are appended here.
        intermediate_artifacts: dict[str, Any] = copy.deepcopy(base_artifacts)
        stages_completed: list[str] = list(base_stages)

        # Tail-only metadata accumulators (set ONLY within this tail).
        meta_step_critique_score: float | None = None
        meta_verification_passed: bool | None = None
        meta_refine_iterations: int | None = None
        meta_refine_quality_delta: float | None = None
        meta_tot_branches_explored: int | None = None
        meta_mcts_simulations_run: int | None = None
        meta_mcts_best_reward: float | None = None

        if emit_step_planning:
            # CARE: persist the chosen plan as an intermediate artifact.
            intermediate_artifacts["step_plan"] = step_plan.model_dump()

            self._progress.on_stage_complete("step_planning", step_plan)
            stages_completed.append("step_planning")

        # P1.10: enforce the operator budget — truncate unjustified runaway plans.
        step_plan = _enforce_step_budget(step_plan, self._config.simplicity_max_steps)

        _check_cancel(cancel, "dag_building")
        # 3. DAG building — ToT or standard
        if self._config.enable_tot_planning:
            self._progress.on_stage_start("tot_planning")
            tot = ToTPlanner(self._llm)
            tot_result = await tot.plan(
                query,
                step_plan,
                domain_analysis,
                num_branches=self._config.tot_branches,
                depth=self._config.tot_depth,
            )
            dag = tot_result.best_dag
            meta_tot_branches_explored = tot_result.branches_explored
            self._progress.on_stage_complete("tot_planning", tot_result)
            stages_completed.append("tot_planning")
        elif self._config.enable_dag_optimization:
            self._progress.on_stage_start("dag_building")
            builder = DAGBuilder(self._llm)
            dag = await builder.build(step_plan, topology_hint=topology_hint)
            intermediate_artifacts["dag"] = dag.model_dump()
            self._progress.on_stage_complete("dag_building", dag)
            stages_completed.append("dag_building")
        else:
            dag = self._fallback_dag(step_plan)

        # P5.8 — emit a lightweight chain_preview once the skeleton/DAG is
        # assembled (BEFORE describe + serialize) so a UI can render the chain
        # forming. Optional hook — probe via getattr for legacy callbacks;
        # best-effort (a failing preview never aborts generation).
        _on_preview = getattr(self._progress, "on_chain_preview", None)
        if callable(_on_preview):
            try:
                _on_preview(
                    {
                        "topology": getattr(dag, "pattern", None),
                        "max_workers": getattr(dag, "max_workers", None),
                        "steps": [
                            {
                                "number": getattr(s, "number", None),
                                "title": getattr(s, "title", ""),
                                "step_type": getattr(s, "step_type", ""),
                                "dependencies": list(
                                    getattr(s, "dependencies", []) or []
                                ),
                            }
                            for s in getattr(dag, "steps", []) or []
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001 — preview is best-effort
                logger.info("on_chain_preview hook failed: %s", exc)

        _check_cancel(cancel, "step_describing")
        # 4. Step descriptions — MCTS or standard
        if self._config.enable_mcts:
            self._progress.on_stage_start("mcts_construction")
            mcts = MCTSConstructor(self._llm)
            mcts_result = await mcts.construct(query, domain_analysis, step_plan, self._config)
            carl_steps = mcts_result.best_chain_steps
            meta_mcts_simulations_run = mcts_result.simulations_run
            meta_mcts_best_reward = mcts_result.best_reward
            self._progress.on_stage_complete("mcts_construction", mcts_result)
            stages_completed.append("mcts_construction")
        elif self._config.enable_step_descriptions:
            self._progress.on_stage_start("step_describing")
            describer = StepDescriber(self._llm)
            # Probe via getattr — user-supplied callbacks may implement
            # only the legacy three methods (on_stage_start, _complete,
            # _error) and the Protocol additions are optional.
            on_llm_chunk = getattr(self._progress, "on_llm_chunk", None) if self._config.enable_streaming else None
            on_stage_progress = getattr(self._progress, "on_stage_progress", None)
            carl_steps = await describer.describe(
                query,
                dag,
                allowed_step_types=self._config.allowed_step_types,
                stream_callback=on_llm_chunk,
                on_step=on_stage_progress,
                iterative=self._config.enable_iterative_step_describing,
                mode=_resolve_describe_mode(self._config, len(dag.steps)),
                tools=tools,
            )
            intermediate_artifacts["described_steps"] = [s.model_dump() for s in carl_steps]
            self._progress.on_stage_complete("step_describing", carl_steps)
            stages_completed.append("step_describing")
        else:
            carl_steps = self._fallback_step_descriptions(dag)

        # 4.5. Step Critique (optional — NEW)
        # T11 — cheap verifier: critique only the risky step types when
        # step_critique_step_types is set (the care_default compromise), else all.
        critique_steps = _select_critique_steps(carl_steps, self._config.step_critique_step_types)
        if self._config.enable_step_critique and critique_steps:
            self._progress.on_stage_start("step_critique")
            critic = StepCriticAgent(self._llm)
            critique_result = await critic.critique(
                query,
                critique_steps,
                domain_analysis,
                threshold=self._config.step_critique_threshold,
                revision_threshold=self._config.step_critic_revision_threshold,
            )
            intermediate_artifacts["critique"] = critique_result.model_dump()
            meta_step_critique_score = critique_result.overall_quality
            if critique_result.steps_needing_revision:
                describer = StepDescriber(self._llm)
                carl_steps = await describer.revise(query, dag, carl_steps, critique_result)
                intermediate_artifacts["revised_steps"] = [s.model_dump() for s in carl_steps]
            self._progress.on_stage_complete("step_critique", critique_result)
            stages_completed.append("step_critique")

        _check_cancel(cancel, "serialization")
        # 5. Serialize to CARL JSON
        _name_query = _clean_query_for_name(query)  # T16: keep memory blocks out of the name
        chain_name = (
            f"{domain_analysis.domain.title()} — {_name_query[:60]}"
            if _name_query
            else domain_analysis.domain.title()
        )
        chain_schema = CARLChainSchema(
            name=chain_name,
            max_workers=dag.max_workers,
            timeout=self._config.default_timeout,
            steps=carl_steps,
        )
        chain_dict = chain_schema.model_dump(exclude_none=True)
        chain_dict["_domain"] = domain_analysis.domain

        # Optional: attach a default RE-PLAN policy. Promote a risk step to a
        # checkpoint first (P1.14) so the policy has a rollback target, then build.
        if self._config.enable_replan_generation:
            _promote_checkpoints(chain_dict.get("steps") or [])
            replan_policy = _build_default_replan_policy(
                chain_dict,
                max_replans=self._config.replan_max_replans,
                action_on_failure=self._config.replan_action_on_failure,
            )
            if replan_policy is not None:
                chain_dict["replan_policy"] = replan_policy

        chain_json = json.dumps(chain_dict, ensure_ascii=False, indent=2)

        # Failure log accumulator (§5 P2). Populated when verifier
        # rejects the chain OR self-refine hits its iteration cap without
        # reaching the quality threshold. Surfaced via quality_meta.
        failure_meta: list[dict[str, Any]] = []

        # 5.5. Chain Verification (optional — NEW)
        if self._config.enable_chain_verification:
            self._progress.on_stage_start("chain_verification")
            verifier = ChainVerifier(self._llm)
            vresult = await verifier.verify(
                query, chain_dict, domain_analysis, max_questions=self._config.max_verification_questions
            )
            intermediate_artifacts["verification"] = vresult.model_dump()
            meta_verification_passed = vresult.chain_passed
            if not vresult.chain_passed:
                failure_meta.append(
                    {
                        "stage": "chain_verification",
                        "reason": "Verifier rejected the chain; auto-fix applied.",
                        "issues_found": list(vresult.issues_found or []),
                        "num_questions": len(vresult.questions or []),
                    }
                )
                chain_dict = await verifier.fix(chain_dict, vresult)
                chain_json = json.dumps(chain_dict, ensure_ascii=False, indent=2)
            self._progress.on_stage_complete("chain_verification", vresult)
            stages_completed.append("chain_verification")

        # 5.6. Self-Refinement (optional — NEW)
        if self._config.enable_self_refine:
            self._progress.on_stage_start("self_refine")
            refiner = SelfRefiner(self._llm)
            refine_result = await refiner.refine(
                query,
                chain_dict,
                domain_analysis,
                max_iterations=self._config.max_refine_iterations,
                quality_threshold=self._config.refine_quality_threshold,
            )
            chain_dict = refine_result.chain_dict
            chain_json = refine_result.chain_json
            intermediate_artifacts["refine"] = refine_result.model_dump()
            meta_refine_iterations = refine_result.iterations_used
            meta_refine_quality_delta = refine_result.final_quality - refine_result.initial_quality
            # Failure: iterations used == cap AND final quality < threshold.
            if (
                refine_result.iterations_used >= self._config.max_refine_iterations
                and refine_result.final_quality < self._config.refine_quality_threshold
            ):
                failure_meta.append(
                    {
                        "stage": "self_refine",
                        "reason": (
                            f"Self-refine exhausted {refine_result.iterations_used} iterations "
                            f"without reaching quality threshold "
                            f"{self._config.refine_quality_threshold:.2f} "
                            f"(final_quality={refine_result.final_quality:.2f})."
                        ),
                        "initial_quality": refine_result.initial_quality,
                        "final_quality": refine_result.final_quality,
                    }
                )
            self._progress.on_stage_complete("self_refine", refine_result)
            stages_completed.append("self_refine")

        # Post-generation: save ideas to memory if cold start
        if memory_was_cold_start and self._config.enable_memory:
            ideas = [{"title": s.title, "aim": s.rough_aim, "type": s.step_type} for s in step_plan.steps]
            await self._memory.save_ideas(ideas, domain=domain_analysis.domain, query=query)

        # Collect quality enhancement metadata
        quality_meta = {
            "step_critique_score": meta_step_critique_score,
            "verification_passed": meta_verification_passed,
            "refine_iterations": meta_refine_iterations,
            "refine_quality_delta": meta_refine_quality_delta,
            "tot_branches_explored": meta_tot_branches_explored,
            "mcts_simulations_run": meta_mcts_simulations_run,
            "mcts_best_reward": meta_mcts_best_reward,
            "feedback_recalled": meta_feedback_recalled,
            # Pass the domain analysis up to `generate()` so it can drive
            # CARE library-ready naming (§7.2). Stripped before reaching
            # `MAGEMetadata`.
            "_domain_analysis": domain_analysis,
            # §5 P2 — failure incidents (verifier rejection / refine cap
            # hit). Stripped from MAGEMetadata; read by
            # `_persist_ecosystem_artifacts` when failure-card save is on.
            "_failure_meta": failure_meta,
            # §7.1 — per-stage outputs captured during deep generation,
            # surfaced onto `MAGEResult.intermediate_artifacts` for
            # `to_care_dict()`.
            "_intermediate_artifacts": intermediate_artifacts,
        }

        return (
            chain_dict,
            chain_json,
            stages_completed,
            memory_hits_used,
            web_results_used,
            was_cold_start,
            quality_meta,
        )

    # ------------------------------------------------------------------
    # Cold start plan scoring
    # ------------------------------------------------------------------

    async def _score_plans(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        candidates: list[StepPlan],
    ) -> StepPlan:
        """Select the best candidate plan via a pairwise tournament (P1.5).

        Reasoning models compare two options far more reliably than they score
        many options against an absolute rubric, so instead of one multi-way
        scoring call we run a single-elimination tournament: the incumbent best
        is compared head-to-head against each remaining candidate (N-1 calls),
        and the winner advances. A failed comparison keeps the incumbent.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        candidates : list[StepPlan]
            Candidate step plans to score.

        Returns
        -------
        StepPlan
            The tournament winner.
        """
        if len(candidates) == 1:
            return candidates[0]

        best = candidates[0]
        for challenger in candidates[1:]:
            best = await self._pick_better_plan(query, domain_analysis, best, challenger)
        return best

    async def _pick_better_plan(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        plan_a: StepPlan,
        plan_b: StepPlan,
    ) -> StepPlan:
        """Compare two plans head-to-head; return the winner (default: incumbent A)."""
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Plan A:\n{json.dumps(plan_a.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Plan B:\n{json.dumps(plan_b.model_dump(), ensure_ascii=False, indent=2)}"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=PLAN_PAIRWISE_PROMPT,
                user_prompt=context,
                stage="plan_scoring",
            )
            winner = str(data.get("winner", "A")).strip().upper()
            return plan_b if winner == "B" else plan_a
        except Exception as exc:
            logger.warning("Pairwise plan comparison failed, keeping incumbent: %s", exc)
            return plan_a

    # ------------------------------------------------------------------
    # CARE library-ready naming (§7.2)
    # ------------------------------------------------------------------

    async def _generate_suggested_naming(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        capability_tags: list[str],
        *,
        fallback_display_name: str,
        fallback_description: str,
        fallback_tags: list[str],
    ) -> tuple[str, str, list[str]]:
        """Deep-mode LLM call producing CARE library-ready naming.

        Failure is non-fatal — any exception (incl. ``StopAsyncIteration``
        from exhausted test mocks) falls back to the heuristic triple
        passed in.
        """
        context = (
            f"User query: {_clean_query_for_name(query)}\n\n"  # T16: name from the task, not memory
            f"Domain analysis:\n"
            f"{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Detected capability tags: {json.dumps(capability_tags)}"
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=SUGGESTED_NAMING_PROMPT,
                user_prompt=context,
                stage="suggested_naming",
            )
        except Exception as exc:
            logger.warning("Suggested naming LLM call failed, using heuristic: %s", exc)
            return fallback_display_name, fallback_description, fallback_tags

        return _coerce_naming_payload(
            data,
            fallback_display_name=fallback_display_name,
            fallback_description=fallback_description,
            fallback_tags=fallback_tags,
        )

    # ------------------------------------------------------------------
    # Fallbacks (when agents are disabled)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_step_plan(domain: DomainAnalysis) -> StepPlan:
        """Create a minimal but MEANINGFUL step plan when StepPlanner is disabled (P3.1).

        Uses a domain/task-type-keyed template (e.g. extraction → Parse/Extract/
        Validate/Emit) sized to ``suggested_step_count``, instead of useless
        ``Step N / Process part N`` placeholders. The meaningful titles + aims
        propagate through ``_fallback_dag`` and ``_fallback_step_descriptions``.
        """
        count = domain.suggested_step_count
        fitted = _fit_template(_select_fallback_template(domain), count)
        steps = [
            PlannedStep(number=i + 1, title=title, step_type="llm", rough_aim=aim)
            for i, (title, aim) in enumerate(fitted)
        ]
        label = domain.task_type or domain.domain or "general"
        return StepPlan(steps=steps, rationale=f"Fallback plan ({label} template; step planning disabled)")

    @staticmethod
    def _fallback_dag(plan: StepPlan) -> DAGStructure:
        """Create a simple pipeline DAG when DAGBuilder is disabled."""
        dag_steps = []
        for i, s in enumerate(plan.steps):
            dag_steps.append(
                DAGStep(
                    number=s.number,
                    title=s.title,
                    step_type=s.step_type,
                    dependencies=[plan.steps[i - 1].number] if i > 0 else [],
                    rough_aim=s.rough_aim,
                )
            )
        return DAGStructure(steps=dag_steps, max_workers=1, pattern="pipeline")

    @staticmethod
    def _fallback_step_descriptions(dag: DAGStructure) -> list[CARLStepSchema]:
        """Create minimal step descriptions when StepDescriber is disabled."""
        return [
            CARLStepSchema(
                # A tool node with no describer to fill its tool_name/args can't be
                # a real call — degrade it to an llm reasoning step (T23 keeps tool
                # steps required to name a tool).
                step_type="llm" if s.step_type == "tool" else s.step_type,
                number=s.number,
                title=s.title,
                aim=s.rough_aim or f"Complete {s.title}",
                reasoning_questions=f"What is needed to {s.rough_aim.lower()}?" if s.rough_aim else "",
                step_context_queries=[],
                stage_action=s.rough_aim or s.title,
                example_reasoning="",
                dependencies=s.dependencies,
            )
            for s in dag.steps
        ]
