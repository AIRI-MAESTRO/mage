"""
Pydantic models for MAGE input, output, and intermediate data structures.

Defines the data contracts used throughout the MAGE pipeline:
- MAGEConfig: configuration for generation
- MAGEResult / MAGEMetadata: output contract
- DomainAnalysis, StepPlan, DAGStructure: intermediate deep-mode structures
- CARL JSON schema helpers
- MemoryHit, MemoryResearchResult, WebResearchResult: research results
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid step types
# ---------------------------------------------------------------------------

# Step types accepted by the current `mmar_carl` runtime (agent-features
# branch — incl. agent_skill / evaluation / parallel_sampling / etc.).
# Keep in sync with `mmar_carl.models.enums.StepType`.
CARL_VALID_STEP_TYPES: frozenset[str] = frozenset(
    [
        # Stable / pre-agent-features types
        "llm",
        "tool",
        "mcp",
        "memory",
        "transform",
        "conditional",
        "structured_output",
        # agent-features additions
        "agent_skill",  # AgentSkills format skill execution
        "evaluation",  # inline quality gate
        "agent_handoff",  # delegate to a complete sub-chain
        "parallel_sampling",  # sample N independent LLM responses + aggregate
        "tool_discovery",  # discover and register tools at runtime
        "human_input",  # pause for human input
        "supervisor",  # LLM routes a task to one of N sub-chains
        "debate",  # round-robin multi-agent debate + judge
        "mcp_resource",  # read a named MCP resource into memory/history
    ]
)

# MAGE-internal step types — what intermediate stages emit.
# Includes CARL-valid types PLUS MAGE-only semantic labels ("eval", "aggregator")
# that are translated into CARL-loadable forms by `to_carl_compat()`.
VALID_STEP_TYPES: frozenset[str] = CARL_VALID_STEP_TYPES | frozenset(["eval", "aggregator"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class StageLLMOverride(BaseModel):
    """Typed per-stage LLM configuration (§6 P2).

    Replaces the previous ``dict[str, dict]`` shape of
    ``MAGEConfig.stage_llm_overrides``, eliminating silent typos like
    ``temprature`` that would otherwise be ignored. Backwards-compatible:
    ``MAGEConfig.stage_llm_overrides`` accepts bare dicts and a field
    validator promotes them into ``StageLLMOverride`` instances.

    Parameters
    ----------
    model : str | None
        Model identifier to use for the stage (defaults to
        ``MAGEConfig.model`` when ``None``).
    temperature : float | None
        Sampling temperature for the stage (defaults to
        ``MAGEConfig.temperature`` when ``None``).
    """

    model_config = {"extra": "forbid"}  # reject unknown keys to catch typos

    model: str | None = None
    temperature: float | None = None


class MAGEConfig(BaseModel):
    """Configuration for MAGE chain generation.

    All deep-mode sub-stages can be toggled individually, which is useful for
    debugging or running only part of the pipeline.

    Flag orthogonality (P3.2)
    -------------------------
    The research / bootstrap / tool-synthesis stages exist solely to feed the
    StepPlanner, so they are **skipped automatically when**
    ``enable_step_planning=False`` (their digests would otherwise be computed at
    LLM cost and then ignored by the fallback plan). This applies to
    ``enable_memory_research``, ``enable_web_research``, ``enable_capability_lookup``,
    ``enable_template_assembly``, ``enable_feedback_loop``, ``enable_skill_discovery``,
    ``enable_demo_bootstrap``, and ``enable_tool_synthesis``. ``enable_topology_selection``
    is skipped only when **both** ``enable_step_planning`` and ``enable_dag_optimization``
    are off (the topology hint also informs the DAGBuilder).

    Parameters
    ----------
    mode : Literal["fast", "deep"]
        Generation mode.  ``"deep"`` runs the full agentic pipeline;
        ``"fast"`` performs a single LLM call.
    model : str
        Model identifier in ``provider/model`` format (OpenRouter style) or
        plain model name for direct OpenAI usage.
    temperature : float
        Sampling temperature for LLM calls.
    api_key : str | None
        API key.  Falls back to ``MAGE_API_KEY`` / ``OPENAI_API_KEY`` env vars.
    base_url : str | None
        Override base URL for OpenAI-compatible endpoints.
    enable_domain_analysis : bool
        Run the DomainAnalyzer agent in deep mode.
    enable_step_planning : bool
        Run the StepPlanner agent in deep mode.
    enable_dag_optimization : bool
        Run the DAGBuilder agent in deep mode.
    enable_step_descriptions : bool
        Run the StepDescriber agent in deep mode.
    enable_memory : bool
        Save the generated chain to gigaevo-memory after generation.
    memory_base_url : str
        Base URL of the gigaevo-memory server.
    memory_tags : list[str]
        Additional tags to attach to the memory entry.
    default_max_workers : int
        Default ``max_workers`` for generated chains.
    default_timeout : float
        Default ``timeout`` (seconds) for generated chains.
    max_steps : int
        Upper bound on number of steps the generator may produce.
    min_steps : int
        Lower bound on number of steps.
    max_retries : int
        How many times to retry a failed LLM call.
    provider : Literal["openai", "openrouter", "local", "custom"]
        LLM provider type.
    ssl_verify : bool
        Whether to verify SSL certificates for HTTP connections.
    http_timeout : float
        HTTP request timeout in seconds.
    openrouter_site_url : str | None
        Site URL header for OpenRouter requests.
    openrouter_app_name : str | None
        App name header for OpenRouter requests.
    enable_memory_research : bool
        Enable memory research stage in deep mode.
    memory_recall_top_k : int
        Number of top results to recall from memory search.
    memory_relevance_threshold : float
        Minimum similarity score for memory hits.
    memory_research_sub_queries : int
        Number of sub-queries to generate for memory research.
    enable_web_research : bool
        Enable web research stage in deep mode.
    web_search_provider : str
        Web search provider (tavily, serpapi, brave).
    web_search_api_key : str | None
        API key for the web search provider.
    web_research_max_results : int
        Maximum number of web search results per sub-query.
    web_research_sub_queries : int
        Number of sub-queries for web research.
    cold_start_candidates : int
        Number of plan candidates for cold-start multi-sampling.
    stage_llm_overrides : dict[str, dict]
        Per-stage LLM configuration overrides.
    """

    mode: Literal["fast", "deep"] = "deep"

    # Benchmark profile (P3.5): authoring-time preset for a headline benchmark.
    # When set, the profile's field overrides fill any config field NOT explicitly
    # provided (see _apply_benchmark_profile). Out of scope: benchmark execution.
    benchmark_profile: Literal["bfcl", "seal_tools", "rotbench", "toolalpaca", "custom"] | None = None

    # LLM
    model: str = "gpt-4o"
    temperature: float = 0.2
    api_key: str | None = None
    base_url: str | None = None

    # Provider
    provider: Literal["openai", "openrouter", "local", "custom"] = "openai"
    ssl_verify: bool = True
    http_timeout: float = 60.0
    openrouter_site_url: str | None = None
    openrouter_app_name: str | None = None

    # Deep-mode toggles
    enable_domain_analysis: bool = True
    enable_step_planning: bool = True
    enable_dag_optimization: bool = True
    enable_step_descriptions: bool = True
    # P1.5: describe steps ONE AT A TIME, feeding the fully-described prior
    # steps as context so step N can reference step N-1's real output / keys /
    # format. Costs N LLM calls instead of 1; default off (batch describe).
    enable_iterative_step_describing: bool = False
    # WP-7: step-description strategy. "batch" = one call for the whole chain
    # (default); "iterative" = N sequential calls (== enable_iterative_step_describing);
    # "parallel" = describe DAG levels concurrently, one small per-step call each
    # (only that step's type section in the prompt). LAT↓ on multi-level DAGs.
    # "auto" (T44) = iterative for a long chain on a model WITHOUT structured output
    # (where one batch JSON reply tends to break on flash models), else batch.
    step_describe_mode: Literal["batch", "iterative", "parallel", "auto"] = "batch"

    # Topology selection (P1.15): stage 0.5 picks a structural topology (pipeline/
    # diamond/debate/…) after domain analysis and constrains the StepPlanner +
    # DAGBuilder toward it. OFF by default in the CARE-integrated build so the CARE
    # pipeline's call sequence is unchanged unless opted in; adds one LLM call per
    # deep generation when on; falls back to pipeline on failure.
    # ``topology_max_candidates`` caps how many ranked picks are kept.
    enable_topology_selection: bool = False
    topology_max_candidates: int = 3
    # P1.15 part 4: when the selector returns >1 distinct topology, fan out one
    # planning pass per topology and keep the best (pairwise judge, P1.5). A K×
    # planning-token multiplier, so OFF by default — turn on for hard benchmarks.
    enable_parallel_topology_sampling: bool = False

    # Tool synthesis (P1.17 Phase A): when a query needs a tool that doesn't exist,
    # detect that and synthesize a *proposed* ToolSpec (text artifact, no code
    # execution) surfaced to the StepPlanner. OFF by default — opting in is the
    # security discipline (Phases B/C generate+sandbox real code). Phase A alone
    # runs no code, but the gate covers the whole feature.
    enable_tool_synthesis: bool = False

    # Memory
    enable_memory: bool = True
    memory_base_url: str = "http://localhost:8002"
    memory_tags: list[str] = Field(default_factory=list)

    # Deep Research: Memory
    enable_memory_research: bool = False
    memory_recall_top_k: int = 5
    memory_relevance_threshold: float = 0.4
    memory_research_sub_queries: int = 3

    # Retrieval mode (P1.18): how MemoryManager queries gigaevo-memory.
    # "hybrid" (BM25 + vector) gives the best recall when the server has vector
    # search enabled; MemoryManager._search auto-falls-back to BM25 per call if
    # the server rejects the mode. memory_namespace scopes recall to a curated
    # partition (None = no scoping).
    memory_search_mode: Literal["bm25", "vector", "hybrid"] = "hybrid"
    memory_hybrid_weights: tuple[float, float] = (0.5, 0.5)
    memory_namespace: str | None = None
    # T33: opaque session id. When set, MemoryResearchAgent recalls facts saved by
    # earlier runs in this session (MemoryManager.save_run_facts) first — so a
    # follow-up question reuses them instead of cold-starting. None disables it.
    memory_session_id: str | None = None

    # Applicability gate (P1.1): after recall, an LLM judge rates each hit's
    # applicability to the current query (0-1); hits below the threshold are
    # dropped before the digest. Guards against superficial keyword matches
    # polluting the planner's context. Gate failures are non-fatal.
    enable_memory_applicability_gate: bool = True
    memory_applicability_threshold: float = 0.5

    # Second-wave recall (P1.1): when the applicability gate leaves fewer than
    # this many hits, MemoryResearchAgent generates gap-targeting sub-queries and
    # recalls again before declaring a cold start. 0 disables the second wave.
    memory_min_applicable_hits: int = 2

    # Deep Research: Internet
    enable_web_research: bool = False
    web_search_provider: str = "tavily"
    web_search_api_key: str | None = None
    web_research_max_results: int = 5
    web_research_sub_queries: int = 2

    # Source-grounded second wave (P1.2): if the first web digest cites fewer
    # than this many distinct sources, WebResearchAgent runs a gap-targeting
    # second wave of searches and re-digests. 0 disables the second wave.
    web_min_sourced_claims: int = 3

    # Benchmark mode (P1.16): self-verify each generated chain against a
    # synthesized mini-benchmark (stage 7) and attach a quality_report to the
    # result. Off by default (adds LLM calls). benchmark_n_cases = test cases.
    enable_benchmark_mode: bool = False
    benchmark_n_cases: int = 5

    # Execution-feedback loop (P1.9): verify the chain against a synthesized
    # benchmark, repair the weakest chain when it falls below the threshold, and
    # re-verify — looping up to N times, keeping the best. Closes the
    # run-once / author-once gap. Off by default (adds LLM calls); supersedes the
    # single-shot benchmark verification when on.
    enable_execution_feedback: bool = False
    execution_feedback_threshold: float = 0.7
    max_execution_feedback_iterations: int = 2

    # Bootstrapped few-shot demos (P1.7, MIPRO-style): before the authoritative
    # StepPlanner call, generate ``demo_bootstrap_candidates`` candidate chains
    # (fast mode), score them on a synthesized benchmark, and inject the top-K
    # (pass-rate ≥ ``demo_bootstrap_min_pass_rate``) as planner few-shots. Off by
    # default (adds LLM calls — candidates × (1 gen + n_cases judge) + 1 synth).
    enable_demo_bootstrap: bool = False
    demo_bootstrap_candidates: int = 4
    demo_bootstrap_top_k: int = 2
    demo_bootstrap_min_pass_rate: float = 0.5

    # Simplicity bias (P1.10, AFlow-style): bias the generator toward simpler
    # chains. ``suggested_step_count`` from domain analysis is scaled by
    # ``(1 - simplicity_bias)`` (floored at 2) before it reaches the StepPlanner,
    # and any plan exceeding ``simplicity_max_steps`` *without* a filled rationale
    # is truncated back to the cap. AFlow wins on bfcl_v3 with ≤4-operator chains,
    # so this defaults ON. Set ``simplicity_bias=0.0`` to disable.
    simplicity_bias: float = 0.5
    simplicity_max_steps: int = 7

    # Cold start
    cold_start_candidates: int = 1

    # Research-digest hygiene: cap each digest (memory/web/skill/capability/
    # template) handed to the StepPlanner to this many characters (~4 chars
    # per token, so 1200 ≈ 300 tokens). Prevents unbounded digests from
    # bloating the planner prompt and diluting the query. 0 disables capping.
    max_digest_chars: int = 1200

    # CARL load policy (P0.3): MAGE's own validate_carl_json(strict_carl=True)
    # is the authority for chain validity. ReasoningChain.from_json is a
    # best-effort cross-check that currently lags its own runtime (it cannot
    # round-trip several agent-features step types from JSON — see P0.2). When
    # False (default), a from_json failure on an already-strict-valid chain is
    # logged as a warning and generation continues. Set True to hard-fail.
    raise_on_carl_load_error: bool = False

    # Per-stage LLM override (§6 P2 — typed for typo-safety).
    stage_llm_overrides: dict[str, StageLLMOverride] = Field(default_factory=dict)

    # §6 P2 — fallback for local providers that don't honor
    # `response_format={"type": "json_object"}`. When True, MAGE skips
    # the JSON-mode flag and appends a strict-JSON instruction to the
    # prompt; `generate_json` then extracts the first valid JSON
    # object via regex. Auto-flips on when `provider == "local"` and
    # the user hasn't explicitly set it.
    json_mode_fallback: bool | None = None
    # P-7: when True and the provider supports it, structured stages pass their
    # pydantic schema via response_format={"type":"json_schema",...} (vLLM guided
    # decoding / OpenRouter structured outputs) instead of plain json_object —
    # cuts JSON-parse retries. Opt-in (default off); ignored on the local
    # json_mode_fallback path (which sends no response_format at all).
    use_schema_constrained_output: bool = False
    # T8: on a JSON parse failure, re-request the SAME stage this many extra
    # times before raising — a single noisy reply re-runs only that stage's LLM
    # call, not the whole 7-stage pipeline (runs 731/773). 0 disables.
    json_parse_retries: int = 1
    # T42: cache the deterministic deep-mode stages (domain / topology / capability)
    # by a hash of their input, so the same query within a session or a pipeline
    # retry reuses the result instead of recomputing it. Off by default.
    enable_stage_cache: bool = False
    # T13: in deep mode, route a genuinely simple query (DomainAnalyzer says
    # complexity="simple" and <= fast_path_max_steps) to a single fast-mode call
    # instead of the full 7-stage pipeline (deep spent 88-144s on "what's my IP").
    enable_fast_path_routing: bool = False
    fast_path_max_steps: int = 2

    # Capability lookup — single source of truth for the planner's
    # capability surface (§2 P0). Merges tools/MCP servers, memory
    # AgentSkill entries, locally-installed skills, and SkillDiscovery
    # recommendations into one CapabilityContext.
    enable_capability_lookup: bool = False
    capability_environment_id: str = "default"
    # On a registry miss (P1.3): synthesize plausible tool/MCP suggestions from
    # the query via an LLM, so the planner has a stronger seed than nothing.
    # Suggestions are clearly labelled (the planner may need to provide them).
    enable_capability_synthesis: bool = True

    # AgentSkill source 1 — memory. Queries gigaevo-client's
    # MemoryClient.find_capability_matches for skills ranked by the
    # chain's rough aim. Off by default because it needs gigaevo-client
    # installed AND the memory server populated with `agent_skill`
    # entities.
    enable_memory_skill_lookup: bool = False
    memory_skill_top_k: int = 5

    # AgentSkill source 2 — local catalog. Catalogs skills installed on
    # the host via mmar_carl.skill_loader.SkillLoader.catalog_all. Off
    # by default because it requires the local skills directory
    # convention to be set up.
    enable_local_skill_catalog: bool = False

    # AgentSkill source 3 — discovery. Now invoked INSIDE the unified
    # CapabilityLookupAgent rather than as a separate stage (§2 P0
    # "fold SkillDiscoveryAgent"). Built-in registry pre-filter + LLM
    # rerank + optional web fallback.
    enable_skill_discovery: bool = False
    skill_discovery_max_recommendations: int = 3
    # Off by default — only enable when web_search_api_key is set AND you
    # want to find skills not in the built-in registry.
    enable_skill_web_discovery: bool = False

    # Memory write-back for ecosystem reuse (§5 P1). Off by default
    # because both helpers need either gigaevo-client or
    # gigaevo-memory installed AND a writable memory server.
    enable_skill_provenance_save: bool = False
    """When True, every `agent_skill` step in a finalized chain is
    persisted via `MemoryManager.save_agent_skill` so subsequent runs
    can find it. Idempotent — duplicate SHAs are no-oped by the SDK."""
    enable_reusable_step_save: bool = False
    """When True, every reusable LLM step (no dynamic refs) in a
    finalized chain is persisted via `MemoryManager.save_reusable_step`
    so `TemplateAssemblerAgent` can recall it later."""

    # §5 P2 — failure-card + evolution_meta writes.
    enable_failure_card_save: bool = False
    """When True, a `memory_card` describing the failure is persisted
    via `MemoryManager.save_failure_card` whenever chain verification
    fails OR self-refine hits its iteration cap. FeedbackRecallAgent
    surfaces these on similar future queries."""
    enable_evolution_meta_save: bool = False
    """When True, the chain row carries an `evolution_meta` block
    (step_critique_score, verification_passed, refine_quality_delta,
    mcts_best_reward, etc.) so the platform's evolution engine can use
    them as fitness priors."""

    # §4 P2 — intermediate-artifact persistence. When enabled, every
    # entry in the generator's `intermediate_artifacts` dict is saved
    # as a separate `memory_card` tagged `mage_intermediate` so a CARE
    # audit UI can reconstruct the per-stage history of any chain.
    enable_intermediate_artifact_save: bool = False

    # §6 P3 — pre-flight cost estimate. When enabled the generator
    # builds a per-stage cost estimate before the first LLM call
    # fires and surfaces it via `MAGEProgressCallback.on_cost_estimate`
    # so CARE can show a "this will cost ~$X" panel.
    enable_preflight_cost_estimate: bool = False
    # Model name → (input_per_1k_usd, output_per_1k_usd). When None
    # the estimator returns token counts but cost fields stay at 0.
    pricing: dict[str, tuple[float, float]] | None = None

    # RE-PLAN policy generation (agent-features integration).
    # When enabled, StepDescriber is instructed to mark critical steps as
    # checkpoints (checkpoint=true + checkpoint_name), and a default
    # rule-based ReplanPolicy is attached at chain level. The runtime then
    # rolls back to nearest checkpoint on failure.
    enable_replan_generation: bool = False
    replan_max_replans: int = 3
    replan_action_on_failure: str = "replan_from_checkpoint"  # see ReplanAction enum

    # Step memoization (P2.2): auto-mark read-only steps (MCP resources) with a
    # result cache TTL so repeated fetches are memoized. On by default — safe,
    # since only side-effect-free reads are auto-cached. step_cache_default_ttl
    # is the TTL (seconds) applied.
    enable_step_caching: bool = True
    step_cache_default_ttl: int = 300

    # Template assembly
    enable_template_assembly: bool = False

    # Output format
    output_format: str = "json"  # "json" | "python" | "both"

    # Chain defaults
    default_max_workers: int = 2
    default_timeout: float = 300.0
    max_steps: int = 10
    min_steps: int = 2
    max_retries: int = 2

    # Quality Enhancement: Step-level Self-Critique
    enable_step_critique: bool = False
    # T11: when set, the step critic only sees these step types (e.g.
    # ["tool", "structured_output"]) instead of the whole chain — the cheap
    # care_default verifier that checks just the risky steps. None = critique all.
    step_critique_step_types: list[str] | None = None
    step_critique_threshold: float = 0.6
    # P1.11: a step is also flagged for revision when a *critical* dimension
    # (aim_clarity / action_specificity) falls below this threshold, even if its
    # averaged overall_score passes — averaging can mask a load-bearing failure.
    step_critic_revision_threshold: float = 0.6

    # Quality Enhancement: Chain of Verification
    enable_chain_verification: bool = False
    max_verification_questions: int = 3

    # Quality Enhancement: Iterative Self-Refinement
    enable_self_refine: bool = False
    max_refine_iterations: int = 3
    refine_quality_threshold: float = 0.8

    # Quality Enhancement: Tree of Thoughts DAG Planning
    enable_tot_planning: bool = False
    tot_branches: int = 3
    tot_depth: int = 2

    # Quality Enhancement: MCTS Chain Construction
    enable_mcts: bool = False
    mcts_simulations: int = 8
    mcts_exploration: float = 1.41
    mcts_max_depth: int = 8
    mcts_progressive_widening: bool = True

    # Quality Enhancement: Feedback Loop
    enable_feedback_loop: bool = False
    feedback_recall_top_k: int = 3
    feedback_min_quality: float = 0.3

    # CARE TUI streaming (§3 P0). When enabled, the generator forwards
    # the progress callback's `on_llm_chunk` into the underlying LLM
    # client so chunked deltas are surfaced as they arrive. Off by
    # default because most mock-based tests + the CLI don't render
    # streaming output. CARE flips this to True when running.
    enable_streaming: bool = False

    # CARE library-ready naming (§7.2). Deep mode uses a dedicated LLM call
    # to produce a punchy `suggested_display_name` + 1-2 sentence
    # `suggested_description`. Disable to fall back to the heuristic
    # (`"{Domain.title()} — {first 50 chars of query}"`) and save an LLM
    # round-trip.
    enable_suggested_naming_llm: bool = True

    # Applicability gate for recalled feedback (P1.3): an LLM judge rates how
    # relevant each past-execution lesson is to the *current* query; records
    # below the threshold are dropped before the digest, so feedback matched
    # only by domain/task-type doesn't mislead the planner. Non-fatal.
    enable_feedback_applicability_gate: bool = True
    feedback_applicability_threshold: float = 0.5

    # Step type restriction
    allowed_step_types: list[str] | None = Field(
        default=None,
        description="If set, restricts generation to only these step types. "
        "None means all types allowed. Example: ['llm'] for pure LLM-only chains.",
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_benchmark_profile(cls, data: Any) -> Any:
        """P3.5: fill config fields from the selected benchmark profile.

        Runs before field validation, so profile values are validated normally
        and the after-validators (allowed-types check, cold-start bump) see them.
        Only fills fields the user did NOT pass explicitly.
        """
        if isinstance(data, dict):
            profile_name = data.get("benchmark_profile")
            if profile_name:
                from .profiles import get_profile

                profile = get_profile(profile_name)
                if profile is not None:
                    for field, value in profile.config.items():
                        data.setdefault(field, value)  # explicit user values win
        return data

    @model_validator(mode="after")
    def _validate_allowed_step_types(self) -> MAGEConfig:
        if self.allowed_step_types is not None:
            invalid = set(self.allowed_step_types) - VALID_STEP_TYPES
            if invalid:
                msg = f"Invalid step types: {sorted(invalid)}. Valid types: {sorted(VALID_STEP_TYPES)}"
                raise ValueError(msg)
            if self.allowed_step_types == ["llm"]:
                logger.info("LLM-only mode is active: chains will contain only 'llm' step types")
        return self

    @field_validator("stage_llm_overrides", mode="before")
    @classmethod
    def _coerce_stage_overrides(cls, v: Any) -> Any:
        """Accept bare dicts for backwards compatibility (§6 P2).

        Pre-§6 callers passed ``{"step_planning": {"model": "x", "temperature": 0.1}}``.
        Promote each value into a ``StageLLMOverride`` so unknown keys
        (e.g. ``temprature``) raise instead of being silently ignored.
        """
        if not isinstance(v, dict):
            return v
        out: dict[str, Any] = {}
        for stage_name, override in v.items():
            if isinstance(override, StageLLMOverride):
                out[stage_name] = override
            elif isinstance(override, dict):
                out[stage_name] = StageLLMOverride(**override)
            else:
                out[stage_name] = override  # let Pydantic produce its native error
        return out

    @model_validator(mode="after")
    def _resolve_json_mode_fallback_default(self) -> MAGEConfig:
        """Auto-enable `json_mode_fallback` for local providers (§6 P2).

        Local servers (vLLM / Ollama / llama.cpp HTTP) often don't
        honor OpenAI's ``response_format={"type": "json_object"}``. If
        the caller hasn't explicitly set the flag, default it to True
        when ``provider == "local"``, False otherwise.
        """
        if self.json_mode_fallback is None:
            object.__setattr__(self, "json_mode_fallback", self.provider == "local")
        return self

    @model_validator(mode="after")
    def _default_cold_start_candidates_for_research(self) -> MAGEConfig:
        # P1.5: when memory research is enabled, cold-start multi-sampling pays
        # off (the planner has no recalled context to anchor on), so bump the
        # default 1 → 3. Explicit values other than the default are respected.
        if self.enable_memory_research and self.cold_start_candidates == 1:
            self.cold_start_candidates = 3
        return self

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | Path) -> MAGEConfig:
        """Load configuration from a TOML file.

        Parameters
        ----------
        path : str | Path
            Path to the TOML config file.

        Returns
        -------
        MAGEConfig
            Configuration populated from the TOML ``[mage]`` section.
        """
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)

        mage_data = data.get("mage", {})
        return cls.model_validate(mage_data)

    @classmethod
    def _env_var_map(cls) -> dict[str, str]:
        """``MAGE_*`` env var → field name — the single source of truth for both
        :meth:`from_env` and the generated config docs (T39)."""
        return {
            "MAGE_API_KEY": "api_key",
            "MAGE_BASE_URL": "base_url",
            "MAGE_MODEL": "model",
            "MAGE_PROVIDER": "provider",
            "MAGE_OPENROUTER_SITE_URL": "openrouter_site_url",
            "MAGE_OPENROUTER_APP_NAME": "openrouter_app_name",
            "MAGE_ENABLE_MEMORY": "enable_memory",
            "MAGE_MEMORY_BASE_URL": "memory_base_url",
            "MAGE_ENABLE_MEMORY_RESEARCH": "enable_memory_research",
            "MAGE_ENABLE_WEB_RESEARCH": "enable_web_research",
            "MAGE_WEB_SEARCH_API_KEY": "web_search_api_key",
            "MAGE_WEB_SEARCH_PROVIDER": "web_search_provider",
            "MAGE_MEMORY_RELEVANCE_THRESHOLD": "memory_relevance_threshold",
            "MAGE_MEMORY_RECALL_TOP_K": "memory_recall_top_k",
            "MAGE_MEMORY_SEARCH_MODE": "memory_search_mode",
            "MAGE_MEMORY_NAMESPACE": "memory_namespace",
            "MAGE_MEMORY_APPLICABILITY_THRESHOLD": "memory_applicability_threshold",
            "MAGE_MEMORY_MIN_APPLICABLE_HITS": "memory_min_applicable_hits",
            "MAGE_WEB_MIN_SOURCED_CLAIMS": "web_min_sourced_claims",
            "MAGE_FEEDBACK_APPLICABILITY_THRESHOLD": "feedback_applicability_threshold",
            "MAGE_STEP_CRITIC_REVISION_THRESHOLD": "step_critic_revision_threshold",
            "MAGE_COLD_START_CANDIDATES": "cold_start_candidates",
            "MAGE_BENCHMARK_N_CASES": "benchmark_n_cases",
            "MAGE_EXECUTION_FEEDBACK_THRESHOLD": "execution_feedback_threshold",
            "MAGE_MAX_EXECUTION_FEEDBACK_ITERATIONS": "max_execution_feedback_iterations",
            "MAGE_ENABLE_DEMO_BOOTSTRAP": "enable_demo_bootstrap",
            "MAGE_DEMO_BOOTSTRAP_CANDIDATES": "demo_bootstrap_candidates",
            "MAGE_DEMO_BOOTSTRAP_TOP_K": "demo_bootstrap_top_k",
            "MAGE_DEMO_BOOTSTRAP_MIN_PASS_RATE": "demo_bootstrap_min_pass_rate",
            "MAGE_SIMPLICITY_BIAS": "simplicity_bias",
            "MAGE_SIMPLICITY_MAX_STEPS": "simplicity_max_steps",
            "MAGE_ENABLE_TOPOLOGY_SELECTION": "enable_topology_selection",
            "MAGE_TOPOLOGY_MAX_CANDIDATES": "topology_max_candidates",
            "MAGE_ENABLE_PARALLEL_TOPOLOGY_SAMPLING": "enable_parallel_topology_sampling",
            "MAGE_ENABLE_TOOL_SYNTHESIS": "enable_tool_synthesis",
            "MAGE_MAX_DIGEST_CHARS": "max_digest_chars",
            "MAGE_SSL_VERIFY": "ssl_verify",
            "MAGE_HTTP_TIMEOUT": "http_timeout",
        }

    @classmethod
    def from_env(cls) -> MAGEConfig:
        """Build configuration from environment variables.

        Environment variables are prefixed with ``MAGE_`` and mapped to field
        names (lowercase, underscored).

        Returns
        -------
        MAGEConfig
            Configuration populated from environment variables.
        """
        env_map: dict[str, str] = cls._env_var_map()

        kwargs: dict[str, Any] = {}
        bool_fields = {
            "enable_memory",
            "enable_memory_research",
            "enable_web_research",
            "ssl_verify",
            "enable_demo_bootstrap",
            "enable_topology_selection",
            "enable_parallel_topology_sampling",
            "enable_tool_synthesis",
        }
        float_fields = {
            "memory_relevance_threshold",
            "http_timeout",
            "memory_applicability_threshold",
            "feedback_applicability_threshold",
            "step_critic_revision_threshold",
            "execution_feedback_threshold",
            "demo_bootstrap_min_pass_rate",
            "simplicity_bias",
        }
        int_fields = {
            "memory_recall_top_k",
            "cold_start_candidates",
            "max_digest_chars",
            "memory_min_applicable_hits",
            "web_min_sourced_claims",
            "benchmark_n_cases",
            "max_execution_feedback_iterations",
            "demo_bootstrap_candidates",
            "demo_bootstrap_top_k",
            "simplicity_max_steps",
            "topology_max_candidates",
        }

        for env_var, field_name in env_map.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                if field_name in bool_fields:
                    kwargs[field_name] = value.lower() in ("true", "1", "yes")
                elif field_name in float_fields:
                    kwargs[field_name] = float(value)
                elif field_name in int_fields:
                    kwargs[field_name] = int(value)
                else:
                    kwargs[field_name] = value

        # Fallback for API key
        if "api_key" not in kwargs:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                kwargs["api_key"] = openai_key

        return cls.model_validate(kwargs)

    @classmethod
    def for_openrouter(
        cls,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        **kwargs: Any,
    ) -> MAGEConfig:
        """Create a config for OpenRouter.

        Parameters
        ----------
        api_key : str
            OpenRouter API key.
        model : str
            Model identifier.
        **kwargs
            Additional config overrides.

        Returns
        -------
        MAGEConfig
            Configuration for OpenRouter.
        """
        return cls(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model,
            openrouter_site_url=kwargs.pop("openrouter_site_url", "https://github.com/Glazkoff/carl-mage"),
            openrouter_app_name=kwargs.pop("openrouter_app_name", "carl-mage"),
            **kwargs,
        )

    @classmethod
    def for_local(
        cls,
        base_url: str,
        api_key: str = "EMPTY",
        model: str | None = None,
        ssl_verify: bool = True,
        **kwargs: Any,
    ) -> MAGEConfig:
        """Create a config for a local OpenAI-compatible server.

        Parameters
        ----------
        base_url : str
            Server endpoint URL.
        api_key : str
            API key (use ``"EMPTY"`` for servers that don't require auth).
        model : str | None
            Model name, or ``None`` for ``"__auto__"`` detection.
        ssl_verify : bool
            Whether to verify SSL certificates.
        **kwargs
            Additional config overrides.

        Returns
        -------
        MAGEConfig
            Configuration for a local server.
        """
        return cls(
            provider="local",
            base_url=base_url,
            api_key=api_key,
            model=model or "__auto__",
            ssl_verify=ssl_verify,
            **kwargs,
        )

    @classmethod
    def for_benchmark(cls, profile: str, **kwargs: Any) -> MAGEConfig:
        """Create a config preset for a headline benchmark (plan item P3.5).

        Applies the named profile's authoring-time overrides (``allowed_step_types``,
        simplicity knobs, …) for any field not in ``kwargs`` — explicit ``kwargs``
        always win. See :data:`mmar_mage.profiles.BENCHMARK_PROFILES`.

        Parameters
        ----------
        profile : str
            One of ``bfcl`` / ``seal_tools`` / ``rotbench`` / ``toolalpaca`` / ``custom``.
        **kwargs
            Config overrides (take precedence over the profile).
        """
        profile_literal = cast(
            'Literal["bfcl", "seal_tools", "rotbench", "toolalpaca", "custom"]',
            profile,
        )
        return cls(benchmark_profile=profile_literal, **kwargs)


# ---------------------------------------------------------------------------
# Replay helpers (§4 P1) — reconstruct typed artifacts from a bundle
# ---------------------------------------------------------------------------


def _reconstruct_domain_analysis(value: Any) -> DomainAnalysis | None:
    """Coerce a bundle entry back into a typed :class:`DomainAnalysis`.

    Accepts ``None`` (stage wasn't run), an already-typed instance, or
    a Pydantic dump dict. Returns ``None`` when the value is missing.
    """
    if value is None:
        return None
    if isinstance(value, DomainAnalysis):
        return value
    if isinstance(value, dict):
        return DomainAnalysis.model_validate(value)
    logger.warning("Unexpected domain_analysis value type %s; ignoring", type(value).__name__)
    return None


def _reconstruct_step_plan(value: Any) -> StepPlan | None:
    if value is None:
        return None
    if isinstance(value, StepPlan):
        return value
    if isinstance(value, dict):
        return StepPlan.model_validate(value)
    logger.warning("Unexpected step_plan value type %s; ignoring", type(value).__name__)
    return None


def _reconstruct_dag(value: Any) -> DAGStructure | None:
    if value is None:
        return None
    if isinstance(value, DAGStructure):
        return value
    if isinstance(value, dict):
        return DAGStructure.model_validate(value)
    logger.warning("Unexpected dag value type %s; ignoring", type(value).__name__)
    return None


def _recover_query_from_result(result: MAGEResult) -> str:
    """Best-effort recovery of the original query from a MAGEResult.

    Priority:
      1. Caller-supplied ``query=`` kwarg (handled by the caller, not
         here)
      2. ``chain_dict.metadata.task_description`` (set by §7.3
         CARE metadata block)
      3. ``intermediate_artifacts.domain_analysis.task_type`` (last
         resort, lossy)

    Returns an empty string when no recovery succeeds — the caller
    must supply ``query`` explicitly.
    """
    chain_meta = result.chain_dict.get("metadata") if isinstance(result.chain_dict, dict) else None
    if isinstance(chain_meta, dict):
        td = chain_meta.get("task_description")
        if isinstance(td, str) and td.strip():
            return td.strip()
    da = result.intermediate_artifacts.get("domain_analysis")
    if isinstance(da, dict):
        # Use the task_type as a degraded proxy — not the original
        # query, but enough to keep deep mode happy on a replay.
        tt = da.get("task_type")
        if isinstance(tt, str) and tt:
            return tt
    return ""


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class StageUsage(BaseModel):
    """Token usage accumulated for one pipeline stage (WP-5).

    Populated from each call's ``response.usage``. ``estimated`` flips True when
    a contributing call returned no usage (the provider didn't report it).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class MAGEMetadata(BaseModel):
    """Metadata about a generation run.

    Parameters
    ----------
    domain : str
        Detected knowledge domain (e.g. ``"finance"``, ``"medicine"``).
    num_steps : int
        Number of steps in the generated chain.
    generation_time_seconds : float
        Wall-clock time for the entire generation.
    model : str
        LLM model used.
    deep_stages_completed : list[str] | None
        Names of deep-mode stages that ran (``None`` for fast mode).
    memory_hits_used : int
        Number of memory hits incorporated into generation.
    web_results_used : int
        Number of web search results incorporated into generation.
    was_cold_start : bool
        Whether the generation was a cold start (no memory hits).
    """

    domain: str = "general"
    num_steps: int = 0
    generation_time_seconds: float = 0.0
    model: str = ""
    deep_stages_completed: list[str] | None = None
    memory_hits_used: int = 0
    web_results_used: int = 0
    was_cold_start: bool = False

    # Quality enhancement metadata
    step_critique_score: float | None = None
    verification_passed: bool | None = None
    refine_iterations: int | None = None
    refine_quality_delta: float | None = None
    tot_branches_explored: int | None = None
    mcts_simulations_run: int | None = None
    mcts_best_reward: float | None = None
    feedback_recalled: int | None = None

    # CARE library-ready naming (§7.2). Pre-filled into CARE's
    # `SaveAgentModal` as editable defaults so the user is never faced with
    # a blank "Name this agent" prompt. Default heuristic uses domain +
    # query; deep mode replaces it with a dedicated LLM call.
    suggested_display_name: str = ""
    suggested_description: str = ""
    suggested_tags: list[str] = Field(default_factory=list)

    # WP-5: actual token usage (from response.usage) across all stages of this
    # generation. ``usage_estimated`` is True if any call lacked usage.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    usage_estimated: bool = False
    usage_by_stage: dict[str, Any] = Field(default_factory=dict)
    # T45: wall-clock seconds per pipeline stage (between its on_stage_start and
    # on_stage_complete), so a replay bundle shows where the time went.
    stage_timings: dict[str, float] = Field(default_factory=dict)

    def stage_telemetry(self) -> dict[str, dict[str, Any]]:
        """Per-stage timing + token usage merged into one dict (T45), keyed by
        stage name — the shape CARE writes to a replay bundle / ledger as
        ``extra.stage_timings`` so a post-mortem reads from one file instead of
        digging through the session log."""
        out: dict[str, dict[str, Any]] = {}
        for stage in sorted(set(self.stage_timings) | set(self.usage_by_stage)):
            entry: dict[str, Any] = {}
            if stage in self.stage_timings:
                entry["time_seconds"] = self.stage_timings[stage]
            if stage in self.usage_by_stage:
                entry["usage"] = self.usage_by_stage[stage]
            out[stage] = entry
        return out


class MAGEResult(BaseModel):
    """Result of chain generation — the primary output of MAGE.

    Parameters
    ----------
    chain_json : str
        Serialized CARL chain as a JSON string.
    chain_dict : dict[str, Any]
        The same chain as a Python dictionary (for programmatic use).
    code : str | None
        Python code using ChainBuilder fluent API, or ``None`` if output_format is ``"json"``.
    memory_key : str | None
        Entity ID in gigaevo-memory, or ``None`` if memory was disabled / failed.
    mode : Literal["fast", "deep"]
        Which mode produced this result.
    metadata : MAGEMetadata
        Generation metadata.
    intermediate_artifacts : dict[str, Any]
        Per-stage outputs collected during deep generation (§7.1):
        ``domain_analysis``, ``step_plan``, ``dag``, ``critique``,
        ``verification``, ``refine``. Each value is the agent's
        ``model_dump()`` so it round-trips through JSON. Empty in
        fast mode.
    memory_keys : dict[str, str | list[str]]
        Entity IDs for every artifact persisted during generation
        (§7.1): ``chain`` (the main entity), ``skills`` (list),
        ``failure_cards`` (list), ``reusable_steps`` (list),
        ``ideas`` (cold-start ideas). Empty when memory is disabled.
    """

    chain_json: str
    chain_dict: dict[str, Any] = Field(default_factory=dict)
    code: str | None = None
    memory_key: str | None = None
    mode: Literal["fast", "deep"] = "deep"
    metadata: MAGEMetadata = Field(default_factory=MAGEMetadata)
    intermediate_artifacts: dict[str, Any] = Field(default_factory=dict)
    memory_keys: dict[str, Any] = Field(default_factory=dict)

    # P1.16: self-verification report (a ``benchmark.QualityReport`` dumped to a
    # dict) when ``enable_benchmark_mode`` is set; ``None`` otherwise. Stored as a
    # plain dict to keep ``schemas`` free of a benchmark-module import cycle.
    quality_report: dict[str, Any] | None = None
    _config: MAGEConfig | None = None

    async def submit_feedback(
        self,
        quality_score: float,
        success: bool,
        failed_at_step: int | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Submit execution feedback to memory for future learning."""
        if not self.memory_key or self._config is None:
            return

        from .memory import MemoryManager

        mgr = MemoryManager(self._config)
        feedback = {
            "quality_score": quality_score,
            "success": success,
            "failed_at_step": failed_at_step,
            "failure_reason": failure_reason,
        }
        await mgr.submit_feedback(self.memory_key, feedback)

    async def replay_from(
        self,
        stage_name: str,
        *,
        partial_state: dict[str, Any] | None = None,
        query: str | None = None,
    ) -> MAGEResult:
        """Re-run the pipeline starting at ``stage_name`` (§4 P1).

        Reuses prior agent outputs cached in :attr:`intermediate_artifacts`
        for every stage *before* ``stage_name`` so CARE can regenerate
        just the part the user wants — e.g. "regenerate the DAG only"
        becomes ``await result.replay_from("dag_building")``. The
        caller can override the artifact feeding the replay point via
        ``partial_state`` (e.g. ``{"step_plan": edited_plan_dict}`` for
        ``stage_name="dag_building"``).

        Parameters
        ----------
        stage_name : str
            One of ``"domain_analysis"``, ``"step_planning"``,
            ``"dag_building"``, ``"step_describing"``. Stages earlier
            than the chosen one are skipped; later stages (serialise +
            optional verify/refine) always run.
        partial_state : dict[str, Any] | None
            Overrides for the cached artifacts. Keys are stage names
            (``"domain_analysis"``, ``"step_plan"``, ``"dag"``). When
            a key is supplied, its value replaces the cached artifact;
            other keys keep the cached value. Use this to edit the
            plan / DAG before re-running downstream stages.
        query : str | None
            User query. Falls back to ``metadata.suggested_description``
            slug or the first 80 chars of
            ``intermediate_artifacts.domain_analysis.task_type``. Pass
            it explicitly when CARE has the original user query at hand.

        Returns
        -------
        MAGEResult
            A fresh result carrying the regenerated chain plus
            ``intermediate_artifacts`` populated from this replay (a
            mix of cached + freshly-generated stages).

        Raises
        ------
        ValueError
            If ``stage_name`` is unknown, if ``_config`` is missing
            (result wasn't produced by ``generate()``), or if the
            stage being replayed needs a prior artifact that is
            neither cached nor supplied via ``partial_state``.
        """
        if self._config is None:
            raise ValueError(
                "Cannot replay: MAGEResult has no _config — was this result built from"
                " from_care_dict() rather than generate()? Pass a fresh config via"
                " result._config = ... before calling replay_from()."
            )

        valid_stages = ("domain_analysis", "step_planning", "dag_building", "step_describing")
        if stage_name not in valid_stages:
            raise ValueError(f"Unknown replay stage {stage_name!r}. Valid stages: {valid_stages}")

        # Lazy imports to break the circular dep with the generator.
        from .agents.capability_lookup_agent import CapabilityContext
        from .carl_export import to_carl_compat
        from .generator import MAGEGenerator, validate_carl_json

        effective_query = query or _recover_query_from_result(self)
        if not effective_query:
            raise ValueError(
                "Cannot replay: no query available. Pass query=... explicitly or"
                " ensure metadata.suggested_description / task_description is set."
            )

        state = dict(self.intermediate_artifacts)
        if partial_state:
            state.update(partial_state)

        # Reconstruct prior-stage artifacts when they exist on the
        # bundle. Each helper accepts either a Pydantic dump (dict) or
        # the already-typed instance for tolerance with `partial_state`.
        cached_domain = _reconstruct_domain_analysis(state.get("domain_analysis"))
        cached_plan = _reconstruct_step_plan(state.get("step_plan"))
        cached_dag = _reconstruct_dag(state.get("dag"))

        gen = MAGEGenerator(config=self._config)

        # Decide which stages to re-run. Each branch leaves the four
        # locals (domain, plan, dag, steps) ready for serialisation.
        # `plan` is optional: the `step_describing` branch reuses the cached
        # plan (which may be absent — it's only echoed into the artifact bundle).
        plan: StepPlan | None
        if stage_name == "domain_analysis":
            domain = await gen.analyze_domain(effective_query)
            plan = await gen.plan_steps(effective_query, domain)
            dag = await gen.build_dag(plan)
            steps = await gen.describe_steps(effective_query, dag)
        elif stage_name == "step_planning":
            if cached_domain is None:
                raise ValueError(
                    "replay_from('step_planning') needs a cached domain_analysis"
                    " in intermediate_artifacts or partial_state."
                )
            domain = cached_domain
            plan = await gen.plan_steps(effective_query, domain)
            dag = await gen.build_dag(plan)
            steps = await gen.describe_steps(effective_query, dag)
        elif stage_name == "dag_building":
            if cached_domain is None or cached_plan is None:
                raise ValueError("replay_from('dag_building') needs cached domain_analysis + step_plan.")
            domain = cached_domain
            plan = cached_plan
            dag = await gen.build_dag(plan)
            steps = await gen.describe_steps(effective_query, dag)
        else:  # step_describing
            if cached_domain is None or cached_dag is None:
                raise ValueError("replay_from('step_describing') needs cached domain_analysis + dag.")
            domain = cached_domain
            plan = cached_plan  # may be None — only used for the artifact bundle
            dag = cached_dag
            steps = await gen.describe_steps(effective_query, dag)

        chain_schema = CARLChainSchema(
            name=f"{domain.domain.title()} — {effective_query[:60]}",
            max_workers=dag.max_workers,
            timeout=self._config.default_timeout,
            steps=steps,
        )
        chain_dict = chain_schema.model_dump(exclude_none=True)
        chain_dict = to_carl_compat(chain_dict)
        validate_carl_json(chain_dict, strict_carl=True)
        chain_json = json.dumps(chain_dict, ensure_ascii=False, indent=2)

        # Rebuild intermediate_artifacts with the new outputs, falling
        # back to cached dumps where the stage was skipped.
        new_artifacts: dict[str, Any] = {"domain_analysis": domain.model_dump()}
        if plan is not None:
            new_artifacts["step_plan"] = plan.model_dump()
        new_artifacts["dag"] = dag.model_dump()
        new_artifacts["described_steps"] = [s.model_dump() for s in steps]

        # Preserve the §7.2 suggested-* fields from the prior run so
        # CARE's library row stays stable across replays. Stage list
        # marks this run as a replay for telemetry.
        metadata = MAGEMetadata(
            domain=domain.domain,
            num_steps=len(steps),
            model=self._config.model,
            deep_stages_completed=[f"{stage_name}_replay"],
            suggested_display_name=self.metadata.suggested_display_name,
            suggested_description=self.metadata.suggested_description,
            suggested_tags=list(self.metadata.suggested_tags),
        )

        new_result = MAGEResult(
            chain_json=chain_json,
            chain_dict=chain_dict,
            mode="deep",
            metadata=metadata,
            intermediate_artifacts=new_artifacts,
        )
        new_result._config = self._config
        # Best-effort: keep the chain memory key from the prior run so
        # follow-up `submit_feedback` calls land on the right entity.
        new_result.memory_key = self.memory_key
        # Surfacing the new context-injection path (§2 P1) for the
        # next stage if the caller already had capabilities resolved.
        # We don't re-run the capability lookup automatically on
        # replay — that's a deliberate scope choice; CARE can pass
        # `capabilities=` to `MAGEGenerator.generate()` on a fresh run
        # if it needs that.
        _ = CapabilityContext  # silence unused-import in non-typing path
        return new_result

    def to_care_dict(self) -> dict[str, Any]:
        """Serialize the full result as a single CARE replay bundle (§7.1).

        CARE persists this as one ``agent_run`` artifact so the user
        can re-run a chain from the library without re-driving the
        pipeline. Round-trippable via :meth:`from_care_dict`.

        Returns
        -------
        dict[str, Any]
            ``{"schema_version": 1, "mode": "...", "chain_dict": ...,
            "chain_json": ..., "code": ..., "metadata": ...,
            "intermediate_artifacts": ..., "memory_keys": ...}``.
        """
        return {
            "schema_version": 1,
            "mode": self.mode,
            "chain_json": self.chain_json,
            "chain_dict": self.chain_dict,
            "code": self.code,
            "memory_key": self.memory_key,
            "metadata": self.metadata.model_dump(),
            "intermediate_artifacts": dict(self.intermediate_artifacts),
            "memory_keys": dict(self.memory_keys),
        }

    @classmethod
    def from_care_dict(cls, data: dict[str, Any]) -> MAGEResult:
        """Reconstruct a :class:`MAGEResult` from a CARE replay bundle (§7.1).

        Tolerant of missing optional keys so a bundle from a future
        MAGE version with extra fields still loads (forward-compat).
        Unknown ``schema_version`` values log a warning but don't
        raise — the caller already chose to load the bundle.
        """
        if not isinstance(data, dict):
            msg = f"from_care_dict expects a dict, got {type(data).__name__}"
            raise TypeError(msg)
        sv = data.get("schema_version", 1)
        if sv != 1:
            logger.warning("MAGEResult.from_care_dict: unknown schema_version=%s; best-effort", sv)

        metadata_raw = data.get("metadata") or {}
        if isinstance(metadata_raw, MAGEMetadata):
            metadata = metadata_raw
        else:
            metadata = MAGEMetadata.model_validate(metadata_raw)

        return cls(
            chain_json=str(data.get("chain_json") or ""),
            chain_dict=dict(data.get("chain_dict") or {}),
            code=data.get("code"),
            memory_key=data.get("memory_key"),
            mode=data.get("mode", "deep"),
            metadata=metadata,
            intermediate_artifacts=dict(data.get("intermediate_artifacts") or {}),
            memory_keys=dict(data.get("memory_keys") or {}),
        )


# ---------------------------------------------------------------------------
# Edit-chain models (NL-driven targeted editing)
# ---------------------------------------------------------------------------


# The eight edit operations the ChainEditPlanner may emit. The first five
# mirror the evolutionary repair vocabulary in
# ``mmar_mage.evolve.operators._apply_edit`` (so there is one source of truth
# for structural mutation + renumbering); the last three are lightweight
# additions for natural-language editing.
EditOp = Literal[
    "remove_step",
    "insert_step",
    "rewrite_step",
    "change_step_type",
    "swap_steps",
    "edit_field",
    "set_dependencies",
    "set_chain_field",
]


class ChainEdit(BaseModel):
    """One targeted edit to an existing chain, produced by the planner.

    Parameters
    ----------
    op : EditOp
        Which edit to apply.
    target_step_number : int | None
        The step the edit acts on (1-based, in the *shown* numbering). ``None``
        for ``set_chain_field`` and (optionally) ``insert_step``.
    new_step : dict | None
        Full step object for ``insert_step`` / ``rewrite_step`` (and the
        ``step_type`` source for ``change_step_type``).
    fields : dict | None
        Key→value patch for ``edit_field`` (per-step) or ``set_chain_field``
        (chain-level, e.g. ``{"name": "..."}``).
    value : Any
        Free-form payload — used by ``set_dependencies`` (``list[int]``).
    rationale : str
        One sentence: why this edit satisfies the instruction.
    """

    op: EditOp
    target_step_number: int | None = None
    new_step: dict[str, Any] | None = None
    fields: dict[str, Any] | None = None
    value: Any = None
    rationale: str = ""


class ChainEditPlan(BaseModel):
    """An ordered list of edits plus a plain-language summary of the change."""

    summary: str = ""
    edits: list[ChainEdit] = Field(default_factory=list)


class ChainEditCandidate(BaseModel):
    """A possible target chain when an instruction resolves ambiguously."""

    entity_id: str
    name: str = ""
    score: float = 0.0


class MAGEEditResult(BaseModel):
    """Result of an edit run — the output of :meth:`MAGEGenerator.edit`.

    Mirrors :class:`MAGEResult` but for the edit flow: it carries the chain
    *before* and *after* the edit, the applied edit plan, and the new memory
    version (when ``save=True``). When the instruction matched several chains
    and no id was given, ``needs_disambiguation`` is ``True`` and
    ``candidates`` lists the options — no edit was performed.
    """

    chain_json: str = ""
    chain_dict: dict[str, Any] = Field(default_factory=dict)
    before_chain_dict: dict[str, Any] = Field(default_factory=dict)
    edits: list[ChainEdit] = Field(default_factory=list)
    summary: str = ""
    entity_id: str | None = None
    memory_key: str | None = None
    saved: bool = False
    resolved_by: Literal["id", "search", "provided"] | None = None
    needs_disambiguation: bool = False
    candidates: list[ChainEditCandidate] = Field(default_factory=list)
    metadata: MAGEMetadata = Field(default_factory=MAGEMetadata)


# ---------------------------------------------------------------------------
# Intermediate deep-mode models
# ---------------------------------------------------------------------------


class DomainAnalysis(BaseModel):
    """Output of the DomainAnalyzer agent.

    Parameters
    ----------
    domain : str
        Knowledge domain of the user query.
    task_type : str
        High-level task type (``"analysis"``, ``"generation"``, ``"qa"``, etc.).
    complexity : Literal["simple", "medium", "complex"]
        Estimated complexity.
    key_concepts : list[str]
        Domain-specific concepts extracted from the query.
    suggested_step_count : int
        Recommended number of reasoning steps.
    """

    domain: str = "general"
    task_type: str = "analysis"
    complexity: Literal["simple", "medium", "complex"] = "medium"
    key_concepts: list[str] = Field(default_factory=list)
    suggested_step_count: int = 3
    preferred_step_types: list[str] = Field(default_factory=list)


class NeedsAnalysis(BaseModel):
    """P6.2 — the capability/input requirements the model declares for a task
    (the 'pull' side of capability resolution). The pipeline resolves what it
    can (skills/tools/MCP via the registries) and flags the rest — data only
    the user can give, or credentials to fetch. The model isn't locked to the
    surfaced catalog; this is how it asks for more."""

    needs_skills: list[str] = Field(default_factory=list)
    needs_tools: list[str] = Field(default_factory=list)
    needs_services: list[str] = Field(default_factory=list)
    needs_from_user: list[str] = Field(default_factory=list)
    needs_credentials: list[str] = Field(default_factory=list)


class PlannedStep(BaseModel):
    """A single step in the StepPlanner output.

    Parameters
    ----------
    number : int
        Step number (1-based).
    title : str
        Human-readable step title.
    step_type : str
        CARL step type (``"LLM"``, ``"TOOL"``, ``"MEMORY"``, etc.).
    rough_aim : str
        Brief description of what the step should achieve.
    """

    number: int
    title: str
    step_type: str = "llm"
    rough_aim: str = ""


class StepPlan(BaseModel):
    """Output of the StepPlanner agent.

    Parameters
    ----------
    steps : list[PlannedStep]
        Ordered list of planned steps.
    rationale : str
        Why this plan was chosen.
    """

    steps: list[PlannedStep] = Field(default_factory=list)
    rationale: str = ""


class DAGStep(BaseModel):
    """A step with dependency information.

    Parameters
    ----------
    number : int
        Step number.
    title : str
        Step title.
    step_type : str
        CARL step type.
    dependencies : list[int]
        Numbers of steps this step depends on.
    rough_aim : str
        Brief description of what the step should achieve.
    """

    number: int
    title: str
    step_type: str = "llm"
    dependencies: list[int] = Field(default_factory=list)
    rough_aim: str = ""


class DAGStructure(BaseModel):
    """Output of the DAGBuilder agent.

    Parameters
    ----------
    steps : list[DAGStep]
        Steps with dependency edges.
    max_workers : int
        Suggested max parallel workers.
    pattern : str
        Detected DAG pattern (``"pipeline"``, ``"funnel"``, ``"diamond"``,
        ``"critique_revise"``, ``"map_reduce"``, ``"multi_hop"``).
    """

    steps: list[DAGStep] = Field(default_factory=list)
    max_workers: int = 2
    pattern: str = "pipeline"


class CARLStepSchema(BaseModel):
    """Full CARL step description ready for JSON serialization.

    Parameters
    ----------
    step_type : str
        CARL step type.
    number : int
        Step number.
    title : str
        Step title.
    aim : str
        What the step should establish.
    reasoning_questions : str
        Questions the LLM asks itself at this step.
    step_context_queries : list[str]
        RAG keywords for ``outer_context`` retrieval.
    stage_action : str
        Concrete action performed at this stage.
    example_reasoning : str
        Example reasoning thought (few-shot).
    dependencies : list[int]
        Dependency step numbers.
    retry_max : int
        Max retries on failure.
    llm_config : dict[str, Any] | None
        Optional per-step LLM config override.
    """

    step_type: str = "llm"
    number: int = 1
    title: str = ""
    aim: str = ""
    reasoning_questions: str = ""
    step_context_queries: list[str] = Field(default_factory=list)
    stage_action: str = ""
    example_reasoning: str = ""
    dependencies: list[int] = Field(default_factory=list)
    retry_max: int = 3
    llm_config: dict[str, Any] | None = None

    # tool step fields
    tool_name: str | None = None
    tool_description: str | None = None
    tool_input_mapping: dict[str, str] | None = None
    tool_output_key: str | None = None
    tool_timeout: float | None = None

    # mcp step fields
    mcp_server_name: str | None = None
    mcp_server_transport: str | None = None  # "stdio" | "http" | "sse" (P2.2)
    mcp_server_command: str | None = None
    mcp_server_args: list[str] | None = None
    mcp_server_url: str | None = None
    mcp_server_headers: dict[str, str] | None = None  # auth/etc. for http/sse transports (P2.2)
    mcp_tool_name: str | None = None
    mcp_arguments: dict[str, Any] | None = None
    mcp_argument_mapping: dict[str, str] | None = None
    mcp_timeout: float | None = None

    # eval step fields
    eval_target_steps: list[int] | None = None
    eval_criteria: str | None = None
    eval_on_fail: str | None = None

    # conditional step fields
    condition_branches: list[dict[str, Any]] | None = None
    condition_default_step: int | None = None
    condition_context_key: str | None = None

    # aggregator step fields
    aggregate_from_steps: list[int] | None = None
    aggregation_strategy: str | None = None

    # memory step fields
    memory_operation: str | None = None
    memory_key: str | None = None
    memory_value_source: str | None = None
    memory_namespace: str | None = None

    # transform step fields
    transform_type: str | None = None
    transform_input_key: str | None = None
    transform_output_format: str | None = None
    transform_expression: str | None = None

    # structured_output step fields (CARL StructuredOutputStepConfig)
    structured_output_schema: dict[str, Any] | None = None
    structured_output_schema_name: str | None = None
    structured_output_input_source: str | None = None
    structured_output_instruction: str | None = None
    structured_output_strict_json: bool | None = None

    # RE-PLAN / checkpoint fields (per-step)
    checkpoint: bool | None = None
    checkpoint_name: str | None = None
    replan_enabled: bool | None = None

    # Memoization (P2.2): TTL (seconds) for a per-step result cache. Exported to a
    # top-level CARL ``cache: {ttl}`` (StepCache). Set on pure steps (read-only MCP
    # resources, deterministic tools). ``key_fn`` is runtime-only, so only ttl rides.
    step_cache_ttl: int | None = None

    # Success metrics (P3.4): declared, MAGE-side metric specs for downstream
    # analytics — a list of ``{name, type, threshold}``. CARL's runtime ``metrics``
    # field holds live evaluator objects (``exclude=True``, no JSON spec), so these
    # ride as MAGE analytics metadata that CARL ignores on load — they never drive
    # CARL execution (that needs an upstream serializable MetricSpec).
    step_metrics: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # agent-features step types (carl-experiments @ agent-features branch)
    # All fields are optional; only the subset relevant to the step's
    # step_type is read by to_carl_compat() during serialization.
    # ------------------------------------------------------------------

    # agent_skill — AgentSkillStepConfig (skill identity + execution mode)
    skill: str | dict[str, Any] | None = None  # URI string OR AgentSkillSource dict
    skill_task: str | None = None
    skill_execution_mode: str | None = None  # llm | script | hybrid | subagent | llm_agent
    skill_input_mapping: dict[str, str] | None = None
    skill_output_key: str | None = None
    skill_output_memory_key: str | None = None
    skill_output_memory_namespace: str | None = None
    skill_system_prompt_prefix: str | None = None
    skill_script_name: str | None = None
    skill_script_args: dict[str, str] | None = None
    skill_timeout: float | None = None
    skill_llm_max_iterations: int | None = None
    skill_runtime: str | None = None  # local | docker | e2b
    skill_runtime_config: dict[str, Any] | None = None
    skill_output_schema: dict[str, Any] | None = None
    skill_trust_policy: str | None = None  # any | sha_pinned
    skill_sha256: str | None = None
    skill_extra_pip: list[str] | None = None

    # evaluation — inline quality gate over another step's output
    evaluation_target_step: int | None = None
    evaluation_method: str | None = None  # rule | llm | python
    # CARL's EvaluationStepConfig.criteria is list[str]; the planner emits a
    # list per the few-shot example. A stray bare string is coerced to a
    # single-element list (see _coerce_evaluation_criteria) so a weak model
    # that returns "..." instead of ["..."] doesn't fail validation.
    evaluation_criteria: list[str] | None = None
    evaluation_fail_action: str | None = None  # continue | fail | replan | retry
    evaluation_threshold: float | None = None
    evaluation_rule: str | None = None  # for method=rule

    # parallel_sampling — N samples + aggregation
    sampling_n: int | None = None
    sampling_temperature: float | None = None
    sampling_aggregation: str | None = None  # vote | best_of | mean | concat
    sampling_judge_prompt: str | None = None

    # agent_handoff — delegate to sub-chain
    handoff_chain_path: str | None = None  # JSON file or registry key
    handoff_input_mapping: dict[str, str] | None = None
    handoff_output_key: str | None = None

    # tool_discovery — register tools at runtime
    discovery_sources: list[dict[str, Any]] | None = None  # list of ToolSource dicts

    # human_input — pause for human
    human_prompt: str | None = None
    human_input_timeout: float | None = None
    human_default: str | None = None

    # supervisor — LLM router over named sub-chains
    supervisor_routes: dict[str, str] | None = None  # name → chain path / registry key
    supervisor_routing_prompt: str | None = None

    # debate — round-robin multi-agent
    debate_agents: list[dict[str, Any]] | None = None  # [{name, llm_config, role}]
    debate_rounds: int | None = None
    debate_judge_prompt: str | None = None

    # mcp_resource — read a named MCP resource
    mcp_resource_uri: str | None = None
    mcp_resource_memory_key: str | None = None

    @field_validator("evaluation_criteria", mode="before")
    @classmethod
    def _coerce_evaluation_criteria(cls, v: Any) -> Any:
        """Accept a bare string for ``evaluation_criteria`` and wrap it in a
        one-element list. CARL requires ``list[str]`` and the planner is told to
        emit a list, but weak models sometimes return a single string (or a
        delimited one). Wrapping keeps the step valid instead of failing the
        whole generation. ``None`` and lists pass through untouched."""
        if isinstance(v, str):
            text = v.strip()
            return [text] if text else None
        return v

    @field_validator("title", "aim", "reasoning_questions", "stage_action", "example_reasoning", mode="before")
    @classmethod
    def _coerce_text_field(cls, v: Any) -> str:
        """Coerce a free-text field to ``str`` (T22). Weak models sometimes return
        a list / None / number for these — run 731 failed with "4 validation
        errors ... reasoning_questions". Coerce instead of failing the whole
        generation: a list is joined with "; ", ``None`` becomes "", any other
        scalar is ``str()``-ed; a real string passes through."""
        if v is None:
            return ""
        if isinstance(v, list):
            return "; ".join(str(x) for x in v)
        return v if isinstance(v, str) else str(v)

    @field_validator("step_context_queries", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        """Coerce ``step_context_queries`` to ``list[str]`` (T22): a bare string
        becomes a one-element list, ``None`` becomes [], and list elements are
        ``str()``-ed so a weak model returning ``"finance"`` (or ``[1, 2]``)
        doesn't fail validation."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    @model_validator(mode="after")
    def _validate_tool_step(self) -> CARLStepSchema:
        """Schema-level guards mirroring the deterministic lint (T5/T23) so the
        09.06 failure class is rejected at every point the schema is built: a
        ``tool`` step must name a tool, and a url-shaped field
        (``url`` / ``uri`` / ``endpoint``) must not carry the raw user query
        (``$outer_context``). Cheap, deterministic, no LLM."""
        if self.step_type == "tool" and not (self.tool_name or "").strip():
            raise ValueError("tool step requires a non-empty tool_name")
        for field, value in (self.tool_input_mapping or {}).items():
            if (
                isinstance(value, str)
                and value == "$outer_context"
                and re.search(r"(?:^|_)(url|uri|endpoint)$", str(field), re.IGNORECASE)
            ):
                raise ValueError(
                    f"tool_input_mapping['{field}'] must not be $outer_context (the raw user "
                    "query) — use a literal URL or a $history[N] reference"
                )
        return self


class CARLChainSchema(BaseModel):
    """Top-level CARL chain schema for JSON serialization.

    Parameters
    ----------
    name : str
        Human-readable chain name.
    max_workers : int
        Max parallel step workers.
    timeout : float | None
        Chain execution timeout in seconds.
    search_config : dict[str, Any] | None
        Context search configuration.
    steps : list[CARLStepSchema]
        Ordered list of chain steps.
    """

    name: str = ""
    max_workers: int = 2
    timeout: float | None = 300.0
    search_config: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "strategy": "substring",
            "substring_config": {
                "case_sensitive": False,
                "min_word_length": 3,
                "max_matches_per_query": 5,
            },
        }
    )
    steps: list[CARLStepSchema] = Field(default_factory=list)

    # Chain-level RE-PLAN policy (mmar_carl ReplanPolicy). Opaque dict here —
    # MAGE doesn't validate the schema; mmar_carl does at chain load time.
    # When set, the runtime evaluates RE-PLAN checkers after each step and
    # may trigger replan / restart / retry actions per their verdicts.
    replan_policy: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Research result schemas
# ---------------------------------------------------------------------------


class MemoryHit(BaseModel):
    """A single memory search hit.

    Parameters
    ----------
    entity_id : str
        Memory entity identifier.
    name : str
        Entity name.
    score : float
        Relevance score.
    abstract_structure : dict[str, Any]
        Structural metadata from the memory entry.
    """

    entity_id: str
    name: str = ""
    score: float = 0.0
    abstract_structure: dict[str, Any] = Field(default_factory=dict)


class MemoryResearchResult(BaseModel):
    """Output of the MemoryResearchAgent.

    Parameters
    ----------
    digest : str
        Compressed memory digest (max 300 words).
    hits_count : int
        Number of memory hits found.
    was_cold_start : bool
        True if no relevant memory hits were found.
    sub_queries_used : list[str]
        Sub-queries generated for memory search.
    """

    digest: str = ""
    hits_count: int = 0
    was_cold_start: bool = True
    sub_queries_used: list[str] = Field(default_factory=list)


class WebResearchResult(BaseModel):
    """Output of the WebResearchAgent.

    Parameters
    ----------
    digest : str
        Compressed web research digest (max 300 words).
    sources : list[str]
        Source URLs used.
    sub_queries_used : list[str]
        Sub-queries generated for web search.
    was_used : bool
        Whether web research results were actually used.
    """

    digest: str = ""
    sources: list[str] = Field(default_factory=list)
    sub_queries_used: list[str] = Field(default_factory=list)
    was_used: bool = False
