# MAGE — Implementation TODO for the CARE ecosystem

MAGE (Mesh Architecture Generation Engine) turns natural-language queries into
CARL reasoning-chain JSON. To serve as CARE's chain-generation engine, MAGE
needs to expose a CARE-shaped serialisation surface (so CARE's `SaveAgentModal`
and "Re-run from library" work), add streaming + cancellation hooks for the
TUI, and grow capability-aware planning that knows about AgentSkills.

Priorities: **[P0]** blocker for CARE MVP · **[P1]** required for v0.1 ·
**[P2]** required for full ecosystem · **[P3]** quality / polish ·
**[P4]** future / research · **[DONE]** shipped on `main` with tests.

**Snapshot 2026-05-18.** §1 (step-type coverage), **§2 (capability-aware
planning — P0+P1)**, **§3 (streaming + progress + cancellation + retry
— P0+P1)**, **§4 (per-stage entrypoints + `replay_from` + intermediate-
artifact persistence — P1+P2)**, **§5 (memory writes for ecosystem
reuse — P1+P2)**, **§6 (multi-provider robustness + cost UX — P2+P3)**,
**§7 (CARE-friendly serialization — P0+P2 done)**, **§8 (profile
presets + YAML prompts + `mage validate` CLI — P2+P3 done)**, **§9
(testing — P1+P2 done)**, **§10 (docs + README "MAGE in CARE" —
P2+P3 done)** all shipped on `who-cares`. **EVERY P0-P3 ITEM IN THE
PLAN IS NOW DONE.** All CARE-side integration work
on the MAGE module is complete (M0-M3 + ergonomics + audit). All
upstream prep on **Memory**, **Platform** and **CARL** sides has
shipped — see `~/Development/care/PREPARE.md`.

---

## 1. Step-type coverage (DONE)

CARL today supports 16 `StepType` values. MAGE's `VALID_STEP_TYPES`
(`schemas.py:30-56`) now covers every one of them plus the MAGE-internal
`eval` / `aggregator` semantic labels (translated to CARL-loadable `llm`
steps by `to_carl_compat()` in `carl_export.py`).

- **[DONE]** `CARL_VALID_STEP_TYPES` is the source of truth for CARL-loadable
  step types (`llm`, `tool`, `mcp`, `memory`, `transform`, `conditional`,
  `structured_output`, `agent_skill`, `evaluation`, `agent_handoff`,
  `parallel_sampling`, `tool_discovery`, `human_input`, `supervisor`,
  `debate`, `mcp_resource`). `VALID_STEP_TYPES = CARL_VALID_STEP_TYPES |
{"eval", "aggregator"}` covers what intermediate MAGE stages emit.
- **[DONE]** Per-type fields plumbed through `CARLStepSchema` and
  re-nested under `step_config` in `to_carl_compat()` for every type from
  `{agent_skill, evaluation, parallel_sampling, agent_handoff, supervisor,
debate, human_input, tool_discovery, mcp_resource}`.
- **[DONE]** `STEP_PLANNER_PROMPT` + `STEP_DESCRIBER_PROMPT` + fast-mode
  prompt extended with descriptions and few-shot examples for the new
  types.
- **[DONE]** Skill-discovery surface (`mmar_mage/skills/registry.py` +
  `discovery.py`): built-in `SkillRegistry` (PDF, DOCX, PPTX, XLSX,
  mcp-builder, …) + optional web fallback gated by
  `MAGEConfig.enable_skill_discovery` / `enable_skill_web_discovery`.

---

## 2. Capability-aware planning (DONE — P0 + P1)

**§2 P0 fully shipped on `who-cares`** — the planner now sees ONE
unified `CapabilityContext` from a single `CapabilityLookupAgent` stage
instead of two overlapping `capability_lookup` + `skill_discovery`
stages.

- **[DONE] Extended `CapabilityContext`** with `agent_skills:
  list[AgentSkillEntry]`. New `AgentSkillEntry` Pydantic model carries
  `name`, `description`, `uri`, `manifest_summary`, `sha256`, `tags`,
  `allowed_tools`, plus `source` (`memory` | `local` | `discovered`),
  `relevance`, `why`. Default `agent_skills=[]` keeps existing callers
  backwards-compatible. New `skills_digest()` helper renders the list
  into a planner-prompt-friendly string.
- **[DONE] AgentSkill-aware `CapabilityLookupAgent.lookup()`** now
  accepts `query` + `domain_analysis` and orchestrates four sources:
  1. Tools / MCP servers from `capability_registry` (legacy).
  2. AgentSkills via `gigaevo_client.MemoryClient.find_capability_matches`
     (lazy-imported, gated on `MAGEConfig.enable_memory_skill_lookup`).
  3. Locally-installed skills via
     `mmar_carl.skill_loader.SkillLoader.catalog_all(payloads=True)`
     (lazy-imported, gated on `enable_local_skill_catalog`).
  4. SkillDiscoveryAgent recommendations (gated on
     `enable_skill_discovery`).
  Sources merge and dedupe by SHA256 (with `(uri, name)` fallback) —
  memory entries beat local entries beat discovered entries on ties.
  Every external integration is lazy-imported and exception-isolated;
  missing deps are non-fatal.
- **[DONE] SkillDiscoveryAgent folded into `CapabilityLookupAgent`** —
  the standalone `skill_discovery` Stage 0f is gone. The planner reads
  one digest from `capability_ctx.skills_digest()` instead of merging
  two overlapping ones. `enable_skill_discovery=True` now turns on the
  unified `capability_lookup` stage (no separate stage entry in
  `deep_stages_completed`).
- **[DONE]** 22 tests in `tests/test_capability_unified.py` covering
  schema defaults, digest format, payload conversion (dict + object
  shapes), SHA256 dedup (memory wins ties), tools+MCP graceful
  degradation, memory skill source (mocked `gigaevo_client`), local
  skill source (mocked `mmar_carl.skill_loader`), folded
  SkillDiscovery, generator stage emission, planner-digest wiring.
- **[DONE] Capability-driven step selection.** New helpers
  `_tokenize()` (stop-word + short-token filter) and
  `_score_capability_match(rough_aim, capability)` in
  `agents/capability_lookup_agent.py` produce a deterministic
  Jaccard-based score (0-1) blending pure token overlap (0.7 weight)
  with the upstream ranker's `relevance` (0.3 weight). The new
  **CAPABILITY-DRIVEN STEP SELECTION** rule in `STEP_PLANNER_PROMPT`
  tells the LLM it MUST emit `agent_skill` steps when a listed skill
  fits the sub-goal, and use the skill URI verbatim.
- **[DONE] External `CapabilityContext` injection.**
  `MAGEGenerator.generate(query, *, capabilities=CapabilityContext(...))`
  now bypasses the internal `CapabilityLookupAgent` entirely when the
  caller supplies a context. Precedence: **caller-supplied > Memory +
  local merge > SkillDiscovery**. The injected path emits a synthetic
  `capability_lookup_injected` entry in `deep_stages_completed` so
  CARE can distinguish the two paths in telemetry. The injected
  `agent_skills` flow into the planner's `skill_digest` exactly like
  internally-discovered ones do.
- **[DONE]** 16 tests in `tests/test_capability_p1.py` covering
  tokenizer behaviour, scoring edges (strong / unrelated / empty /
  capped), relevance blending math, planner-prompt rule presence,
  bypass via `capabilities=`, empty-context injection still bypasses
  lookup, planner-digest visibility of injected skills, and legacy
  no-kwarg path still uses the internal lookup.

---

## 3. Streaming & progress (DONE — P0 + P1)

MAGE already exposed `MAGEProgressCallback` with `on_stage_start`,
`on_stage_complete`, `on_error`. **§3 P0 items now shipped on
`who-cares`** — CARE TUI streaming & per-step progress are both wired.

- **[DONE] `on_llm_chunk(stage, delta)` token streaming** —
  `LLMClient.generate(stream_callback=...)` opt-in path runs the
  underlying call with `stream=True`, accumulates deltas, and fires the
  callback per chunk. Falls back to a non-streaming request when the
  provider rejects `stream=True` (cosmetic-only feature, never
  fatal). Buggy user callbacks are isolated — first exception swaps the
  callback for a no-op so the LLM call still completes. Opt-in via
  `MAGEConfig.enable_streaming` (default `False`) so existing tests /
  CLI behaviour is unaffected.
- **[DONE] `on_stage_progress(stage, artifact)` intermediate artifact
  reporting** — `StepDescriber.describe(on_step=...)` fires once per
  parsed step. Wired in `MAGEGenerator` so CARE sees "step N ready"
  events without waiting for the whole stage to finish.
- **[DONE]** Both methods added to `MAGEProgressCallback` Protocol +
  `_DefaultProgressCallback` no-op stubs. Backwards-compatible: legacy
  callbacks that only implement the three original methods are probed
  with `getattr(..., None)` and skipped if the optional method is
  missing — see test
  `test_generator_works_with_legacy_callback_missing_optional_methods`.
- **[DONE]** Helpers `_extract_chunk_delta` (handles both SDK-object
  and dict chunk shapes) and `_noop_stream_callback` in `llm.py`.
- **[DONE]** 14 tests in `tests/test_streaming_progress.py` covering
  chunk extraction, streaming → callback fan-out, provider-rejection
  fallback, buggy-callback isolation, `on_step` per-step firing,
  generator wiring with streaming on/off, legacy-callback compat.

All §3 P1 items are now shipped on `who-cares`:

- **[DONE] Cancellation:** `MAGEGenerator.generate(query, *, cancel:
  asyncio.Event | None = None)`. The new `_check_cancel(cancel, stage)`
  helper fires `MAGECancelledError` whenever the caller flips the event;
  generator probes between stages (`startup`, `domain_analysis`,
  `step_planning`, `dag_building`, `step_describing`, `serialization`,
  `finalize`) and inside the fast-mode retry loop. Stage attribute on
  the exception identifies where the abort happened so CARE can show
  "Cancelled at step_planning". Pre-set events abort before any LLM
  call. New `MAGECancelledError` exported from the package public API.
- **[DONE] Retry visibility:** `MAGEProgressCallback.on_retry(stage,
  attempt, error)` Protocol method + `_DefaultProgressCallback`
  logging stub. `LLMClient.generate(retry_callback=...)` /
  `set_retry_callback()` fire the callback for every failed attempt
  that has a retry queued (NOT on the final terminal failure).
  Generator wires `progress.on_retry` via `set_retry_callback` for the
  duration of each `generate()` call; callback exceptions are isolated.
- **[DONE]** 14 new tests in `tests/test_cancel_retry.py` (helper
  semantics, pre-set cancel, mid-pipeline cancel, retry callback shape,
  final-attempt-doesn't-fire, callback exception isolation,
  client-level default, generator wiring, legacy-callback compat).

---

## 4. Per-stage entrypoints (DONE — P1 per-stage methods + replay_from + P2 intermediate-artifact persistence)

CARE's "regenerate stage X" flow is now unblocked on MAGE's side.

- **[DONE]** Seven public per-stage methods on `MAGEGenerator`, each a
  thin stateless wrapper around the existing agent (auto-detects the
  model on first call, no chain save, no MAGEResult wrapping):
  - `await mg.analyze_domain(query) -> DomainAnalysis`
  - `await mg.plan_steps(query, domain, *, memory_digest='',
    web_digest='', capability_context='', template_skeleton='',
    skill_digest='', allowed_step_types=None) -> StepPlan`
  - `await mg.build_dag(plan) -> DAGStructure`
  - `await mg.describe_steps(query, dag, *, allowed_step_types=None) ->
    list[CARLStepSchema]` — propagates `progress.on_llm_chunk` /
    `on_stage_progress` exactly like the inline call inside
    `generate()`.
  - `await mg.critique_steps(query, steps, domain, *, threshold=None) ->
    StepCritiqueResult` — falls back to `MAGEConfig.step_critique_threshold`.
  - `await mg.verify_chain(query, chain_dict, domain, *, max_questions=None) ->
    VerificationResult` — falls back to
    `MAGEConfig.max_verification_questions`.
  - `await mg.refine(query, chain_dict, domain, *, max_iterations=None,
    quality_threshold=None) -> RefinementResult` — falls back to
    `MAGEConfig.max_refine_iterations` / `refine_quality_threshold`.
- **[DONE]** 14 tests in `tests/test_per_stage_entrypoints.py` covering
  exact-one-LLM-call discipline, stateless behaviour (no
  `MemoryManager.save_chain`), kwarg forwarding, config defaults,
  on_stage_progress propagation in `describe_steps`, and end-to-end
  composability (4 chained per-stage calls reproducing a full deep run).
- **[DONE]** `MAGEResult.replay_from(stage_name, *, partial_state=None,
  query=None)` re-runs the pipeline starting at one of
  ``domain_analysis | step_planning | dag_building | step_describing``,
  reusing cached agents from ``intermediate_artifacts`` for every
  prior stage. ``partial_state`` overrides the cached artifact at the
  replay point (e.g. ``{"step_plan": edited_plan_dict}`` for CARE's
  "edit plan, regenerate DAG" UX). New
  `_reconstruct_domain_analysis/_step_plan/_dag` helpers accept both
  Pydantic dumps and typed instances. `_recover_query_from_result`
  falls back to ``chain_dict.metadata.task_description`` (§7.3 CARE
  block) when the caller omits ``query=``. The new result preserves
  the prior run's ``suggested_*`` metadata and ``memory_key`` so the
  CARE library row stays stable; ``deep_stages_completed`` records
  e.g. ``"dag_building_replay"`` for telemetry. Built on top of the
  existing §4 per-stage entrypoints (`analyze_domain`, `plan_steps`,
  `build_dag`, `describe_steps`) — no new agent machinery.
- **[DONE]** 19 tests in `tests/test_replay_from.py` covering all
  three reconstruction helpers (dict + typed + None), query
  recovery from three sources, every error path (unknown stage,
  missing _config, missing cached artifact, missing query), each of
  the four replay-stage happy paths with exact LLM-call-count
  expectations, partial_state override flowing into the right agent,
  and explicit-query precedence over recovery.
- **[DONE]** `MemoryManager.save_intermediate_artifacts(artifacts, *,
  domain, query=None, chain_memory_key=None)` persists each entry of
  `MAGEResult.intermediate_artifacts` as a `memory_card` tagged
  ``mage_intermediate`` + ``mage_intermediate_{stage}`` + ``{domain}``.
  Per-stage save failures are isolated (one bad stage doesn't break
  the others); top-level failures log + return what was saved.
  Each card carries `kind="intermediate_artifact"`, `stage`,
  `domain`, optional `query`, and the parent `chain_memory_key` so an
  audit UI can link cards back to their chain. Generator now wires
  this from `_persist_ecosystem_artifacts` gated on
  `MAGEConfig.enable_intermediate_artifact_save=True` (default
  `False`). Returned entity IDs land on
  `MAGEResult.memory_keys["intermediate_artifacts"]`.
- **[DONE]** 10 tests in `tests/test_intermediate_artifact_save.py`
  covering per-stage persistence with tag verification, empty-input
  no-op, memory-disabled / unavailable no-ops, per-stage failure
  isolation, optional-query omission, generator gating
  (`enable_intermediate_artifact_save` on vs off),
  `chain_memory_key` forwarding, and top-level failure isolation.

---

## 5. Memory writes for ecosystem reuse (DONE — P1 + P2)

- **[DONE] Save AgentSkill provenance.**
  `MemoryManager.save_agent_skill(resolved_skill, *, entity_id=None)`
  prefers the typed helper
  `gigaevo_client.MemoryClient.ingest_skill_from_carl` when the SDK is
  installed (idempotent via SHA) and falls back to the legacy
  `gigaevo_memory.MemoryClient.save_chain` shape otherwise. Both paths
  lazy-imported and exception-isolated. Generator fires this for every
  `agent_skill` step in the finalized chain when
  `MAGEConfig.enable_skill_provenance_save=True` (default `False`).
  `_resolve_skill_payload()` reads the descriptor from both flat
  (`skill_name`, `skill`) and nested (`step_config`) CARL shapes.
- **[DONE] Save individual reusable steps.**
  `MemoryManager.save_reusable_step(step, *, domain, extra_tags=None)`
  persists clean LLM steps as `step` entities tagged `template`. The
  new `is_reusable_step()` module-level helper gates the save —
  rejects non-llm steps, dynamic refs (`{{...}}`, `step_N_output`,
  `_result`, etc.), and steps with a sparse aim. Generator fires this
  per-step when `MAGEConfig.enable_reusable_step_save=True` (default
  `False`).
- **[DONE]** 19 tests in `tests/test_ecosystem_writes.py` covering
  heuristic edges (clean step, non-llm, dynamic refs, sparse aim,
  non-dict), skill payload resolution (flat + nested + uri-only),
  preferred-vs-legacy save path (with shimmed `gigaevo_client` and
  `gigaevo_memory`), `template` tag presence, generator flag gating
  (each flag triggers only its own save), default-off behaviour, and
  isolation of save failures (generation completes even when memory
  raises).
- **[DONE] Save failed-generation cards.**
  `MemoryManager.save_failure_card(*, query, domain, reason, stage,
  chain_dict=None, extra=None)` persists a `failure_card` payload tagged
  `mage-failure` + `mage-failure-{stage}` so `FeedbackRecallAgent`
  surfaces it on similar future queries. The deep-mode generator now
  collects a `failure_meta` list during the pipeline — currently
  populated when chain verification rejects the chain (with the
  verifier's `issues_found` attached) or when self-refine exhausts
  `max_refine_iterations` without reaching
  `refine_quality_threshold`. Each entry triggers one
  `save_failure_card` call. Opt-in via
  `MAGEConfig.enable_failure_card_save=True` (default `False`).
- **[DONE] Quality signals as `evolution_meta`.**
  `MemoryManager.save_chain(evolution_meta=...)` now embeds the block
  under `chain.metadata.evolution_meta` (coexists with the §7.3 CARE
  metadata fields). New helper
  `_build_evolution_meta(metadata: MAGEMetadata)` extracts the
  non-`None` quality signals (`step_critique_score`,
  `verification_passed`, `refine_iterations`, `refine_quality_delta`,
  `tot_branches_explored`, `mcts_simulations_run`,
  `mcts_best_reward`, `feedback_recalled`). Opt-in via
  `MAGEConfig.enable_evolution_meta_save=True` (default `False`) so
  the platform's evolution engine can read these off the chain entity
  as fitness priors.
- **[DONE]** 13 tests in `tests/test_ecosystem_writes_p2.py` covering
  failure-card payload + tags + graceful degradation, evolution_meta
  embedding + CARE-metadata coexistence + omission when None,
  `_build_evolution_meta` None-stripping, generator triggers on both
  verifier rejection and refine cap-hit, default-off behaviour, and
  evolution_meta wiring into `save_chain`.

---

## 6. Multi-provider robustness (DONE — P2 + P3)

CARE running on top of any OpenAI-compatible provider (OpenAI,
OpenRouter, local Ollama/vLLM, AIRI) is now robust.

- **[DONE] Local-provider JSON-mode fallback.**
  `MAGEConfig.json_mode_fallback` (default: `None` → auto-True for
  `provider="local"`, auto-False elsewhere). When True,
  `LLMClient.generate(json_mode=True)` skips
  `response_format={"type": "json_object"}` and appends a strict-JSON
  suffix to the system prompt. `generate_json` runs a brace-balanced
  `_extract_json_object` on parse failure, recovering markdown-fenced
  or prose-wrapped JSON. Handles nested objects, strings containing
  `{}`, and escaped quotes. Explicit `True` / `False` always wins over
  the provider-derived default.
- **[DONE] Typed stage overrides.** New `StageLLMOverride` Pydantic
  model (`extra="forbid"`). `MAGEConfig.stage_llm_overrides` typed as
  `dict[str, StageLLMOverride]`; a `field_validator(mode="before")`
  promotes bare dicts into the typed shape so legacy TOML configs
  keep working. Typos like `temprature` now raise `ValidationError`
  at config construction instead of being silently ignored.
  `LLMClient.get_stage_config` tolerates both attribute and
  subscript shapes during the transition.
- **[DONE]** 30 new tests in `tests/test_multi_provider.py` covering
  `StageLLMOverride` accept/reject paths, `MAGEConfig` promotion of
  bare dicts, `get_stage_config` fallback semantics,
  `json_mode_fallback` per-provider defaults + explicit overrides,
  `_extract_json_object` edge cases (fenced / prose / nested / quoted
  / escaped / unbalanced / empty), and `LLMClient.generate` /
  `generate_json` round-trips for both fallback-on and fallback-off
  paths. 2 existing tests updated to use the typed attribute API.
- **[DONE]** Pre-flight cost estimator UX. New
  `mmar_mage.cost.estimate_pipeline_cost(config, query, *, pricing,
  default_output_tokens, char_per_token)` enumerates the enabled
  deep-mode stages (fast mode collapses to one row), uses each
  stage's actual prompt body length (loaded via the §8 P2 YAML
  prompts) + query length for input-token estimates,
  default_output_tokens for the output budget, and applies per-row
  pricing when supplied. `self_refine` is modelled as
  `max_refine_iterations × 2` LLM calls (critique + revise). Per-
  stage model overrides flow through. New `MAGEConfig` fields:
  `enable_preflight_cost_estimate: bool = False` and `pricing:
  dict[str, tuple[float, float]] | None = None`. New
  `MAGEProgressCallback.on_cost_estimate(estimate)` Protocol method
  + `_DefaultProgressCallback` no-op stub; the generator fires it
  BEFORE any LLM call when the flag is on. Callback failures are
  isolated (caught + logged) so a buggy CARE callback can't break
  generation. Legacy callbacks without the method work unchanged
  (probed via getattr). 16 tests in
  `tests/test_preflight_cost.py` covering fast-mode single row,
  deep-mode stage enumeration, disabled-stage handling, self_refine
  call multiplier, suggested_naming on/off, pricing math
  (applied/missing/total/no-pricing), per-stage model overrides,
  generator integration (fires before any LLM call, no-op when flag
  off, isolated callback failure, legacy-callback compatibility).

---

## 7. CARE-friendly serialization (P0–P2)

**This section concentrates the CARE M0 blockers.** Two small Pydantic
changes (§7.2) plus a one-line save-helper call (§7.3) unblock the entire
save → library → re-run loop.

### 7.1 (DONE) Replay artifact bundle

- **[DONE]** `MAGEResult.to_care_dict()` produces a JSON-serializable
  bundle: ``{schema_version: 1, mode, chain_json, chain_dict, code,
  memory_key, metadata, intermediate_artifacts, memory_keys}``. CARE
  persists this as one ``agent_run`` artifact per generation.
- **[DONE]** `MAGEResult.from_care_dict(data)` round-trips losslessly,
  tolerant of missing optional keys (forward-compat) and of `MAGEMetadata`
  instances passed in instead of dicts. Unknown `schema_version`
  values log a warning but still load best-effort.
- **[DONE]** Two new `MAGEResult` fields: `intermediate_artifacts:
  dict[str, Any]` (per-stage agent outputs) and `memory_keys:
  dict[str, Any]` (chain + skills + reusable_steps + failure_cards
  entity ids). Both default to `{}` so the change is
  backwards-compatible.
- **[DONE]** Generator captures `domain_analysis`, `step_plan`, `dag`,
  `described_steps`, `revised_steps` (post-critique), `critique`,
  `verification`, and `refine` into `intermediate_artifacts` during
  deep mode (fast mode leaves it empty). `_persist_ecosystem_artifacts`
  now returns the persisted entity ids (`skills`,
  `reusable_steps`, `failure_cards`), which the outer `generate()`
  folds into `memory_keys["chain" | "skills" | "reusable_steps" |
  "failure_cards"]`.
- **[DONE]** 18 tests in `tests/test_care_replay_bundle.py` covering
  default field values, `to_care_dict` shape + JSON-serializability,
  `from_care_dict` round-trips (via Python and via JSON),
  missing-key defaults, schema_version warning, deep-mode capture of
  all 4 stages, fast-mode empty intermediate_artifacts, full
  end-to-end round-trip, and memory_keys collection (chain + skills +
  reusable_steps).

### 7.2 (DONE) Library-ready naming — **CARE M0 BLOCKER**

- **[DONE]** `MAGEResult` populates three new fields on `MAGEMetadata`
  that CARE's `SaveAgentModal` pre-fills (`mmar_mage/schemas.py` +
  `mmar_mage/generator.py`):
  - `suggested_display_name: str` — short, human-readable agent name
    (≤ 60 chars). Heuristic: `"{Domain.title()} — {first 50 chars of
    query}"`. In deep mode, replaced by `SUGGESTED_NAMING_PROMPT`
    LLM call (toggleable via `MAGEConfig.enable_suggested_naming_llm`,
    default `True`) returning a punchy noun phrase.
  - `suggested_description: str` — 1–2 sentence summary. Heuristic:
    `"{task_type}: {first 150 chars of query}"`; deep-mode LLM call
    produces a "does X, takes Y, returns Z" sentence.
  - `suggested_tags: list[str]` — domain + task_type + first 3 key
    concepts + detected capability tags (`uses-pdf`, `uses-web`,
    `uses-memory`) derived from chain step inspection by
    `_detect_capability_tags()`. Capability tags are re-injected even
    when the LLM omits them so they always reflect the actual chain.
  - Graceful degradation: if the LLM call fails or the response is
    malformed, the heuristic values stand in. Covered by
    `tests/test_suggested_naming.py` (26 tests, all passing).

### 7.3 (DONE) Persist task + context — **CARE M0 BLOCKER**

- **[DONE] `task_description` + `context_files` + `mage_metadata` +
  `display_name` + `description` are now persisted under the chain's
  `metadata` block** by `MemoryManager.save_chain()` (see
  `mmar_mage/memory.py:_merge_care_chain_metadata`). When
  `gigaevo-client` is installed the typed helper
  `CareChainMetadata.merge_into_content()` is used; otherwise a
  direct dict-merge produces an equivalent shape.

  CARE's "Re-run from library" can now read these back to re-prime
  `ReasoningContext` via CARL's `ReasoningContext.from_chain_inputs()`.

- **[DONE]** `MemoryManager.save_chain()` extended with five keyword
  args: `task_description`, `context_files`, `mage_metadata`,
  `display_name`, `description`. When none are supplied the saver is
  a strict no-op merge — preserves backwards compatibility for the
  `save_template` / `save_ideas` / `save_benchmark` callers.

- **[DONE]** `MAGEGenerator.generate()` now accepts
  `context_files: list[dict] | None = None` and forwards `query` as
  `task_description`. `MAGEMetadata.suggested_*` flows into the chain
  row so the CARE library list view is pre-populated.

- **[DONE]** Tests in `tests/test_care_metadata.py` (12 tests):
  direct-merge fallback, typed-helper preference when shimmed,
  recording `MemoryClient` round-trip, generator end-to-end including
  context_files with sha256 + size_bytes preserved.

---

## 8. Configuration ergonomics (DONE — P2 presets + YAML prompts + P3 CLI)

- **[DONE]** Every prompt body now lives in
  `mmar_mage/prompts_data/prompts.yaml` (one consolidated file, 26
  top-level keys, ~59KB). `mmar_mage/prompt_loader.py` exposes
  `load_prompts(path=None)` and `default_prompts_path()`. The default
  path respects the `MAGE_PROMPTS_PATH` env var so power users can
  swap in a custom YAML without code edits. `mmar_mage/prompts.py`
  was rewritten to load from the YAML at import time and re-export
  every legacy `X_PROMPT` constant byte-for-byte; the `from
  mmar_mage.prompts import STEP_PLANNER_PROMPT` pattern keeps
  working (verified by `test_constants_match_yaml_byte_for_byte`).
  `get_step_types_constraint` helper kept verbatim. `pyyaml>=6.0`
  added to the runtime deps.
- **[DONE]** 16 tests in `tests/test_prompt_loader.py` covering
  default YAML presence + load, every expected name present + non-
  empty, byte-for-byte agreement between YAML and module constants,
  explicit `path=` override, string-path acceptance, env-var
  override, re-import respects env var so module constants reflect
  the override, and four error paths (missing file, non-mapping,
  non-string value, non-string key, empty-string value).
- **[DONE]** Three CARE-shaped profile presets bundled under
  `configs/`:
  - `care_default.toml` — production OpenAI config with streaming,
    capability-lookup, skill-discovery and library-ready naming all
    on; memory/research/ecosystem-saves OFF (opt-in via CARE
    settings).
  - `care_local.toml` — local provider (Ollama/vLLM by default at
    `http://localhost:11434/v1`), auto-model detection, json_mode_fallback
    explicit-on, naming_llm OFF (saves a flaky LLM call on local
    models), `max_retries=3`.
  - `care_research.toml` — power-user preset: memory + memory-research
    + web-research, full capability surface (memory + local + web
    discovery), every quality stage (critique + verify + refine +
    feedback), every ecosystem-save flag including
    `enable_intermediate_artifact_save` and
    `enable_evolution_meta_save`, `cold_start_candidates=3`.
- **[DONE]** 27 tests in `tests/test_care_presets.py` covering
  preset-file existence, clean load via `from_toml`, per-preset
  flag combinations (default = streaming-on + saves-off,
  local = json_mode_fallback + extra retries + naming heuristic,
  research = full memory + full ecosystem saves), `model_dump` →
  `model_validate` round-trip for each preset, and a cross-preset
  "presets actually differ on the right axes" smoke test.
- **[DONE]** `mage validate ./chain.json` CLI subcommand in
  `mmar_mage/cli.py`. Three independently-toggleable validation
  layers: `validate_carl_json` (always, with `--strict/--no-strict`
  controlling whether MAGE-internal `eval`/`aggregator` types are
  rejected), `_try_load_reasoning_chain` (`--check-carl/--no-check-carl`
  — no-ops with a dim note when `mmar_carl` isn't installed), and a
  soft `--check-capabilities` pass that walks every `agent_skill`
  step's URI and looks it up via `CapabilityLookupAgent` against
  `--memory-base-url` + `--environment-id`. Missing skills are
  yellow warnings (exit 0); structural failures are red errors
  (exit 1). Helper `_collect_agent_skill_uris` reads both flat and
  nested CARL shapes. 14 tests in `tests/test_cli_validate.py`
  covering valid/invalid/missing-file/malformed-JSON paths,
  `--no-strict` toggle, capability check with all-known / one-
  missing / no-skill chains, and the missing-`mmar_carl` skip note.

---

## 9. Testing (DONE — P1 + P2)

- **[DONE]** Mock fixtures for the new step types added to
  `tests/conftest.py`: `SAMPLE_STEP_AGENT_SKILL` (with
  `skill_execution_mode`, `skill_input_mapping`, `skill_output_key`,
  `skill_timeout`), `SAMPLE_STEP_STRUCTURED_OUTPUT` (with
  `structured_output_schema`, `schema_name`, `strict_json`,
  `input_source`), `SAMPLE_STEP_AGENT_HANDOFF` (with
  `handoff_chain_path`, `handoff_input_mapping`,
  `handoff_output_key`). Aggregate `SAMPLE_HETEROGENEOUS_CHAIN` bundles
  all three. Existing `SAMPLE_DESCRIBED_STEPS` kept intact for
  backwards compatibility with the 612 existing tests.
- **[DONE]** Round-trip exercised in `tests/test_step_type_round_trips.py`
  (11 tests + 1 mmar_carl-conditional skip): per-exemplar
  `CARLStepSchema.model_validate`, internal `validate_carl_json` on
  the heterogeneous chain, `to_carl_compat` → strict
  `validate_carl_json(strict_carl=True)` (re-nesting per-type fields
  under `step_config`), `CARLChainSchema.model_dump` preserves each
  per-type identifier, `MAGEResult.to_care_dict` / `from_care_dict`
  round-trip via JSON for each step type and for the aggregate
  chain. Conditional `ReasoningChain.from_json` acceptance test
  skips when `mmar_carl` is not installed (the agent-features branch
  + its dependencies sit outside MAGE's hard-dep set; the
  same step types are already exercised by MAGE-side validation).
- **[DONE]** `suggested_*` golden test
  (`tests/test_suggested_naming_golden.py`): a frozen `CASSETTE_NAMING_RESPONSE`
  is replayed offline through the full deep-mode pipeline (5 LLM
  calls) and `MAGEMetadata.suggested_display_name` /
  `suggested_description` / `suggested_tags` are pinned verbatim
  against the cassette. A second test asserts the heuristic fallback
  kicks in when the cassette is malformed (LLM regression), keeping
  the fields populated.
- **[DONE]** `save_chain` ↔ `get_chain` round-trip test
  (`tests/test_save_chain_roundtrip.py`): in-process
  `_RoundTripMemoryClient` simulates server-side persist + retrieve
  via `MemoryClient.save_chain` then `MemoryClient.get_chain`. Three
  scenarios: (1) full CARE block (`task_description`, `context_files`
  with sha256+size_bytes, `mage_metadata` dump, `display_name`,
  `description`, `evolution_meta` § 5 P2) round-trips byte-for-byte;
  (2) minimal payload (just `task_description`) round-trips with
  `generated_by="mage"` and no spurious optional keys; (3) backwards-
  compat — calling `save_chain` with NO CARE kwargs persists the
  chain WITHOUT a `metadata` block, so legacy callers
  (`save_template`, `save_ideas`) aren't accidentally CARE-tagged.
- **[DONE]** Capability-aware planning integration test
  (`tests/test_integration_p2.py::test_capability_aware_planning_emits_agent_skill_for_pdf_query`):
  injects a `CapabilityContext` with a `pdf` `AgentSkillEntry`, mocks
  the planner to verify the skill_digest is reached, asserts the
  final chain contains an `agent_skill` step referencing the PDF
  skill URI (after `to_carl_compat` runs).
- **[DONE]** Streaming integration test
  (`test_streaming_fires_on_llm_chunk_with_nonempty_deltas`):
  full deep-mode run with `enable_streaming=True` and a
  stream-capable mock. Asserts `on_llm_chunk` fires exactly 3 times
  (one per chunk), every delta is non-empty, all chunks tag the
  `step_describing` stage, and the reassembled deltas exactly
  reproduce the StepDescriber JSON response.
- **[DONE]** Cancellation integration tests
  (`test_cancel_mid_stage_stops_within_one_llm_call` +
  `test_cancel_set_before_any_stage_aborts_with_zero_llm_calls`):
  fires `cancel.set()` from inside the first LLM call's side_effect,
  asserts (a) generation raises `MAGECancelledError(stage="step_planning")`,
  (b) exactly ONE LLM call ran (the in-flight one), (c) end-to-end
  latency is < 2s. The pre-set-cancel variant asserts ZERO LLM calls
  fire and the error's stage is `"startup"`.

---

## 10. Documentation (DONE — P2 + P3)

- **[DONE]** `docs/AGENT_SKILLS.md` documents the four
  capability sources (`capability_registry`, memory
  `find_capability_matches`, local `SkillLoader.catalog_all`,
  `SkillDiscoveryAgent`), the memory → local → discovered merge order,
  SHA-256 dedup, the `_score_capability_match` blend
  (0.7 Jaccard + 0.3 relevance), how to write a planner-rankable
  skill description, both registration paths (typed
  `save_agent_skill` + auto-save via
  `enable_skill_provenance_save`), and the §2 P1 caller-supplied
  `capabilities=` bypass with the
  `capability_lookup_injected` telemetry label.
- **[DONE]** `docs/CARE_INTEGRATION.md` is the explicit runtime
  contract: full `generate()` signature, the seven per-stage methods
  + `replay_from`, the `MAGEProgressCallback` protocol (three legacy
  + three optional methods probed via `getattr`), streaming
  semantics, cancellation semantics including `MAGECancelledError.stage`,
  retry-visibility timing, `CapabilityContext`/`AgentSkillEntry`
  shapes, `MAGEResult` field-by-field including
  `intermediate_artifacts` + `memory_keys`, the
  `CareChainMetadata` content.metadata block from §7.3, the three
  config presets, error taxonomy, and the full set of
  `deep_stages_completed` labels (including
  `capability_lookup_injected` and `*_replay`). All symbol references
  verified against the live API.
- **[DONE]** README now has a "MAGE in CARE" section (between
  "Memory Integration" and "Development") with a Quick-start
  snippet for the three preset paths, a feature checklist mapping
  each CARE-integration capability to its callback / config flag /
  CLI surface, and explicit links to
  [`docs/CARE_INTEGRATION.md`](docs/CARE_INTEGRATION.md),
  [`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md), and
  [`configs/`](configs/). The "Project Structure" tree was refreshed
  to reflect the new `prompts_data/`, `prompt_loader.py`, `cost.py`,
  `capability_lookup_agent.py`, `chain_verifier.py`, `self_refiner.py`,
  `step_critic.py` files and the three new `care_*.toml` presets, and
  marked the doc files added in §10 P2.

---

## Cross-module dependencies

| MAGE task                | Needs                                                                          |
| ------------------------ | ------------------------------------------------------------------------------ |
| §1 step-type coverage    | DONE — built on CARL's `mmar_carl.models.enums.StepType`                       |
| §2 AgentSkill discovery  | `MemoryClient.find_capability_matches` (DONE in gigaevo-client §4)             |
| §3 streaming             | none (internal)                                                                |
| §3 cancellation          | CARL `ReasoningContext.cancel()` (DONE) for execution-side parity              |
| §5 save_agent_skill      | `MemoryClient.ingest_skill_from_carl` (DONE in gigaevo-client §1.3)            |
| §7.2 suggested_* fields  | none — pure MAGE-internal Pydantic + one LLM call                              |
| §7.3 task + context      | `CareChainMetadata.merge_into_content()` (DONE in gigaevo-client §1.4)         |

---

## Suggested milestones

- **M0 — CARE walking-skeleton unblocker (2 days):** §7.2 + §7.3.
  Tiny PR — three new fields on `MAGEMetadata`, one helper call in
  `MemoryManager.save_chain`, plus the deep-mode naming LLM call.
- **M1 — CARE execution + re-run (1 week):** §3 P0 streaming hooks +
  §3 P1 cancellation. Optional: §5 P1 save_agent_skill provenance.
- **M2 — CARE editing + evolution (2 weeks):** §4 per-stage entrypoints,
  §2 P0 + §2 P1 capability discovery + injection, §5 P2 quality signals
  to `evolution_meta`.
- **M3 — Polish (ongoing):** §6, §7.1, §8, §9 P2, §10.
