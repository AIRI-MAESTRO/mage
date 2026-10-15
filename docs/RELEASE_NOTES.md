# MAGE Release Notes

## 0.1.2 — chain-edit deep-merge + real CARL eval runtime (2026-06-11)

- **`fix`: `chain_edit.edit_field` deep-merges nested dicts.** A flat
  `s[k] = v` let an `edit_field` patch to a nested key (e.g.
  `step_config.input_mapping.query`) replace the WHOLE `step_config` and drop
  sibling keys — so an NL `/revise` (CARE) on a deployed chain silently lost
  `tool_name` (ToolStepConfig 400) or `output_schema` (StructuredOutputStepConfig
  400), and the version failed to load. New `_merge_field` recurses into dicts
  (scalars/lists still replace; `number` stays owned by the renumberer).
- Includes **T35** (below): `evaluation/` runs chains through the real
  mmar-carl runtime; scores → `evolution_meta` via `to_evolution_meta`.

## T35: real CARL runtime in evaluation (2026-06-10)

On branch `t35-real-executor` (off `mage-deps-carl-required`). Unblocked by
making `mmar-carl` a required dependency.

- **T35** — `evaluation/` executes chains through the REAL mmar-carl DAG
  runtime instead of the flattened-prompt stub (which never ran tool steps —
  evaluation was blind to the whole 09.06 incident class). New
  `evaluation/carl_executor.py`: `execute_chain_via_carl` (from_dict
  use_typed_steps → execute_async under a timeout; parse errors return failed
  executions), `build_eval_llm_client` (OpenAI-compatible CARL client from
  EvalConfig), and `register_benchmark_tools` — a deterministic mock-HTTP tool
  set (canned ip/geo/weather payloads, frozen clock) so composite chains like
  run-096's ip→geo→weather execute end-to-end with zero network. Wired into
  `execute_chains` behind `EvalConfig.use_carl_runtime` (default **True**; the
  legacy stub stays as an escape hatch and its tests pin False).
  `to_evolution_meta(fitness_score, …)` formats eval scores into the
  standardised evolution_meta schema Memory's `versions/beating` reads (the
  C1 promotion-gate eval check) — versions are immutable, so callers pass it
  to `save_chain(evolution_meta=…)` at save time. Tests in
  `tests/test_eval_carl_executor.py` (the deciding one: a tool step REALLY
  executes; + determinism, honest failures, timeout, execute_chains
  integration, schema). Suite: 1405 passed.

## Unreleased — dependencies: gigaevo-client + required mmar-carl (2026-06-10)

On branch `mage-deps-carl-required` (off `mage-failure-fixes`).

- **`gigaevo-memory` → `gigaevo-client`.** The optional `memory` extra now declares
  `gigaevo-client>=0.3.0` (the renamed SDK) instead of `gigaevo-memory`.
  `MemoryManager` already preferred `gigaevo_client` (legacy `gigaevo_memory` shim as
  fallback); this aligns the *declared* dependency. The old memory monorepo's heavy
  transitive tail (grpcio, protobuf, mmar-llm/mapi/mimpl, tiktoken, …) drops out of the lock.
- **`mmar-carl` is now REQUIRED.** Moved from the `carl` optional extra into
  `[project.dependencies]` (`mmar-carl>=0.2`); the redundant `carl` extra is removed.
  Both deps resolve from **PyPI** (mmar-carl 0.3.0, gigaevo-client 0.3.0 published
  2026-06-10; the client's source moved to the standalone `gigaevo-client` repo) —
  no `[tool.uv.sources]`, matching care's same-day switch to PyPI-first. The suite
  runs green against the published mmar-carl 0.3.0 wheel.
- **Two latent CARL-0.3.0 round-trip bugs surfaced and fixed** — they were hidden while
  `mmar_carl` was absent and the relevant tests `skipif`-skipped:
  - `agent_skill` export now emits the required `task` (defaulted from the step's
    `aim`/`title` in `refine_agent_skill_step`) so `AgentSkillStepConfig` validates.
  - the code-gen tool step no longer passes `output_key` to
    `ChainBuilder.add_tool_step()` (which has no such param — unlike
    `add_memory_step`/`add_conditional_step`); a custom `tool_output_key` is preserved
    by the JSON export path instead.
  - `test_step_type_round_trips` loads agent-features chains via
    `from_dict(use_typed_steps=True)` (legacy `from_json` builds the unified
    `StepDescription`, whose config union predates `AgentSkillStepConfig`).
- Suite: **1398 passed**, 15 skipped (down from 57 — the previously CARL-gated tests now
  run against `mmar_carl` 0.3.0), ruff + mypy clean.

## Unreleased — failure-audit fixes (2026-06-10)

Follow-up to the 09.06 composite-query failure
(`mage-failure-audit-and-todo-20260610.md`), on branch `mage-failure-fixes`
(off `mage-audit`).

**Incident in one paragraph.** "Weather where my IP is, what should I wear?" needs
three chained external calls (IP → geolocate → weather), but the DomainAnalyzer
called it `simple`/2-step, the StepPlanner packed three HTTP calls into one tool
step, and the StepDescriber — having no tool signatures and only a `web_search`
worked-example — put the raw question into a `url` field (`GET https://<the
question>` → `Invalid IDNA hostname`). Nothing validated the url, the tool error
came back as a string with `success: true`, retry/replan never fired, and the
ledger recorded `success` while the user got an apology. The fixes below close
that class on every layer: planning/atomicity (T1–T2), the describer's tool
awareness + url discipline (T3–T4), deterministic + schema-level validation
(T5/T22/T23), JSON/stage resilience (T8/T9/T21), the run-096 e2e regression (T36),
tool-synthesis hardening (T28–T31), evolve fixes (T24–T27), memory (T32–T34) and
config visibility (T38–T40).

- **T1** — `STEP_PLANNER_PROMPT`: explicit rule "ONE tool/mcp/agent_skill step =
  EXACTLY ONE external call" + an anti-pattern example — a live composite
  (IP → geolocate → weather) is N dependent tool steps, never one mega-step that
  "gets IP, geo and weather"; "then call …" in a tool step's rough_aim is the
  violation marker. Snapshot test `tests/test_planner_tool_atomicity.py`.
- **T2** — step count by atomic external calls. `DOMAIN_ANALYZER_PROMPT` gains a
  "COUNT ATOMIC EXTERNAL CALLS" rule + Example 5 (a composite live query
  "weather where my IP is" = 4 steps, complexity medium); `STEP_PLANNER_PROMPT`
  softens "Number of steps should match suggested_step_count" to a LOWER-BOUND
  hint (one tool/mcp call per step, expand past the count when needed). Snapshot
  test `tests/test_atomic_step_counting.py`.
- **T3** — tool signatures reach the StepDescriber. New
  `prompts.render_tool_signatures(tools)` renders each capability tool's call
  signature / input fields; `StepDescriber.describe` (+ iterative/parallel) takes
  a `tools=` arg and appends the block, wired from `capability_ctx.tools` in the
  generator (also threaded through WP-6's `_finalize_chain_from_plan`). A new
  `STEP_DESCRIBER_PROMPT` rule restricts `$outer_context`/free text to text
  fields only — `url`/`uri`/`endpoint`/`path` need a literal or `$history[N]`,
  never free text (the exact `GET https://<question text>` failure). Tests in
  `tests/test_describer_tool_signatures.py`.
- **T4** — data-chain worked example in `STEP_DESCRIBER_PROMPT`. Verified against
  CARL's resolver (`mmar_carl.step_executors.resolve_context_reference`): `$history[N]`
  resolves the WHOLE field value — it is **not** interpolated inside a longer
  string, so the audit's suggested `url: "http://.../$history[1]"` would not
  execute. The new rule + worked example teach the correct shape instead:
  tool(get IP) → `transform`(compose the geolocation URL) → tool(geo, `url`=`"$history[-1]"`
  whole-value) → llm — i.e. build a URL from a prior output via a transform step,
  then pass it whole. Snapshot test `tests/test_describer_data_chain.py`.
- **T5** — deterministic pre-export tool-step lint (no LLM). New
  `chain_repair.lint_chain(chain_dict, known_tools=...)` flags the 09.06 failure
  class before a chain ships: empty `tool_name`, a url-shaped field holding
  `$outer_context` / free text / left empty, or an unknown tool — reading both the
  flat and CARL-compat nested forms. `validate_or_repair` now routes lint failures
  through its one-shot repair pass (re-linting after); the generate path
  (`_assemble_result`) surfaces lint failures non-fatally in logs + `failure_meta`.
  Tests in `tests/test_chain_lint.py`. **Closes the P0 correctness batch (T1–T5).**
- **T22** — `CARLStepSchema` before-validators coerce weak-model outputs instead
  of failing the whole generation (run 731 raised "4 validation errors …
  reasoning_questions"). The free-text fields (`title` / `aim` /
  `reasoning_questions` / `stage_action` / `example_reasoning`) accept a
  list (joined with "; "), `None` (→ ""), or a scalar (→ `str()`); and
  `step_context_queries` accepts a bare string (→ one-element list) or `None`
  (→ []). Valid values pass through unchanged. Tests in
  `tests/test_step_schema_coercers.py`.
- **T23** — schema-level tool-step guards on `CARLStepSchema` (mirror the T5 lint
  at every construction point, no LLM): an `after` `model_validator` rejects a
  `tool` step with an empty `tool_name`, and a url-shaped field (`url`/`uri`/
  `endpoint`) holding `$outer_context` (the raw query). Non-tool steps and
  `$outer_context` in text fields are unaffected. This exposed a latent bug —
  `_fallback_step_descriptions` built degenerate `tool` steps with no `tool_name`;
  the describer-off fallback now degrades a tool node to an `llm` reasoning step.
  (The audit's hard "aim non-empty" check was deliberately omitted: `aim=""` is
  the schema default, so a raise would reject minimal/fallback steps — T22 coerces
  + T5 flags it non-fatally instead.) Tests in
  `tests/test_tool_step_schema_validator.py`.
- **T21** — JSON-repair cascade in `llm.generate_json` (no new dependency). A
  single malformed/truncated reply used to restart the whole 7-stage pipeline
  (run 773). New `_repair_json` tries the brace-balanced extraction (strips
  markdown fences + trailing prose), then `_balance_json` (closes a cut-off
  object — appends missing brackets and an open string). It now runs for ALL
  providers (not just the local `json_mode_fallback` path — run 773 was on
  OpenRouter), logging the raw output before raising a clear error when nothing
  parses. Tests in `tests/test_json_repair.py`.
- **T8** — stage-level retry on JSON-parse failure (`json_parse_retries`, default
  1). After the T21 repair fails, `generate_json` re-requests the SAME stage's
  LLM call rather than letting the error bubble up and trigger CARE's whole-
  pipeline restart (runs 731/773 re-ran all 7 stages). The earlier stages
  (domain / planning / dag) stay valid — only the failed call repeats — so a
  transient bad reply on a weak model costs one extra call, not a full
  regeneration. `json_parse_retries=0` restores the old single-shot behaviour.
  Tests in `tests/test_stage_retry.py`.
- **T9** — schema-constrained decoding enabled for the care profiles
  (`use_schema_constrained_output = true` in `care_default.toml` +
  `care_research.toml`), so the structured stages (domain / planning / dag, which
  call `generate_structured`) get schema-guided decoding where the provider
  supports it. Made safe to enable anywhere: when a `json_schema` request fails
  (the runtime model doesn't honor structured outputs — e.g. deepseek-flash on
  OpenRouter), `generate` now degrades `response_format` to plain `json_object`
  for the remaining attempts (the T21 repair then handles any noisy JSON) instead
  of hard-failing. New tests in `tests/test_schema_constrained_output.py`.
- **T36** — incident regression tests for run 096. New
  `tests/test_incident_regression.py` drives the composite "weather where my IP
  is, what should I wear?" query through the deep pipeline (mocked stage
  responses) and asserts the invariants that failed in 096 now hold: ≥4 atomic
  steps, a clean `lint_chain`, and no `$outer_context` in any url field; plus a
  guard that the exact run-096 describer output (`url=$outer_context`) is rejected
  by both the T5 lint and the T23 schema validator. Parts (a) describer / (b)
  lint / (c) stage-retry of the audit item were already covered by
  `test_describer_tool_signatures.py` / `test_tool_step_schema_validator.py`,
  `test_chain_lint.py` and `test_stage_retry.py`.
- **T28** — atomic, concurrency-safe writes to the generated-tool index.
  `generated_tool_index.register_tool` did a non-atomic read-modify-write of
  `index.json`, so parallel generation could corrupt the file or lose entries.
  The whole upsert now runs under an in-process `threading.Lock` plus a
  best-effort cross-process `fcntl` lock (no-op where unavailable, e.g. Windows),
  and the index is replaced via a temp file + `os.fsync` + `os.replace` (atomic)
  so a concurrent reader never sees a half-written file. Used stdlib rather than
  the audit-suggested `filelock` package to avoid a new dependency. Tests in
  `tests/test_tool_index_atomic.py` (incl. a 24-thread no-lost-updates check).
- **T29** — verified the generated-tool index reaches the planner (already wired
  as WP-9: `CapabilityLookupAgent.lookup` appends `_fetch_generated_tools()`,
  tagged `source: "generated"`, into the `CapabilityContext`) and closed a test
  gap: a new case in `tests/test_generated_tool_index.py` asserts generated tools
  are *added alongside* the memory-registry tools (not replacing them) and pass
  `signature`/`path` through so the planner can reuse them.
- **T30** — verified spec example-cases gate tool registration (already wired as
  WP-8: `validate_tool_source` runs `spec.examples` in the sandbox subprocess and
  fails `report.ok` on any mismatch; `ToolImplementationGenerator` only
  `register_tool`s on `ok`, retries on failure up to `max_attempts`, and persists
  the cases as a `<name>_<hash>_tests.json` regression set) and closed the
  failure-path gap: a new test asserts that when every attempt fails its
  examples, `generate()` returns `None` and ships nothing — no module, no
  regression set, no index entry (never register raw). In
  `tests/test_tool_examples.py`.
- **T31** — `code_generator.py` was silently dropping required CARL fields when
  emitting the Python build script: a `tool` step lost `tool_output_key` (→
  `output_key`, leaving its output unreferenceable), a `memory` step lost
  `memory_namespace`, and a `conditional` step lost `condition_context_key`. All
  three are now emitted. New `tests/test_code_generator_roundtrip.py` checks every
  `CARL_FIELD_MAP` field for each codegen-handled type survives into the generated
  source (and stays in sync with the map), plus an `mmar_carl`-gated `skipif` test
  for the literal exec → `to_json` round-trip.
- **T24** — `ChainEvolver` plateau early-stop is now covered + clarified. The
  WP-3 defaults were already correct (`plateau_patience=3`,
  `plateau_min_delta=0.01`) and the logic sound; the inline counter was extracted
  to a pure `ChainEvolver._plateau_step(best, last_best, counter, *, min_delta,
  patience) -> (counter, stop)` so the early-stop behaviour is unit-testable.
  `tests/test_plateau_logic.py` drives synthetic score sequences proving it stops
  after *exactly* `patience` flat generations (not one early), resets on an
  improvement that exceeds `min_delta`, and treats a ≤`min_delta` improvement as a
  plateau.
- **T25** — the judge's `ChainScore.suspect_gaming` flag (P-5) now influences
  selection. A new `ChainEvolver._rank_key` subtracts a `_GAMING_PENALTY` (0.5)
  from a flagged chain's `overall` and is used wherever survivors / parents / the
  reported best are ranked (`_select` simple + Pareto paths, the final ranking,
  parent biasing, and the per-generation snapshot best). Gamed chains sink below
  honest ones of comparable raw score but aren't hard-excluded; honest chains rank
  exactly by `overall`, so non-gamed runs are unchanged. Tests in
  `tests/test_gaming_penalty.py`.
- **T26** — `ChainGrader.grade` now scores a benchmark's tasks concurrently
  instead of awaiting one judge/execution call at a time. The per-task body moved
  to `_grade_one_task`, run under `asyncio.gather` bounded by a semaphore sized by
  the new `EvolveConfig.grading_concurrency` (default 4; 1 = sequential). `gather`
  preserves order so the assembled `per_task_scores` / failures / notes are
  identical to the sequential version — an ~N× speedup on an N-task benchmark with
  no change in results. Tests in `tests/test_grader_concurrency.py` (concurrency
  reached, semaphore cap respected, results not scrambled).
- **T27** — hardened evolve operators against degenerate chains. The named
  boundary cases were already guarded (`remove_step` skips the source/synthesis
  steps and no-ops on ≤2 steps; `swap_steps` needs ≥4; `crossover` needs ≥2 per
  parent), but the shared `_renumber` sink crashed (`KeyError`) on a step missing
  or sharing a `number`. It now renumbers positionally and drops dependencies that
  point at a missing/ambiguous-duplicate number — unchanged for unique-numbered
  chains. New `tests/test_operator_robustness.py` table-tests every mutator +
  crossover across empty / 1- / 2- / 3-step / mixed-type / no-deps chains
  (no crash, always a valid renumbered chain) plus `_renumber` on
  missing/duplicate numbers.
- **T32** — memory research no longer collapses a non-empty recall to an empty
  digest (the run-096 "hits=4, digest=''"). When the applicability gate (or second
  wave) rejects every hit, `MemoryResearchAgent.research` now returns a labelled
  low-confidence `_weak_digest` of the pre-gate hit titles with
  `was_cold_start=False` instead of an empty digest + false cold-start; and a hit
  set that the purifier reduces to whitespace falls back to the same titles.
  `_filter_applicable` also logs each hit's applicability score vs the threshold
  (kept/dropped). Tests in `tests/test_memory_weak_digest.py`.
- **T33** — session run-facts so a follow-up question reuses earlier results
  instead of cold-starting. New `MemoryManager.save_run_facts(question, facts, *,
  ttl_seconds)` persists a session-scoped `run_facts` card (kind/session_id tag +
  `expires_at`), and `recall_run_facts()` returns the live, same-session facts as a
  digest prefix (filtering other sessions + expired cards). Gated by the new
  `MAGEConfig.memory_session_id`; `MemoryResearchAgent.research` recalls them first,
  leads the digest with them, and treats their presence as `was_cold_start=False`
  even when the recall is empty. (CARE wires this by setting `memory_session_id`
  per session and calling `save_run_facts` after a successful run.) Tests in
  `tests/test_session_facts.py`.
- **T34** — auto failure-cards from execution tool errors, so the feedback loop
  learns from masked failures (not only generation-stage rejections). New
  `memory.find_masked_tool_errors(step_results)` detects failing tool steps,
  including the run-096 "error as data with success=true" masking (an error-shaped
  output on a `tool`/`mcp` step that still reported success), while ignoring
  prose mentions of "error" in `llm` steps. `MemoryManager.save_failure_cards_for_run`
  writes one `stage="execution"` failure card per detected error (gated on
  `enable_failure_card_save`). MAGE generates chains but doesn't execute them, so
  CARE passes the replay's step results after a run. Tests in
  `tests/test_auto_failure_card.py`.
- **T38** — care profiles now declare the quality / replan flags explicitly
  instead of silently inheriting `MAGEConfig` defaults. `care_default`
  (lightweight, recommended) and `care_local` add `enable_replan_generation` /
  `enable_step_critique` / `enable_chain_verification` (all off), and `care_local`
  pins `use_schema_constrained_output = false` since the json_mode_fallback path
  ignores `response_format`; `care_research` (thorough) adds the explicit
  `enable_replan_generation = false`. Replan stays off pending T10. New
  `tests/test_care_profile_flags.py` asserts every care profile declares the four
  flags, that they load, that the tiers are coherent (research verifies + critiques,
  default doesn't), and a cross-profile drift check that no config self-refines
  without verification.
- **T39** — auto-generated config reference + drift check. New
  `mmar_mage.config_docs.generate_config_table` introspects every `MAGEConfig`
  field (name, type, default), joins it with the `MAGE_*` env-var map (extracted
  from `from_env` to a shared `MAGEConfig._env_var_map` classmethod) and the
  `configs/*.toml` profiles, and renders `docs/CONFIG.md` (`python -m
  mmar_mage.config_docs`). `tests/test_config_docs.py` fails if the committed file
  is stale, so an added field / profile / changed default surfaces itself — the
  drift check runs in the test suite (the CI surface once T37 lands).
- **T40** — refreshed the pipeline docs to match the code. The README + ARCHITECTURE
  deep-mode pipeline described ~5 stages but the code runs 6+: the optional research
  stages (incl. `CapabilityLookupAgent`) and `TopologySelector` (between planning and
  the DAG) plus the `StepCritic`/`ChainVerifier`/`SelfRefiner` quality gates are now
  listed, and the ARCHITECTURE fallback table gained `CapabilityLookupAgent` +
  `TopologySelector` rows. Added the incident-in-one-paragraph summary at the top of
  this section. `CARE_INTEGRATION.md` already covered capability + quality stages
  (verified, unchanged). New `tests/test_docs_consistency.py` asserts the docs name
  every core stage and that the documented stage agents exist — a doc-drift guard.
- **T42** — content-addressable cache for the deterministic deep-mode stages.
  domain / topology / capability recompute the same result for the same query
  within a session and across pipeline retries (runs 731/773 ran every stage
  twice per retry). New `mmar_mage.stage_cache.StageCache` keys results by
  `sha256(stage + canonical-JSON(input))`; `MAGEGenerator._cached_stage` wraps the
  three stage calls and stores + returns deep copies so a downstream mutation
  can't taint the cache. Gated by the new `MAGEConfig.enable_stage_cache` (off by
  default — no behaviour change; the shared mechanism Layer A of the evolve design
  builds on). Tests in `tests/test_stage_cache.py`.
- **T43** — overlap the independent deep-mode stages. The research stages
  (memory / web / capability) already ran concurrently inside
  `_run_research_stages`, but topology selection + tool synthesis still ran
  sequentially *before* it even though all three depend only on `domain_analysis`.
  Now `_generate_deep` starts research as a task right after domain analysis and
  awaits it where it's first consumed, so it overlaps topology + tool synthesis
  (≈20-30% less deep-mode latency). There is no cancel point between the task start
  and the await, so it can't be orphaned. New `tests/test_stage_parallelism.py`
  drives `generate()` and uses an event barrier to prove research is in-flight
  while topology runs (a sequential pipeline would deadlock the barrier).
- **T44** — `step_describe_mode="auto"` describes one step per call on models that
  break a long batch JSON reply. The iterative describe path already existed; this
  adds the auto policy: `_resolve_describe_mode(config, num_steps)` picks
  `iterative` when the provider isn't trusted for a long batch reply (OpenRouter /
  a local server without guided decoding — the deepseek-flash class from runs
  731/773) AND the chain has ≥4 steps, else `batch`; OpenAI and guided-decoding
  vLLM stay `batch`. A noisy reply then costs one step's re-request, not the whole
  chain. `care_default` + `care_local` now use `"auto"`. Tests in
  `tests/test_describe_mode.py`.
- **T45** — per-stage timing telemetry for the replay bundle. `MAGEMetadata`
  already carried per-stage token `usage_by_stage` but not timings, so a slow run
  meant digging through the session log. A `_StageTimingRecorder` now wraps the
  progress callback (delegating every other call straight through) and times each
  stage between its `on_stage_start` and `on_stage_complete` / `on_error`; the
  durations land in the new `MAGEMetadata.stage_timings`. `MAGEMetadata.stage_telemetry()`
  merges timings + usage per stage into the dict CARE writes to a replay bundle /
  ledger as `extra.stage_timings`, so a post-mortem reads from one file. Tests in
  `tests/test_stage_telemetry.py`.
- **T6** (CARE-side, spec) — tool errors must be errors. CARE / the CARL runtime
  aren't in this working copy, so the fix is specified for the CARE team in the new
  `docs/care-side-fixes.md`: the executor must set `success=false` on a masked tool
  error (so `retry_max` fires), and the ledger must record `failed`/`degraded`, not
  `success` (run 096 logged an apology as a success). The spec reuses the MAGE-side
  `find_masked_tool_errors` detector shipped in T34 so both sides agree on what
  "looks like an error". `tests/test_docs_consistency.py` guards the spec.
- **T7** (CARE-side, spec) — the synthesiser must own failures, not blame the user.
  Spec'd in `docs/care-side-fixes.md` §T7: branch the answer prompt on the run
  status (the T6 `success=false` signal) — on a system failure, say what went wrong
  and what the system did about it, and never tell the user to rephrase a clear
  question (run 096 apologised *and* blamed the user). Best path: replan/regenerate
  (T10) before synthesis, so the answer reports the recovered outcome. Guarded by
  `tests/test_docs_consistency.py`.
- **T10** — replan reanimated. The machinery was complete (`_promote_checkpoints`
  marks a rollback point, `_build_default_replan_policy` synthesises the real
  `mmar_carl.ReplanPolicy`), but `enable_replan_generation` was off everywhere, so
  no checkpoints and no policy were ever attached (R7, run 096). `care_default` now
  sets `enable_replan_generation = true` — a tool-step failure rolls back to an
  auto-marked checkpoint and replans instead of failing, and the recovery only
  costs on failure. Also lowered the deterministic checkpoint promoter's external-
  call threshold from ≥4 to ≥3 steps so a 3-call composite (IP → geolocate →
  advise) gets a rollback point too. The escalation is step-level `retry_max`
  (retry-with-feedback) → chain-level `replan_from_checkpoint`. Tests in
  `tests/test_replan_enabled.py` (+ existing `test_checkpoint_promotion.py` still green).
- **T11** — cheap verifier in `care_default`. The full `ChainVerifier` is expensive,
  so instead of running it (or critiquing every step) `care_default` now enables a
  step critique **scoped to the risky step types** via the new
  `MAGEConfig.step_critique_step_types = ["tool", "structured_output"]` — the
  `_select_critique_steps` helper filters the chain so the critic only sees tool /
  structured_output steps (where run 096 broke), and the critique is skipped
  entirely when a chain has none. The deterministic lint (T5) still runs for free
  underneath. `care_research` stays the thorough tier (unscoped critique +
  verification). Tests in `tests/test_step_critique_scope.py`; the profile-tier test
  updated.
- **T12** — E2E regression for the composite live query. Added a `live_composite`
  benchmark to `evaluation/tasks.py` (the run-096 "weather where my IP is" query +
  a market-API query) as mandatory regression cases carrying their expectations in
  `metadata` (`min_atomic_calls`, `expect_replan_ready`). New
  `tests/test_eval_live_composite.py` asserts the case is registered and that a
  composite chain is ≥4 steps, lints clean (valid urls, no `$outer_context`), and is
  **replan-ready** — `_promote_checkpoints` marks a rollback point,
  `_build_default_replan_policy` attaches a policy, and tool steps keep `retry_max ≥ 1`,
  so a tool failure retries then replans instead of masking as success. (The actual
  runtime retry/replan is exercised once the real CARL executor replaces the
  evaluation stub — T35.)
- **T13** — fast-path routing for simple queries. Deep mode ran all 7 stages on
  "what's my IP" (88-144s / 44k tokens). `_generate_deep` now checks the
  DomainAnalyzer's verdict right after it runs and, when `_should_fast_path` holds
  (the new `MAGEConfig.enable_fast_path_routing` is on, complexity is `"simple"`,
  and `suggested_step_count ≤ fast_path_max_steps`, default 2), routes to a single
  fast-mode call instead of planning / DAG / describe — reusing the domain analysis
  it already ran. It never fires under parallel sampling. Composite queries are now
  classified `"medium"` (T2), so they stay deep. `care_default` enables it; CONFIG.md
  regenerated. Tests in `tests/test_fast_path_routing.py` (decision matrix + an
  integration proving a simple query makes 2 LLM calls, not the 4-stage deep run).
- **T14** (CARE-side, spec + MAGE helper) — token + cost accounting in the
  ledger / replay. The TUI shows real token counts (WP-5) but the ledger records
  `tokens_in/out` and `cost_usd` as null. The actual usage already lives on
  `MAGEResult.metadata` (WP-5) with a per-stage breakdown via `stage_telemetry()`
  (T45); this commit adds `mmar_mage.cost.cost_usd_for_usage(prompt_tokens,
  completion_tokens, in_per_1k, out_per_1k)` so MAGE and CARE compute the actual
  cost with the same per-1k formula `estimate_chain_cost` uses. The CARE-side wiring
  (write the metadata tokens + computed cost into the ledger / replay) is specified
  in `docs/care-side-fixes.md` §T14. Tests in `tests/test_cost_usage.py`; the spec
  is guarded by `tests/test_docs_consistency.py`.
- **T15** — prompt-bloat audit. Verified the StepPlanner already caps all five
  research digests (memory / web / capability / template / skill) to
  `max_digest_chars` (P1.6), and locked it with a regression that drives the planner
  with five oversized digests and asserts none survives uncapped. The remaining
  unbounded channel was the describer's tool-signature block: `render_tool_signatures`
  rendered every tool in the capability context. It now bounds the block to
  `max_chars` (default ~2400, ≈600 tokens) on whole-line boundaries with an
  "(N more tools available)" note, so a large context can't bloat the describer
  prompt — full signatures stay describer-only (T3) but capped. Tests in
  `tests/test_prompt_bloat.py`.
- **T16** — chain name no longer leaks a memory block. A query polluted with a
  leaked context block produced `General — ## What I remember about you (long-term
  memory)\n- role: ML r` as the chain `name`, because both the chain name and the
  naming heuristic sliced the raw query. New `_clean_query_for_name` strips markdown
  headers, bullet / blockquote lines and memory-marker lines, keeping the actual
  task prose (empty → the caller falls back to the domain title). It's applied at
  the three naming sources: the serialized `CARLChainSchema.name`, the
  `_naming_heuristic` (so both display name and description are clean), and the
  suggested-naming LLM context. Tests in `tests/test_chain_name_sanitize.py`.
- **T17** — session memory. The two halves were already shipped: the empty-digest
  bug (`hits=4` → `digest: ""`) in T32, and the session run-facts store
  (`save_run_facts` / `recall_run_facts`, raised first in `memory_research`) in T33.
  This adds the missing bridge: `memory.extract_run_facts(step_results)` pulls a
  run's successful tool / mcp outputs (the IP a step fetched, the country a
  geolocation returned) into a compact facts dict keyed by each step's output key,
  skipping failed and non-tool steps. CARE calls it on a replay and passes the
  result to `save_run_facts`, so a follow-up question reuses the IP instead of
  re-deriving it from the dialogue text (run 773). Tests in
  `tests/test_extract_run_facts.py`.
- **T18** — live data beats memory. Run 096's plan hard-coded "user in Russia" while
  geolocation returned Sweden (a VPN). Two guards: a new anti-pattern rule in
  `STEP_PLANNER_ANTI_PATTERNS_PROMPT` ("don't bake a remembered personal fact —
  country, city, IP — into a tool/mcp step's aim when that step is what discovers it
  live; the step must DETERMINE the value and downstream steps consume it via
  `$history`"), plus a deterministic backstop in `chain_repair.lint_chain`: a
  tool-step aim containing a personal-fact presumption ("since the user is in …",
  "your country is …", …) is flagged, so it goes through the repair loop instead of
  shipping. Tests in `tests/test_live_data_over_memory.py` (prompt snapshot + lint).
- **T19** — prefer a ready tool over a raw-HTTP chain. `web_search` was available and
  answered the weather part in one call, but the run-096 planner hand-rolled an
  `http_request` chain instead. New planner anti-pattern rule: when a single ready
  tool from the capability context already covers the whole task, use it — raw
  `http_request` chains are the fallback for when no listed tool fits. The rule
  explicitly carves out genuine N-call composites so it doesn't contradict T1's
  one-step-per-external-call rule. Prompt-only (no lint — a multi-`http_request`
  chain can be a legitimate composite, so it must not be flagged deterministically).
  Tests in `tests/test_prefer_ready_tool.py`.
- **T20** (CARE-side, spec) — honest run status in the TUI. The progress widget
  showed a green `✓` on every step of run 096, which actually failed — the masked
  tool error on step 1 was invisible until someone dug through the replay. Spec'd in
  `docs/care-side-fixes.md` §T20: the per-step indicator must render the real status
  (`✓` on success, `⚠`/`✗` on a `success=false` step from T6, `↻` on a replan), reusing
  the MAGE detector `find_masked_tool_errors` (T34) so masked errors are flagged too —
  the execution-side equivalent of MAGE's `MAGEProgressCallback.on_error` for
  generation stages. Guarded by `tests/test_docs_consistency.py`. **This closes the
  CARE-side fixes (T6, T7, T14, T20).**
- **T35** *(blocked: `mmar-carl` not installed)* — replacing the evaluation
  stub-executor with the real CARL runtime. The audit asks for the real runner "as
  in CARE" (`mmar_carl.ReasoningChain`) with mock-HTTP; that package isn't in this
  environment, so the integration can't be built or tested here (untested glue
  against its execution API would be guesswork). The generation-side of the incident
  is already caught — the T5 lint, the T23 schema validators and the T36 e2e
  regression all reject a `$outer_context` url before a chain ships; the
  execution-side catching lands once `mmar-carl` is available (CARE's environment).
  `evaluation/chain_executor.py` keeps its flattened-prompt stub until then.
- **T37** — CI. The repo already ran ruff + mypy + pytest on a Python 3.12/3.13
  matrix in `.github/workflows/ci.yml`; this adds the missing piece — a separate,
  non-blocking `carl-roundtrip` job that best-effort installs `mmar-carl` and runs
  the CARL round-trip tests with an xfail/skip report (`-rxXs`), so the
  optional-dependency tests are visible in CI without blocking the merge when the
  package is unavailable. Guarded by `tests/test_ci_workflow.py`.
- **T41** — repo hygiene. The 09.06 archive accidentally bundled `.venv` (55 MB) and
  `__MACOSX`. `.venv/` was already ignored; added `__MACOSX/` and raw-archive patterns
  (`*.zip` / `*.tar.gz` / `*.tgz`) to `.gitignore`, and a `make package` target that
  builds a clean source cut via `git archive` (git-tracked files only — no `.venv`,
  `__MACOSX`, or caches by construction). Tests in `tests/test_repo_hygiene.py`
  (`.gitignore` coverage, the `package` target, and that the cruft isn't tracked).

  **This completes the failure-audit backlog: 44 of 45 tasks done; T35 (real CARL
  runtime in `evaluation/`) is blocked on `mmar-carl` not being installed.**

## 0.0.1 — first PyPI release (2026-06)

Initial public release of `mmar-mage` on PyPI so it installs with
`uvx mmar-mage` / `pip install mmar-mage` (CARE-MVP unblock). Packaging only —
completed `[project]` metadata (added `[project.urls]`), set the version to
`0.0.1`, and verified the sdist + wheel build with `twine check`. The engine
itself is the SOTA-hardening work documented below.

Install the optional extras as needed: `mmar-mage[carl]` (chain execution via
`mmar-carl`), `mmar-mage[memory]` (gigaevo-memory), or `mmar-mage[all]`.

---

# SOTA Hardening (2026-06)

This release hardens MAGE (Mesh Architecture Generation Engine) toward a SOTA
text-to-agent generator: real research, verified self-checking, diverse
topologies, tool/skill synthesis, full `carl-experiments@agent-features`
coverage, and a benchmark-profile system. All work was landed against the latest
CARL (`mmar_carl@agent-features`) and `gigaevo-memory` (`who-cares`), with every
schema verified against the installed packages rather than guessed.

**Health:** test suite **560 → 791 passing** (+6 `xfail` tracking a known upstream
CARL `from_json` gap), `ruff` clean throughout.

---

## TL;DR (short version)

- **Real deep research** — memory & web research now have applicability gates,
  second-wave gap queries, and source grounding (no more passthrough digests).
- **Self-verifying chains** — every `generate()` can return a `quality_report`
  from a synthesized mini-benchmark; an optional execution-feedback loop repairs
  the weakest chain and re-scores.
- **Diverse topologies** — a `TopologySelectorAgent` + 10-pattern library picks a
  structural topology (pipeline / diamond / debate / map-reduce / …) instead of
  always emitting linear pipelines; optional parallel topology sampling.
- **Tool & skill synthesis** — propose a `ToolSpec` when none exists (Phase A),
  generate a sandbox-validated implementation (Phase B), or author a full
  loader-validated AgentSkill (Phase C). Off by default; sandbox mandatory when on.
- **Smarter authoring** — simplicity bias (AFlow-style), pairwise plan tournament,
  reflective (GEPA-style) evolve mutation + Pareto (quality×cost) selection,
  MIPRO-style bootstrapped demos, anti-pattern catalogue, domain-aware fallbacks.
- **Full agent-features coverage** — correct `to_carl_compat` field maps for every
  step type, required-field validation, risk-based `ReplanPolicy`, skill execution
  modes (`llm_agent`/`subagent` + SHA-pinning + docker runtime), `cache`
  memoization, MCP `http`/`sse` + headers, `step_metrics` analytics.
- **Real gigaevo-memory** — configurable hybrid/BM25/vector search, typed memory
  cards for templates/feedback/benchmarks, scored recall (fixed an
  always-empty-recall bug).
- **Benchmark profiles** — `mage generate --benchmark-profile bfcl|seal_tools|
  rotbench|toolalpaca` presets (allowed step types, simplicity, curated demos).
- **Reliability** — CARL-loadable round-trip smoke test, config-flag orthogonality
  (research stages skipped when the planner is off), non-fatal degradation
  everywhere.

---

## Detailed (long version)

Every item is gated by a `MAGEConfig` flag where noted and falls back gracefully
(non-fatal) on error. New tests accompany each.

### Research & context (deep research that actually works)
- **Memory research applicability gate + second wave** (`enable_memory_research`):
  recalled hits are scored for applicability and low-relevance ones dropped; a
  gap-analysis second wave issues follow-up sub-queries when coverage is thin.
- **Web research source grounding + second wave** (`enable_web_research`): the
  digest numbers its sources and enforces a sourced-claim floor
  (`web_min_sourced_claims`); a second wave fills gaps.
- **Capability / template / feedback agents** moved from passthrough to synthesis
  (applicability gates, skeleton synthesis from a built-in library).
- **Digest hygiene** (`max_digest_chars`): every research digest is truncated on a
  word boundary so unbounded context can't dominate the planner prompt.
- **Skill discovery** (`enable_skill_discovery`): gate → rerank → optional web
  fallback over a built-in AgentSkills registry.

### Self-verification & feedback
- **Benchmark mode** (`enable_benchmark_mode`): new `mmar_mage/benchmark/` module
  synthesizes a mini-benchmark per query, design-judges the chain, and attaches a
  `quality_report` (pass-rate, failure modes, fixes) to every result.
- **Execution-feedback loop** (`enable_execution_feedback`): synth → score →
  repair the weakest chain → re-score, keep-best with early stop. Closes the
  author-once/run-once gap (MIPRO/GEPA/AFlow direction).

### Topology & planning
- **Topology library + selector** (`enable_topology_selection`, on by default):
  10 named patterns with metadata; the selector ranks 1–3 per query and constrains
  the StepPlanner + DAGBuilder toward them.
- **Parallel topology sampling** (`enable_parallel_topology_sampling`): fan out one
  planning pass per distinct topology, keep the best via the pairwise judge.
- **Simplicity bias** (`simplicity_bias`, default 0.5; `simplicity_max_steps`):
  AFlow-style — scales the suggested step count and truncates unjustified runaway
  plans.
- **Pairwise plan tournament**: cold-start candidates are compared head-to-head
  (reliable) instead of absolute-scored.
- **Bootstrapped demos** (`enable_demo_bootstrap`): MIPRO-style — generate candidate
  chains, score on a synth benchmark, inject the top-K as planner few-shots.
- **Anti-pattern catalogue** + advanced-step-type worked examples in the prompts.
- **Smarter fallbacks**: domain/task-type-keyed templates (extraction / analysis /
  comparison / qa / generation / verification) replace `Step N` placeholders.

### Evolve (`mage evolve`)
- **Reflective mutation** (GEPA-style): diagnose the systematic weakness, then
  apply one structural edit as a textual "gradient".
- **Pareto selection** (`enable_pareto_selection`): survivor selection over
  (quality ↑, cost ↓) — a cheaper chain of equal quality is never crowded out.

### Full `agent-features` coverage (CARL export correctness)
- **Field maps** (`carl_export.py`): a single correct `CARL_FIELD_MAP` for every
  step type; `eval`/`aggregator`/`agent_handoff`/`supervisor`/`parallel_sampling`
  translated to `llm` for loadability (interim, pending upstream JSON support).
- **Required-field validation**: typed steps missing essential `step_config`
  (e.g. an `evaluation` with no target) are flagged before export.
- **Risk-based `ReplanPolicy`**: scales depth (rule checker → +LLM judge with
  `mandatory_plus_k_of_rest` voting) and budgets by chain risk — and fixed a latent
  bug where the old policy shape silently dropped all checker config on load.
- **Skill execution modes**: `llm_agent` for tool-call loops, `subagent` for
  isolation, SHA-pinning for remote skills, `docker` runtime for untrusted scripts.
- **Orchestration primitives**: `cache` memoization (auto-cache read-only MCP
  resources), MCP `http`/`sse` transports + auth `headers`.
- **`step_metrics`**: declared success metrics carried as analytics metadata.

### Tool & skill synthesis (`enable_tool_synthesis`, off by default)
- **Phase A**: detect a missing-tool need → propose a `ToolSpec` text artifact to
  the planner (no code execution).
- **Phase B**: generate an implementation, validated by a real sandbox
  (`mmar_mage/security/sandbox.py`: AST static scan + resource-limited subprocess
  smoke run), with a repair retry loop; written to `~/.mage/generated_tools/`.
- **Phase C**: author a full AgentSkill (`SKILL.md` + optional script) that
  round-trips through the real `mmar_carl` loader; written to
  `~/.mage/generated_skills/`.

### Real gigaevo-memory integration
- Configurable search mode (`memory_search_mode`: hybrid / bm25 / vector) + weights
  + namespace, with per-call BM25 fallback.
- Typed memory cards for templates, feedback, ideas, and benchmarks (no longer
  abusing the `chain` entity type).
- Scored recall — `iterative_search` now uses real relevance scores (fixed an
  always-empty-recall bug).

### Benchmark profiles (`--benchmark-profile`)
- `bfcl` / `seal_tools` / `rotbench` / `toolalpaca` / `custom` presets bundling
  `allowed_step_types`, simplicity knobs, a topology preference, a tool catalogue,
  known-quirk notes, and **curated few-shot demo plans** seeded into the planner.

### Reliability & hygiene
- **CARL-loadable smoke test**: round-trips real generated chains through
  `ReasoningChain.from_json` to catch drift between MAGE's validation and CARL's
  loader.
- **Config-flag orthogonality**: research / bootstrap / tool-synthesis stages are
  skipped when `enable_step_planning=False` (their only consumer), so no LLM calls
  are wasted.
- **Cost estimation** (`mmar_mage/cost.py`) drives the evolve budget guard and
  Pareto cost axis.

---

## Known upstream blockers (filed, not worked around)
- **CARL `from_json` is lossy** for newer fields/types: the 6 agent-features step
  types, `loop_back_to`/`loop_config`, `step_groups`, `triggered_by`, and step
  `metrics` do not round-trip (loaded as the deprecated `StepDescription` / dropped
  on serialize). MAGE's round-trip test marks the affected types `xfail(strict)` so
  an upstream fix is detected automatically. Needs upstream `_reconstruct_step_config`
  + `StepConfig` union + a serializable `MetricSpec`.
- **gigaevo-memory live-server features** (cached capability probe, run telemetry,
  lineage-aware evolve, SSE live cache) need a running server to implement/test.

See `docs/IMPROVEMENT_PLAN.md` for the full task-by-task record and
`docs/ARCHITECTURE.md` for the pipeline overview.
