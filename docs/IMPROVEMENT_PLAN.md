# MAGE Improvement Plan — Beat MIPROv2 / GEPA / AFlow / AgentSquare

Last updated: 2026-06-01.

## How to use this plan (status & priority convention)

Every actionable item (the `###` sub-headings P0.x–P3.x) carries a **status tag in square brackets** at the start of its heading:

- `[N]` — **not started.** `N` is the execution-order priority (`1` = do first), taken from the [Sequencing](#sequencing) table. Lower number = higher priority. Items sharing the same `N` are in the same phase and may be done in any order.
- `[wip]` — **in progress.** Set this when you actively start an item (optional but recommended so parallel devs don't collide).
- `[done]` — **finished**: implemented, tested, and merged.
- `[merged]` — superseded by another item; kept as a tombstone so cross-references don't shift.
- `[upstream]` — blocked on a PR to another repo (`carl-experiments` / `gigaevo-memory`); not actionable inside MAGE alone.

**Workflow:** pick the lowest-numbered `[N]` item that is not yet `[done]` → set it `[wip]` → implement + test + merge → flip its tag to `[done]` → move to the next-lowest-numbered open item. The priority numbers already encode the dependency order (P0 correctness blockers are `[1]`; the memory substrate `[2]` precedes the research stages `[3]` that build on it), so following the numbers ascending respects the critical path. When all items are `[done]`, the plan is complete.

## Where we are vs where we want to be

Three parallel research passes (see `873ccbd7-e48c-…` session transcript for full reports) revealed:

1. **Current `deep` mode has 5 "real-work" stages and 7 stubby/passthrough stages disabled by default.** The "deep research" pre-planning loop is currently mostly a façade: memory/web research is one-pass query → digest with no validation, capability/template/feedback agents are pure cache lookups with no synthesis, skill-discovery has an unwired gate.
2. **`carl-experiments@agent-features` ships 9 new step types + ReplanPolicy + StepGroup + StepCache + event/loop edges.** MAGE's `mmar_mage/carl_export.py` field maps are wrong for **7 of the 9** new types — chains emitting `evaluation` / `parallel_sampling` / `agent_handoff` / `supervisor` / `debate` / `mcp_resource` / `human_input` will silently produce invalid CARL JSON.
3. **All 28 historical `gen_failed` rows come from one unreliable generator model (`or-deepseek-v4-flash-free`).** The real performance gap vs competitors (Δ on Seal-Tools, ToolAlpaca, BFCL) is in chain *quality*, not validity. The competitors' core advantage is **iterative refinement against an executable signal**, which MAGE currently lacks.
4. **Three first-class capabilities are missing entirely:** MAGE emits almost-always linear pipelines (no explicit topology selection); cannot self-validate a chain it just generated (synthesizing a mini-benchmark + metric + running the chain on it is wired up inside `mmar_mage/evolve/` but not exposed for the plain `generate()` path); and cannot synthesize new tools / skills (only looks them up in a registry). Each is a structural gap vs at least one competitor — AgentSquare for topology, MIPRO for self-evaluating its own candidates, nobody for tool-synthesis (pure greenfield).
5. **The gigaevo-memory integration uses ~10% of the available surface.** `mmar_mage/memory.py` hard-codes `SearchType.BM25` everywhere and crams chains, feedback, templates, benchmarks, and ideas all into the `chain` entity via a `kind`-discriminator hack. The `who-cares` branch offers hybrid BM25+vector search with a reranker, five typed entity routers (incl. Memory Cards and Agent Skills), run telemetry, lineage/versioning, and an SSE event stream — none of which MAGE touches. Since retrieval quality is the input to every research / cold-start / demo stage, this is the substrate gap that quietly caps everything else.

**The goal of this document:** be measurably better than MIPROv2 / GEPA / AFlow / AgentSquare on BFCL / Seal-Tools / RoTBench / ToolAlpaca at the same authoring budget — and gain a *structural* edge by exploiting `agent-features` step types that competitors cannot emit (debate, supervisor, parallel-sampling, replan-policies) and by adding tool / skill synthesis that no current competitor offers.

---

## P0 — Correctness blockers (nothing else works if these are broken)

### [done] P0.1 Rewrite `carl_export.py` field maps to match `mmar_carl@agent-features`

> **Done 2026-06-01.** Rewrote `mmar_mage/carl_export.py` against the real `mmar_carl@agent-features` schemas (`config.py`, `agent_skill.py`). Consolidated into a single `CARL_FIELD_MAP` table. Fixed CARL target names for `evaluation` (`evaluates_step`/`criteria`/`on_fail`/`evaluation_method`, dropped non-existent `threshold`/`rule`), `human_input` (`fallback_value`), `mcp_resource` (`resource_uri`/`output_memory_key` + nested `server`). Reshaped `debate` (`debate_agents` list → `roles`+`role_prompts`) and `tool_discovery` (`discovery_sources` list → single `source`). Kept MAGE flat names (they are contract: `schemas.py`/`prompts.py`/`cost.py`). The 3 runtime-only types (`agent_handoff`/`supervisor`/`parallel_sampling`) translate to `llm` (interim P0.3 option A — see below). New `tests/test_carl_export_field_maps.py` validates produced `step_config` against the **real** CARL `*StepConfig` classes (loaded via importlib stubs; skips if checkout absent). 410 tests pass, ruff clean.

`mmar_mage/carl_export.py` has wrong field names for almost every new step type. Today the chain validates inside MAGE but `ReasoningChain.from_json(chain)` will silently drop or reject fields.

Concrete edits (one PR per row; each is ~10 lines):

| Step type | MAGE emits today | CARL expects (`models/config.py`) |
|---|---|---|
| `evaluation` | `evaluation_target_step` / `evaluation_method` / `evaluation_criteria` / `evaluation_fail_action` / `evaluation_threshold` / `evaluation_rule` (`carl_export.py:84-91`) | `evaluates_step` / `evaluation_method` / `criteria` / `on_fail` / `input_source` / `max_retries` (`config.py:401-477`) |
| `parallel_sampling` | `sampling_n` / `sampling_temperature` / `sampling_aggregation` / `sampling_judge_prompt` (`carl_export.py:94-99`) | `n_samples` / `aggregation` / `judge_prompt` / `normalize_for_vote` — and `base_step: LLMStepDescription` is required at runtime (`config.py:480-513`) |
| `agent_handoff` | `handoff_chain_path` / `handoff_input_mapping` / `handoff_output_key` (`carl_export.py:102-106`) | `input_mapping` / `output_memory_key` / `output_namespace` / `propagate_failure` / `inherit_tools` / `timeout`; `sub_chain` is runtime-only (`config.py:271-338`) |
| `supervisor` | `supervisor_routes` / `supervisor_routing_prompt` (`carl_export.py:117-119`) | `routing_prompt` / `task_source` / `fallback_agent` / `input_mapping` / `output_memory_key` / `output_namespace` / `propagate_failure` / `inherit_tools` / `timeout` / `llm_config`; `agents` is runtime-only (`config.py:877-971`) |
| `debate` | `debate_agents` / `debate_rounds` / `debate_judge_prompt` (`carl_export.py:122-126`) | `roles` / `role_prompts` / `rounds` / `task_source` / `judge_prompt` / `output_memory_key` / `output_namespace` / `llm_config` / `role_llm_configs` (`config.py:974-1062`) |
| `mcp_resource` | `mcp_resource_uri` / `mcp_resource_memory_key` (`carl_export.py:130-132`) | `resource_uri` / `output_memory_key` / `output_namespace` / `timeout` + nested `server: MCPServerConfig` (`config.py:142-187`) |
| `human_input` | `human_default` (`carl_export.py:113`) | `fallback_value` (`config.py:631-685`) |
| `tool_discovery` | `discovery_sources: list[dict]` | `source: ToolSource` (single union, not a list) (`config.py:516-628`) |
| `agent_skill` | mostly correct (`carl_export.py:62-81`) | missing `cache_skills` / `skill_cache_dir` / `persist_workspace` / `output_capture` / `output_files_glob` / `output_schema_strict` / `output_file_key` |

**Implementation note**: do the rewrite *as a table* (`CARL_FIELD_MAP: dict[StepType, dict[str, str]]`) instead of inline branches. Single source of truth, easier to extend, generatable from the CARL schemas.

### [done] P0.2 Round-trip CI test for every step type

> **Done 2026-06-01.** Added `tests/test_carl_round_trip.py`: parametrized over all 16 `CARL_VALID_STEP_TYPES`, builds a minimal 2-step chain, runs `validate_carl_json(strict_carl=True)`, then the **real** `ReasoningChain.from_json(json.dumps(to_carl_compat(chain)))`. Installed `mmar_carl` editable from the checkout (`uv pip install -e … --no-deps` + `simpleeval` + `mmar-utils`; PyPI is reachable via uv now) so the boundary runs for real here; tests `skipif` it's absent. Also de-fanged P0.1's test (`test_carl_export_field_maps.py`) — it stubbed `sys.modules["mmar_carl"]`, which shadowed the real package and made these skip; now both import the real package directly. **437 passed, 6 xfailed, ruff clean.**
>
> **Major finding (real-execution payoff).** CARL's `agent-features` branch **cannot round-trip the new step types through JSON**, on either load path:
> - **6 stable types round-trip with config intact:** `tool`/`mcp`/`memory`/`transform`/`conditional`/`structured_output`.
> - **`agent_skill` CRASHES `from_json`:** `_reconstruct_step_config` builds an `AgentSkillStepConfig`, but the generic `StepDescription.step_config` Union (`config.py` `StepConfig`) omits it → pydantic rejects (6 errors).
> - **`evaluation`/`human_input`/`debate`/`mcp_resource`/`tool_discovery` LOAD BUT SILENTLY DROP CONFIG** (`step_config=None`): `_reconstruct_step_config` has no branch for them, and the typed path (`from_dict_typed`) raises `Unknown step type` for them.
> - These 6 are marked `xfail(strict=True)` in the test, so when CARL is fixed upstream they flip to XPASS and tell us to promote them.
> This gap is **broader than P0.3** (which only covered the 3 runtime-only types) and is the hard blocker for **P2.1** (emit new step types reliably). See the new upstream item in the Cross-cutting section. Interim safety: P0.1 already translates the 3 runtime-only types to `llm`; the safe set MAGE can emit and load today is the 6 stable types + `llm`. The other agent-features types currently degrade (drop config) or crash on load — a follow-up should decide whether to extend the P0.1 `llm`-translation to them until upstream lands (a P2.1 / P0.3 decision).

This is the single highest-ROI quality investment because it catches P0.1-class drift permanently.

### [done] P0.3 Decide the runtime-only-fields policy

> **Decided & shipped 2026-06-01.** The policy, now implemented end-to-end:
> 1. **Runtime-only types** (`agent_handoff`/`supervisor`/`parallel_sampling`) → translated to `llm` with a synthesized aim at export (option A interim; shipped P0.1). They can never round-trip (need a runtime `sub_chain`/`agents`/`base_step`), so `llm` makes them functional today.
> 2. **CARL-load failures are non-fatal by default** (this iteration). `MAGE`'s `validate_carl_json(strict_carl=True)` is the authority for chain validity; `ReasoningChain.from_json` is a best-effort cross-check that currently lags its own runtime (it crashes on `agent_skill`, drops config on `evaluation`/`debate`/`mcp_resource`/`human_input`/`tool_discovery` — P0.2). So `_try_load_reasoning_chain` now logs a warning and returns `None` instead of aborting generation, fixing the latent `agent_skill` generation crash **without** discarding P0.1's correct field maps. New `MAGEConfig.raise_on_carl_load_error: bool = False` restores hard-fail for strict pipelines. New `tests/test_carl_load_policy.py` (5 tests) covers the real `agent_skill` crash path (non-fatal vs strict), valid-chain load, and the not-installed skip. **452 passed, ruff clean.**
> 3. **Option B (upstream) is the long-term answer** and remains open — tracked as the two upstream items in the Cross-cutting section (extend `_reconstruct_step_config` + the `StepConfig` union; load `sub_chain`/`agents`/`base_step` from a registry). When it lands, shrink the export-time `llm`-translation set and the P0.2 `xfail`s flip to XPASS automatically.
>
> Net effect: **every chain MAGE emits today either loads into CARL or degrades gracefully with a warning — none abort generation.** The moat path (first-class agent-features steps) is preserved in code, gated on the upstream fix.

**Original analysis (for reference):**

`agent_handoff.sub_chain`, `supervisor.agents`, `parallel_sampling.base_step` are `Field(..., exclude=True)` runtime Python objects (`mmar_carl/models/steps.py:447-451, 504-508`). CARL's JSON loader does **not** reconstruct them.

Three options:

- **(A) Translate to `llm` like `eval` / `aggregator`** — safe and works today, but loses all the structure that makes these types interesting. Recommended for `human_input` only.
- **(B) Extend `mmar_carl` to load sub-chains from a chain-path registry** — needs a Glazkov-side change to `chain.py:2376+`. **This is the right long-term answer.** File an issue first.
- **(C) Emit these step types only for in-process Python use and never serialize them** — splits the API surface, makes the JSON path a strict subset. Not recommended.

Decision needed before P2.1 starts. Default plan: pursue (B) in a parallel PR on `carl-experiments`, fall back to (A) if blocked.

> **Interim (A) shipped in P0.1; full policy decided in P0.3 (see the `[done]` summary above).** The only remaining open thread is the upstream option-B PR, tracked in the Cross-cutting section.

---

## P1 — Make "deep research" actually deep

### [done] P1.1 Memory research: multi-hop + re-rank + applicability gate

> **Done 2026-06-01.** `MemoryResearchAgent` now does real multi-hop research on top of P1.18's scored recall:
> - **Applicability gate** (`_filter_applicable`): one LLM judge (`MEMORY_APPLICABILITY_PROMPT`) rates each recalled hit 0-1; hits below `memory_applicability_threshold` (0.5) are dropped and the rest ordered best-first. Gated by `enable_memory_applicability_gate` (default True); **non-fatal** (judge failure keeps the unfiltered hits).
> - **Second wave** (`_second_wave` + `_generate_gap_subqueries` + `MEMORY_GAP_SUBQUERY_PROMPT`): when the gate leaves fewer than `memory_min_applicable_hits` (2) hits *and wave 1 found content* (skipped on a true cold start), generate gap-targeting sub-queries, recall again (via new `iterative_search(extra_queries=…)`), re-gate, and merge-dedup. Non-fatal.
> - New `_summarize_structure` keeps both judge/gap prompts compact. New config: `enable_memory_applicability_gate`, `memory_applicability_threshold`, `memory_min_applicable_hits` (+ env). Tests: `tests/test_memory_applicability_gate.py` (12). **482 passed, ruff clean.**
>
> Original change list: ✅ applicability gate · ✅ second-wave gap sub-queries · `was_cold_start` reasoning — *implicit* (an empty digest already signals cold start to the planner) · 500-token digest cap — **covered by P1.6** (planner caps every digest at `max_digest_chars`).

### [done] P1.2 Web research: source-grounded multi-pass with citations

> **Done 2026-06-01.** `WebResearchAgent` is now source-grounded multi-pass:
> - **Citations:** `_create_digest` presents results as a NUMBERED list and the rewritten `WEB_DIGEST_PROMPT` requires every claim to carry a `[N]` source marker and to **omit anything it can't attribute** ("do not include unsourced claims").
> - **Second wave:** if the digest cites fewer than `web_min_sourced_claims` (3) distinct sources, `_second_wave` generates gap-targeting queries (`WEB_GAP_SUBQUERY_PROMPT`), searches them, merges results + source URLs, and re-digests. `0` disables it. Non-fatal.
> - `_count_sourced_claims` counts distinct `[N]` markers as the grounding proxy. New config `web_min_sourced_claims` (+ env). Tests: `tests/test_web_source_grounding.py` (6) — marker counting, numbered-source prompt, second-wave fires/merges/re-digests, skipped-when-well-sourced, disabled-by-zero; 3 provider-mechanics tests in `test_web_research.py` + the skill web-fallback test set `web_min_sourced_claims=0`. **498 passed, ruff clean.**
>
> Note: kept `WebResearchResult.sources` as `list[str]` (URLs) — the structured `{id,url,title,snippet}` form would break `discovery.py` consumers; the `[N]` markers index into the numbered sources, which is enough for traceability.

### [done] P1.3 Capability / Template / Feedback: from passthrough to synthesis

All three currently return empty if the memory cache misses (`capability_lookup_agent.py:64-77`, `template_assembler_agent.py:77-81`, `feedback_recall_agent.py:137-146`).

Changes:
- ✅ **CapabilityLookupAgent** — *done 2026-06-01.* On a registry miss it synthesizes plausible tool/MCP suggestions from the query via an LLM (`CAPABILITY_SYNTHESIS_PROMPT`), tagged `source="synthesized"`. The agent gained optional `llm` + `query`/`domain_analysis` params (backward-compatible — no-LLM/no-query → empty context as before); the generator passes `self._llm` + query + domain and **labels synthesized capabilities as suggestions** in the planner context (not confirmed-available). Gated by `enable_capability_synthesis` (default True); non-fatal. Tests: `tests/test_capability_synthesis.py` (6). **518 passed, ruff clean.**
- ✅ **TemplateAssemblerAgent** — *done 2026-06-01.* On a memory miss it now synthesizes a built-in skeleton instead of returning `None`: new `mmar_mage/templates/` library (`generic_tool_use`, `multi_source_analysis`, `extract_transform_emit`, `critique_revise`, `default_pipeline`) + `select_skeleton(task_type, domain)` keyword selector. The skeleton carries `template_id="builtin:<name>"` and `quality_score=0.0` (soft, unproven guidance — the planner already treats skeletons as hints, not templates). Deterministic, no LLM. Tests: `tests/test_template_skeletons.py` (7) + updated `test_template_assembler.py`. **505 passed, ruff clean.**
- ✅ **FeedbackRecallAgent** — *done 2026-06-01.* After the quality filter, `_filter_applicable` runs an LLM judge (`FEEDBACK_APPLICABILITY_PROMPT`) rating how relevant each past-execution lesson is to the *current* query; records below `feedback_applicability_threshold` (0.5) are dropped before the digest, the rest ordered best-first. Records are addressed by **list index** (robust to missing `chain_entity_id`). Gated by `enable_feedback_applicability_gate` (default True); **non-fatal**; drops-all → empty digest. Reuses the P1.1 pattern. Tests: `tests/test_feedback_applicability_gate.py` (7) + updated 2 recall tests + the full-pipeline integration test. **512 passed, ruff clean.**

**All three sub-agents done → P1.3 complete.** This clears the entire `[3]` block (P1.1, P1.2, P1.3, P1.4, P1.5 all `[done]`).

### [done] P1.4 Wire the SkillDiscovery gate

> **Verified 2026-06-01 — already correct; the plan's premise was an inaccurate inference.** `mmar_mage/skills/discovery.py:160-161` already short-circuits: `if not gate.get("needs_skill"): return SkillSearchResult(...)` — the registry pre-filter + rerank only run when the gate returns `needs_skill=true`. The gate field is `needs_skill` (bool), not `relevance=="no"`. This is locked by an existing test, `tests/test_skills.py::TestDiscoveryGate::test_gate_skips_for_pure_reasoning`, which asserts that on `needs_skill=False` **only one LLM call (the gate) fires — no rerank**. 17 skill tests pass. No code change needed.

### [done] P1.5 Cold-start multi-sampling: default 3 candidates with pairwise scoring

> **Done 2026-06-01.**
> - **Research-aware candidate default:** a `MAGEConfig` `model_validator` bumps `cold_start_candidates` 1→3 when `enable_memory_research=True` (cold-start multi-sampling pays off precisely when the planner has no recalled context). The global default stays 1 (no contract change for non-research configs); explicit values are respected.
> - **Pairwise tournament:** `_score_plans` now runs a single-elimination tournament — the incumbent best is compared head-to-head against each remaining candidate via the new `_pick_better_plan` + `PLAN_PAIRWISE_PROMPT` (N-1 calls), winner advances. Reasoning models judge two options far more reliably than scoring many against an absolute rubric. A failed comparison keeps the incumbent (non-fatal). `PLAN_SCORER_PROMPT` is retired from the generator (kept in `prompts.py`).
> - Tests: `tests/test_plan_scoring.py` (10) — winner A/B, missing-winner→incumbent, error→incumbent, single-candidate-no-LLM, 3-way tournament (N-1 comparisons), incumbent-sweeps, + the research-aware default. **492 passed, ruff clean.**

### [done] P1.6 Length-limit and structure-aware research digests

> **Done 2026-06-01 (length-cap half; structured-fragments half deferred to P1.1/P1.2).** Added `MAGEConfig.max_digest_chars: int = 1200` (~300 tokens; `0` disables) + `MAGE_MAX_DIGEST_CHARS` env mapping. `StepPlanner.plan` now truncates **every** digest channel (memory / web / skill / capability / template) on a word boundary via a new `_truncate_digest` helper before assembling the prompt, with a visible `…[truncated]` marker; the generator passes `max_digest_chars` from config at both planner call sites (cold-start + single). New `tests/test_digest_capping.py` (10 tests): helper unit tests (word-boundary, no-whitespace, disable, marker) + real-path tests that drive `StepPlanner.plan` with a recording LLM and assert oversized digests are capped before the LLM call. **447 passed, ruff clean.**
>
> **Deferred to P1.1/P1.2:** "truncation by relevance score" and "structured fragments (`{retrieved_tools, retrieved_patterns, domain_facts}`)" both require the research agents to emit *scored, structured* digests, which they don't yet (they emit free-text). `_truncate_digest`'s docstring notes the relevance-aware upgrade hook. The char-cap delivered here is the core anti-bloat value and is independent of that work.

Originally:
- Cap each digest to 300 tokens with truncation by relevance score, not by character count.
- Emit digests as *structured fragments* (`{retrieved_tools: [...], retrieved_patterns: [...], domain_facts: [...]}`) instead of free-text — easier for the planner to pick what to use.

---

## P1 — Close the gap to MIPRO / GEPA / AFlow / AgentSquare

These are the four moves competitors make that MAGE structurally cannot today. Closing the gap is necessary to reach parity; the next section (`agent-features` step types) is how we *beat* them.

### [done] P1.7 Bootstrapped few-shot demo synthesis per benchmark (MIPRO-style)

> **Done 2026-06-01.** The single biggest known quality lever for prompt-based agent authoring (MIPROv2 §4.2). **Resolved the deferral's "evolve-loop mismatch"**: the plan first imagined this living in the evolve loop, but `_next_generation` only applies structural operators — it never re-invokes a prompt module, so bootstrapped demos can't help there. The optimizable prompt module in MAGE is the **StepPlanner**, so the bootstrap lives at the generator's StepPlanner stage (which *also* transitively lifts evolve's initial population, since `_build_initial_population` calls `generate()`).
> - **`mmar_mage/bootstrap.py`** (new): `DemoBootstrapper.bootstrap(query, domain, candidate_chains, benchmark, top_k, min_pass_rate)` scores each candidate on a synthetic benchmark via P1.16's `ChainRunner` + `QualityReporter`, keeps the top-K clearing a pass-rate floor (sorted desc), and returns `BootstrapDemo`s (compact step plans). `format_demos_for_planner` renders them; `_compact_plan` strips to `{number,title,step_type,aim,dependencies}`. Decoupled from generation (it *receives* candidates) → no generator import cycle, trivially testable.
> - **StepPlanner** gained `bootstrapped_demos: str = ""`, injected last in the context ("Worked examples (verified on a synthetic benchmark)") — closest to the generation point.
> - **Generator** stage 1.5 (`_bootstrap_demos`, gated by `enable_demo_bootstrap`): synth a benchmark slice → generate `demo_bootstrap_candidates` candidates via **fast mode** (one LLM call each, no deep-pipeline recursion) → score → format → thread into both `planner.plan(...)` calls. Non-fatal: any failure → no demos, pipeline continues.
> - New config: `enable_demo_bootstrap` (off), `demo_bootstrap_candidates=4`, `demo_bootstrap_top_k=2`, `demo_bootstrap_min_pass_rate=0.5` (+ env + int/float/bool coercion).
> - **Real-execution check**: `test_bootstrap_demos_end_to_end` drives the full path — synth (2 tasks) → 2 fast-gen candidates → runner scores A=pass/B=fail → only A's plan survives into the formatted demos, with exactly one fast-gen per candidate.
> - Tests: `tests/test_demo_bootstrap.py` (15) — bootstrap floor/top-K/sort/empty-cases/scoring-failure-drops, compact+format rendering, StepPlanner injection on/off, generator end-to-end + no-tasks + non-fatal. **587 passed, 6 xfailed, ruff clean.**
>
> Cost: `candidates × (1 fast-gen + n_cases judge) + 1 synth` extra calls when on (off by default; turn on for hard benchmarks). Sharper once P1.16's real-executor `ChainRunner` mode lands (the run is currently a design-judge).

### [done] P1.8 Reflective mutation with Pareto filtering (GEPA-style)

> **Done 2026-06-01.** Added a reflection-first evolutionary operator that closes the GEPA gap: instead of guess-and-check, the LLM first emits an explicit `diagnosis` of the *systematic* weakness behind the per-task failures, then proposes ONE structural edit as a textual "gradient" from it.
> - **`REFLECTIVE_MUTATION_PROMPT`** (`prompts.py`): sees query + chain JSON + overall score + per-task failures → returns `{diagnosis, edit_type, target_step_number, new_step, rationale}` where `edit_type ∈ {insert_step, remove_step, swap_steps, rewrite_step, change_step_type}`.
> - **`reflective_mutate(chain, score, query, llm)`** (`evolve/operators.py`): short-circuits when there are no failures, logs the diagnosis, applies the edit. Extracted the per-edit-type application body out of `llm_repair` into a **shared `_apply_edit(chain, edit_type, target, new_step)`** helper (deepcopy-isolated, renumbers + rewires deps, unknown edit → unchanged) — `llm_repair` now also routes through it.
> - **Wired into the engine**: new `EvolveConfig.weight_reflective_mutate=1.2`; `ChainEvolver._pick_operator` weights dict + `_apply_operator` dispatch (`reflective_mutate` → CARL-compat → strict-validate → cost-guard → `Individual`). The invalid/over-budget mutant is dropped exactly like the other operators.
> - **Real-execution check**: `test_engine_dispatches_reflective_mutate_to_valid_child` drives the full path `_pick_operator → _apply_operator → reflective_mutate → _apply_edit → to_carl_compat → validate`, asserting a valid child Individual tagged `operator="reflective_mutate"`.
> - Tests: `tests/test_reflective_mutation.py` (12) — `_apply_edit` (remove/insert/rewrite/unknown/no-mutation-of-input), `reflective_mutate` (applies edit, no-failures short-circuit, LLM-failure & unknown-edit return unchanged, correct `stage`), `llm_repair` refactor regression, engine dispatch. **572 passed, 6 xfailed, ruff clean.**
>
> **Pareto filter on quality × authoring cost — [done 2026-06-02].** Survivor selection (`ChainEvolver._select`) is now Pareto-aware over (quality ↑, cost ↓) when `EvolveConfig.enable_pareto_selection` is on (OFF by default — pure-quality selection unchanged): a `_pareto_front` helper keeps the non-dominated set first (a cheaper chain of equal quality is never crowded out by a pricier one), best-quality first, then fills remaining slots by quality. The global best is always non-dominated, so elitism is preserved. Cost axis = `estimate_chain_cost(chain_dict).total_tokens` (pricing-free). Pure evolve logic — no CARL round-trip, so unblocked. **Real-execution check**: `test_pareto_select_keeps_cost_efficient_over_redundant_pricey` runs the real `ChainEvolver._select` with real cost estimation and shows Pareto keeping the cost-efficient chain where pure-quality keeps the redundant pricey one; + front/elitism/cap tests. Tests: `tests/test_pareto_selection.py` (8). **Also fixed a latent flaky test** the full suite surfaced — `test_engine_dispatches_reflective_mutate_to_valid_child` relied on a high-but-finite operator weight (≈6% chance of a wrong pick under some RNG seeds); now zeroes the competing weights for a deterministic pick. **783 passed, 6 xfailed, ruff clean.**

### [done] P1.9 Execution-feedback loop (close the run-once-author-once gap)

> **Done 2026-06-01.** The single largest structural gap vs MIPRO/GEPA/AFlow — MAGE no longer authors-once/runs-once. New stage 5.7 (`MAGEGenerator._run_execution_feedback`, gated by `enable_execution_feedback`, runs after CARL-export validation and **before serialization** since it may rewrite the chain), built directly on P1.16's verification pipeline:
> 1. Synthesize one mini-benchmark (`BenchmarkSynthesizer`) and score the chain (`ChainRunner` → `QualityReporter`).
> 2. While `pass_rate < execution_feedback_threshold` (0.7), `_repair_chain` rewrites the chain guided by its **failing cases** (input + rubric + reason) via a new whole-chain `EXECUTION_REPAIR_PROMPT`, re-validated strict-CARL (invalid repair → discarded).
> 3. Re-score; **keep the best chain**, stop on no improvement or after `max_execution_feedback_iterations` (2).
> The loop's final `QualityReport` becomes the result's `quality_report` (supersedes the single-shot stage-7 verify when on). Used a whole-chain repair prompt rather than evolve's structural-edit `LLM_REPAIR_PROMPT` (which needs an evolve operator to apply) to keep the loop self-contained. New config: `enable_execution_feedback`, `execution_feedback_threshold`, `max_execution_feedback_iterations` (+ env). Tests: `tests/test_execution_feedback.py` (8) — threshold-met-skips-repair, repair-improves-adopted (4 LLM calls), no-improvement-keeps-best, invalid-repair-discarded, no-tasks, + `_repair_chain` cases. **560 passed, ruff clean.**
>
> Cost: ~3× the per-query authoring budget when on (off by default). Worth it on hard benchmarks (RoTBench, ToolAlpaca). Sharper still once P1.16's real-executor `ChainRunner` mode lands (currently the run is a design-judge).

### [done] P1.10 Simplicity bias + bounded operator vocabulary (AFlow-style)

> **Done 2026-06-01.** AFlow wins on bfcl_v3 with ≤4-operator chains; MAGE defaulted to 5-8 steps. Two levers, both defaulting ON (tunable off):
> - **`simplicity_bias: float = 0.5`** — `_apply_simplicity_bias(count, bias)` (`generator.py`) scales `domain_analysis.suggested_step_count` by `(1 - bias)`, applied right after domain analysis so it flows into both the StepPlanner prompt and the fallback plan. Clamped: `bias` to `[0, 1]`, result rounded and floored at `_SIMPLICITY_MIN_STEPS=2` so a chain never collapses below decompose→act; counts ≤2 and `bias≤0` are no-ops.
> - **`simplicity_max_steps: int = 7`** — `_enforce_step_budget(plan, cap)` truncates a plan exceeding the cap **only when its `rationale` is empty** (the unjustified-runaway case). `PlannedStep` carries no dependencies (DAGBuilder rewires from the truncated plan), so first-N truncation + renumber is structurally safe. A filled `rationale` exempts the plan — that's the model justifying the complexity, matching the plan's "unless rationale filled". Reused the existing `StepPlan.rationale` rather than adding a redundant `reasoning` field.
> - **`STEP_PLANNER_PROMPT`**: replaced "steps should match suggested_step_count" with "SIMPLICITY FIRST: aim ≤4 steps, treat suggested_step_count as an UPPER bound, only exceed 5 for genuine multi-hop and justify in `rationale`; no ceremonial analyze/format steps."
> - New config (+ env + int/float coercion): `simplicity_bias`, `simplicity_max_steps`.
> - **Real-execution check**: `test_generate_applies_default_simplicity_bias` / `…_with_bias_disabled` drive `generate()` on the fallback path — default 0.5 yields a 2-step chain (round(3·0.5)=2), `simplicity_bias=0.0` yields the full 3. Updated `test_deep_mode.py` fallback assertion 3→2 to encode the new default.
> - Tests: `tests/test_simplicity_bias.py` (12) — bias halve/round/floor/noop/clamp + budget truncate/justified-keep/under-cap/disabled + 2 generate() integration. **599 passed, 6 xfailed, ruff clean.**
>
> **Behavioral note:** this makes chains ~half as long *by default* — deliberate (the plan specified `0.5`), since the target benchmarks reward simplicity. Multi-hop tasks are protected by the 2-step floor and the rationale exemption; set `simplicity_bias=0.0` to restore prior behavior.

---

## P1 — New capabilities: diverse topologies, benchmark mode, tool/skill synthesis

Three capabilities the generator needs that don't exist today. Topologies and tool synthesis are *structural* edges over the four competitors; benchmark mode is the integration layer that makes everything else reproducible.

### [done] P1.15 Diverse topology generation (today MAGE mostly emits linear pipelines)

> **Core done 2026-06-01 (parts 1-3, 5-6).** MAGE now makes topology choice *explicit* instead of drifting to linear pipelines. New stage 0.5 (`enable_topology_selection`, **on by default**) runs after `DomainAnalyzer`, before `StepPlanner`:
> - **Topology library** (`mmar_mage/topology/`): `schemas.py` (`Topology`, `TopologyChoice`, `TopologyCandidates`) + `library.py` — all 10 catalog entries from the table, each with `when_to_use` / `required_step_types` / `example_chain` / `complexity_score`. Helpers: `get_topology` (case-insensitive), `list_topologies` (simplest-first), `describe_for_prompt`, `format_topology_hint` (pipeline/unknown → empty hint), `default_choice`.
> - **`TopologySelectorAgent`** (`agents/topology_selector.py`): one `generate_json` call → 1-3 ranked `TopologyChoice`s; `_parse` validates names against the catalog, drops unknowns/dups, clamps confidence, caps at `topology_max_candidates`. Graceful — LLM failure or no-valid-candidate falls back to `pipeline`.
> - **`TOPOLOGY_SELECTOR_PROMPT`** (`prompts.py`): catalog table + "prefer the simplest topology that fits; complex ones must earn their cost in `rationale`" (composes with P1.10's simplicity bias).
> - **Topology-aware planning** (part 3): `StepPlanner.plan(topology_hint=…)` injects the constraint first (frames the whole plan); `DAGBuilder.build(topology_hint=…)` wires dependencies toward the pattern. Generator threads the best choice's hint into both (and the cold-start fan-out).
> - **Prompt rebalancing** (part 5): added explicit per-pattern guidance to the selector; the heavier corpus swap (replacing 3 pipeline few-shots in `STEP_PLANNER_PROMPT` with diamond/debate/critique_revise examples) is a deferred polish.
> - **Part 6**: P2.7 ("Explicit DAG pattern selector") is superseded by this — leave it marked superseded.
> - New config (+ env + coercion): `enable_topology_selection` (True), `topology_max_candidates` (3).
> - **Real-execution check**: `test_generate_runs_topology_stage_and_passes_hint` drives `generate()` with a `diamond` pick and asserts (a) `topology_selection` ∈ `deep_stages_completed`, (b) `"Target topology: diamond"` reaches the StepPlanner prompt; the pipeline-pick test asserts no constraint is injected.
> - Tests: `tests/test_topology.py` (16). Updated the shared `conftest.mock_openai_deep` (+topology slot) and 5 deep-pipeline tests (disable the stage where orthogonal, or insert a pipeline response). **615 passed, 6 xfailed, ruff clean.**
>
> **Part 4 (parallel topology sampling) — [done 2026-06-02].** When the selector returns >1 *distinct* topology and `enable_parallel_topology_sampling` is on (OFF by default — a K× planning-token multiplier), the generator fans out one planning pass per topology and keeps the best via the P1.5 pairwise judge — exploring diverse structures, not just `candidates.best`. The topology stage now captures the distinct candidate hints (`topology_hints`, deduped, pipeline→"" once); the planning stage gained a `_plan(topology_hint)` closure (refactor — also de-duplicated the cold-start branch) and a parallel branch that `asyncio.gather`s a plan per hint → `_score_plans`. New config `enable_parallel_topology_sampling` (+ env). **Real-execution check**: `test_parallel_sampling_fans_out_per_topology_and_picks_winner` drives `generate()` with diamond+debate candidates, asserts 2 planning passes carrying the 2 distinct topology hints, and that the pairwise winner drives the final chain; + off-by-default / single-topology-no-fanout / pipeline+one / env-plumbing. Tests: `tests/test_parallel_topology_sampling.py` (5). **775 passed, 6 xfailed, ruff clean.** P1.15 is now fully complete (parts 1-6).

**Symptom.** `DAGBuilder` knows about `pipeline` / `diamond` / `funnel` / `map-reduce` / `critique-revise` patterns (`DAG_BUILDER_PROMPT` few-shots, `prompts.py:640-713`), but pattern choice is implicit — inferred from whatever the `StepPlanner` happened to emit. `StepPlanner` few-shots are heavy on linear sequences, so most generated chains end up linear regardless of whether the query would benefit from parallel branches, debate, supervisor routing, or replan loops. `ToTPlanner` (`enable_tot_planning`) exists but is disabled by default.

**What to build:**

1. **Topology library** — `mmar_mage/topology/library.py`. A registry of named topologies with explicit metadata:

| Topology | When to use | Emitted step types |
|---|---|---|
| `pipeline` | Sequential single-track reasoning | `llm` / `tool` / `transform` |
| `diamond` | One input → N independent analyses → synthesizer | `llm` ×N + aggregator |
| `funnel` | N independent inputs → one synthesizer | `llm` ×N + aggregator |
| `map_reduce` | One input broadcast over N samples | `parallel_sampling` |
| `tree_of_thought` | Exploratory reasoning with backtracking | `llm` with `loop_back_to` |
| `mixture_of_experts` | Heterogeneous tasks needing routing | `supervisor` + sub-chains |
| `debate` | Close-call decisions, contested reasoning | `debate` |
| `critique_revise_loop` | Output quality matters more than speed | `llm` + `evaluation` + `loop_back_to` |
| `hierarchical` | Multi-level orchestration | `supervisor` of `supervisor`s |
| `orchestrator_with_rollback` | High-stakes chains needing recovery | any + `ReplanPolicy` multi-checker |

Each entry exposes `(name, when_to_use, required_step_types, example_chain, complexity_score)` so downstream agents can reason about it programmatically.

2. **`TopologySelectorAgent`** — `mmar_mage/agents/topology_selector.py`. Runs after `DomainAnalyzer`, before `StepPlanner`. Inputs: `DomainAnalysis` + (optional) bootstrapped demos. Output: `TopologyCandidates(candidates: list[TopologyChoice])` with 1–3 ranked topologies and rationale per choice. Lives behind `enable_topology_selection: bool = True` (on by default).

3. **Topology-aware planning** — `StepPlanner` accepts a `topology: TopologyChoice` constraint and emits steps conforming to the topology's `required_step_types`. `DAGBuilder` accepts the same constraint and skips pattern inference.

4. **Parallel topology sampling** — when `TopologySelectorAgent` proposes K > 1 candidates, fan-out K planning passes (`generator.py`-side, similar to existing cold-start multi-sampling). Score with the same pairwise judge from P1.5. Keep the winner. Cost: K× planning tokens, but K is small (2–3).

5. **Prompt corpus rebalancing** — current `STEP_PLANNER_PROMPT` has 7 examples, 5 of which are effectively pipelines. Replace 3 of the 5 with non-pipeline patterns: one `diamond`, one `debate`, one `critique_revise_loop`. Source from `carl-experiments/examples/` to keep them realistic.

6. **Replaces P2.7** ("Explicit DAG pattern selector") — that section is removed; P1.15 supersedes it.

Files added:
- `mmar_mage/topology/__init__.py`
- `mmar_mage/topology/library.py`
- `mmar_mage/topology/schemas.py` (`Topology`, `TopologyChoice`, `TopologyCandidates`)
- `mmar_mage/agents/topology_selector.py`
- `mmar_mage/prompts.py` → add `TOPOLOGY_SELECTOR_PROMPT` + rebalance `STEP_PLANNER_PROMPT` examples

Files patched: `mmar_mage/generator.py` (insert stage 0.5: topology selection), `mmar_mage/schemas.py` (add `enable_topology_selection` flag), `mmar_mage/agents/step_planner.py` (accept topology constraint), `mmar_mage/agents/dag_builder.py` (accept pattern constraint).

ROI estimate: on RoTBench (where parallel sampling helps) and ToolAlpaca eval (where debate helps), expect +0.05–0.10 on mean quality. Variance reduction on Seal-Tools should be visible.

### [done] P1.16 Benchmark mode — synthetic self-verification of generated chains

> **Core done 2026-06-01 (4 slices).** The headline goal is delivered: with `enable_benchmark_mode=True`, every `generate()` returns a `quality_report`. Pipeline (all in the new owner module `mmar_mage/benchmark/`):
> - **schemas** — `benchmark/` owns `SynthesizedTask`/`BenchmarkSynthesisResult`/`ChainScore` (evolve re-exports → consumer) + new `SyntheticBenchmark` / `QualityReport`.
> - **synthesizer** — `BenchmarkSynthesizer.synthesize(query, domain, n_cases)` → `SyntheticBenchmark` (one LLM call, focused; reuses `BENCHMARK_SYNTHESIZER_PROMPT`).
> - **runner + reporter** — `ChainRunner.run(chain, benchmark, mode="mock")` judges the chain's design per case (`BENCHMARK_RUN_PROMPT`, non-fatal); `QualityReporter.report` → deterministic `QualityReport` (pass-rate, deduped failure modes, fix suggestions).
> - **stage 7** — `MAGEGenerator._run_benchmark_mode` (synthesize→run→report after serialization) attaches `quality_report` (a `QualityReport` dict) to `MAGEResult`; new `enable_benchmark_mode` / `benchmark_n_cases` config (+ env). Non-fatal; skipped on no-tasks. Stored as a dict on `MAGEResult` to avoid a `schemas`→`benchmark` import cycle.
> - Tests: `tests/test_benchmark_{schemas,synthesizer,runner,stage7}.py` (4 files, ~22 tests). **552 passed, ruff clean.**
>
> **Additive follow-ups (not blocking the goal):** real-executor `ChainRunner` mode (load+execute via `mmar_carl` instead of design-judging — sharper signal, needs the executor + tools); the `mage verify --query --chain` CLI; per-task-type test-case templates. This already unblocks P1.9 (execution-feedback can use `_run_benchmark_mode` as its signal) and sharpens P1.5/P1.7.

**Goal.** Every chain MAGE generates comes back with a `quality_report` produced by *synthesizing a mini-benchmark on the fly* for the original query and running the chain against it. No external benchmark required. This is the closest thing MAGE can offer to "is the chain you just gave me actually any good?"

**What's already there.** Stage-1 benchmark synthesis and stage-3 grading are *already implemented* inside `mmar_mage/evolve/`:
- `mmar_mage/evolve/benchmark_synthesizer.py` — generates test cases per query
- `mmar_mage/evolve/grader.py` — structural rubric + LLM-judge + optional real-execution
- `mmar_mage/evolve/engine.py` — wires them into the evolutionary loop

The capability exists but is captive to the evolve mode. The plain `generate()` path has no access to it.

**What to build:**

1. **Extract from `evolve/` into a first-class `mmar_mage/benchmark/` module.** The synthesizer + grader logic moves; `evolve/` becomes a *consumer* of the new module, not its owner. Refactor only — same behavior, new home.

2. **`SyntheticBenchmark` schema** — `mmar_mage/benchmark/schemas.py`:
   ```
   SyntheticBenchmark:
     query: str                      # the original user query
     domain: str                     # inherited from DomainAnalysis
     test_cases: list[TestCase]
     metric: Metric
     generation_provenance: dict     # which prompt / model / seed produced this
   TestCase:
     test_id: str
     input: dict                     # bound to the chain's input schema
     expected_behavior: str          # natural-language spec
     success_criteria: list[str]     # rubric items
     difficulty: Literal["easy","medium","hard"]
   Metric:
     kind: Literal["rule","llm_judge","structural_rubric","hybrid"]
     config: dict                    # kind-specific (e.g. judge_prompt, regex, etc.)
   QualityReport:
     n_cases: int
     pass_rate: float
     per_case_results: list[TestResult]
     failure_modes: list[str]        # diagnosed clusters
     suggested_fixes: list[str]      # rationale for next iteration
   ```

3. **Three agents in `mmar_mage/benchmark/`:**
   - `BenchmarkSynthesizer` — `synthesize(query, domain_analysis, n_cases) → SyntheticBenchmark`. Uses a new `BENCHMARK_SYNTHESIZER_PROMPT` (current evolve version is good seed but needs explicit "vary difficulty" instruction and "produce success_criteria as a checklist not prose").
   - `ChainRunner` — `run(chain_dict, benchmark) → list[TestResult]`. Two modes: `mock` (LLM simulates executor) and `real` (`ReasoningChain.from_json` + execute, requires `mmar-carl` installed).
   - `QualityReporter` — `aggregate(results) → QualityReport`. Clusters failure modes via cheap LLM call, suggests fixes targeted at the weak steps.

4. **Wire into `MAGEGenerator`.** New optional stage 7 (after serialization, before return):
   ```
   if config.enable_benchmark_mode:
       bench = await synthesizer.synthesize(query, domain_analysis, n_cases=config.benchmark_n_cases)
       results = await runner.run(chain.dict, bench)
       report = await reporter.aggregate(results)
       result.quality_report = report
   ```
   `MAGEResult` gains an optional `quality_report: QualityReport | None` field.

5. **New `MAGEConfig` fields:**
   - `enable_benchmark_mode: bool = False`
   - `benchmark_n_cases: int = 5`
   - `benchmark_runner: Literal["mock","real"] = "mock"` — `real` requires `mmar-carl` installed
   - `benchmark_metric_kind: Literal["rule","llm_judge","structural_rubric","hybrid"] = "hybrid"`

6. **Standalone `mage verify` CLI** — `mage verify --query "..." --chain chain.json [--n 10] [--executor real]`. Useful when the user already has a chain and just wants the quality report without regenerating.

7. **Per-task-type test-case templates** — `mmar_mage/benchmark/templates/`:
   - `tool_use.json` — TestCase shape for tool-calling tasks (input → expected tool sequence)
   - `reasoning.json` — for chain-of-thought / multi-hop
   - `structured_output.json` — for JSON-schema outputs
   - `transform.json` — for data-transformation tasks
   The synthesizer reads the domain and picks the matching template.

**Critical wiring downstream.** The synthetic benchmark unlocks several other plan items:
- **P1.5 (cold-start pairwise scoring)** — use the synthetic benchmark as the scoring signal instead of an absolute LLM judge. Strictly stronger.
- **P1.7 (MIPRO-style bootstrapped demos)** — filter candidate demos by their pass rate on the synthetic benchmark. Strictly stronger than current `was_cold_start`-only filtering.
- **P1.9 (execution-feedback loop)** — the synthetic benchmark *is* the feedback signal. P1.9 becomes "run chain on benchmark → diagnose failures → apply `LLM_REPAIR_PROMPT` to weakest step → re-run". No external benchmark plumbing required.
- **Evolve mode** — refactor `evolve/engine.py` to consume `mmar_mage/benchmark/` instead of owning the synthesis logic. Removes duplication, makes evolve's fitness signal exchangeable.

Files added:
- `mmar_mage/benchmark/__init__.py`
- `mmar_mage/benchmark/schemas.py`
- `mmar_mage/benchmark/synthesizer.py` (extracted + extended from `evolve/benchmark_synthesizer.py`)
- `mmar_mage/benchmark/runner.py`
- `mmar_mage/benchmark/grader.py` (extracted + extended from `evolve/grader.py`)
- `mmar_mage/benchmark/reporter.py`
- `mmar_mage/benchmark/templates/{tool_use,reasoning,structured_output,transform}.json`
- `tests/test_benchmark_mode.py` — end-to-end stage-7 smoke test
- `tests/test_benchmark_synth_quality.py` — golden-dataset check that synthesizer produces non-trivial test cases (not just "ensure the chain runs")

Files patched: `mmar_mage/generator.py` (add stage 7), `mmar_mage/schemas.py` (config fields + `MAGEResult.quality_report`), `mmar_mage/cli.py` (`mage verify` command), `mmar_mage/evolve/{engine,benchmark_synthesizer,grader}.py` (refactor to consume `mmar_mage/benchmark/`).

ROI estimate: this is the most ROI-dense item in the plan. It directly unblocks P1.5, P1.7, P1.9 (closing the gap to MIPRO / GEPA / AFlow); turns every `generate()` call into a verifiable artifact; and makes evolve's fitness signal swappable.

**Limits (worth knowing up front):**
- Synthetic test cases are biased toward what the synthesizer LLM thinks the task is. They are NOT a substitute for real benchmark numbers — those still come from `experiments/baselines/`. Treat `quality_report` as authoring-time signal, like unit tests, not as a headline metric.
- Mock-executor mode hallucinates. `real`-executor mode is the source of truth when `mmar-carl` is installed.

### [done] P1.17 Tool / Skill synthesis (currently MAGE only looks them up)  — Phase A=[done], Phase B=[done], Phase C=[done]

**Symptom.** `CapabilityLookupAgent` and `SkillDiscoveryAgent` both gracefully return empty when nothing matches in the registry. So when the user's query needs a tool that doesn't exist (custom data transform, niche API call, ad-hoc parser), MAGE quietly fails to surface it and the planner is forced to fall back to generic `llm` steps. No competitor synthesizes new tools either — this is greenfield.

**What to build, in three phases:**

**Phase A — `ToolSpec` generation (text artifact, no code execution). [done 2026-06-01]**

> **Done.** The safe, no-code-execution half of the moat. New stage 0.6 (`enable_tool_synthesis`, **OFF by default** — opting in is the security discipline; Phases B/C add real codegen+sandbox):
> - **`mmar_mage/agents/tool_spec_generator.py`**: `ToolNeed` + `ToolSpec(name, signature, docstring, input_schema, output_schema, implementation_hint, dependencies)`. `ToolSpecGenerator.detect_need` → `generate_spec` → combined `synthesize(query, domain) -> ToolSpec | None`. All non-fatal (LLM failure / missing name / not-needed → `None`); `synthesize` short-circuits with a single detector call when no tool is warranted. `format_tool_spec_for_planner` renders the proposal with an explicit "does NOT exist yet / treat as a design proposal" marker.
> - **Prompts**: `TOOL_NEED_DETECTOR_PROMPT` (conservative — default `needed:false`, don't propose tools for things `llm` steps do well) + `TOOL_SPEC_PROMPT` (single-purpose, deterministic, prefer stdlib).
> - **StepPlanner** gained `proposed_tools: str = ""`, injected under "Proposed synthesized tool" with the not-existing marker so the planner can choose a `tool` step or ignore it.
> - **Generator** stage 0.6 (after topology, before research): `synthesize` → format → thread into both `planner.plan(...)` calls. `tool_synthesis` lands in `deep_stages_completed` only when a tool was actually proposed.
> - New config (+ env + bool coercion): `enable_tool_synthesis` (False).
> - **Real-execution check**: `test_generate_surfaces_proposed_tool_to_planner` drives `generate()` (detector→spec→plan), asserts `tool_synthesis` ∈ stages and `parse_cobol_records` + "Proposed synthesized tool" reach the planner prompt; the not-needed test asserts a single detector call and no injection.
> - Tests: `tests/test_tool_synthesis.py` (11). Default-off → **zero existing-test churn**. **626 passed, 6 xfailed, ruff clean.**
>
> **Not in Phase A (by design):** no code generated/executed; the `tool_discovery` step emission (registering a real tool with the executor) belongs to Phase B once an implementation exists — Phase A only *proposes* to the planner.

- `mmar_mage/agents/tool_spec_generator.py` — `ToolSpecGenerator`. Input: a "tool need" extracted from `DomainAnalysis` (e.g. "extract tables from a PDF and emit them as CSV"). Output: `ToolSpec(name, signature, docstring, input_schema, output_schema, implementation_hint, dependencies)`.
- New prompts in `prompts.py`: `TOOL_NEED_DETECTOR_PROMPT` (decides whether the chain *needs* a new tool versus existing ones), `TOOL_SPEC_PROMPT` (fills the `ToolSpec`).
- Output gets fed into `STEP_PLANNER_PROMPT` as a *proposed* tool, with a marker that the tool is synthesized-not-existing.

**Phase B — `ToolImplementation` generation with sandboxed validation. [done 2026-06-01]**

> **Done.** The security-critical phase — real code generation, behind the mandatory sandbox. Still gated by `enable_tool_synthesis` (off by default).
> - **`mmar_mage/security/sandbox.py`** — defence-in-depth: `scan_source` (AST) rejects banned imports (`subprocess`/`ctypes`/`pickle`/`marshal`/`importlib`/…), **undeclared** network imports (`socket`/`requests`/`urllib`/… allowed only if in the spec's `dependencies`), banned builtins (`eval`/`exec`/`compile`/`__import__`), and dangerous calls (`os.system`/`os.popen`/`os.exec*`/file-destroying `os.*`/`subprocess.*`). Then `run_module_function` does a **resource-limited subprocess smoke run**: `python -I` isolated mode, minimal env (`PATH=""`), `resource` rlimits (RLIMIT_CPU + best-effort RLIMIT_AS via `preexec_fn`), and a wall-clock `timeout`. Passing = imports cleanly + function defined/callable; import error / missing func / timeout / crash = fail; a call exception from approximate `synth_inputs` is recorded (`call_error`) but non-fatal. `validate_tool_source` runs static first and **short-circuits before executing** any flagged code.
> - **`ToolImplementationGenerator`** (`agents/tool_implementation_generator.py`): codegen via `TOOL_IMPL_PROMPT` (anti-pattern catalogue) → strip fences → `validate_tool_source` → on failure feed `report.feedback()` back and retry (max 3). Writes the winner to `~/.mage/generated_tools/<name>_<spechash>.py` (user dir, never the repo) with a provenance header (`# Generated by MAGE … DO NOT COMMIT`). Content-hash cache (re-runs return `cached=True`, no LLM call); `regenerate=True` bypasses.
> - **Real-execution check**: tests drive the generator through the *real* sandbox (real AST scan + real subprocess) — clean code validates+writes on attempt 1; an `os.system` candidate is rejected then repaired on attempt 2 (asserting "REJECTED" feedback reached the 2nd prompt); all-banned gives up → `None`; cache hit skips the LLM. Sandbox tests cover banned imports/calls, declared-network exemption, safe execution, missing-func, import-error, infinite-loop **timeout**, and static-short-circuit.
> - Tests: `tests/test_sandbox.py` (17) + `tests/test_tool_implementation.py` (6). Default-off → no existing-test churn. **649 passed, 6 xfailed, ruff clean.**
>
> **Deferred follow-ups:** (1) OS-level network/FS isolation (seccomp/namespaces) — macOS has no portable primitive, so network is **statically** enforced today (the honest contract); (2) generator-side wiring (`generate()` auto-implements a Phase-A-proposed tool + emits a `tool_discovery` step to register it with the executor) — the generator is a standalone capability for now, keeping code-execution out of the default `generate()` path. Phase C ([12]) builds skill manifests on top.

- `mmar_mage/agents/tool_implementation_generator.py` — `ToolImplementationGenerator`. Input: `ToolSpec`. Output: a Python module file written to `~/.mage/generated_tools/<name>.py` (user-config-dir, not the repo).
- New prompt: `TOOL_IMPL_PROMPT` with anti-pattern catalogue (no `eval`, no shell-out, no network unless declared in `dependencies`).
- Sandbox validation in `mmar_mage/security/sandbox.py`:
  - Static checks: AST scan for banned imports / calls.
  - Dynamic check: run in subprocess with `--no-network`, `--read-only-fs`, time limit, memory limit.
  - Test with synthetic inputs derived from `input_schema`.
- Retry loop: if sandbox fails, feed the trace back to `ToolImplementationGenerator` for repair, max 3 attempts.

**Phase C — Full skill synthesis with manifest. [done 2026-06-01]**

> **Done.** MAGE can now author a complete AgentSkill, matching the real `mmar_carl` `SkillManifest` format (verified by introspecting the installed `carl-experiments@agent-features` — *not* guessed).
> - **`mmar_mage/skills/manifest.py`**: `SkillManifestData` + `render_skill_md` / `parse_skill_md` (stdlib frontmatter parser mirroring `mmar_carl`'s `skill_loader` — keys at col 0, hyphens→underscores, unknown keys→metadata, body=instructions) + `write_skill` (writes `<dir>/SKILL.md` + `scripts/<file>`) + `validate_with_carl` (round-trips through the real `SkillLoader.load_sync(AgentSkillSource(path=…))` when installed; cleanly skipped if not — `mmar_carl` is optional). Renders the AgentSkills-convention `allowed-tools:` frontmatter key; `scripts`/`references`/`assets` are loader-discovered from subdirs, not frontmatter.
> - **`SkillSynthesizer`** (`agents/skill_synthesizer.py`) + `SKILL_SYNTHESIZER_PROMPT`: one LLM call → `{name, description, instructions, allowed_tools, script?}`. Writes the skill to `~/.mage/generated_skills/<name>/`, validates via the real loader, and returns `SynthesizedSkill(name, skill_dir, uri="local://<dir>", has_script)`. A proposed helper `script` is **statically sandbox-scanned (Phase B `scan_source`)** — an unsafe script (e.g. `os.system`) is *dropped* and the skill is still authored instructions-only; code is never written unscanned. All non-fatal.
> - **Real-execution check**: tests author skills end-to-end and assert the written `SKILL.md` **loads back through the real `mmar_carl` loader** (`validate_with_carl is None`); the safe-script case writes `scripts/<file>.py`, the unsafe-script case drops it but keeps the skill.
> - Tests: `tests/test_skill_synthesis.py` (9) — manifest render/parse round-trip, hyphenated key, missing-delimiter raise, carl loader round-trip, + synthesizer (instructions-only, safe-script, unsafe-script-dropped, no-name, llm-failure). **658 passed, 6 xfailed, ruff clean.**
>
> **P1.17 is now complete (A+B+C).** Deferred integration follow-ups (shared with Phase B): wiring the synthesizers into `generate()` so a chain auto-adopts a generated tool/skill and emits the `tool_discovery` / `agent_skill` step that registers it with the executor. Kept standalone for now — code generation stays out of the default `generate()` path; the capabilities are invoked explicitly behind `enable_tool_synthesis`.

- `mmar_mage/agents/skill_synthesizer.py` — `SkillSynthesizer`. Output: a full agent-skill directory matching the AgentSkills SKILL.md format (`carl-experiments`'s `SkillManifest` schema, `models/agent_skill.py:95-147`): `SKILL.md` with frontmatter (`allowed-tools`, `scripts`, `references`, `assets`), optional `module.py`, optional `script.sh`.
- Skill goes to `~/.mage/generated_skills/<name>/`.
- The chain integrates the skill via an `agent_skill` step with `source = local:~/.mage/generated_skills/<name>`.

**Cross-cutting requirements:**

- **CARL integration.** When MAGE emits a generated tool, it must also emit a `tool_discovery` step at the start of the chain that registers the tool with the executor. This is exactly what `tool_discovery` (`carl_export.py:148`) is for. Wire it up; today MAGE never emits this step type.
- **Caching.** Synthesized tools/skills cache by content-hash of the `ToolSpec`. Re-use across runs. Optional `--regenerate-tools` CLI flag to force regeneration.
- **Provenance.** Every generated file has a top-comment block with `# Generated by MAGE on YYYY-MM-DD from query: …`. Never commit unreviewed generated tools to the repo.
- **Security default.** Generation is gated behind `enable_tool_synthesis: bool = False`. Users opt in. When on, sandbox is *mandatory*, not optional.

Files added (Phase A): `mmar_mage/agents/tool_spec_generator.py`, prompts.
Files added (Phase B): `mmar_mage/agents/tool_implementation_generator.py`, `mmar_mage/security/sandbox.py`, prompts.
Files added (Phase C): `mmar_mage/agents/skill_synthesizer.py`, `mmar_mage/skills/manifest.py` (parser matching `mmar_carl`'s `SkillManifest`), prompts.

ROI estimate: hardest item in the plan, biggest structural moat. Direct quality lift expected on tasks where benchmark catalogs are incomplete (ToolAlpaca's long tail, custom Seal-Tools categories). Indirect lift: lets MAGE attack out-of-distribution use cases competitors can't touch.

**Risks:**
- *Security.* Generated code is RCE-shaped. Sandboxing must be solid before this ships; default-off is the discipline.
- *Quality drift.* First-pass synthesized tools often fail; rely on Phase B's retry loop and P1.9's execution-feedback loop to converge.
- *Schema-vs-implementation drift.* Tool signature has to match what the executor actually calls. JSON-schema validation at sandbox time covers this.

### [done] P1.18 Real gigaevo-memory integration — core (hybrid search, typed entities, scored recall)

> **In progress — hybrid-search recall slice done 2026-06-01.** Installed the real SDK (`gigaevo-memory==0.2.3`, deps `httpx`/`httpx-sse`/`mmar-carl`/`pydantic`) and confirmed its API: `SearchType` = `{BM25, VECTOR, HYBRID}`; `search(...)` natively accepts `search_type`, `hybrid_weights`, `namespace`, `entity_type`. Landed:
> - **[done] Configurable hybrid search (item 2) + curation namespace (item 8).** New `MAGEConfig.memory_search_mode` (`"hybrid"` default), `memory_hybrid_weights` `(0.5,0.5)`, `memory_namespace`, + env mappings. New `MemoryManager._search` centralizes search-type/weights/namespace and replaces the hard-coded `SearchType.BM25` at all 5 recall sites (`search_similar`, `get_capabilities`, `search_templates`, `search_feedback`, `recall_benchmarks`).
> - **[done] Per-call graceful degrade (lightweight item 1).** If the server rejects a `vector`/`hybrid` request (e.g. vector search disabled), `_search` retries once as BM25 instead of returning empty — a cheap stand-in for the full cached capability probe.
> - Tests: `tests/test_memory_search_mode.py` (8) — mode→`SearchType` resolution, hybrid weights, namespace, BM25 fallback-on-rejection, BM25-failure-propagates, unknown-mode fallback. **460 passed, ruff clean.**
>
> **Item 3 (typed entities) — started 2026-06-01: template round-trip migrated.** Critical API findings (verified against v0.2.3): valid `entity_type` values are only `{chain, step, agent, memory_card}` — so the legacy `"chain_template"` / `"chain_feedback"` / `"capability_registry"` recalls were **silently broken** (no such type → no results). And `MemoryCardSpec` is a *fixed* GigaEvo schema (`category`/`description`/`keywords`/`code`/`fitness`/…), **not** a generic dict store: the typed `search()`/`search_memory_cards()` path validates hits into `MemoryCardSpec` and would **drop** MAGE's custom keys. The clean path: `save_memory_card(memory_card=<raw dict>)` stores content verbatim, and `search_hits(entity_type="memory_card")` returns `SearchHitData` with raw `.content` (dict) + `.score` (relevance, useful for P1.1/P1.5). Implemented:
> - New `MemoryManager._save_card` (raw-dict card persistence + `kind` tag + namespace) and `_search_cards` (search_hits on memory_card, reusing the configured hybrid mode + BM25 fallback). Refactored `_search` to share resolution via `_search_params`/`_run_search`.
> - Migrated `save_template`+`search_templates`, `submit_feedback`+`search_feedback`, and `save_ideas` to typed memory cards (kinds `"mage_template"`/`"mage_feedback"`/`"mage_ideas"`), fixing the broken `chain_template`/`chain_feedback` recalls and establishing the reusable pattern. Feedback fields are flattened into the card content so recall reads `quality_score`/`success`/`failure_reason` directly. Tests: `tests/test_memory_typed_entities.py` (7). **467 passed, ruff clean.**
> - **Item 3 is effectively done for all MAGE-*written* reusable knowledge.** Not migrated: `get_capabilities` reads an externally-written registry → confirm that card's real `content` shape against a live server before migrating; `save_benchmark`/`recall_benchmarks` work correctly as `chain` entities today → migrate for consistency only (low priority).
> - **[done] Item 4 (scored recall) — fixed a latent always-empty bug.** `iterative_search` filtered on a `score` that `search_similar` never returned, so every hit scored `0.0`, failed the `>= threshold` check, and the method **always returned `[]`** — the entire memory-research recall path was dead. Rewrote it to use the new scored `_search_hits` (real relevance scores) over a single client connection, with dedup, threshold filter, and best-first sort. New helper `_search_hits(client, query, entity_type, top_k)` (generalized from `_search_cards`). Tests: `tests/test_memory_typed_entities.py` +3 (scored/dedup/sorted, threshold filter, disabled); updated `tests/test_memory_research.py` to the new path. **470 passed, ruff clean.** The agent-level multi-hop (LLM-generated sub-queries + applicability gate) remains P1.1.
>
> **Done = the offline-completable core** (hybrid search, namespace, typed entities for MAGE-written knowledge, scored recall, per-call BM25 fallback). The remaining items all need a **running gigaevo-memory server** or a downstream consumer to implement+test meaningfully, so they are split into **P1.19** below rather than blocking this `[done]`.

Target branch: `AIRI-Institute/gigaevo-memory@who-cares`. The `who-cares` API is far richer than what `mmar_mage/memory.py` exercises today, and our integration is both *shallow* (BM25-only) and *wrong-shaped* (everything is crammed into the `chain` entity).

**Current state of `mmar_mage/memory.py` (the problem):**
- Single retrieval mode: every `client.search(...)` call hard-codes `search_type=SearchType.BM25` (`memory.py:155, 250, 295, 423, 581`). No vector, no hybrid, no reranking.
- Entity-type abuse: chains, feedback, templates, benchmarks, and ideas are **all** persisted through `client.save_chain(chain=…, name=…, tags=…)` with a `kind` discriminator stuffed into the payload (`memory.py:330, 374, 473, 535`). There is no use of dedicated entity types.
- `iterative_search` (`memory.py:163-225`) fakes multi-query recall by *string-concatenating* the query with domain concepts (`f"{query} {concept}"`). That's keyword expansion, not semantic retrieval.
- No use of run telemetry, lineage, versioning, favourites, or the event stream.
- 10 methods, ~18 call-sites across 9 files: `generator.py`, `evolve/{engine,benchmark_synthesizer}.py`, and the 4 research agents (`capability_lookup`, `feedback_recall`, `memory_research`, `template_assembler`).

**What `who-cares` actually offers (and we ignore):**

| Capability | `who-cares` endpoint / feature | MAGE today |
|---|---|---|
| Hybrid retrieval | `GET /v1/search/unified` — BM25 (`tsvector`) **+ pgvector**, configurable weights, **pluggable reranker** | `SearchType.BM25` only |
| First-class entity types | `/v1/{steps,chains,agents,agent-skills,memory-cards}` — five typed routers | everything is a `chain` |
| Reusable knowledge | **Memory Cards** — "reusable knowledge patterns" | faked via chain payloads with `kind` |
| Portable skills | **Agent Skills** — SKILL.md bundles (ties to P1.17) | not used |
| Run telemetry | `POST /v1/{type}/{id}/run-recorded`, `run_count`, `last_run_at` | not used |
| Lineage & promotion | `GET /v1/chains/{id}/lineage`, `GET /v1/chains/{id}/versions/beating` | not used |
| Live updates | `GET /v1/events/stream` — SSE firehose of entity mutations | not used |
| Curation & scoping | `favourites_only`, `namespace`, `tags`, `sort_by`/`sort_dir` filters | tags only |

**What to build:**

1. **Pin the SDK + add a capability probe.** `who-cares` ships a `client/` Python SDK wrapping the HTTP API. Pin its version in `pyproject.toml`'s optional `[memory]` extra. Add `MemoryManager._detect_features()` that probes which endpoints the connected server supports (unified search, vector, events) so MAGE degrades per-feature, not all-or-nothing. (Confirm exact SDK class + method names against the branch's `client/` and `openapi.yaml` before coding — the README does not show signatures, so do not hard-code guessed method names.)

2. **Switch recall to unified hybrid search.** Replace the BM25-only `client.search(...)` calls with the unified endpoint. New config:
   - `memory_search_mode: Literal["bm25","vector","hybrid"] = "hybrid"`
   - `memory_hybrid_weights: tuple[float,float] = (0.5, 0.5)` — (bm25, vector)
   - `memory_reranker: str | None = None` — name of server-side reranker
   Falls back to `bm25` automatically if the probe in (1) says vector is off (`ENABLE_VECTOR_SEARCH=false`).

3. **Use real entity types instead of overloading `chain`.** Map MAGE's five payload kinds onto their natural `who-cares` homes:
   - generated chains → `chains` router (already correct intent, fix the call)
   - templates / ideas / feedback patterns → **Memory Cards** (`memory-cards` router) — these *are* reusable knowledge patterns
   - synthesized skills (from P1.17 Phase C) → **Agent Skills** (`agent-skills` router)
   - synthetic benchmarks (from P1.16) → a typed Memory Card with a `mage_benchmark` schema, not a chain payload
   This kills the `kind`-discriminator hack and lets retrieval filter by entity type natively.

4. **Make `iterative_search` genuinely multi-hop (folds into P1.1).** Replace string-concatenation sub-queries with: (a) LLM-generated semantic sub-queries, (b) unified hybrid search per sub-query, (c) applicability re-rank via the server reranker *or* a cheap LLM judge, (d) a gap-targeting second wave when retained hits < N_min. This is the concrete realization of P1.1 against the real API.

5. **Quality/recency-weighted retrieval via run telemetry.** When MAGE recalls a chain or card, weight the ranking by `run_count` × `success_rate` × recency (`last_run_at`). After a chain executes (P1.9 / P1.16), call `run-recorded` to write back the outcome. This closes the loop: chains that demonstrably work surface first next time.

6. **Lineage-aware evolve (folds into evolve mode).** Use `GET /v1/chains/{id}/versions/beating` to seed the evolve population with *known-good ancestors* instead of cold-starting, and write evolved offspring back with proper lineage edges so the "beating" query stays meaningful. This makes evolve resumable across runs.

7. **Optional live cache via SSE.** A long-running `mage` process (server mode, or a batch evolve run) can subscribe to `GET /v1/events/stream` to keep a warm in-process cache of recently promoted chains/cards. Behind `enable_memory_event_stream: bool = False`. Nice-to-have; deprioritize.

8. **Curation.** Surface `favourites_only` and `namespace` in recall config so a user can scope MAGE to their own curated memory partition (`memory_namespace: str | None = None`, `memory_favourites_only: bool = False`).

**Cross-references** — this item is the substrate several others stand on:
- **P1.1 / P1.2** (deep memory/web research) — (4) above *is* the memory half of P1.1.
- **P1.3** (capability/template/feedback synthesis) — those three agents stop faking entity types and read/write real Memory Cards.
- **P1.5** (cold-start pairwise scoring) — recall quality from hybrid search directly improves the cold-start signal.
- **P1.7** (bootstrapped demos) — demos live as Memory Cards, retrieved by hybrid search, ranked by run telemetry.
- **P1.16** (synthetic benchmark) — benchmarks persist as typed cards; `recall_benchmarks` (`memory.py:547`) gets a real home instead of the `kind=="mage_benchmark"` payload scan.
- **P1.17** (skill synthesis) — synthesized skills register as Agent Skills entities and become discoverable by `SkillDiscoveryAgent`.

Files patched: `mmar_mage/memory.py` (the big one — rewrite around typed entities + unified search; keep the graceful-degradation contract), `mmar_mage/schemas.py` (new config fields above), the 4 research agents (consume typed entities), `mmar_mage/evolve/engine.py` (lineage-aware seeding), `pyproject.toml` (`[memory]` extra version pin).
Files added: `tests/test_memory_who_cares.py` — mock the `who-cares` SDK, assert: hybrid-search path used when available, BM25 fallback when not, each payload kind lands in the correct entity type, `run-recorded` called after execution.

**ROI estimate:** retrieval quality is the input to every research/cold-start/demo stage, so this lifts P1.1, P1.3, P1.5, P1.7 simultaneously. Hybrid search alone (BM25→BM25+vector+rerank) typically moves recall@5 by a wide margin on semantic queries where keyword overlap is low — exactly MAGE's failure regime.

**Risks / unknowns:**
- *Branch volatility.* `who-cares` is an active feature branch — the SDK surface may move. Pin a commit, not just the branch. Re-verify endpoint names against `openapi.yaml` at integration time.
- *Server availability.* All memory work stays best-effort (the existing `return None`/`[]` contract). A missing or older server must never block generation.
- *Secrets.* If `namespace`/auth requires headers, route them through config, never hard-code; mirror the `MCPServerConfig.headers_secret_ref` concern from the cross-cutting section.

### [blocked] P1.19 gigaevo-memory live-server features (needs a running server / consumer)

Split out of P1.18 — these need a **running gigaevo-memory server** (or a downstream consumer) to implement and test meaningfully, so they can't be done in the offline/mock dev loop and are deprioritized to hygiene-tier. The one offline-completable slice (benchmark typed-entity migration) is **done**; the remaining four are genuinely server-blocked.

- **[done 2026-06-02] Typed entities — benchmark migration.** `save_benchmark`/`recall_benchmarks` no longer abuse the `chain` entity type (which conflated synthesized benchmarks with real reasoning chains and polluted chain search). They now use the P1.18 typed-memory-card path: `save_benchmark` → `_save_card` (kind `mage_benchmark`); `recall_benchmarks` → `_search_cards` (`search_hits` on `memory_card`, reading the raw `.content` + relevance `.score`, filtered by kind). Verified the installed `MemoryClient` exposes `save_memory_card`/`search_hits` (introspected, not guessed). Tests: rewrote `tests/test_memory_benchmarks.py` to the card path (the old tests encoded the abused `chain` path) + a **save→recall round-trip** through a shared in-memory fake store. **770 passed, 6 xfailed, ruff clean.**
- **[blocked] Cached capability probe** (`_detect_features`): call `health_check() -> dict` once, cache which search modes the server supports, and have `_search` pick the best available mode up front (instead of the current per-call hybrid→BM25 retry). `MemoryClient.health_check` *exists* (confirmed) but returns an **opaque `/health` dict** whose dependency/feature keys aren't knowable without a live server — implementing against guessed keys would be hollow (the per-call BM25 fallback already covers the failure case safely).
- **[blocked] Run telemetry**: weight recall by `run_count` × `success_rate` × recency, and write outcomes back after execution. The 0.2.3 client has no `run-recorded` method — confirm the telemetry surface against the server before building.
- **[blocked] Lineage-aware evolve**: use `promote`/`revert`/`diff_versions`/`list_versions`/`get_version` (all present in the client) to seed the evolve population from known-good ancestors and write offspring back with lineage. Needs the evolve loop wired as the consumer + real versioned data.
- **[blocked] SSE live cache**: `watch_chain`/`pin_channel` for a warm in-process cache of recently promoted entities. Needs a live event stream + a long-running process.
- **[blocked] Typed entities, `get_capabilities` remainder**: migrate `get_capabilities` — needs the externally-written registry card's real `content` shape confirmed against a live server first.

---

## P1 — Make quality-enhancement stages do real work

### [done] P1.11 StepCritic: make revision *mandatory* above threshold

> **Done 2026-06-01.** The plan's "revision is opt-in" premise was outdated — the generator already auto-revises whenever `steps_needing_revision` is non-empty. The real gap: that list used only the *averaged* `overall_score < threshold`, which can mask a load-bearing failure in a single critical dimension. Now a step is **also** flagged when `aim_clarity` **or** `action_specificity` falls below the new `step_critic_revision_threshold` (0.6), even if its average passes. `StepCriticAgent.critique` gained a `revision_threshold` param (`_CRITICAL_DIMENSIONS` constant); the generator passes `config.step_critic_revision_threshold` (+ env). Tests: 2 added to `test_quality_enhancements.py` (flag-on-critical-dim-despite-passing-average; threshold configurability) + 3 existing critic tests unchanged. **523 passed, ruff clean.**

### [done] P1.12 ChainVerifier: actually answer the verification questions independently

> **Verified + improved 2026-06-01.** The plan's premise was partly outdated: `ChainVerifier.verify` **already** answers each question with its *own* independent LLM call (one per question, scoped context — not a single batched fixer call), and the module documents the CoVe generate→answer→compare→fix flow. The real gap was that the independent answerer saw only the single step, so it could not judge **cross-step** questions (dependencies / ordering / consistency). Fixed: the answer context now includes the step under review **plus its dependency steps** + the query, via a new `_answer_question` method and a named `VERIFICATION_ANSWER_PROMPT` (independent-answerer persona; promoted from an inline string). Tests: `tests/test_chain_verifier_deps.py` (3) — dependency-step inclusion, no-deps-section when none, independent-per-question; existing verifier tests unchanged. **521 passed, ruff clean.**

### [done] P1.13 SelfRefiner: batch critiques, not 1-at-a-time

> **Done 2026-06-01.** The plan's "1 critique + 1 revision per iteration is inefficient (per-step)" premise was outdated — `SelfRefiner` was **already** batched at the chain level (one whole-chain critique + one whole-chain revise per iteration, not per-step). The real defects: it returned the *last* chain even when a revision **regressed** quality, and it kept iterating through plateaus. Fixed: (1) track the **best** chain/quality across iterations and return that (a bad revision is now discarded); (2) **early-stop** when a revision fails to improve quality (`q <= prev_quality`) — saves LLM calls at large `max_iterations`. Tests: 2 added to `test_quality_enhancements.py` (best-not-regression returns the original + stops early; plateau stops at iter 2 instead of running all 5) + 4 existing refine tests unchanged. **531 passed, ruff clean.**

### [done] P1.14 Risk-driven checkpoint promotion

> **Done 2026-06-01.** New `_promote_checkpoints` (in `generator.py`, run inside the `enable_replan_generation` block before `_build_default_replan_policy`) promotes one risk step to `checkpoint=True` + `checkpoint_name="primary"` so the default ReplanPolicy finally has a rollback target (it always returned `None` before, since nothing set a checkpoint). Deterministic heuristic over the serialized steps: (1) the latest *synthesizer* step (≥2 dependencies — a funnel/diamond merge); (2) else, in a chain of ≥4 steps, the first external call (`tool`/`mcp`/`agent_skill`/`mcp_resource`). No-op when a checkpoint already exists or nothing fits. Implemented as a generator post-process rather than inside `StepDescriber` (it has the resolved DAG + is deterministic — no extra LLM call). Tests: `tests/test_checkpoint_promotion.py` (6) — incl. the end-to-end "promotion enables the policy that was `None` before". **529 passed, ruff clean.**

---

## P2 — Full `agent-features` coverage in the generator

### [done] P2.1 Emit new step types reliably

> **Done 2026-06-01 — premise was mostly already satisfied; closed the real gap.** On inspection, two of the three sub-deliverables were already in place from earlier work: the type *descriptions* exist in the planner/describer corpus (all agent-features types listed with "when to use"), `CARL_FIELD_MAP` nests every type (P0.1), and `test_carl_round_trip.py` round-trips all of them with `xfail(strict)` on the CARL-`from_json` upstream gap (P0.2). The genuine remaining reliability hole: a typed step could be *emitted without its essential `step_config` fields* (an `evaluation` with no target, a `parallel_sampling` with no sample count) and silently export to a no-op `llm`. Closed it:
> - **`REQUIRED_STEP_CONFIG_FIELDS` + `missing_step_config_fields(chain_dict)`** (`carl_export.py`): per-type essential MAGE-flat fields (conservative — only the field(s) without which the step is meaningless), and a detector returning `{step_number: [missing fields]}`. Counts a field present whether it's flat on the step or nested in an existing `step_config`; blank string/empty collection counts as missing.
> - **Generator integration**: runs on the MAGE-flat chain right after `validate_carl_json` (before export) and logs a warning per incomplete typed step. Non-fatal — graceful degradation, since the step still exports (degraded) rather than crashing.
> - **Describer/planner corpus**: added a "REQUIRED fields per non-LLM type" block to the prompt so the LLM emits the essentials (`evaluation_target_step`/`evaluation_criteria`, `sampling_n`/`sampling_aggregation`, `skill`, `supervisor_routes`, `debate_agents`, etc.). *(Caught a `.format()` brace-escaping bug here — literal `{…}` in a `.format()`-ed prompt became placeholders; the full suite surfaced it where the isolated fast-mode test didn't.)*
> - **Real-execution check**: `test_generator_warns_on_incomplete_typed_step` drives `generate()` with an evaluation step missing its config — generation still succeeds and the gap is logged.
> - Tests: `tests/test_step_config_required.py` (12) — per-type missing/complete/partial detection, blank-value handling, nested-config recognition, all required types export without crashing, + the generator caplog integration. **670 passed, 6 xfailed, ruff clean.**
>
> **Note vs. the original plan:** per-type step-planner few-shot *examples* (sub-deliverable a) overlap with **P2.5** ([15] "Add few-shot examples for advanced step types") and are left to that item; this task delivered the describer field-templates + the deterministic reliability enforcement, which is the higher-leverage half.

Today `STEP_PLANNER_PROMPT` lists all 17 types but only the original 7 have working examples in `STEP_DESCRIBER_PROMPT`. For each new type below, add:
- A step-planner example showing *when* to choose this type (with a one-line rationale).
- A step-describer template showing every required `step_config` field with a plausible value.
- A `tests/test_step_type_X.py` round-trip test (covers P0.2).

Coverage order (by ROI):
1. `evaluation` — quick wins on RoTBench (catch malformed outputs early); already partially supported, just needs field map fix.
2. `parallel_sampling` — direct quality lever on hard reasoning (Seal-Tools).
3. `agent_skill` with `LLM_AGENT` mode — unlocks PDF/DOCX/code-exec tasks.
4. `supervisor` — multi-route dispatch for heterogeneous benchmarks.
5. `debate` — last-mile quality on close-call decisions.
6. `tool_discovery` — runtime tool registration for ToolAlpaca's huge tool catalogue.
7. `mcp_resource` — read-only data fetch (config files, schemas).
8. `human_input` — only for interactive use cases; deprioritize.
9. `agent_handoff` — depends on P0.3 resolution.

### [done] P2.2 Emit new orchestration primitives  — cache + MCP-transport slice done; 3 LLM-agent primitives deferred

> **Done 2026-06-01 (the two deterministic, schema-verified primitives).** Verified each field against the installed `mmar_carl` rather than guessing.
> - **`cache: StepCache` memoization**: `StepCache` = `{ttl, key_fn}` but `key_fn` is a runtime-only Python callable, so only `ttl` is JSON-serializable. Added a flat `CARLStepSchema.step_cache_ttl` → exported (`_apply_cache`) to CARL's **top-level** `cache: {ttl}` (confirmed it round-trips a `tool` step through `ReasoningChain.from_json`). New `mark_cacheable_steps` pass auto-marks **only read-only `mcp_resource`** steps (side-effect-free by definition) with `step_cache_default_ttl` (300s) — never side-effecting `tool`/`mcp` (those opt in via the describer prompt). Runs in `generate()` for both fast + deep modes, gated by `enable_step_caching` (on by default). New config: `enable_step_caching`, `step_cache_default_ttl`.
> - **MCP `http`/`sse` transports + `headers`**: `MCPServerConfig` supports `transport ∈ {stdio,http,sse}` + `url` + `headers` (auth), but MAGE's `_MCP_SERVER_FIELDS` lacked `headers`. Added `mcp_server_headers` → `headers` (new `CARLStepSchema.mcp_server_headers`) + describer guidance for remote (http/sse + bearer header) vs local (stdio + command). Confirmed the exported `server` block loads into the real `MCPServerConfig` with headers + transport preserved.
> - **Real-execution checks**: `test_cached_tool_round_trips_via_real_carl` (cache survives `ReasoningChain.from_json`), `test_mcp_server_config_loads_into_real_carl` (sse + headers validate against `MCPServerConfig`), `test_generator_auto_caches_mcp_resource` (a real `generate()` auto-caches the resource step).
> - Tests: `tests/test_orchestration_primitives.py` (9). **698 passed, 6 xfailed, ruff clean.**
>
> **Deferred (the 3 remaining primitives) — [blocked: lossy `from_json`, verified 2026-06-02]:** `triggered_by` event edges, `loop_back_to`/`loop_config`, and `step_groups` per-group LLM overrides. Beyond needing new agents/prompts, all three hit the **same upstream lossy-loader wall as P0.2** — directly verified against the live checkout: `ReasoningChain.from_json` rebuilds steps as the deprecated `StepDescription` (which lacks `loop_back_to`/`triggered_by` entirely) and drops `step_groups` (loads as `[]`), and `to_json` re-drops all of them. So MAGE could *emit* them but CARL would silently discard them on the round-trip (the same "emit-but-dropped" situation as `metrics`/P3.4). **Genuinely blocked until the upstream `from_json` fix lands** (already filed in the cross-cutting section). Not worth a hollow MAGE-emits-it/CARL-drops-it implementation.

Beyond step types, `agent-features` adds:

- **`triggered_by` event edges** (`models/steps.py:35-97`) — name-based event triggers (`$event.<name>`) instead of step-number dependencies. Add `EventEdgeBuilder` agent + few-shot example.
- **`loop_back_to` / `loop_config`** (`models/config.py:341-384`) — bounded retry loops with `condition_key` / `max_iterations` / `negate_condition`. Add `LOOP_DESIGN_PROMPT` + example chain.
- **`cache: StepCache`** (`models/config.py:706-755`) — memoization with `ttl` + `key_fn`. Add to `StepDescriber` for any `tool` or `mcp` step that's pure given its inputs.
- **`step_groups`** (`models/config.py:1065-1111`) — per-group `LLMStepConfig` overrides. Useful when one phase of the chain needs a cheaper model than another. Add `GroupAssignmentAgent` (optional).
- **MCP `http` / `sse` transports + `headers`** — currently only `stdio` is in the prompt corpus. Add transport selection + auth-header guidance to `STEP_DESCRIBER_PROMPT`.

### [done] P2.3 Deep `ReplanPolicy` synthesis

> **Done 2026-06-01 — and fixed a latent silent-config-loss bug found while verifying the schema.** Introspecting the real `mmar_carl.ReplanPolicy` (not guessing) revealed the old `_build_default_replan_policy` emitted a **wrong shape that loaded but silently dropped every checker setting**: it nested the checker fields under `config: {...}`, but `RuleBasedReplanCheckerConfig` has them *flat* — so on `ReplanPolicy.model_validate` the whole `config` was ignored and the checker reverted to defaults (`action_on_failure` → `retry_current_step`, not `replan_from_checkpoint`; rollback target + feedback lost). It also wrote a top-level `max_replans` that isn't a field (real home: `budgets.max_replans_per_chain`), so the `max_replans` param had **no effect**.
> - **Shape fix**: flat `RuleBasedReplanCheckerConfig` fields, a `budgets` (`ReplanBudgetConfig`) sub-object, and a `default_checkpoint_target`. Verified: the emitted policy now round-trips through the real `ReplanPolicy` with the checker action, rollback target, feedback, and budget all **preserved**.
> - **Risk-based depth** (`_chain_replan_risk` = external/side-effecting steps + 2·debate + len≥6): low-risk → one rule checker, `aggregation="any"`; high-risk (≥3) → adds an `LLMReplanCheckerConfig` judge and switches to `mandatory_plus_k_of_rest` voting (rule checker mandatory + k=1 of the rest).
> - **Budget scaling**: `max_replans_per_chain = max(arg, min(n, 6))`, `max_visits_per_checkpoint = max(2, min(n//2, 5))` — longer chains get more headroom.
> - **Real-execution check**: `test_settings_preserved_on_load` / `test_high_risk_policy_loads` round-trip both shapes through the real `mmar_carl.ReplanPolicy` and assert action/rollback/voting/budget survive. Rewrote the existing `tests/test_replan_policy.py` (it encoded the old buggy `config`-nested + top-level-`max_replans` shape) to the corrected shape, same as the P0.1/P0.2 test updates.
> - Tests: `tests/test_replan_policy.py` (12). **677 passed, 6 xfailed, ruff clean.**
>
> **Deferred follow-up:** `REPLAN_POLICY_PROMPT` (LLM proposes a policy when the heuristic is uncertain) and per-trigger `step_types` allow-lists — the deterministic risk heuristic covers the common cases; the LLM path is additive.

Today `_build_default_replan_policy` (`generator.py:206`) emits a single-checker default. `agent-features` supports `mandatory_plus_k_of_rest` voting, per-trigger `step_types` allow-lists, granular budget knobs.

Changes:
- Make the policy depth a function of chain *risk* (longer chains, tool-heavy chains, debate steps → multi-checker policy).
- Emit `ReplanBudgetConfig.max_visits_per_checkpoint` based on chain length.
- Add `REPLAN_POLICY_PROMPT` so the LLM can propose a policy when the heuristic is uncertain.

### [done] P2.4 Skill execution modes

> **Done 2026-06-01.** Introspected the real `mmar_carl.AgentSkillStepConfig` (modes `{llm, script, hybrid, subagent, llm_agent}`; `runtime` `{local, docker, e2b}`; `trust_policy` `{any, sha_pinned}` where `sha_pinned` *requires* `skill_sha256`). The export side already nests all these fields (P0.1 field map) — the gap was that MAGE never *chose* them. Added **`refine_agent_skill_step`** (`carl_export.py`), a deterministic default-filler invoked inside `to_carl_compat` for every agent_skill step. Each rule fires only on a clear signal and **never overrides an explicit LLM choice**:
> - `skill_output_schema` present → `execution_mode = "llm_agent"` (structured output needs the tool-call loop, not single-shot).
> - remote skill (`github://`/`https://`) that runs a bundled script → `runtime = "docker"` (isolate untrusted remote code).
> - remote skill with a known `skill_sha256` → `trust_policy = "sha_pinned"`. Crucially, a remote skill *without* a digest is left at `any` — forcing `sha_pinned` without a sha would make the config invalid (the digest is required).
> - Describer-prompt guidance added so the LLM picks modes/runtime/trust deliberately (subagent for isolation, docker + sha-pin for untrusted remotes).
> - **Real-execution check**: `test_refined_config_loads_into_real_carl` exports a remote skill with output-schema + script + sha and asserts the result loads into the real `AgentSkillStepConfig` as `llm_agent` / `docker` / `sha_pinned`.
> - Tests: `tests/test_skill_execution_modes.py` (12) — mode/runtime/trust rules, explicit-choice preserved, local-stays-trusted, dict-URI extraction, to_carl_compat integration, + the real-CARL round-trip. **689 passed, 6 xfailed, ruff clean.**
>
> **Deferred follow-up:** `e2b` runtime + manifest-`allowed-tools`-driven runtime selection (needs loading the skill manifest at generation time) — the conservative `docker`-for-remote-scripts rule covers the security-critical case; finer runtime routing is additive.

Today MAGE only emits the default `agent_skill` mode. Add:
- `LLM_AGENT` mode with `output_schema` for skills that need tool-call loops.
- `SUBAGENT` mode for skills running in isolated context.
- SHA-pinning for `github://` URIs (trust requirement).
- `docker` / `e2b` runtime selection based on skill manifest's `allowed-tools`.

---

## P2 — Prompt quality

### [done] P2.5 Add few-shot examples for advanced step types

> **Done 2026-06-01.** The describer prompt had only type *definitions* (and, since P2.1, a required-fields list) but **zero concrete examples**. Added **`ADVANCED_STEP_EXAMPLES`** (`prompts.py`) — a single-source list of correct, minimal step objects for the five priority advanced types (`evaluation`, `parallel_sampling`, `agent_skill`, `supervisor`, `debate`), distilled from `carl-experiments/examples/` (agent_skill→SUBAGENT pdf, llm_council→vote, supervisor routing). Rendered into `STEP_DESCRIBER_PROMPT` via `_render_advanced_examples` under an explicit "REFERENCE ONLY — adapt the SHAPE, don't echo" header. (`STEP_DESCRIBER_PROMPT` is concatenated, not `.format()`-ed, so the literal JSON braces are safe — no P2.1-style escaping needed.)
> - **Single source of truth → examples can't drift into wrongness**: the same list is rendered into the prompt AND validated in tests. Every example is proven to (a) carry all its required `step_config` fields (cross-checked against P2.1's `missing_step_config_fields`) and (b) export to a CARL-valid step (`to_carl_compat` + `validate_carl_json(strict_carl=True)`). So the few-shots can never teach the model a field name that produces a broken chain.
> - **Real-execution check**: `test_describer_call_includes_examples` drives a real `StepDescriber.describe(...)` and asserts the worked examples reach the system prompt; the parametrized export-validity tests are the substantive guarantee.
> - Tests: `tests/test_describer_examples.py` (13 — coverage, render, per-example required-fields + export-validity ×5, describer integration). **719 passed, 6 xfailed, ruff clean.**
>
> **Note:** `replan` example patterns (voting / budget-guard) are already covered structurally by P2.3's risk-based policy synthesis + its real-CARL round-trip tests, so they're not re-added as describer few-shots here.

`STEP_DESCRIBER_PROMPT` (`prompts.py:719-802`) has *zero* concrete examples — just type definitions. Add at least one example per type covering all required `step_config` fields. Source from `carl-experiments/examples/`:
- `agent_skill` from `examples/agents/agent_skill_example.py`
- `parallel_sampling` from `examples/agents/llm_council_example.py:1-150`
- `supervisor` from `examples/agents/supervisor_routing_example.py`
- `debate` from `examples/agents/llm_council_example.py` (judge synthesis section)
- `replan` patterns from `examples/replan/replan_voting_example.py`, `replan_budget_guard_example.py`

### [done] P2.6 Anti-pattern catalogue

> **Done 2026-06-01.** Added the **`STEP_PLANNER_ANTI_PATTERNS_PROMPT`** constant (`prompts.py`) and appended it to the planner system prompt (`StepPlanner.plan`, after the type constraint). Covers the four from the plan plus two reinforcing ones — needless `supervisor` (use a single step / `conditional` / `transform`), `parallel_sampling` on deterministic logic (token waste), 6+ steps for a single-tool task (prefer ≤4), `debate` on non-contested factual tasks, checkpoint spam (1-3 at phase boundaries, never the final step), and ceremonial analyze/format steps. Reinforces P1.10 (simplicity), P1.15 (topology), P1.14 (one checkpoint per phase).
> - It's concatenated **after** `STEP_PLANNER_PROMPT.format(...)`, so it's never `.format()`-ed (no repeat of the P2.1 brace bug); a test also guards that the constant has no stray `{`/`}`.
> - **Real-execution check**: `test_planner_prompt_includes_anti_patterns` drives a real `StepPlanner.plan(...)` and asserts the catalogue ("Avoid these anti-patterns", "Don't use `supervisor`") reaches the LLM's `system_prompt`, composed with the rest of the planner prompt.
> - Tests: `tests/test_anti_patterns.py` (3). **722 passed, 6 xfailed, ruff clean.**

### [merged] P2.7 (merged into P1.15)

Originally "Explicit DAG pattern selector before `DAGBuilder`". Subsumed by the topology-library + selector agent in P1.15; pattern choice now flows from `TopologySelectorAgent`, not a separate DAG-only inference. Leaving the section number as a tombstone so external references don't shift.

### [done] P2.8 Enforce `allowed_step_types` at planner output validation

> **Done 2026-06-01.** The constraint was prompt-only — `StepPlanner` never checked the LLM's output, so a disallowed type slipped through and `to_carl_compat` silently downgraded it to `llm` (lossy, and invisible to the caller). Added **`StepPlanner._enforce_allowed_step_types`**, invoked inside `plan()` after the generation call:
> - On any step type ∉ `allowed_step_types`, **re-prompt** with explicit feedback naming the offending types (`"YOUR PREVIOUS PLAN WAS INVALID: it used step types [...]"`), up to `max_type_retries` (default 2).
> - If violations remain after retries, **coerce** the offenders to a safe allowed type (`"llm"` if permitted, else the first allowed type) with a warning — graceful, never fails generation over this.
> - No-op when `allowed_step_types` is unset (the common case) → zero extra LLM calls, zero churn to existing deep-mode tests.
> - **Real-execution check**: the tests drive the real `plan()` path with a mocked LLM — `test_violation_reprompts_then_accepts` (violation → 1 re-prompt → accepted, 2 calls), `test_coerces_to_llm_after_retries` (always-violating → coerced after 2 retries), `test_coerces_to_first_allowed_when_llm_not_allowed`, partial-violation, zero-retries-coerces-immediately.
> - Tests: `tests/test_allowed_step_types.py` (8). **706 passed, 6 xfailed, ruff clean.**

`prompts.py:169-187` appends an allowed-types constraint to the prompt but `StepPlanner` never validates output against it. If `allowed_step_types=['llm']` and LLM emits `agent_skill`, it passes through unchanged. Today `carl_export.py:189-192` silently downgrades to `llm` — lossy.

Add post-LLM validation: reject the plan and re-prompt with the violation if any step type is not in `allowed_step_types`. Max 2 retries.

---

## P3 — Hygiene

### [done] P3.1 Smarter fallback paths

> **Done 2026-06-01.** The fallback plan emitted `Step N` / `Process part N` placeholders; the DAG + description fallbacks copied them verbatim → useless chains when agents are disabled. Replaced with **domain/task-type-keyed templates** (`_FALLBACK_TEMPLATES` + `_select_fallback_template` + `_fit_template` in `generator.py`):
> - Six templates (`extraction` → Parse/Extract/Validate/Emit, `analysis`, `comparison`, `qa`, `generation`, `verification`) + a generic fallback, each ordered **core-first with the synthesis/output step last**. Selected by keyword-matching the `task_type`/`domain` (specific→broad), else generic.
> - `_fit_template` sizes the template to `suggested_step_count` while **always keeping the output step last** — fewer steps → first `count-1` + output; more → pad "refine" steps *before* the output. This preserves P1.10's count-driven sizing (so the simplicity-bias tests still see 2/3 steps) while making every title/aim meaningful.
> - Meaningfulness **propagates for free** through `_fallback_dag` + `_fallback_step_descriptions` (they copy title/rough_aim), so fixing only `_fallback_step_plan` upgrades the whole fallback path.
> - **Real-execution check**: `test_generate_fallback_chain_is_meaningful` drives a fully-stage-disabled deep `generate()` and asserts no `Step N`/`Process part` placeholders and that the chain ends on a synthesis step (`"Synthesize findings"` — default `DomainAnalysis()` has `task_type="analysis"`).
> - Tests: `tests/test_fallback_templates.py` (12 — selection, `_fit_template` sizing/output-preservation, meaningful titles, count-respect, generate() integration). **734 passed, 6 xfailed, ruff clean.**

### [done] P3.2 Config flag orthogonality

> **Done 2026-06-01.** Chose the "skip research when the planner is off" option (cheaper + cleaner than teaching the fallback to consume digests). Verified the digests' consumers first: the memory/web/capability/template/feedback/skill digests + demo-bootstrap + tool-synthesis output are consumed **only** by the two `planner.plan(...)` calls — so with `enable_step_planning=False` they were pure wasted LLM cost (the fallback ignores them).
> - In `_generate_deep`, computed `plan_will_run = enable_step_planning` and `topology_useful = enable_step_planning or enable_dag_optimization` once, and ANDed them into the relevant stage gates: the 6 research stages + `demo_bootstrap` + `tool_synthesis` now require `plan_will_run`; `topology_selection` requires `topology_useful` (the topology hint also feeds the DAGBuilder, so it's only skipped when *both* planning and dag-opt are off). Logs once when stages are skipped.
> - Documented the full interaction matrix in `MAGEConfig`'s docstring (new "Flag orthogonality (P3.2)" section).
> - **Real-execution check**: `tests/test_flag_orthogonality.py` drives real `generate()` calls — research/bootstrap/tool-synthesis skipped (absent from `deep_stages_completed`, zero LLM calls) when planning off; topology skipped when planning+dag both off but **runs** when dag-opt is on (it feeds the builder); research still runs when planning is on (no regression).
> - Tests: `tests/test_flag_orthogonality.py` (5). **739 passed, 6 xfailed, ruff clean.** No existing test churned (none had planning-off + research-on).

### [done] P3.3 Round-trip `from_json` smoke test in CI (subset of P0.2 wider)

> **Done 2026-06-02.** Added `tests/test_carl_loadable_smoke.py` — round-trips chains that `generate()` *actually produces* (not hand-built fixtures) through the real `mmar_carl.ReasoningChain.from_json`, catching drift between MAGE's `validate_carl_json` and CARL's loader. Two layers: (1) the tests set `raise_on_carl_load_error=True`, so `generate()` itself raises if the loader rejects its output (exercises the P0.3 `_try_load_reasoning_chain` path end-to-end); (2) an explicit `from_json(result.chain_json)` afterwards.
> - Parametrized over a spread of **round-trippable** step types via real fast-mode `generate()` — `llm_only`, `tool`, `mcp`, `transform`, `structured_output`, and a multi-type chain — each `to_carl_compat`-exported then loaded. Plus a deep-mode round-trip via the shared `mock_openai_deep` fixture.
> - The 6 agent-features types that crash `from_json` stay out of scope here (already tracked `xfail(strict)` in `test_carl_round_trip.py`); this smoke test guards the *common* generation paths against silent loader drift.
> - `pytest.importorskip("mmar_carl")` so it's skipped cleanly when the optional loader isn't installed.
> - Tests: `tests/test_carl_loadable_smoke.py` (8). **747 passed, 6 xfailed, ruff clean.**

### [done] P3.4 Chain-level metrics emission

> **Done 2026-06-02 — with an upstream caveat found by introspecting the real schema.** `StepDescriptionBase.metrics` turns out to be `list[Any]` with **`exclude=True`** and holds live `MetricBase` *ABC instances* (not a Pydantic spec — `ContainsMetric.__init__(self)` takes no args). So it is **runtime-only**: anything put there is dropped on serialize and can't round-trip as JSON specs (same class of blocker as P0.2 / the P0.3 runtime-only fields). Emitting metrics that *drive CARL execution* is therefore blocked upstream.
> - **MAGE-side support delivered** (faithful to the plan's literal "Add MAGE-side support … downstream analytics has structure"): new `CARLStepSchema.step_metrics` — a list of `{name, type, threshold}` specs carried as analytics metadata. Confirmed CARL **tolerates+ignores** it on load (it's not the runtime `metrics` field), so it rides in the emitted chain and never breaks execution.
> - `carl_export.py`: `VALID_METRIC_TYPES` (contains/exact_match/case_insensitive_match/regex_match/numeric/llm_judge — mirroring CARL's `MetricBase` subclasses), `validate_step_metrics(chain)` (flags malformed specs per step), `chain_metrics_summary(chain)` (`{total, by_type, per_step}` for analytics). Describer-prompt guidance added so the LLM declares metrics on steps with a checkable criterion.
> - Generator logs malformed metrics (non-fatal, beside the P2.1 reliability check).
> - **Real-execution check**: `test_generate_carries_metrics_and_flags_malformed` drives `generate()` — a valid metric is carried in the returned chain, a malformed one is logged; `test_metrics_chain_still_loads_in_carl` confirms the metrics-bearing chain loads via `ReasoningChain.from_json`.
> - Tests: `tests/test_step_metrics.py` (11). **758 passed, 6 xfailed, ruff clean.**
>
> **Upstream ask (filed in cross-cutting):** a serializable `MetricSpec` (`{name, type, threshold}`) + a `from_dict` that CARL reconstructs into `MetricBase` instances, so declared metrics can drive execution-time evaluation, not just MAGE analytics.

### [done] P3.5 Shipped profiles for the four headline benchmarks (factored out of the original P1.16)

> **Done 2026-06-02** (its ingredients — P1.7/P1.15/P1.16 — all landed). Authoring-time presets that bias generation per benchmark.
> - **`mmar_mage/profiles.py`**: `BenchmarkProfile(name, config, default_topology, tool_catalog, notes)` + `BENCHMARK_PROFILES` for `bfcl` / `seal_tools` / `rotbench` / `toolalpaca` / `custom`, with `config` overrides mapped onto **real** MAGEConfig fields (tested) — e.g. `bfcl` → `allowed_step_types=[llm,tool]`, `simplicity_bias=0.6`, `simplicity_max_steps=5`; `rotbench` → adds `evaluation` (early validation); `toolalpaca` → adds `mcp`+`tool_discovery` (long-tail APIs); `seal_tools` → adds `parallel_sampling` (self-consistency). Each carries a `default_topology` preference + `notes` on known quirks ("BFCL simple_python rewards ≤3 steps").
> - **`MAGEConfig.benchmark_profile`** field (Literal) + a `mode="before"` validator that fills any **unset** field from the profile (so values are validated normally + the after-validators see them) — explicit user values always win. Plus a `MAGEConfig.for_benchmark(profile, **overrides)` factory (mirrors `for_openrouter`/`for_local`).
> - **CLI**: `mage generate --benchmark-profile bfcl "query"` (validated against the registry).
> - **Real-execution check**: `test_generate_with_profile_respects_allowed_types` drives `generate()` under the bfcl profile and confirms the output stays within `{llm, tool}` and `simplicity_bias=0.6` was applied; other tests cover unset-fill / explicit-override-wins / factory / every-profile-key-is-a-real-field / all-profiles-validate.
> - Tests: `tests/test_benchmark_profiles.py` (11). **769 passed, 6 xfailed, ruff clean.**
>
> **Scope (as planned):** authoring-time settings only — benchmark *execution* stays in `experiments/baselines/`. `default_topology`/`tool_catalog` ride as profile metadata (the TopologySelectorAgent still chooses).
>
> **Curated per-profile demo sets (P1.7 enrichment) — [done 2026-06-02].** Each headline profile now ships `demos` — proven `{query, plan:[{number,title,step_type,aim}]}` reference plans whose step types stay within that profile's `allowed_step_types` (tested). When a profile is selected, `format_profile_demos` seeds them into a **new `StepPlanner.curated_demos` param** (its own honest "Curated reference plans (proven structure for this benchmark)" header — distinct from P1.7's synthetic-benchmark-`bootstrapped_demos` header), cheaply and benchmark-specifically (no runtime bootstrap needed). The generator computes them from `benchmark_profile` and threads them through the shared `_plan` closure (so they compose with cold-start + parallel-topology sampling). **Real-execution check**: `test_profiled_generate_seeds_curated_demos_into_planner` drives `generate()` under the bfcl profile and asserts the curated block reaches the planner prompt; the no-profile case asserts none. Tests: `tests/test_profile_demos.py` (8 — demos/consistency/format/injection + 2 generate() integration). **791 passed, 6 xfailed, ruff clean.**

---

## Cross-cutting: changes that need `carl-experiments` PRs

Filed as upstream issues, not done in MAGE:

- **[upstream] CARL `from_json` cannot round-trip the agent-features step types** (discovered in P0.2, verified against the live `agent-features` checkout). Two concrete bugs in `mmar_carl/chain.py`: (1) `_reconstruct_step_config` (used by the default `from_dict`/`from_json`) has **no branch** for `evaluation` / `human_input` / `debate` / `mcp_resource` / `tool_discovery` → their `step_config` loads as `None` (silent drop); (2) `agent_skill` IS reconstructed but `AgentSkillStepConfig` is **missing from the generic `StepDescription.step_config` Union** (`config.py` `StepConfig = Union[...]`) → `from_json` raises. The typed path (`from_dict_typed`/`_create_typed_step_from_dict`) raises `Unknown step type` for all of these too. **Fix upstream:** add the missing branches to `_reconstruct_step_config`, add `AgentSkillStepConfig`/`SupervisorStepConfig`/`DebateStepConfig`/`MCPResourceStepConfig` to the `StepConfig` Union, and extend `_create_typed_step_from_dict`. This is the hard blocker for **P2.1**; MAGE's round-trip test (`test_carl_round_trip.py`) already marks the 6 affected types `xfail(strict=True)` so a fix is detected automatically. File this as the priority upstream issue alongside the runtime-only-fields one below.
- **CARL JSON loader for `sub_chain` / `agents` / `base_step`** — unblocks P0.3 option (B). Suggest a `chain_registry: dict[str, ReasoningChain]` constructor argument with sub-chain references emitted as `"sub_chain_ref": "registered-name"`.
- **`MCPServerConfig.headers_secret_ref`** — current `headers: dict[str, str]` would leak secrets in serialized JSON. Add a secret-ref form.
- **JSON-loadable `parallel_sampling.base_step`** — same problem as sub-chains. Could be inline in JSON since `LLMStepDescription` is JSON-serializable already.
- **[upstream] Serializable `MetricSpec` for `StepDescriptionBase.metrics`** (discovered in P3.4). The `metrics` field is `list[Any]` with `exclude=True` holding live `MetricBase` ABC instances, so declared metric specs cannot persist in JSON or drive execution after a `from_json` round-trip. **Fix upstream:** add a Pydantic `MetricSpec` (`{name, type, threshold}`) + a `from_dict`/registry that reconstructs the concrete `ContainsMetric`/`ExactMatchMetric`/… from a spec, and make the field serialize the specs (not `exclude=True`). MAGE already emits `step_metrics` specs (P3.4) ready to map onto this once it exists.
- **gigaevo-memory `who-cares`: pin a commit, not the branch** (P1.18). It's an active feature branch; confirm the `client/` SDK class + method names and the `/v1/search/unified` request schema against `openapi.yaml` at the pinned commit, and ask the AIRI maintainers whether the SDK surface is considered stable. If a typed `agent-skills` / `memory-cards` create call isn't in the SDK yet, that's an upstream ask before P1.17 Phase C and P1.3 can fully land.

---

## Validation plan

Per change, retest grid:
- benchmarks: BFCL / Seal-Tools / RoTBench / ToolAlpaca
- executors: gpt-oss-20b / Llama-3.3-70B / gpt-oss-120b / Qwen3-8B-AWQ
- baseline: current MAGE-best (0.750 mean, fast-mode authoring cost)
- target: > 0.78 mean (beats next non-MAGE method by > 0.07)

Regression suite:
- run-on-PR: P0.2 round-trip tests (every step type)
- nightly: small benchmark subset (10 tasks × 4 benchmarks × 1 executor)
- weekly: full grid

Track in `experiments/baselines/results/improvement_plan_tracker.csv`:
columns: `change_id, status, baseline_score, post_change_score, delta, retest_date`.

---

## Sequencing

Estimated calendar from a single full-time developer:

| Phase | Items | Weeks |
|---|---|---|
| 1 | P0.1, P0.2, P0.3 (decision), P1.6 | 2 |
| **2** | **P1.18 (real gigaevo-memory: hybrid search + typed entities)** | **2** |
| 3 | P1.1, P1.2, P1.3, P1.4, P1.5 (now backed by P1.18) | 2 |
| 4 | P1.11, P1.12, P1.13, P1.14 | 1 |
| 5 | **P1.16 (synthetic benchmark mode)** — extract from `evolve/` + new stage 7 | 2 |
| 6 | P1.9 (execution-feedback loop, fueled by P1.16) | 2 |
| 7 | P1.7, P1.8 (MIPRO + GEPA-style, fueled by P1.16 + P1.18) | 2 |
| 8 | P1.10 (simplicity bias) | 0.5 |
| 9 | **P1.15 (diverse topologies)** | 2 |
| 10 | **P1.17 Phase A (tool-spec synthesis)** | 1 |
| 11 | **P1.17 Phase B (impl + sandbox)** | 2 |
| 12 | **P1.17 Phase C (full skill synthesis, registers as Agent Skills via P1.18)** | 1 |
| 13 | P2.1 (step types 1-5 by ROI) | 2 |
| 14 | P2.2, P2.3, P2.4 | 2 |
| 15 | P2.5, P2.6, P2.8 (P2.7 merged into P1.15) | 1 |
| 16 | P3.x cleanup (incl. P3.5 shipped profiles) | 0.5 |

**Total: ~25 weeks single-track.** With 2 devs parallel (one on memory / retrieval / benchmark / loop track, one on topology + tool-synthesis track): ~14 weeks.

Critical path is now **P0 → P1.18 (real memory) → P1.16 (synthetic benchmark) → P1.9 (execution-feedback) → P1.7 + P1.8**. P1.18 comes early because hybrid retrieval + typed entities are the substrate under P1.1, P1.3, P1.5, P1.7 — doing those first against the BM25-only API means redoing them. P1.16 must land before P1.9, P1.7, P1.8 because all three need a self-synthesized fitness signal. P1.15 (topologies) and P1.17 (tool synthesis) run on a parallel track and don't gate the main quality loop; note P1.17 Phase C writes its skills back through P1.18's Agent Skills entity, so it should follow P1.18.

After P1.16 + P1.9 we should already match MIPRO / GEPA on Seal-Tools and ToolAlpaca (closed the execution-feedback gap). After P1.7 + P1.8 we should pull ahead (better demo synthesis + reflective mutation). P1.15 widens the moat on RoTBench (parallel sampling, debate). P1.17 opens up tasks no competitor can play (custom tools).

---

## What we will NOT do (in scope discipline)

- No new generation modes beyond `fast` / `deep`. If a new combination is needed, it's a config flag (e.g. `enable_benchmark_mode`), not a code path.
- No matplotlib/Python plotting in the generator. Plotting is for the paper, not the runtime.
- No new prompt languages, no DSL invention. CARL JSON is the contract.
- No support for closed-weight executor tuning. We don't control them; design assumes drop-in OpenAI-API executors.
- No committing of generated tools / skills to the MAGE repo. They live in `~/.mage/generated_*/` per-user, not in source control. Any tool used in the paper or in shipped benchmark profiles must go through human review and be re-published as a regular CARL tool.
- **No reporting synthetic-benchmark `quality_report` numbers as headline results.** P1.16 is an authoring-time signal — same role as unit tests. Headline numbers stay sourced from `experiments/baselines/` against real benchmarks. Conflating the two is the fastest way to lose reviewer trust.
- No "benchmark profile execution" in `MAGEConfig` (P3.5). Profiles are authoring-time settings, runners stay in `experiments/baselines/`. Mixing these would couple the library to evaluation harnesses.
- No topology defaults that auto-select advanced types (debate / supervisor / replan-policies) without `enable_topology_selection=True`. The library ships safe-by-default (linear pipeline) and lights up complexity on explicit opt-in.
- No code-execution-by-default in the sandbox for P1.17 Phase B. Default is `enable_tool_synthesis=False`; users opt in to the synthesis pipeline and the sandbox stays mandatory whenever it's on.
