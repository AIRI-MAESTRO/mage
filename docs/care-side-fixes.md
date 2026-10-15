# CARE-side fixes (from the 09.06.2026 failure audit)

Tasks **T6, T7, T14, T20** from `mage-failure-audit-and-todo-20260610.md` live in the
**CARE** app (`../care`) and the **CARL runtime** (`../carl-experiments` → `mmar_carl`),
not in `carl-mage`'s working copy. Per the audit's Регламент they are **specified
here** for the CARE team instead of implemented in MAGE, so the autonomous MAGE run
doesn't block on a repo it can't touch. Each references the MAGE-side support that
already landed on branch `mage-failure-fixes`.

> Status legend: ☐ to do · ◑ spec ready (this doc) · ☑ done in CARE.

---

## T6 — Tool errors must be errors (executor + ledger) ◑

**Problem (run 096).** A `tool` step (`http_request` → `GET https://<the question>`)
returned the string `"… Invalid IDNA hostname"` but the step result carried
`success: true`. Because the result reported success, `retry_max=3` never fired,
replan never fired, and the ledger recorded `status: success` — a user-facing failure
(an apology) was logged as a successful run.

**Root cause (R6 — "error as data").** The CARL tool executor treats any non-exception
return value as success. A tool that returns an error *string* (or a JSON body with an
error field) without raising is recorded as `success=true`.

**Fix.**

1. **Executor (`mmar_carl` step executors).** After a `tool`/`mcp` step runs, classify
   the result *before* setting `success`:
   - raised exception → `success=false`, `error=<exc>` (already the case);
   - returned value that *looks like* a tool error (HTTP error body; `Invalid IDNA
     hostname`; connection/timeout/`traceback`/`errno`/`max retries exceeded` markers;
     an `{"error": …}` or `{"success": false}` payload) → `success=false`,
     `error=<extracted message>`;
   - otherwise → `success=true`.

   **Reuse the MAGE detector so both sides agree on "looks like an error":**
   `mmar_mage.memory.find_masked_tool_errors(step_results)` (landed in **T34**) already
   encodes the markers and the "success=true but error-shaped output" rule. Either call
   it on the executed steps, or port its `_TOOL_ERROR_MARKERS` / `_looks_like_tool_error`
   constants into the executor so the classification is identical.

2. **Retry.** With `success=false`, the existing `retry_max` path re-runs the step —
   now it actually triggers on the masked failures.

3. **Ledger / replay bundle.** The run's `status` must reflect the real outcome: any
   step ending `success=false` after retries ⇒ `status: failed` (hard) or `degraded`
   (a partial answer was synthesised). **Never `success` when a step failed.** Also
   write `extra.tool_errors` (from the detector) into the bundle so the failures are
   visible without re-running.

**Acceptance.**
- Replaying run 096: the masked `Invalid IDNA hostname` step has `success=false`;
  `retry_max` fires; ledger `status != success`.
- A genuinely successful run is unchanged (`status: success`).
- A tool returning `{"success": false, "error": "…"}` is recorded as a failure, not data.

**Files.** CARL executor: `carl-experiments/src/mmar_carl/step_executors.py` (tool/mcp
execution + result construction). CARE ledger / replay writer: `care/runtime/…`. MAGE
support already shipped: `carl-mage/mmar_mage/memory.py::find_masked_tool_errors` (T34)
and `MemoryManager.save_failure_cards_for_run` (writes a failure card per detected error).

**Ties into:** T7 (synthesiser must own the failure instead of blaming the user),
T20 (surface the ⚠ in the TUI), T10 (a real failure should trigger replan).

---

## T7 — Synthesiser must own failures, not blame the user ◑

**Problem (run 096, step 6).** After the run failed, the synthesiser apologised **and
told the user to "rephrase the question"** — shifting blame to the user for a *system*
failure (MAGE/CARE built a broken URL, `GET https://<the question>`). The user asked a
perfectly clear question.

**Root cause.** The synthesiser prompt has one failure path: "I couldn't answer →
ask the user to clarify." It doesn't separate **system failure** (a step errored — the
system's fault) from **genuine input ambiguity** (the request really is under-specified).
On any empty/failed result it defaults to "please rephrase."

**Fix.**

1. **Failure-aware synthesis (depends on T6).** Branch the synthesiser on the run
   status. If any step ended `success=false` (the T6 signal), the answer must:
   - say *what went wrong* in plain language (e.g. "I built an invalid request URL");
   - say *what the system did / will do* about it (retried / replanned / regenerated);
   - **not** advise the user to rephrase, unless the failure is genuinely an
     input-ambiguity class (e.g. a required parameter the user alone can supply).
   Pass the failed step's `error` (from T6) into the synthesiser so it can be specific
   rather than generic.

2. **Replan-first (depends on T10).** Better: a step failure should trigger automatic
   replan / regenerate **before** synthesis runs, so the synthesiser reports the final
   recovered outcome — not a raw failure. Synthesis is the *last* stage, after recovery.

3. **Prompt change.** The CARE answer/synthesiser prompt needs the explicit rule
   "system failure → own it + state the next action; never tell the user to rephrase a
   clear question." Cover it with a CARE-side prompt snapshot test (mirror MAGE's
   `test_describer_examples.py` pattern).

**Acceptance.**
- Run 096 replay: the answer names the URL-construction failure + the system's recovery
  (retry / replan), **not** "please rephrase."
- A genuinely ambiguous query (no step failure, just low confidence) may still ask for
  clarification — that path stays legitimate.

**Files.** CARE synthesiser / final-answer step + its prompt. Depends on T6 (the
`success=false` signal) and T10 (replan before synthesis).

**Ties into:** T6 (provides the failure signal), T10 (recover before synthesising),
T20 (the TUI should already show the ⚠ by the time the answer lands).

---

## T14 — Token + cost accounting in the ledger / replay ◑

**Problem (R10).** The TUI shows real numbers (`in 44013 / out 8029`, WP-5), but the
replay bundle and ledger record `tokens_in: null`, `tokens_out: null`,
`cost_usd: null`. So cost is invisible after the fact — and the evolve branch, which
uses cost as a fitness signal, runs blind.

**Root cause.** The run's actual usage lives on `MAGEResult.metadata` (WP-5) but is
never written into the ledger / replay bundle the run writer emits.

**Fix (CARE).** After a run, read the already-available metadata and persist it:
- `metadata.prompt_tokens` → `tokens_in`, `metadata.completion_tokens` → `tokens_out`
  (and `metadata.total_tokens`); `metadata.usage_estimated` → an `estimated` flag.
- per-stage breakdown: `metadata.stage_telemetry()` (T45) → `extra.stage_timings`
  (each stage's `time_seconds` + `usage`), so a post-mortem reads from one file.
- `cost_usd`: compute it from the actual usage with
  `mmar_mage.cost.cost_usd_for_usage(prompt_tokens, completion_tokens, in_per_1k,
  out_per_1k)` (shipped this commit) using CARE's per-model price table — the same
  per-1k formula MAGE's `estimate_chain_cost` uses, so estimate and actual agree.
  Unknown model price → `0.0` (record it, don't leave `null`).

**MAGE support already shipped.** WP-5 token fields on `MAGEMetadata`, T45's
`stage_telemetry()` per-stage timing+usage bundle, and the new
`cost_usd_for_usage` helper. CARE only needs the price table + the ledger write.

**Acceptance.**
- A fresh replay / ledger has non-null `tokens_in` / `tokens_out` / `cost_usd`
  matching the TUI for the same run.
- `extra.stage_timings` is populated per stage.
- The evolve branch can read a real `cost_usd` as a fitness term.

**Files.** CARE run / ledger writer (`care/runtime/…`), CARE's per-model price table.
MAGE support: `carl-mage/mmar_mage/cost.py::cost_usd_for_usage`,
`mmar_mage/schemas.py::MAGEMetadata` (`usage_by_stage`, `stage_telemetry`).

---

## T20 — Honest run status in the TUI ◑

**Problem (run 096).** The TUI rendered a green `✓` on every step of a run that
actually failed — the masked tool error on step 1 (`Invalid IDNA hostname`) never
showed up in the progress widget, so the failure was invisible until someone dug
through the replay bundle.

**Root cause.** The execution progress widget marks a step done as soon as it
returns, regardless of the step's real outcome. With the "error as data with
`success: true`" masking (R6) every step looked successful; even once T6 sets
`success=false`, the widget still has to *render* that failed state.

**Fix (CARE).** The per-step progress indicator must reflect the step's real status:
- `✓` on success;
- `⚠` (or `✗`) on a step that ended `success=false` (the T6 signal) — shown
  **immediately** on that step, not buried in the replay;
- a degraded run (recovered via replan, T10) can show `↻`/`⚠` so the user sees a
  retry happened.
Drive it from the same per-step result CARE already has after execution, and reuse
the MAGE detector `mmar_mage.memory.find_masked_tool_errors(step_results)` (T34) so
the widget flags the *masked* errors too, consistently with the ledger (T6) and the
auto failure-card (T34). This mirrors MAGE's own generation-stage
`MAGEProgressCallback.on_error(stage, error)` hook, which already surfaces failed
*generation* stages — execution steps need the equivalent.

**Acceptance.**
- Run 096 replay drives the TUI: step 1 shows `⚠`, not `✓`; the overall run is not
  rendered as a clean success.
- A genuinely successful run is unchanged (all `✓`).

**Files.** CARE TUI progress widget + the step-execution callback (`care/screens/…`,
`care/runtime/…`). Depends on T6 (the `success=false` signal). MAGE support:
`carl-mage/mmar_mage/memory.py::find_masked_tool_errors` (T34), and the
`MAGEProgressCallback.on_error` pattern in `mmar_mage/generator.py`.

**Ties into:** T6 (the failure signal), T7 (the answer should already own it), T34
(the same masked-error detector feeds the failure card).
