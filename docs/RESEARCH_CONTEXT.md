# MAGE — Research Context & Handoff Document

> Self-contained briefing for moving MAGE-related research between projects.
> Covers: project positioning, the CARL/GigaEvo paper that grounds the work,
> the current codebase state, the CARL JSON spec we conform to, the evolve-mode
> we built on top, open research questions, and a concrete next-steps roadmap.

---

## 1. Project positioning — one paragraph

**MAGE (Mesh Architecture Generation Engine)** turns a natural-language task
description into a fully specified, structurally-valid, runnable multi-step
LLM-agent program — a CARL chain. Today it does this in two modes: a single-shot
deep pipeline (`MAGEGenerator.generate`), and an evolutionary
benchmark-driven optimizer (`ChainEvolver.evolve`) that synthesizes a per-task
benchmark, generates a population, scores via real or simulated execution, and
iterates with mutation/crossover/LLM-driven repair operators.

We position MAGE as **the autonomous-agent constructor**: text in, working
multi-step agent out. The MMAR ecosystem splits responsibilities cleanly:
- **mmar-carl** — chain runtime / executor (the paper's CARL format).
- **gigaevo-memory** — memory service for cross-run knowledge accumulation.
- **GigaEvo** — the evolutionary backbone formalized in the paper (CARL/GigaEvo,
  NeurIPS 2026 submission).
- **MAGE (this repo)** — the upstream constructor + a complementary evolutionary
  loop (autonomous benchmark synthesis is the differentiator).

The research thesis: *the manual seed chain that GigaEvo requires is a hidden
bottleneck on what's claimed to be an "automated" pipeline. MAGE removes it
end-to-end — a user types a task, MAGE both decomposes it into a CARL chain
and synthesizes the benchmark to evolve against.*

---

## 2. The CARL/GigaEvo paper — what to know

Citation key in this codebase: **"CARL paper"** = "Evolving Multi-Step Agent
Reasoning: CARL Chains as a Unified Evolutionary Genome" (NeurIPS 2026,
anonymous submission). PDF lives in the user's Downloads folder.

### 2.1 Core claim

**The genome representation matters more than the evolutionary algorithm.**
CARL — a typed, JSON-serializable DAG of structured steps where every LLM step
exposes four semantic fields (`aim`, `stage_action`, `reasoning_questions`,
`example_reasoning`) — outperforms or matches state-of-the-art prompt-only
optimizers (GEPA, MIPROv2) under matched compute on 5 benchmarks. The gain comes
from the representation, not the optimizer (proved by an ablation that swaps
GigaEvo for OpenEvolve).

### 2.2 CARL format properties

Three properties make CARL viable as an evolutionary genome:

1. **Structural validity is decidable** at parse time — schema parser rejects
   malformed candidates without execution.
2. **Prompt-only optimization is subsumed** — a single-step CARL chain with a
   free-form `stage_action` exactly recovers prompt evolution as a special case.
3. **Step-batched execution is natural** — DAG topology lets all N evaluation
   samples advance through level k together, producing homogeneous LLM
   request batches that vLLM serves efficiently.

### 2.3 Atomic CARL step schema (paper Figure 2)

```
Structural fields:    title, number, step_type, dependencies, frozen
Semantic (LLM-only):  aim, stage_action, reasoning_questions, example_reasoning
Tool-specific:        tool_name, tool_parameters, behavior_on_failure
Context references:   $outer_context, $history[k]
```

`step_type ∈ {llm, tool}` in the paper; in our implementation we add `mcp`,
`memory`, `transform`, `conditional`, `structured_output` (matching the current
`mmar_carl` library which has evolved beyond what's documented in the paper).

### 2.4 GigaEvo loop (Algorithms 1-2, paper §3)

```
Outer loop (G generations × K parents/gen = B mutator-pipeline calls budget):
  for g in 1..G:
    select K elites from MAP-Elites archive A
    for each parent in parallel:
      r ← PROPOSE-&-EVALUATE(parent, archive, runner, mutator)
      if r is valid: A ← A ∪ {r}
  return arg max F(Φ; D_score)

PROPOSE-&-EVALUATE (1 mutant):
  I ← INSIGHTS(parent, archive)             # 1 LLM call
  L ← {LINEAGE(parent, co-elite_j)}_{j=1,2} # 2 LLM calls
  Φ' ← MUTATION(parent, I, L)               # 1 LLM call (= 4 total per candidate)
  validate(Φ'); if invalid → ⊥
  for k in 1..L:
    Ω[k] ← runner.step(s_k, Φ', x, prior_results)   # batched over D_score
  return (Φ', score)
```

Paper budget: B = 200 candidate proposals (G=25, K=8). Inference LLM = Qwen3-8B
(frozen). Mutator LLM = Qwen3-235B-A22B-Thinking-2507. **Crucially, ALL compared
methods (GigaEvo-Chain, GigaEvo-Prompt, GEPA) share the same B** — the comparison
is matched-compute.

### 2.5 Two evolution modes

- **Static mode** — chain topology and tool calls frozen; only LLM semantic
  fields and the system prompt mutate. Used when task structure mandates
  specific tool calls (HotpotQA, HoVer, IFBench, GSM8K).
- **Full mode** — optimizer can insert/remove/retype steps and rewire
  dependencies subject to DAG validity. Used on AIME (no structural constraints).

### 2.6 Four feedback modes (paper §3, last paragraph)

Inference-time feedback wrappers around `R_φ.step`:
- `without-feedback` — no retry on step output.
- `self-reason` — fixed phrase in the chain triggers one retry.
- `metric-aware` — retries when an external scorer falls below threshold.
- `data-aware` — retries with ground-truth `y` injected as a counter-example
  critique (oracle access, only at training).

Each wrapper performs **at most one retry per step** before consumption by step
k+1.

### 2.7 Headline results (paper Table 1)

| Bench    | Best prompt | Best CARL chain | Δ (test) | p     |
|----------|-------------|-----------------|----------|-------|
| GSM8K    | 95.0        | 98.5 (data-aware) | +3.5  | 0.048 |
| HoVer    | 49.0        | 68.7 (data-aware) | +19.7 | <0.001|
| IFBench  | 52.2        | 53.2 (metric-aware) | +1.0  | 0.806 (val: +17.0pp, p<0.001) |
| HotpotQA | **82.0**    | 78.0 (data-aware) | -4.0  | 0.011 (prompt wins) |
| AIME     | 57.8        | 81.1 (data-aware) | +23.3 | <0.001|

**Optimal feedback mode is task-dependent** (no universal winner): without
beats on IFBench, data-aware on HotpotQA/AIME/HoVer, metric-aware best
non-oracle on HoVer.

### 2.8 Limitations the paper acknowledges (research handles)

> **Quote (§Limitations)**:
> 1. *Every LLM step uses a single frozen inference model.* No per-step model
>    identifier. **→ MAGE could add per-step model selection as evolved field.**
> 2. *Every benchmark begins from a manually constructed seed chain.*
>    "LLM-based auto-seeding from the task description can remove this bias and
>    enable zero-effort application to new tasks; we leave it to future work."
>    **→ This is exactly the MAGE thesis. We claim this contribution.**
> 3. *MAP-Elites archive uses one-dimensional fitness descriptor.* No diversity
>    pressure on structural axes (chain length, tool composition).
>    **→ MAGE could add multi-axis behavioral descriptors.**

These three limitations form the natural research frontier for MAGE.

---

## 3. Current MAGE codebase — module map

### 3.1 Top-level directories

```
mmar_mage/                  # Main package — 8979 LoC
  agents/                   # Per-stage LLM-driven agents (15 files)
  evolve/                   # Evolutionary chain optimization (6 files, ~1400 LoC)
  generator.py              # MAGEGenerator orchestrator (deep + fast modes)
  schemas.py                # All Pydantic models (CARLChainSchema, etc.)
  prompts.py                # All LLM prompts as constants
  carl_export.py            # MAGE-format → CARL-loadable JSON converter
  llm.py                    # AsyncOpenAI wrapper (LLMClient)
  memory.py                 # gigaevo-memory wrapper (MemoryManager)
  cli.py                    # `mage generate`, `mage evolve`, `mage version`
  code_generator.py         # CARL JSON → Python code (ChainBuilder API)

evaluation/                 # Stub-v1 benchmark runner (separate from evolve)
  runner.py                 # CLI for batch generation + execution
  chain_generator.py        # Generates chains for fixed benchmark tasks
  chain_executor.py         # Stub v1: flatten chain → 1 LLM call (TODO: real CARL exec)
  baseline_generator.py     # Plain-prompt baselines for comparison
  tasks.py                  # BenchmarkTask schema + seed registry (hover, hotpotqa, etc.)

tests/                      # 350 tests, all passing as of this snapshot
```

### 3.2 Agents inventory (mmar_mage/agents/)

| Agent | Stage | Purpose | LoC |
|-------|-------|---------|-----|
| DomainAnalyzer | 1 | classify domain/complexity/preferred step types | 66 |
| StepPlanner | 2 | decompose into rough step sequence | 101 |
| DAGBuilder | 3 | wire dependencies into a DAG | 65 |
| StepDescriber | 4 | flesh out each step with full CARL semantic fields | 157 |
| MemoryResearchAgent | 0a | recall similar past chains from gigaevo-memory | 149 |
| WebResearchAgent | 0b | web search digest for domain context (Tavily/SerpAPI/Brave) | 230 |
| CapabilityLookupAgent | 0c | query memory for available tools/MCP servers | 77 |
| TemplateAssemblerAgent | 0d | assemble skeleton from past successful chains | 97 |
| FeedbackRecallAgent | 0e | tap past execution feedback (success/failure) | 187 |
| StepCriticAgent | 4.5 | per-step critique → revise loop | 146 |
| ChainVerifier | 5.5 | chain-of-verification (yes/no questions about chain) | 240 |
| SelfRefiner | 5.6 | iterative whole-chain refinement | 249 |
| ToTPlanner | 3 (alt) | Tree-of-Thoughts DAG planning (replaces DAGBuilder) | 276 |
| MCTSConstructor | 4 (alt) | MCTS over step descriptions (replaces StepDescriber) | 574 |

### 3.3 Evolve module (mmar_mage/evolve/) — what we built

Built on top of the static MAGE pipeline. ~1400 LoC, ~32 tests.

| File | Role | LoC |
|------|------|-----|
| `schemas.py` | EvolveConfig, EvolveResult, Individual, ChainScore, SynthesizedTask, BenchmarkSynthesisResult, EvolveGeneration | 150 |
| `benchmark_synthesizer.py` | Stage 1: synthesize tasks via LLM + memory recall + (optional) web | 311 |
| `grader.py` | Stage 3: score chains via structural rubric + LLM-judge + (optional) real-execution | 286 |
| `operators.py` | Stage 4: mutate_aim, change_step_type, insert_step, remove_step, swap_steps, crossover, llm_repair | 342 |
| `engine.py` | Stage 0/2/4-5: orchestrator (domain → benchmark → population → grade → evolve loop) | 399 |

### 3.4 Public Python API

```python
from mmar_mage import (
    # Single-shot generation
    MAGEGenerator, MAGEConfig, MAGEResult, MAGEMetadata,

    # Evolutionary search
    ChainEvolver, EvolveConfig, EvolveResult, Individual, ChainScore,
    BenchmarkSynthesisResult, SynthesizedTask, EvolveGeneration,

    # Validation + CARL-compat
    validate_carl_json,           # accepts strict_carl=True for runtime check
    to_carl_compat,               # MAGE-format dict → CARL-loadable dict

    # Schemas
    CARLChainSchema, CARLStepSchema,
    DomainAnalysis, StepPlan, DAGStructure,

    # Constants
    VALID_STEP_TYPES,             # MAGE-internal incl. eval/aggregator
    CARL_VALID_STEP_TYPES,        # CARL runtime: llm/tool/mcp/memory/transform/conditional/structured_output

    # Quality enhancement agents (all opt-in via MAGEConfig flags)
    StepCriticAgent, ChainVerifier, SelfRefiner,
    ToTPlanner, MCTSConstructor, FeedbackRecallAgent,
)
```

### 3.5 CLI surface

```bash
# Single-shot generation
mage generate "Verify a multi-hop fact" --mode deep --output chain.json

# Evolutionary search with autonomous benchmark
mage evolve "Extract structured invoice data" \
  --population 5 --generations 5 --benchmark-tasks 8 \
  --execution-model gpt-4o-mini \              # real chain execution (vs LLM-judge forecast)
  --memory --benchmark-output bench.json -o best.json
```

---

## 4. CARL JSON format — concrete spec we implement

### 4.1 Two step-type sets

```python
# MAGE-internal — what intermediate stages emit
VALID_STEP_TYPES = {
  "llm", "tool", "mcp", "memory", "transform", "conditional",
  "structured_output", "eval", "aggregator"
}

# CARL runtime — what mmar_carl.ReasoningChain.from_json() actually accepts
CARL_VALID_STEP_TYPES = {
  "llm", "tool", "mcp", "memory", "transform", "conditional", "structured_output"
}
```

`eval`/`aggregator` are MAGE-only semantic labels. `to_carl_compat()`
([mmar_mage/carl_export.py](../mmar_mage/carl_export.py)) translates them to `llm`
with a synthesized aim before serialization.

### 4.2 LLM step (canonical CARL)

```json
{
  "step_type": "llm",
  "number": 1,
  "title": "Extract entities",
  "aim": "Identify all named entities in the source text",
  "reasoning_questions": "What is the entity, role, and span?",
  "step_context_queries": ["entity", "named entity", "person", "organization"],
  "stage_action": "Extract and categorize entities",
  "example_reasoning": "Given a person mentioned with title and affiliation, extract all three.",
  "dependencies": [],
  "retry_max": 3,
  "llm_config": null,                    # optional per-step model override
  "checkpoint": false,                   # RE-PLAN rollback marker
  "checkpoint_name": null,
  "replan_enabled": null
}
```

`aim` MUST be non-empty for `llm` and `structured_output` — runtime
`LLMStepDescription` validator rejects empty `aim` with `ValueError`. Our
`validate_carl_json` enforces this pre-flight.

### 4.3 Non-LLM step — nested step_config

CARL runtime expects type-specific config nested under `step_config`. Our
intermediate `CARLStepSchema` keeps it flat for ergonomics; `to_carl_compat`
re-nests:

```json
{
  "step_type": "tool",
  "number": 2,
  "title": "Fetch invoice text",
  "dependencies": [1],
  "step_config": {
    "tool_name": "http_get",
    "tool_description": "...",
    "input_mapping": {"url": "$history[-1]"},
    "output_key": "response",
    "timeout": 30.0,
    "parameters": [...]
  }
}
```

MCP requires a doubly-nested `server` object inside `step_config`; memory uses
`{operation, memory_key, value_source, namespace}`; structured_output uses
`{output_schema, schema_name, input_source, instruction, strict_json}`. See
[mmar_mage/carl_export.py](../mmar_mage/carl_export.py) for the complete mapping
table.

### 4.4 Chain-level fields

```json
{
  "name": "Invoice extraction chain",
  "max_workers": 2,
  "timeout": 300.0,
  "search_config": {
    "strategy": "substring",
    "substring_config": {
      "case_sensitive": false,
      "min_word_length": 3,
      "max_matches_per_query": 5
    }
  },
  "steps": [...],
  "replan_policy": null      # optional chain-level RE-PLAN config
}
```

### 4.5 Validation guarantees

`validate_carl_json` ([mmar_mage/generator.py:101](../mmar_mage/generator.py)):
1. Pass 1 — structural: `number` unique, `title` non-empty, dependencies
   reference existing steps, no self-references.
2. Pass 2 — DFS cycle detection (catches `1→3→2→1` regardless of order).
3. Pass 3 — semantic: `step_type ∈ valid_set`, non-empty `aim` for `llm`/
   `structured_output`.

Two modes via `strict_carl: bool = False`. False = MAGE-internal types allowed
(`eval`, `aggregator` legal). True = CARL runtime types only — call this AFTER
`to_carl_compat`.

---

## 5. Evolve mode — architecture in detail

### 5.1 Pipeline (mmar_mage/evolve/engine.py:ChainEvolver.evolve)

```
Input: query (str)
   │
Stage 0  DomainAnalyzer.analyze(query) → DomainAnalysis
   │
Stage 1  BenchmarkSynthesizer.synthesize(query, domain, evolve_cfg, mage_cfg)
         │  ├─ user_provided_tasks    (always survive)
         │  ├─ LLM synthesis          (primary, BENCHMARK_SYNTHESIZER_PROMPT)
         │  ├─ memory recall          (recall_benchmarks for same domain)
         │  └─ web discovery          (opt-in, via WebResearchAgent)
         │  └→ dedupe by input, renumber → BenchmarkSynthesisResult
         ▼
Stage 2  _build_initial_population
         │  k=population_size MAGEGenerator.generate(query) calls
         │  diversity via varying temperature [0.2, 0.5, 0.7, 0.9, 1.0]
         ▼
Stage 3  ChainGrader.grade(individual, benchmark, evolve_cfg)
         │  per task: structural_rubric + (LLM-judge OR real-execution + judge)
         │  per task → 0-1 score; overall = mean
         ▼
Stage 4  Loop until plateau or num_generations:
         │  pick operator by weight (mutate_aim, change_step_type, insert_step,
         │    remove_step, swap_steps, crossover, llm_repair)
         │  apply, validate (skip invalid), grade child
         │  selection: top-N by score (elitism)
         ▼
Stage 5  Persist: optionally save_benchmark to memory
         Return: EvolveResult{best_individual, benchmark, history, top_k}
```

### 5.2 Operators (mmar_mage/evolve/operators.py)

| Operator | Type | What it does |
|----------|------|--------------|
| `mutate_aim` | semantic | rewrites one LLM step's `aim` via heuristic perturbation |
| `change_step_type` | semantic | promotes `llm` ↔ `structured_output` (auto-fills schema) |
| `insert_step` | structural | adds a sanity-check LLM step before the final synthesis |
| `remove_step` | structural | drops a non-terminal step, rewires deps through it |
| `swap_steps` | structural | swaps two adjacent middle steps (only if independent) |
| `crossover` | structural | prefix(A) + suffix(B), bridge dependency added |
| `llm_repair` | LLM-driven | shows score + per-task failures, asks LLM for ONE targeted edit |

All operators preserve DAG validity via `_renumber()`; `to_carl_compat()` +
`validate_carl_json(strict_carl=True)` runs on every child to filter invalid
mutants.

### 5.3 Grading modes (mmar_mage/evolve/grader.py)

Two non-exclusive components:

**Structural rubric** (deterministic, fast):
- Reward chain length ≥ 2.
- Reward step contents containing rubric keywords.
- Reward `structured_output` step when reference is JSON-shaped.

**LLM-judge** (two flavors):
- *Forecast* (default): judge sees chain summary, predicts what it would emit.
- *Real-execution* (opt-in via `EvolveConfig.execution_model`): execute chain
  via stub-v1 flatten ([prompts.py:CHAIN_FLATTEN_EXECUTION_PROMPT]), feed actual
  response to judge. Mirrors `evaluation/chain_executor.py` so both pipelines
  are comparable.

`overall = 0.3 * structural + 0.7 * judge` when both enabled.

### 5.4 Memory-feedback closing the loop

`MemoryManager.save_benchmark(domain, tasks, best_chain, best_score, query)` —
persists synthesized benchmark + best evolved chain after each evolve run.
`MemoryManager.recall_benchmarks(domain, top_k)` — pulls past benchmarks,
filters by `kind == "mage_benchmark"` and domain match.

`BenchmarkSynthesizer` automatically pulls recalled tasks and folds them into
the synthesis (deduped by input). Two consequences:
- Cold start in domain D becomes "warm" after one prior run in D.
- Best chain from prior run is in the payload (not yet used as seed — see §7).

---

## 6. Research questions — the frontier

Frame all of these as: *"What changes when MAGE handles end-to-end constructor
+ benchmark synthesis + evolution, vs. the paper's pipeline that needs a manual
seed chain and a manually-curated benchmark?"*

### RQ1 — Does autonomous benchmark synthesis preserve evolutionary gains?

The paper's results assume 200-300 manually-curated tasks per benchmark. MAGE
synthesizes 8-50 tasks per query. **Hypothesis**: at small N, judge-based
grading is high-variance, so evolutionary gains will be smaller than +17pp
(IFBench validation) seen in the paper. Test: run evolve for 25 generations on
a query corresponding to a paper benchmark (e.g., a HoVer-flavor multi-hop
verification query); compare best evolved chain on the *real* HoVer test set
against (a) seed chain, (b) GigaEvo with manual seed, (c) random search.
**This is the first thing to measure.**

### RQ2 — Does LLM-based auto-seeding remove decomposition bias?

The paper's §Limitations explicitly notes that the manual seed chain encodes
prior beliefs about task decomposition, tool placement, semantic fields. MAGE's
deep mode (DomainAnalyzer → StepPlanner → DAGBuilder → StepDescriber) is exactly
the auto-seeding the paper says it doesn't have. **Hypothesis**: LLM-seeded
chains underperform manually-seeded chains at generation 0 but converge to
similar quality after evolution; the gap closes faster with more aggressive
structural operators (`insert_step`, `change_step_type`). Test: per-benchmark
comparison of seed quality and convergence curves; ablate operator weights.

### RQ3 — MAP-Elites quality-diversity vs. single-best selection

We use simple top-K elitism. The paper uses single-island MAP-Elites with
1D fitness as descriptor (no structural diversity pressure — the paper's own
limitation). **Hypothesis**: switching to multi-axis behavioral descriptors
(chain length × tool composition × prompt length) reduces premature
convergence on extraction-style tasks where multiple decompositions are valid.
Test: implement MAP-Elites variant; compare diversity at convergence + held-out
test scores. **This is the cleanest experimental contribution because it
addresses an explicit limitation in the paper.**

### RQ4 — Insight-aware mutation (lineage-conditioning)

GigaEvo's mutator is conditioned on (a) `INSIGHTS(parent, archive)` — a
1-LLM-call summary of past failures and successes; (b) `LINEAGE(parent,
co_elite)` for two co-elites — fitness deltas. Our `llm_repair` only sees the
parent's per-task failures. **Hypothesis**: lineage conditioning is the secret
sauce of the paper's mutator-LLM approach. Adding it should give us a step-
function improvement vs. our current operator selection. Test: implement
INSIGHTS + LINEAGE calls in `evolve/operators.py:llm_repair`; A/B against
current.

### RQ5 — Feedback-mode interaction on synthesized benchmarks

The paper's headline finding: optimal feedback mode is task-dependent. But all
their experiments use static benchmarks. **Hypothesis**: when the benchmark is
itself LLM-synthesized, `data-aware` feedback (which uses ground-truth `y` as a
critique anchor) is artificially advantaged because the same model that
synthesized `y` is likely to "approve" it. Test: ablation across feedback modes
on synthesized vs. manual benchmarks for the same query. If hypothesis holds,
this is a methodological warning the paper doesn't surface.

### RQ6 — End-to-end "type a task, get an agent" UX claim

The big-picture differentiator. **Hypothesis**: a single-command pipeline
(`mage evolve "..." --execution-model X`) produces agent chains that are
non-trivially better than a single LLM call on tasks the user didn't write a
benchmark for. Test: human-in-the-loop study — ask N users to type tasks, run
evolve, ask a separate LLM-judge (or the user) whether the evolved chain
output beats `gpt-5-direct(query)`.

### RQ7 — Per-step model assignment as evolved field

Paper's first listed limitation. Currently every step uses the same inference
model. Add `llm_config.model` as an evolvable field; `change_step_type` becomes
generalized to `change_step_config`. **Hypothesis**: small/cheap model on
parsing/extraction steps + large model on synthesis steps Pareto-dominates
fixed-model chains on cost/quality. Test: implement; report Pareto front on
HoVer or AIME.

---

## 7. Gaps from paper baseline — what we don't yet have

### 7.1 Architecture gaps

| Paper has | We have | Gap |
|-----------|---------|-----|
| MAP-Elites archive (1D descriptor) | Top-K elitism | No diversity pressure |
| INSIGHTS + LINEAGE conditioning | Plain LLM repair | Mutator is "blind" |
| Step-batched runner (vLLM-friendly) | Per-individual sequential grading | Inefficient at scale |
| Frozen inference LLM Qwen3-8B | Whatever's configured (gpt-4o default) | Not directly comparable to paper numbers |
| 4 explicit feedback modes in runner | Only post-hoc rubric in grader | Can't replicate paper's feedback-mode ablations |
| Static + Full evolution modes | Only Full mode (insert/remove allowed) | Can't run "matched-static" comparison |
| Reproducible randomness seeds throughout pipeline | rng_seed only in ChainEvolver | Initial population still varies across runs |

### 7.2 Things present-but-stubbed

- `evaluation/chain_executor.py` — explicit "stub v1" comment: flatten to one
  prompt rather than batched per-step execution. **TODO in source.**
- `MemoryManager.recall_benchmarks` — returns past tasks but ChainEvolver
  doesn't use the recalled `best_chain` as seed. Trivial extension.
- Only 7 operators implemented vs. paper's open-ended mutator-LLM that can
  emit any valid CARL JSON.

### 7.3 Datasets

`evaluation/tasks.py` has seed task registries for `hover`, `hotpotqa`,
`livebench_math`, plus per-category aggregators. **No HF/Kaggle/HoVer-full
loaders.** For paper-comparable experiments we need full splits, not the 1-5
seed tasks per benchmark currently shipped.

---

## 8. Concrete next-steps roadmap (research-oriented)

### Phase 1 — Baseline parity (~2 weeks)

Reproduce the paper's loop closely enough that any MAGE delta is methodological,
not implementation gap.

- [ ] `mmar_mage/evolve/archive.py` — MAP-Elites archive class (1D descriptor
      first, multi-axis as flag).
- [ ] `evolve/engine.py` — replace `_select` with MAP-Elites tournament-or-
      roulette (paper §3 algorithm 1, lines 6-8).
- [ ] `evolve/operators.py` — `insights_from_archive(archive) -> str` + per-
      lineage call. Plumb both into `llm_repair`.
- [ ] `evolve/runner.py` — step-batched runner that processes all N benchmark
      tasks at level k together before advancing. Mock-friendly, async.
- [ ] Static evolution mode flag (`EvolveConfig.evolution_mode = "static"|"full"`).
- [ ] Reproducible seeds end-to-end (DomainAnalyzer/Synthesizer → temp lists →
      operator RNG).
- [ ] Add full HoVer / IFBench / AIME loaders (HF datasets) to
      `evaluation/tasks.py`.

### Phase 2 — Novel contributions (~2-4 weeks)

- [ ] **Autonomous benchmark synthesis evaluation**: implement RQ1 protocol —
      run evolve on auto-synthesized benchmark, measure on official held-out
      test set of the corresponding paper benchmark.
- [ ] **End-to-end auto-seeding study (RQ2)**: ablate seed quality at gen 0;
      report convergence rate vs. manually-seeded GigaEvo.
- [ ] **Per-step model field (RQ7)**: schema extension + evolutionary operator
      that swaps step model. Pareto-front analysis on cost/quality.
- [ ] **Multi-axis MAP-Elites (RQ3)**: extend descriptor to (length × tool-
      count × cost). Diversity-vs-quality trade-off analysis.

### Phase 3 — Comparison + writeup (~2 weeks)

- [ ] Replicate Table 1 of the paper with our pipeline using their seed +
      their settings, then with our auto-seeded variant. Two columns side-by-
      side.
- [ ] Cost/latency analysis: paper's B=200; our autonomous variant's effective
      compute including benchmark synthesis.
- [ ] Failure-mode analysis: where does auto-seeding hurt (RQ2 negative
      cases)?
- [ ] Write up as either standalone paper ("Auto-seeded CARL chain evolution")
      or as expanded section in collaboration with the CARL paper authors.

---

## 9. Conventions and file paths to remember

### 9.1 Code style
- ruff line-length 120, rules `E,F,I,N,W,UP,B`. `uv run ruff check mmar_mage/ tests/`.
- mypy on Python 3.12+, `ignore_missing_imports = true`.
- pytest `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed.
- All modules: `from __future__ import annotations`.

### 9.2 Mocking conventions
- LLM mock target: `mmar_mage.llm.AsyncOpenAI` — never `openai.AsyncOpenAI`.
- gigaevo-memory: patch `sys.modules["gigaevo_memory"]` with a `MagicMock`.

### 9.3 Key files for new contributors
| Concern | File |
|---------|------|
| Generation pipeline orchestrator | mmar_mage/generator.py |
| All Pydantic models | mmar_mage/schemas.py |
| All LLM prompts | mmar_mage/prompts.py |
| MAGE → CARL JSON conversion | mmar_mage/carl_export.py |
| Validation | `validate_carl_json` in generator.py |
| Evolve entry point | mmar_mage/evolve/engine.py — `ChainEvolver` |
| Evolve schemas | mmar_mage/evolve/schemas.py |
| Evolve operators | mmar_mage/evolve/operators.py |
| Memory hooks | mmar_mage/memory.py |
| Stub chain executor (paper-comparable) | evaluation/chain_executor.py |
| Benchmark seed tasks | evaluation/tasks.py |
| Tests for evolve | tests/test_evolve.py + tests/test_evolve_operators.py |
| Tests for memory hooks | tests/test_memory_benchmarks.py |

### 9.4 Bringing up the dev env

```bash
uv sync --group dev          # creates .venv, installs deps + dev tools
uv run pytest tests/ -v      # ~350 tests, ~3s
uv run ruff check mmar_mage/ tests/
uv run mypy mmar_mage/        # 17 pre-existing errors, ignore them
make all                      # runs lint + typecheck + tests
```

For real evolve runs you also need:
```bash
export OPENAI_API_KEY=...                  # or MAGE_API_KEY
# optional: connect to running gigaevo-memory at http://localhost:8002
# optional: pip install /path/to/mmar-carl  # for ReasoningChain.from_json validation
```

---

## 10. Glossary

- **CARL** — Collaborative Agent Reasoning Library. Both the JSON format
  (this repo conforms) and the Python runtime (sibling package `mmar-carl`).
- **GigaEvo** — MAP-Elites evolutionary framework formalized in the paper;
  also a sibling package `gigaevo-memory` for cross-run memory (different
  deliverable, same name umbrella).
- **MAGE** — this repo. Mesh Architecture Generation Engine. Text-to-CARL.
- **Static / Full mode** (paper) — what the optimizer is allowed to mutate.
  Static = semantic fields only. Full = topology too. We default to Full.
- **Genome** — paper's term for the optimized object. In CARL, a genome is a
  JSON chain dict. The paper's central claim is that this genome representation
  is a better optimization target than raw prompt strings.
- **Insight-aware mutator** — paper's name for the mutator-LLM call that's
  conditioned on summarized past failures/successes via the INSIGHTS call.
- **Step-batched runner** — paper's runner that gathers all N samples at
  level k and dispatches them in one homogeneous LLM batch.
- **MAP-Elites** — Mouret & Clune 2015. Quality-diversity archive that bins
  candidates by behavioral descriptor and keeps the elite per bin.
- **Lineage conditioning** — the mutator sees not just the parent but the
  parent's path through the archive (fitness deltas of co-elites).

---

## 11. Headline numbers to cite from the paper

- HoVer (multi-hop fact verification): **chain evolution 68.7% vs. prompt 49.0%
  on held-out test, p<0.001** (data-aware feedback).
- IFBench (instruction following): **+17.0pp validation gain, p<0.001**
  (without-feedback).
- AIME (olympiad math): **chain 81.1% vs. prompt 57.8%, p<0.001** (data-aware,
  full-mode evolution).
- GSM8K: matched (98.5 vs. 95.0, p=0.048) — chain advantage saturates near
  ceiling.
- HotpotQA: prompt wins (82.0 vs. 78.0, p=0.011) — only benchmark where prompt
  evolution dominates; signal lives in instruction wording, not topology.
- Engine ablation: with OpenEvolve (no GigaEvo specifics) instead of GigaEvo,
  CARL chains still beat prompts on 4/5 benchmarks — gain comes from the
  representation, not the optimizer.

---

## 12. Agent-features integration (snapshot 2026-05-15)

After the original handoff, MAGE was synced with the `agent-features` branch
of `carl-experiments` to support the new CARL primitives and the wider
agentic-systems surface area. This is independent of the evolutionary loop —
both single-shot deep mode and `ChainEvolver.evolve` can produce chains that
use any of these.

### 12.1 New step types now supported

| Type | What it does | Nested step_config fields |
|------|-------------|-------------------------|
| `agent_skill` | Execute an AgentSkills-format skill | skill, task, execution_mode (llm/script/hybrid/subagent/llm_agent), input_mapping, output_key, timeout, llm_max_iterations, runtime, output_schema, trust_policy, extra_pip |
| `evaluation` | Inline quality gate over a previous step's output | target_step, method (rule/llm/python), criteria, fail_action, threshold |
| `parallel_sampling` | Sample N independent LLM responses | n, temperature, aggregation (vote/best_of/mean/concat), judge_prompt |
| `agent_handoff` | Delegate to a complete sub-chain | chain_path, input_mapping, output_key |
| `supervisor` | LLM router over N named sub-chains | routes, routing_prompt |
| `debate` | Round-robin multi-agent + judge synthesis | agents, rounds, judge_prompt |
| `human_input` | Pause for human (HITL) | prompt, timeout, default |
| `tool_discovery` | Register tools at runtime | sources |
| `mcp_resource` | Read a named MCP resource | uri, memory_key |

All MAGE-flat fields are mapped to nested `step_config` by
[`carl_export.py`](../mmar_mage/carl_export.py).

### 12.2 New modules

| Module | Purpose | LoC |
|--------|---------|-----|
| `mmar_mage/cost.py` | Dry-run cost & token estimation; `fits_in_budget` for evolve guard | ~280 |
| `mmar_mage/skills/registry.py` | `SkillRegistry` + built-in catalog (PDF, DOCX, PPTX, XLSX, mcp-builder, ...) | ~230 |
| `mmar_mage/skills/discovery.py` | `SkillDiscoveryAgent`: gate → registry rerank → optional web fallback | ~280 |
| Patches in `generator.py` | Stage 0f (skill discovery); `_build_default_replan_policy` | — |

### 12.3 New `MAGEConfig` flags

```python
# Skill discovery
enable_skill_discovery: bool = False
skill_discovery_max_recommendations: int = 3
enable_skill_web_discovery: bool = False

# RE-PLAN auto-policy
enable_replan_generation: bool = False
replan_max_replans: int = 3
replan_action_on_failure: str = "replan_from_checkpoint"
```

### 12.4 New `EvolveConfig` fields

```python
# Real-execution grading (already present)
execution_model: str | None = None

# Budget guard (NEW)
max_cost_usd: float | None = None
max_tokens_per_chain: int | None = None
pricing: dict[str, tuple[float, float]] = {}
```

When `max_cost_usd` or `max_tokens_per_chain` is set, `ChainEvolver` rejects
mutants exceeding the limit *before* grading — saves compute on obviously
bloated candidates.

### 12.5 Built-in skill catalog (default)

Includes the official Anthropic Skills set:

* `pdf` — PDF text/table extraction, form filling, OCR
* `docx` — Word documents
* `pptx` — PowerPoint
* `xlsx` — Spreadsheets, CSV/TSV cleaning
* `skill-creator` — Author and improve skills
* `mcp-builder` — Build MCP servers
* `webdev` — HTML/CSS/JS editing
* `canva` — Design via Canva API

Plus community: `brave-search`. Add your own with `SkillRegistry.add(...)`.

### 12.6 CLI flags added

```bash
mage generate "..." --skills              # turn on registry-based discovery
mage generate "..." --skills --skills-web # also allow web fallback
mage evolve   "..." --execution-model gpt-4o-mini  # real-execution grading
```

### 12.7 Research implications

The agent-features integration changes which research questions are
addressable. Updated to RQs 1-7 from §6:

* **RQ7 update — per-step model assignment is now feasible** because every
  CARL step admits an `llm_config.model` override and the runtime resolves it.
  The MAGE-side missing piece is an evolutionary operator that mutates
  `llm_config.model` — straightforward addition.

* **RQ8 (new) — Skill grounding as decomposition prior.** When
  `SkillDiscoveryAgent` is enabled, generated chains anchor decomposition
  around the surfaced skills' capability boundaries. Hypothesis: this
  reduces variance in chain structure and improves cost/quality trade-off
  on tasks where skills exist (documents, MCP), but hurts on pure-reasoning
  tasks where the gate misfires. Test: ablate `enable_skill_discovery` on
  HotpotQA (no-skill) vs an invoice-extraction benchmark (skill-heavy).

* **RQ9 (new) — Cost-bounded evolution.** With `max_cost_usd` enabled, the
  evolutionary search becomes a **constrained optimisation** rather than
  free-form. Hypothesis: under a tight budget, evolution converges to
  structurally simpler chains that use cheaper models on extraction steps
  and the expensive model only on synthesis. Test: sweep `max_cost_usd ∈
  {$0.005, $0.05, $0.50}` on a fixed query; report Pareto front of
  (score, cost) at convergence.

* **RQ10 (new) — RE-PLAN policy quality.** The current default policy is
  a coarse "any-step-failure → roll back to nearest checkpoint" rule. The
  paper doesn't explore RE-PLAN at all (`replan_policy` was added after
  Table 1's experiments). Hypothesis: even the dumb default policy
  measurably improves end-to-end success on chains with retrieval steps
  (HotpotQA, HoVer) because BM25 transient failures are the dominant
  error mode. Test: A/B on HotpotQA with vs without `enable_replan_generation`.

### 12.8 Test snapshot

`uv run pytest tests/ -v` — 400+ tests, all passing. The new test files:

| File | Tests | What |
|------|-------|------|
| `tests/test_cost.py` | 15 | dry-run estimation, agent-features step types, budget guard |
| `tests/test_skills.py` | 17 | registry, URI extraction, discovery gate, web fallback |
| `tests/test_replan_policy.py` | 5 | default-policy synthesis edge cases |
| `tests/test_carl_export.py` (extended) | +5 | nested step_config for agent-features types |

### 12.9 Where to look in carl-experiments

The reference implementation lives at
`/Users/pchelin/Documents/carl-experiments` on the `agent-features` branch.
Key files we mirror in MAGE:

* `src/mmar_carl/models/agent_skill.py` — definitive `AgentSkillStepConfig`
* `src/mmar_carl/skill_resolver.py` — URI resolver runtime (we don't mirror, just produce URIs)
* `src/mmar_carl/cost.py` — original `estimate_chain_cost` (ours mirrors logic for dict-form chains)
* `src/mmar_carl/models/replan.py` — `ReplanPolicy`, `ReplanAction`, etc.
* `src/mmar_carl/models/config.py` — all per-type config classes (use as ground truth)

When a new step type lands in `carl-experiments`, the integration checklist
in MAGE is:

1. Add type to `CARL_VALID_STEP_TYPES` in `mmar_mage/schemas.py`
2. Add flat fields to `CARLStepSchema`
3. Add mapping table + branch to `carl_export.py:_convert_step`
4. Add type-specific instruction to `STEP_DESCRIBER_PROMPT`
5. Add cost model branch in `cost.py:_calls_llm_for_step`
6. Add a `test_carl_export.py` nesting test + cost test
7. Optionally — promotion path in `evolve/operators.py:change_step_type`

---

## 13. What this document does NOT cover

- Concrete implementation of MAP-Elites archive or insight-aware mutator —
  see RQ3/RQ4, Phase-1 roadmap.
- The full paper bibliography — see `References` section in the paper.
- mmar-carl runtime internals — see that package's docs.
- gigaevo-memory schema/API — see `mmar_mage/memory.py` for our usage and the
  gigaevo-memory package for full API.
- Production deployment / scaling concerns — none of the present code is
  production-grade; everything is research-quality.

---

*Last verified: 350/350 tests passing, ruff All checks passed. Code surface
~9000 LoC main package + ~2400 LoC evaluation/tests. Snapshot taken on the
`claude/blissful-shannon-9cde9f` worktree against `origin/initial-version`.*
