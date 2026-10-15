# CLAUDE.md — Project Context for Claude Code

## Project Overview

**MAGE** (Mesh Architecture Generation Engine) converts natural-language queries into structured
[CARL](https://github.com/Glazkoff/carl) reasoning chains (JSON). Part of the MMAR ecosystem
alongside `mmar-carl` (chain execution) and `gigaevo-memory` (memory service).

- **Package**: `mmar-mage` v0.1.0
- **Python**: ≥ 3.12
- **License**: MIT
- **Author**: glazkov (glazkov@airi.net)

## Quick Commands

```bash
# Setup
uv sync --group dev          # install all deps (creates .venv automatically)

# Quality
uv run pytest tests/ -v      # ~320 tests
uv run ruff check mmar_mage/ tests/ examples/  # lint
uv run ruff format mmar_mage/ tests/           # auto-format
uv run mypy mmar_mage/       # type-check

# Or via Makefile
make install    # uv sync --group dev
make test       # pytest
make lint       # ruff check
make all        # lint + typecheck + tests
make examples   # run all example scripts
```

## Architecture

```
User query (str)
    │
    ▼
┌────────────────────────────────────────────────┐
│             MAGEGenerator                       │
│                                                 │
│  mode=fast ──► single LLM call (with retries)   │
│  mode=deep ──► agentic pipeline:                │
│    0a. MemoryResearchAgent     → digest (opt)   │
│    0b. WebResearchAgent        → digest (opt)   │
│    0c. CapabilityLookupAgent   → tools (opt)    │
│    0d. TemplateAssemblerAgent  → skel (opt)     │
│    0e. FeedbackRecallAgent     → digest (opt)   │
│    0f. SkillDiscoveryAgent     → URIs   (opt)   │
│    1.  DomainAnalyzer  → DomainAnalysis         │
│    2.  StepPlanner     → StepPlan               │
│    3.  DAGBuilder | ToTPlanner → DAGStructure   │
│    4.  StepDescriber | MCTSConstructor          │
│    4.5 StepCriticAgent        → revise  (opt)   │
│    5.  Serialization → CARLChainSchema          │
│         + default ReplanPolicy attach (opt)     │
│    5.5 ChainVerifier          → fix     (opt)   │
│    5.6 SelfRefiner            → refine  (opt)   │
│                                                 │
│  + validate_carl_json() — MAGE-internal types   │
│  + to_carl_compat() — eval/aggregator → llm,    │
│    flat fields → nested step_config             │
│  + validate_carl_json(strict_carl=True)         │
│  + ReasoningChain.from_json() (if installed)    │
│  + MemoryManager.save_chain() (optional)        │
└────────────────────────────────────────────────┘

agent-features integration (carl-experiments @ agent-features):
  • Step types: llm/tool/mcp/memory/transform/conditional/structured_output
    + agent_skill, evaluation, agent_handoff, parallel_sampling, supervisor,
    debate, human_input, tool_discovery, mcp_resource
  • Skill discovery via SkillDiscoveryAgent + built-in SkillRegistry (PDF,
    DOCX, PPTX, XLSX, mcp-builder, ...) + optional web fallback
  • Cost estimation via mmar_mage.cost.estimate_chain_cost (dry-run)
  • Evolve loop has budget guard (max_cost_usd / max_tokens_per_chain)
  • RE-PLAN: per-step checkpoint hints + chain-level default policy synth
    │
    ▼
MAGEResult { chain_json, chain_dict, memory_key, mode, metadata }
```

### Edit mode (NL-driven chain editing)

Besides *generating*, MAGE can **edit an existing chain**:
`MAGEGenerator.edit(instruction, *, entity_id=None, chain=None, save=False)` →
`MAGEEditResult`. It resolves the target (`entity_id` → `MemoryManager.load_chain`;
else `search_similar`, disambiguating on ties → `needs_disambiguation` +
`candidates`), asks `ChainEditPlanner` (`CHAIN_EDIT_PROMPT`) for a **minimal** edit
plan, applies it via `mmar_mage.chain_edit.apply_edit_plan` — which reuses
`evolve.operators._apply_edit` for the 5 structural ops and adds 3 lightweight ones
(`edit_field` / `set_dependencies` / `set_chain_field`) — re-validates with
`validate_or_repair`, and (with `save=True`) versions it via
`MemoryManager.save_chain(entity_id=…)`. CLI: `mage edit`. Convenience:
`edit_chain(...)`. Note: `chain_edit` lazy-imports `evolve.operators` inside
`apply_edit` to avoid the `evolve → engine → generator → chain_edit` import cycle
(same trick as `chain_repair`'s lazy `from .generator import validate_carl_json`).

### Key Design Principles

1. **Every deep-mode stage is independently toggleable** via `MAGEConfig` booleans
   (`enable_domain_analysis`, `enable_step_planning`, `enable_dag_optimization`,
   `enable_step_descriptions`, `enable_memory_research`, `enable_web_research`).
   Disabled stages use deterministic fallbacks.

2. **Memory research and web research are COMPLETELY SEPARATE features**.
   They are independent stages (0a and 0b), can be enabled/disabled individually,
   and feed their digests into StepPlanner as optional context.

3. **Structured output everywhere** — all LLM calls use `response_format=json_object`,
   validated against Pydantic models before propagation.

4. **Graceful degradation** — memory and web research failures are non-fatal (`try/except`
   with logging, generation continues without the research results).

5. **Cold-start multi-sampling** — when `memory_research` is enabled but finds no hits
   (`was_cold_start=True`), generates `cold_start_candidates` plan variants and uses
   `PLAN_SCORER_PROMPT` to pick the best.

## File Structure

```
mmar_mage/
  __init__.py              # public API: MAGEGenerator, MAGEConfig, MAGEResult, etc.
  generator.py             # MAGEGenerator orchestrator (fast + deep modes)
  schemas.py               # Pydantic models (MAGEConfig, MAGEResult, DomainAnalysis, etc.)
  llm.py                   # LLMClient — async wrapper around AsyncOpenAI
  prompts.py               # ALL LLM prompt templates (system messages)
  memory.py                # MemoryManager — gigaevo-memory wrapper (incl. save/recall_benchmark)
  cli.py                   # Typer CLI: `mage generate`, `mage evolve`, `mage version`
  exceptions.py            # MAGEError, MAGEValidationError, MAGELLMError, MAGEMemoryError
  carl_export.py           # MAGE-format → CARL-loadable JSON (eval/aggregator → llm, nested step_config)
  cost.py                  # estimate_chain_cost / fits_in_budget (dry-run cost)
  agents/                  # see Agents section
  skills/
    __init__.py
    registry.py            # SkillRegistry + built-in catalog (PDF, DOCX, PPTX, XLSX, ...)
    discovery.py           # SkillDiscoveryAgent (gate → rerank → web fallback)
  evolve/
    __init__.py
    schemas.py             # EvolveConfig, EvolveResult, Individual, ChainScore
    benchmark_synthesizer.py  # Stage 1: synthesize benchmark per query
    grader.py              # Stage 3: structural rubric + LLM-judge (+ real-execution)
    operators.py           # Mutation/crossover/LLM-repair operators on chain_dict
    engine.py              # ChainEvolver — main evolutionary loop

configs/                   # Preset TOML configurations
  fast.toml                # Fast mode, no research
  deep_local.toml          # Deep mode, no research, no memory
  deep_research.toml       # Deep mode + memory research + web research
  deep_openrouter.toml     # Deep mode via OpenRouter
  deep_local_airi.toml     # Deep mode via AIRI inference server

examples/                  # Runnable example scripts
  README.md
  fast_mode/               # run.py + config.toml → result.json
  deep_mode_local/
  deep_mode_web_research/
  deep_mode_memory_research/
  deep_mode_full/

tests/
  conftest.py              # Shared fixtures, mock LLM responses, sample data
  test_schemas.py          # MAGEConfig, MAGEResult, Pydantic model tests
  test_fast_mode.py        # Fast mode generation tests
  test_deep_mode.py        # Deep mode pipeline tests
  test_generator.py        # Generator validation, edge cases
  test_memory.py           # MemoryManager tests
  test_cli.py              # CLI command tests
  test_config.py           # Config loading (TOML, env, factories)
  test_memory_research.py  # MemoryResearchAgent tests
  test_web_research.py     # WebResearchAgent tests

docs/ARCHITECTURE.md       # Detailed architecture documentation
```

## Code Style & Conventions

- **Formatter/Linter**: ruff, line-length=120, ignore E501
- **Ruff rules**: E, F, I, N, W, UP, B
- **Type checking**: mypy, python 3.12, ignore_missing_imports=true
- **Async**: all LLM calls are async (`AsyncOpenAI`), pytest uses `asyncio_mode = "auto"`
- **Docstrings**: NumPy-style with Parameters/Returns/Raises sections
- **CARL step_type**: MUST be lowercase (`"llm"`, not `"LLM"`) — this is enforced in
  `PlannedStep`, `DAGStep`, and `CARLStepSchema` defaults
- **Imports**: use `from __future__ import annotations` everywhere
- **Dependencies**: Pydantic v2 (`model_validate`, `model_dump`), not v1 API

## Testing

- **94 tests**, **87% coverage**, all passing
- **Mock target**: `mmar_mage.llm.AsyncOpenAI` — always mock at this path
- **Pattern**: `conftest.py` defines `mock_openai_fast` and `mock_openai_deep` fixtures
  that return `AsyncMock` for `chat.completions.create` with side_effect lists
- **All LLM and HTTP calls must be mocked** — no real API calls in tests
- **Coverage requirement**: ≥ 80%

### Test fixture structure

```python
# conftest.py provides:
# - SAMPLE_CARL_CHAIN — valid 3-step CARL chain dict
# - SAMPLE_DOMAIN_ANALYSIS — finance domain analysis
# - SAMPLE_STEP_PLAN — 3-step plan
# - SAMPLE_DAG — funnel pattern DAG
# - SAMPLE_DESCRIBED_STEPS — full CARL step descriptions
# - config() — fast mode test config (memory disabled)
# - deep_config() — deep mode test config (memory disabled)
# - mock_openai_fast — patches AsyncOpenAI for fast mode
# - mock_openai_deep — patches AsyncOpenAI with 4-response side_effect for deep mode
```

### Running tests

```bash
uv run pytest tests/ -v                  # all tests
uv run pytest tests/test_fast_mode.py -v # specific file
uv run pytest -k "test_deep" -v          # by keyword
uv run pytest --cov=mmar_mage --cov-report=term-missing  # with coverage
```

## Configuration System

`MAGEConfig` (Pydantic BaseModel) supports multiple loading methods:

```python
# From TOML
config = MAGEConfig.from_toml("configs/deep_local.toml")

# From environment variables (MAGE_* prefix)
config = MAGEConfig.from_env()

# Factory: OpenRouter
config = MAGEConfig.for_openrouter(api_key="sk-or-...", model="anthropic/claude-3.5-sonnet")

# Factory: Local server (vLLM, Ollama, etc.)
config = MAGEConfig.for_local(base_url="http://localhost:8000/v1", model="__auto__")
```

### Key config fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"fast"` \| `"deep"` | `"deep"` | Generation mode |
| `model` | str | `"gpt-4o"` | Model ID or `"__auto__"` for detection |
| `provider` | `"openai"` \| `"openrouter"` \| `"local"` \| `"custom"` | `"openai"` | LLM provider |
| `base_url` | str \| None | None | Custom API endpoint |
| `enable_memory_research` | bool | False | Stage 0a: memory recall |
| `enable_web_research` | bool | False | Stage 0b: web search |
| `web_search_provider` | str | `"tavily"` | `tavily` \| `serpapi` \| `brave` |
| `web_search_api_key` | str \| None | None | API key for web search provider |
| `cold_start_candidates` | int | 1 | Plan candidates for cold-start scoring |
| `stage_llm_overrides` | dict | {} | Per-stage model/temperature overrides |

## LLM Client

`LLMClient` (in `llm.py`) wraps `AsyncOpenAI` with:

- **Per-stage overrides**: `get_stage_config(stage)` merges `stage_llm_overrides`
- **Auto-detection**: `resolve_model()` calls `/v1/models` when model is `"__auto__"`
- **Retries**: configurable via `max_retries`
- **Provider support**: OpenAI, OpenRouter (extra headers), local (vLLM/Ollama), custom

Key methods:
```python
await client.generate(system_prompt, user_prompt, json_mode=True, stage="...")  → str
await client.generate_json(system_prompt, user_prompt, stage="...")  → dict
await client.generate_structured(system_prompt, user_prompt, SomeModel, stage="...")  → BaseModel
```

## Prompts (in `prompts.py`)

All prompts are constants. Each includes few-shot examples and format specifications:

| Constant | Used by | Purpose |
|----------|---------|---------|
| `FAST_MODE_SYSTEM_PROMPT` | `generator._generate_fast` | Single-shot chain generation (3 examples) |
| `DOMAIN_ANALYZER_PROMPT` | `DomainAnalyzer` | Domain/complexity analysis (3 examples) |
| `STEP_PLANNER_PROMPT` | `StepPlanner` | Step sequence planning (2 examples) |
| `DAG_BUILDER_PROMPT` | `DAGBuilder` | Dependency graph construction (1 diamond example) |
| `STEP_DESCRIBER_PROMPT` | `StepDescriber` | CARL step detail generation |
| `MEMORY_SUB_QUERY_GENERATOR_PROMPT` | `MemoryResearchAgent` | Sub-query generation for memory |
| `MEMORY_PURIFIER_PROMPT` | `MemoryResearchAgent` | Memory digest with anti-hallucination |
| `WEB_SUB_QUERY_GENERATOR_PROMPT` | `WebResearchAgent` | Sub-query generation for web search |
| `WEB_DIGEST_PROMPT` | `WebResearchAgent` | Web digest with anti-hallucination |
| `PLAN_SCORER_PROMPT` | `generator._score_plans` | Cold-start plan scoring (weighted criteria) |

## CARL JSON Format

Generated chains must conform to:

```json
{
  "name": "...",
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
  "steps": [
    {
      "step_type": "llm",
      "number": 1,
      "title": "...",
      "aim": "...",
      "reasoning_questions": "...",
      "step_context_queries": ["...", "..."],
      "stage_action": "...",
      "example_reasoning": "...",
      "dependencies": [],
      "retry_max": 3,
      "llm_config": null
    }
  ]
}
```

**Validation** (in `generator.py`):
1. `validate_carl_json()` — structure, step numbers, DAG acyclicity
2. `ReasoningChain.from_json()` — full CARL library validation (if `mmar-carl` installed)

## Dependencies

### Runtime
- `pydantic>=2.0.0` — schemas and validation
- `openai>=1.0.0` — AsyncOpenAI client
- `httpx>=0.27.0` — HTTP client (SSL, timeouts)
- `typer>=0.15.0` — CLI framework
- `rich>=13.0.0` — CLI output formatting
- `pyyaml>=6.0` — prompt YAML loader
- `mmar-carl>=0.2` — **required** (no longer optional): CARL runtime types
  (`ReasoningChain`, `ChainBuilder`, `SkillLoader`, `AgentSkillSource`) back
  validation, the code-gen round-trip, skill loading, and the strict-CARL export.
  Resolves from PyPI (0.3.0 published); for cross-repo dev against the local
  `../carl-experiments` tree use a temporary `uv add --editable`.

### Optional
- `gigaevo-client>=0.3.0` (extra `memory`) — memory-service SDK, **renamed from
  `gigaevo-memory`**. `MemoryManager` prefers `gigaevo_client`, falling back to the
  legacy `gigaevo_memory` shim if only that is present. Resolves from PyPI; the
  source lives in the standalone `../gigaevo-client` repo (moved out of the
  gigaevo-memory monorepo 2026-06-10).

### Dev
- `ruff>=0.4` — linter/formatter
- `mypy>=1.10` — type checker
- `pytest>=8.2` — test runner
- `pytest-asyncio>=1.0.0` — async test support
- `pytest-cov>=5.0.0` — coverage
- `pytest-mock>=3.14.0` — mocker fixture

## Git Workflow

- **Branches**: `main` → `dev` → `initial-version` (active)
- **PR #1**: `initial-version` → `dev` (open, not merged)
- Do NOT merge PRs without explicit request
- Commit messages: conventional commits (`feat:`, `fix:`, `build:`, `test:`, `docs:`)

## Common Pitfalls

1. **step_type must be lowercase** — `"llm"` not `"LLM"`. The CARL library rejects uppercase.
2. **Mock target path** — always `mmar_mage.llm.AsyncOpenAI`, not `openai.AsyncOpenAI`.
3. **Memory is optional** — all memory code must handle the memory SDK
   (`gigaevo-client`, formerly `gigaevo-memory`) not being installed. `mmar-carl`, by
   contrast, is now a **required** runtime dependency.
4. **Web research needs `web_search_api_key` in config** — environment variables alone
   are not auto-read by `MAGEConfig.from_toml()` for web search keys.
5. **Config TOML section** — all fields live under `[mage]`.
6. **Async tests** — `asyncio_mode = "auto"` in pyproject.toml, no need for `@pytest.mark.asyncio`.
7. **`from __future__ import annotations`** — required in every module for `X | None` syntax.
8. **Two step-type sets** —
   - `VALID_STEP_TYPES`: what MAGE-internal logic emits, includes `eval` and `aggregator`.
   - `CARL_VALID_STEP_TYPES`: what `mmar_carl.ReasoningChain.from_json()` actually accepts.
   `eval`/`aggregator` are MAGE-only labels translated to `llm` during CARL-compat export
   in `mmar_mage/carl_export.py`. Never put `eval`/`aggregator` into the *final* serialized
   chain JSON — only into intermediate MAGE structures.
9. **Non-LLM step config is nested under `step_config`** in the final CARL JSON, not flat.
   `CARLStepSchema` keeps flat fields (`tool_name`, `mcp_server_name`, etc.) for ergonomics
   during generation; `to_carl_compat()` re-nests them before serialization.

10. **agent-features step types require nested config** just like the stable types.
    Every type from {agent_skill, evaluation, parallel_sampling, agent_handoff,
    supervisor, debate, human_input, tool_discovery, mcp_resource} has a dedicated
    mapping table in `mmar_mage/carl_export.py`. To add a new type: update both
    `CARL_VALID_STEP_TYPES` and the per-type field map.

11. **Skill discovery is OFF by default.** Set `enable_skill_discovery=True` in
    `MAGEConfig` to surface packaged AgentSkills via the built-in registry.
    `enable_skill_web_discovery=True` adds the web-search fallback — needs
    `web_search_api_key` set.

12. **RE-PLAN auto-policy needs at least one `checkpoint=True` step** in the
    generated chain. If the StepDescriber doesn't mark any (e.g. the chain is
    short / trivial), `_build_default_replan_policy` returns `None` and no
    policy is attached — that's intentional.
