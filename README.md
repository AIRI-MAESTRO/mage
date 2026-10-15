# MMAR MAGE — Mesh Architecture Generation Engine

[![PyPI](https://img.shields.io/pypi/v/mmar-mage?logo=pypi&logoColor=white)](https://pypi.org/project/mmar-mage/)

**MAGE** generates initial [CARL](https://pypi.org/project/mmar-carl/) reasoning chains from plain-text user queries.

```
text query  →  [MAGE]  →  CARL JSON chain  +  write to gigaevo-memory
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/Glazkoff/carl-mage.git
cd carl-mage
uv sync --group dev        # install project + dev dependencies
```

Optional integrations:

```bash
uv sync --group dev --all-extras   # include gigaevo-memory + mmar-carl
```

Alternatively, via pip:

```bash
pip install mmar-mage              # core only
pip install mmar-mage[memory]      # + gigaevo-memory
pip install mmar-mage[carl]        # + mmar-carl validation
pip install mmar-mage[all]         # everything
```

## Quick Start

### Python API

```python
import asyncio
from mmar_mage import generate_chain, MAGEConfig

# Fast mode — single LLM call
result = asyncio.run(generate_chain(
    "Analyze customer feedback and identify top-3 issues",
    MAGEConfig(mode="fast", api_key="sk-..."),
))
print(result.chain_json)

# Deep mode (default) — agentic pipeline
result = asyncio.run(generate_chain(
    "Analyze customer feedback and identify top-3 issues",
    MAGEConfig(api_key="sk-..."),
))
print(f"Domain: {result.metadata.domain}")
print(f"Steps: {result.metadata.num_steps}")
```

### CLI

```bash
# Fast mode
mage generate "Analyze a financial report" --mode fast --api-key sk-...

# Deep mode (default)
mage generate "Analyze a financial report" --api-key sk-...

# Save to file
mage generate "Classify reviews" --output chain.json --no-memory

# Version
mage version
```

### Editing an existing chain (NL-driven)

Take a *saved* chain and change it with a natural-language instruction — MAGE
plans a **minimal, targeted** edit (it preserves everything you didn't mention),
applies it, re-validates (one repair pass if needed), and can persist the result
as a **new version** under the same `entity_id` (history preserved).

```python
import asyncio
from mmar_mage import edit_chain, MAGEConfig

result = asyncio.run(edit_chain(
    "Rename to 'Invoice QA' and add a final validation step",
    entity_id="chain-abc123",
    save=True,                       # persist as a NEW VERSION (same entity_id)
    config=MAGEConfig(api_key="sk-..."),
))
print(result.summary)                # plain-language summary of the change
for e in result.edits:               # the applied edit ops
    print(e.op, e.target_step_number, e.rationale)
```

```bash
# By id (saved chain) → persist a new version
mage edit "make step 2 cheaper and add a validation step" --id chain-abc123 --save

# Offline: edit a local chain JSON, write the result out (no memory)
mage edit "rename to 'Invoice QA'" --chain chain.json -o edited.json

# No id → MAGE resolves the chain from your words (disambiguates on ties)
mage edit "tweak the PDF summariser to also extract tables"
```

Edit ops: `edit_field` / `rewrite_step` / `change_step_type` / `insert_step` /
`remove_step` / `swap_steps` / `set_dependencies` / `set_chain_field`. The first
five reuse the evolutionary `_apply_edit` primitive; the planner emits a plan,
`mmar_mage.chain_edit.apply_edit_plan` applies it.

## Modes

### Fast Mode

Single LLM call with structured output. The system prompt includes the full
CARL format specification and few-shot examples. Returns a CARL JSON chain
directly.

- **1 LLM call**
- Best for simple tasks or rapid prototyping
- Less precise step decomposition

### Deep Mode (default)

Agentic multi-stage pipeline inspired by the Claude Code tool-use loop:

```
Stage 0a:  MemoryResearchAgent (optional) — recalls past experience from memory
Stage 0b:  WebResearchAgent (optional) — searches the web for domain context
Stage 0c:  CapabilityLookupAgent (optional) — unifies the available tools / MCP /
           skills into one CapabilityContext for the planner + describer
Stage 1:   DomainAnalyzer — detects domain, task type, complexity, key concepts
Stage 2:   StepPlanner — plans reasoning steps (uses memory / web / capability digests)
Stage 2.5: TopologySelector (optional) — picks the reasoning topology
           (linear / funnel / diamond / tree-of-thought) before edges are drawn
Stage 3:   DAGBuilder — determines dependency edges (funnel / pipeline / diamond)
Stage 4:   StepDescriber — generates detailed CARL step descriptions
Stage 4.5: StepCritic / ChainVerifier / SelfRefiner (optional) — quality gates
           that revise the chain before it ships
```

> The numbered stages are the core; the `0*` research stages and the `2.5` / `4.5`
> quality stages are optional and individually toggleable (see below) — the full
> deep pipeline is six-plus agents, not five.

Each sub-stage can be individually toggled via `MAGEConfig`:

```python
config = MAGEConfig(
    mode="deep",
    enable_domain_analysis=True,
    enable_step_planning=True,
    enable_dag_optimization=False,  # skip DAG optimization
    enable_step_descriptions=True,
    enable_memory_research=True,    # enable memory research
    enable_web_research=False,      # disable web research
)
```

- **4–7 LLM calls** (depends on optional stages)
- Better decomposition and DAG structure
- Progress callbacks for real-time status

## Deep Research Mode

### Memory Research (Stage 0a)

When `enable_memory_research=True`, MAGE performs iterative multi-query recall
from gigaevo-memory (CoRAG-style):

1. Generates targeted sub-queries from domain analysis
2. Searches memory with BM25 for each sub-query
3. Compresses raw hits into a "memory digest" via an LLM call
4. Passes the digest to the StepPlanner for structural guidance

If no relevant hits are found (cold start) and `cold_start_candidates > 1`,
MAGE generates multiple candidate plans and selects the best one via an LLM
scoring call.

```python
config = MAGEConfig(
    enable_memory_research=True,
    memory_recall_top_k=5,
    memory_relevance_threshold=0.4,
    memory_research_sub_queries=3,
    cold_start_candidates=3,
)
```

### Web Research (Stage 0b)

When `enable_web_research=True`, MAGE searches the internet for domain context:

1. Generates web search sub-queries from domain analysis
2. Calls the configured search provider (Tavily, SerpAPI, or Brave)
3. Summarizes results into a "web digest" via an LLM call
4. Passes the digest to the StepPlanner

```python
config = MAGEConfig(
    enable_web_research=True,
    web_search_provider="tavily",  # or "serpapi", "brave"
    web_search_api_key="your-api-key",
    web_research_max_results=5,
    web_research_sub_queries=2,
)
```

## Configuration

### Config File (TOML)

```python
from mmar_mage import MAGEConfig

# Load from TOML
config = MAGEConfig.from_toml("configs/deep_research.toml")

# Load from environment variables
config = MAGEConfig.from_env()
```

See `.env.example` and the `configs/` directory for example configurations:
- `configs/fast.toml` — minimal, single LLM call
- `configs/deep_local.toml` — deep mode with Ollama
- `configs/deep_research.toml` — deep mode + memory + web research
- `configs/deep_openrouter.toml` — deep mode via OpenRouter
- `configs/deep_local_airi.toml` — local AIRI inference server

### Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"deep"` | `"fast"` or `"deep"` |
| `model` | `"gpt-4o"` | Any OpenAI-compatible model (use `"__auto__"` for local) |
| `temperature` | `0.2` | Sampling temperature |
| `api_key` | `None` | Falls back to `MAGE_API_KEY` / `OPENAI_API_KEY` |
| `base_url` | `None` | Override for OpenRouter, Ollama, vLLM, etc. |
| `provider` | `"openai"` | `"openai"`, `"openrouter"`, `"local"`, `"custom"` |
| `ssl_verify` | `True` | Verify SSL certificates |
| `http_timeout` | `60.0` | HTTP request timeout (seconds) |
| `enable_memory` | `True` | Save chain to gigaevo-memory |
| `memory_base_url` | `localhost:8002` | gigaevo-memory server URL |
| `enable_memory_research` | `False` | Enable memory research stage |
| `enable_web_research` | `False` | Enable web research stage |
| `web_search_provider` | `"tavily"` | Web search provider |
| `web_search_api_key` | `None` | Web search API key |
| `cold_start_candidates` | `1` | Plan candidates for cold-start |
| `stage_llm_overrides` | `{}` | Per-stage model/temperature overrides |
| `default_max_workers` | `2` | Default parallel workers in chains |
| `max_steps` | `10` | Max steps the generator may produce |

## Connecting to Providers

### OpenAI (default)

```python
config = MAGEConfig(api_key="sk-...")
```

### OpenRouter

```python
config = MAGEConfig.for_openrouter(
    api_key="or-...",
    model="anthropic/claude-3.5-sonnet",
)
```

### Local (Ollama, vLLM, AIRI)

```python
config = MAGEConfig.for_local(
    base_url="http://localhost:11434/v1",
    model="__auto__",  # auto-detect first available model
)

# AIRI with self-signed cert
config = MAGEConfig.for_local(
    base_url="https://inference.airi.net:46783/v1",
    api_key="your-key",
    ssl_verify=False,
)
```

### Per-stage LLM Overrides

Use a smaller/faster model for certain stages:

```python
config = MAGEConfig(
    model="gpt-4o",
    stage_llm_overrides={
        "domain_analysis": {"model": "gpt-4o-mini", "temperature": 0.0},
        "step_planning": {"model": "gpt-4o", "temperature": 0.2},
    },
)
```

## Output Contract

```python
class MAGEResult:
    chain_json: str              # valid CARL JSON
    chain_dict: dict             # same chain as a dict
    memory_key: str | None       # gigaevo-memory entity ID
    mode: "fast" | "deep"
    metadata: MAGEMetadata       # domain, num_steps, timing, model,
                                 # memory_hits_used, web_results_used,
                                 # was_cold_start
```

## Memory Integration

When `enable_memory=True` and `gigaevo-memory` is installed, MAGE
automatically saves each generated chain:

```python
from gigaevo_memory import MemoryClient

with MemoryClient(base_url="http://localhost:8002") as client:
    chain = client.get_chain(result.memory_key)
```

Memory failures are handled gracefully — generation always succeeds.

## MAGE in CARE

MAGE is the chain-generation engine inside
[CARE](https://github.com/Glazkoff/care) — Anthropic-style authoring UI
for CARL chains. The CARE integration is contract-stable and feature-
complete: every M0-M3 piece on the MAGE side is shipped.

### Quick start for CARE integrators

Pick a profile preset that matches your deployment shape:

```python
from mmar_mage import MAGEConfig, MAGEGenerator

# Production OpenAI — streaming + library-ready naming, memory off
cfg = MAGEConfig.from_toml("configs/care_default.toml")

# Local Ollama / vLLM — json_mode_fallback on, extra retries
cfg = MAGEConfig.from_toml("configs/care_local.toml")

# Power user — full memory + web research + ecosystem persistence
cfg = MAGEConfig.from_toml("configs/care_research.toml")

result = await MAGEGenerator(cfg).generate(
    "Analyze a quarterly revenue report",
    context_files=[{"path": "/tmp/q3.pdf", "sha256": "...", "size_bytes": 4096}],
    cancel=esc_event,                  # CARE's Esc-key abort
    capabilities=preinstalled_skills,  # bypass internal lookup
)
```

### What CARE gets out of the box

- **Pre-flight cost estimate** (`on_cost_estimate`) — shows
  "this will cost ~$X — continue?" before any LLM call fires.
- **TUI streaming** (`on_llm_chunk` + `on_stage_progress`) — token-
  level streaming on `step_describing`, per-step progress events.
- **Esc cancellation** (`cancel: asyncio.Event`) — stops within one
  LLM call worth of latency; `MAGECancelledError.stage` identifies
  the abort point.
- **Library-ready naming** (`MAGEMetadata.suggested_*`) — punchy
  display name + description + tags pre-filling
  `SaveAgentModal`.
- **Replay bundle** (`MAGEResult.to_care_dict` /
  `from_care_dict`) — JSON-serializable artifact CARE persists as
  one `agent_run` row.
- **Resume from any stage** (`MAGEResult.replay_from`) — re-run
  from `domain_analysis | step_planning | dag_building |
  step_describing` reusing cached `intermediate_artifacts`;
  `partial_state=` lets users edit the plan / DAG before
  regenerating downstream.
- **Per-stage entrypoints** (`analyze_domain`, `plan_steps`,
  `build_dag`, `describe_steps`, `critique_steps`, `verify_chain`,
  `refine`) — stateless wrappers for "regenerate stage X only" UX.
- **Capability injection** (`capabilities=CapabilityContext(...)`) —
  inject skills CARE has installed locally; bypasses the internal
  Memory + registry lookup; the synthetic
  `capability_lookup_injected` stage label keeps telemetry distinct.
- **Ecosystem write-back** (opt-in flags) — `agent_skill` provenance,
  reusable LLM step templates, failure cards, intermediate audit
  cards, `evolution_meta` fitness priors. All saves are best-effort
  and surface their entity IDs in `MAGEResult.memory_keys`.
- **Customisable prompts** — every prompt lives in
  `mmar_mage/prompts_data/prompts.yaml`. Point
  `MAGE_PROMPTS_PATH` at a custom YAML to override without
  forking. Loader is `mmar_mage.prompt_loader.load_prompts`.
- **Chain validation CLI** — `mage validate ./chain.json` runs
  `validate_carl_json`, optional `ReasoningChain.from_json`, and
  an optional soft capability-existence check
  (`--check-capabilities`). Exit 0 on pass, 1 on fail.

### Read next

- **[`docs/CARE_INTEGRATION.md`](docs/CARE_INTEGRATION.md)** — the
  full runtime contract: every callback signature, the
  `CapabilityContext` shape, `MAGEResult` field-by-field, the
  `CareChainMetadata` block CARE reads back on "Re-run from library",
  error taxonomy, and the complete set of `deep_stages_completed`
  labels.
- **[`docs/AGENT_SKILLS.md`](docs/AGENT_SKILLS.md)** — how MAGE
  ranks and selects skills: the four capability sources, the
  scoring blend (`0.7 * Jaccard + 0.3 * relevance`), how to write
  a discoverable description, both registration paths.
- **[`configs/care_*.toml`](configs/)** — the three production-
  recommended presets.

## Development

All commands run through `uv run` which auto-activates the virtual environment.

```bash
# Setup (first time)
uv sync --group dev

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ -v --cov=mmar_mage --cov-report=term-missing

# Lint
uv run ruff check mmar_mage/ tests/

# Auto-format
uv run ruff format mmar_mage/ tests/

# Type check
uv run mypy mmar_mage/
```

Or use Makefile shortcuts:

```bash
make install    # uv sync --group dev
make test       # run tests
make test-cov   # tests + coverage report
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
make all        # lint + typecheck + tests
make help       # show all commands
```

## Examples

The `examples/` directory contains runnable scripts demonstrating each MAGE mode.

| Example | Command | Description |
|---------|---------|-------------|
| Fast mode | `make example-fast` | Single LLM call, no research stages |
| Deep mode (local) | `make example-deep` | Multi-stage pipeline, no internet |
| Deep + web research | `make example-web` | Deep mode with web search (Tavily) |
| Deep + memory research | `make example-memory` | Deep mode with memory recall |
| Full deep mode | `make example-full` | All features enabled |

Run all examples at once:

```bash
make examples
```

Each example reads its own `config.toml`, generates a CARL chain, and saves the result as `result.json` in its directory. See [`examples/README.md`](examples/README.md) for details on configuration options and required environment variables.

## Project Structure

```
mmar_mage/
  __init__.py          # public API
  generator.py         # MAGEGenerator orchestrator
  schemas.py           # Pydantic models (MAGEConfig, MAGEResult, ...)
  llm.py               # LLM client abstraction (AsyncOpenAI + streaming + retries)
  prompts.py           # legacy constants — loaded from prompts_data/prompts.yaml
  prompt_loader.py     # YAML prompt loader (§8 P2)
  prompts_data/
    prompts.yaml       # source of truth for all 26 prompts
  memory.py            # gigaevo-memory wrapper + ecosystem-save helpers
  cli.py               # typer CLI (generate, evolve, validate, version)
  exceptions.py        # typed exceptions incl. MAGECancelledError
  cost.py              # estimate_chain_cost + estimate_pipeline_cost (§6 P3)
  agents/
    domain_analyzer.py
    step_planner.py
    dag_builder.py
    step_describer.py
    capability_lookup_agent.py  # unified §2 capability surface
    chain_verifier.py           # §3.5 Chain-of-Verification
    self_refiner.py             # §3.6 Iterative Self-Refinement
    step_critic.py              # §3.4 per-step quality critique
    memory_research_agent.py    # Stage 0a
    web_research_agent.py       # Stage 0b
configs/
  fast.toml            # quick single-call config
  deep_local.toml      # local provider preset
  deep_research.toml   # research-on preset
  care_default.toml    # CARE production preset (§8 P2)
  care_local.toml      # CARE local-LLM preset
  care_research.toml   # CARE full-stack preset
examples/              # runnable example scripts
tests/                 # 718 tests
docs/
  ARCHITECTURE.md
  AGENT_SKILLS.md      # §10 P2 — skill discovery + scoring guide
  CARE_INTEGRATION.md  # §10 P2 — full runtime contract
```

## License

MIT
