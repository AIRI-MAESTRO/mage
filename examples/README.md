# MAGE Examples

Each subdirectory contains a self-contained example with:

- `run.py` — script that generates a CARL chain and saves it as `result.json`
- `config.toml` — MAGE configuration for that example

## Prerequisites

```bash
uv sync --group dev
export OPENAI_API_KEY="sk-..."          # required for all examples
export TAVILY_API_KEY="tvly-..."        # required for web research examples
```

## Examples

### fast_mode

Single LLM call, no research stages. Fastest option for simple tasks.

- **Query:** "Classify customer support tickets by urgency and route to appropriate department"
- **Config:** `mode = "fast"`, memory and web research disabled
- **Run:** `make example-fast` or `uv run python examples/fast_mode/run.py`

### deep_mode_local

Multi-stage agentic pipeline (DomainAnalyzer → StepPlanner → DAGBuilder → StepDescriber) without any internet access or memory research.

- **Query:** "Design a microservice architecture for an e-commerce platform"
- **Config:** `mode = "deep"`, memory and web research disabled
- **Run:** `make example-deep` or `uv run python examples/deep_mode_local/run.py`

### deep_mode_web_research

Deep mode with web search enabled. The WebResearchAgent searches the internet for domain context before planning steps.

- **Query:** "Analyze current trends in renewable energy investment for 2025"
- **Config:** `mode = "deep"`, `enable_web_research = true`, provider = Tavily
- **Requires:** `TAVILY_API_KEY` (or `MAGE_WEB_SEARCH_API_KEY`)
- **Run:** `make example-web` or `uv run python examples/deep_mode_web_research/run.py`

### deep_mode_memory_research

Deep mode with memory research enabled. The MemoryResearchAgent recalls past experience from gigaevo-memory before planning. Includes cold-start candidate generation.

- **Query:** "Build a data pipeline for real-time fraud detection"
- **Config:** `mode = "deep"`, `enable_memory_research = true`, `cold_start_candidates = 3`
- **Requires:** A running gigaevo-memory server (defaults to `localhost:8002`)
- **Run:** `make example-memory` or `uv run python examples/deep_mode_memory_research/run.py`

### deep_mode_full

All features enabled: deep mode with both memory research and web research.

- **Query:** "Create a comprehensive evaluation framework for large language models"
- **Config:** `mode = "deep"`, all research stages enabled
- **Requires:** `TAVILY_API_KEY` + gigaevo-memory server
- **Run:** `make example-full` or `uv run python examples/deep_mode_full/run.py`

## Run All Examples

```bash
make examples
```

## Output

Each example prints a summary to stdout and saves the full CARL JSON chain to `result.json` in its directory. The `result.json` files are git-ignored.
