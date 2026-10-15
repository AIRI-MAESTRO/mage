# MAGE Architecture

## Overview

MAGE (Mesh Architecture Generation Engine) converts text queries into
structured CARL (Collaborative Agent Reasoning Library) reasoning chains.

```
User query (str)
    │
    ▼
┌─────────────────────────────────────────────┐
│             MAGEGenerator                    │
│                                              │
│  mode=fast ──► single LLM call               │
│  mode=deep ──► agentic pipeline:             │
│    0. research (opt): memory / web /         │
│       capability lookup / templates / skills │
│    1. DomainAnalyzer    → DomainAnalysis     │
│    2. StepPlanner       → StepPlan           │
│    2.5 TopologySelector → topology (opt)     │
│    3. DAGBuilder        → DAGStructure       │
│    4. StepDescriber     → [CARLStepSchema]   │
│    4.5 StepCritic / ChainVerifier /          │
│        SelfRefiner — quality gates (opt)     │
│    5. Serialization     → CARLChainSchema    │
│                                              │
│  + validation via validate_carl_json()       │
│  + optional ReasoningChain.from_json()       │
│  + optional gigaevo-memory save              │
│                                              │
└─────────────────────────────────────────────┘
    │
    ▼
MAGEResult {
    chain_json: str          # CARL JSON
    chain_dict: dict         # parsed dict
    memory_key: str | None   # gigaevo-memory ID
    mode: "fast" | "deep"
    metadata: MAGEMetadata
}
```

## Design Principles

### Inspired by Claude Code

MAGE borrows several architectural patterns from the Claude Code agent:

1. **Tool Use Loop** — deep mode agents call the LLM iteratively, validate
   intermediate results, and retry on failure.  Each agent gets a focused
   prompt and single responsibility.

2. **Context Window Management** — each agent receives only the context it
   needs (user query + output from previous agents), keeping token usage
   predictable.

3. **Structured Output** — all LLM calls use `response_format=json_object`
   and output is validated against Pydantic models before propagation.

4. **Progress Streaming** — `MAGEProgressCallback` protocol lets callers
   observe stage transitions in real time.

### Individually Toggleable Sub-stages

Every deep-mode agent can be enabled or disabled independently in
`MAGEConfig`.  When an agent is disabled, MAGE uses deterministic
fallbacks:

| Stage | Disabled fallback |
|-------|-------------------|
| CapabilityLookupAgent | Empty `CapabilityContext` (no tools / MCP / skills surfaced) |
| DomainAnalyzer | Default `DomainAnalysis(domain="general", ...)` |
| StepPlanner | Generates N generic steps from `suggested_step_count` |
| TopologySelector | Falls back to the planner's linear / funnel default |
| DAGBuilder | Creates a simple linear pipeline |
| StepDescriber | Creates minimal descriptions from `rough_aim` |

This allows partial pipeline execution for debugging, cost savings,
or when only some agents are needed.

## Data Flow

```
DomainAnalysis
    │
    ├──► StepPlan
    │        │
    │        ├──► DAGStructure
    │        │        │
    │        │        ├──► [CARLStepSchema]
    │        │        │        │
    │        │        │        ▼
    │        │        │    CARLChainSchema
    │        │        │        │
    │        │        │        ├──► validate_carl_json()
    │        │        │        ├──► ReasoningChain.from_json() (optional)
    │        │        │        └──► MemoryManager.save_chain() (optional)
    │        │        │                 │
    │        │        │                 ▼
    │        │        │            MAGEResult
```

## Component Details

### LLMClient (`llm.py`)

Thin async wrapper around `openai.OpenAI`.  Supports:
- JSON mode (`response_format`)
- Pydantic model validation (`generate_structured`)
- Configurable retries
- Any OpenAI-compatible endpoint

### MemoryManager (`memory.py`)

Graceful wrapper around `gigaevo-memory`:
- Returns `None` if package not installed
- Returns `None` on connection/save errors
- Never blocks chain generation

### Validation (`generator.py`)

Two-tier validation:
1. `validate_carl_json()` — checks structure, step numbers, DAG acyclicity
2. `ReasoningChain.from_json()` — full CARL library validation (if installed)

### CLI (`cli.py`)

Built with `typer` + `rich`.  Entry point: `mage generate "query"`.
Progress displayed via `RichProgressCallback`.

## CARL JSON Format

The generated JSON follows the CARL specification:

```json
{
  "name": "...",
  "max_workers": 2,
  "timeout": 300.0,
  "search_config": {...},
  "steps": [
    {
      "step_type": "LLM",
      "number": 1,
      "title": "...",
      "aim": "...",
      "reasoning_questions": "...",
      "step_context_queries": [...],
      "stage_action": "...",
      "example_reasoning": "...",
      "dependencies": [],
      "retry_max": 3,
      "llm_config": null
    }
  ]
}
```

Step types supported: `LLM`, `TOOL`, `MEMORY`, `TRANSFORM`, `CONDITIONAL`, `MCP`.

## Future Directions

- Integration with GigaEvo for chain evolution
- Multi-model routing (fast model for analysis, powerful for synthesis)
- Interactive mode with user clarification questions
- Chain templates and caching
