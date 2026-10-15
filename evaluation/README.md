# MAGE Evaluation

Batch generation of CARL chains, baseline prompts, and real LLM execution for NeurIPS 2026 experiments.

## Experimental Design

Two model roles are separated:

- **Generation models** (large, e.g. DeepSeek-V3.2, Kimi-K2.5):
  - Create MAGE CARL chains
  - Generate baseline prompt text (deterministic — no LLM call)

- **Execution models** (small ≤15B, e.g. Qwen3.5-9B, Phi-4):
  - Execute MAGE chains on benchmark tasks
  - Answer baseline prompts (zero-shot, CoT, plan-and-solve)

This lets us measure: does a MAGE-generated chain from a big model help a small model more than a plain CoT prompt?

## Usage

### Option 1: Two-stage (recommended for experiments)

**Stage A — Generate artifacts** (only generation models, no benchmark execution):
```bash
export OPENROUTER_API_KEY="sk-or-..."
uv run python -m evaluation.runner generate-artifacts
```

Output: MAGE chains in `results/chains/`, baseline prompts in `results/baselines/`.

**Stage B — Run benchmarks** (execution models run the artifacts):
```bash
uv run python -m evaluation.runner run-benchmarks
```

Output: chain executions in `results/chain_executions/`, baseline executions in `results/baseline_executions/`.

### Option 2: One-command end-to-end

```bash
uv run python -m evaluation.runner run-all
```

### Quick smoke test

Each command supports `--quick` (2 tasks per benchmark, 1 model):

```bash
uv run python -m evaluation.runner generate-artifacts --quick
uv run python -m evaluation.runner run-benchmarks --quick
# or
uv run python -m evaluation.runner run-all --quick
```

### Selective benchmark execution

Skip chain or baseline execution during Stage B:

```bash
# Only execute baselines (skip chains)
uv run python -m evaluation.runner run-benchmarks --skip-chains

# Only execute chains (skip baselines)
uv run python -m evaluation.runner run-benchmarks --skip-baselines
```

## Other commands

```bash
# List available benchmark tasks
python -m evaluation.runner list-tasks

# List tasks for specific benchmarks
python -m evaluation.runner list-tasks hover hotpotqa
```

## Configuration

Edit `evaluation/eval_config.toml` or provide your own TOML file:

```toml
[evaluation]
api_key = ""  # or set OPENROUTER_API_KEY env var
mage_mode = "deep"
temperature = 0.2

generation_models = [
    "deepseek/deepseek-v3.2",
    "minimax/minimax-m2.7",
]

benchmarks = ["hover", "ifbench", "hotpotqa"]
output_dir = "evaluation/results"
max_tasks_per_benchmark = 2  # quick testing
```

## Chain Generation Modes

The pipeline supports two chain generation modes, controlled by the `chain_mode` config setting or the `--mode` CLI flag:

### Category mode (default, recommended for experiments)

Generates **one abstract reasoning chain per benchmark** instead of one chain per question. This is the correct setup for NeurIPS experiments because:

- **No answer contamination**: the chain describes a *process*, not specific facts
- **Generalisation**: the same chain template works for any question of that benchmark type
- **Scientific rigour**: the generation model never sees the specific question being answered

```bash
# Generate category chains (default)
uv run python -m evaluation.runner generate-artifacts

# Or explicitly
uv run python -m evaluation.runner generate-artifacts --mode category

# Standalone command
uv run python -m evaluation.runner generate-category-chains
```

The category chain prompt uses placeholders like `{question}`, `{entity}`, `{topic}` and includes a contamination checker that rejects chains containing specific entities from example questions.

### Per-task mode (legacy)

Generates one chain per question — the original behaviour. May leak answer-specific entities from the generation model into the chain steps.

```bash
uv run python -m evaluation.runner generate-artifacts --mode per-task
```

### Configuration

```toml
[evaluation]
chain_mode = "category"          # "category" or "per_task"
category_example_count = 5       # examples shown to generation model
contamination_check_enabled = true
```

## Output structure

```
evaluation/results/
├── category_chains/<gen_model>/<benchmark>.json            # NEW: category-level chains
│   └── <benchmark>_meta.json
├── chains/<gen_model>/<benchmark>/                         # Legacy: per-task chains
│   ├── <task_id>.json
│   ├── <task_id>.py
│   └── <task_id>_meta.json
├── baselines/<type>/<benchmark>/                           # Stage A: baseline prompts
│   ├── <task_id>.txt
│   └── <task_id>_meta.json
├── chain_executions/<gen_model>/<exec_model>/<benchmark>/  # Stage B: chain results
│   ├── <task_id>.txt           # response from EXECUTION model
│   ├── <task_id>_prompt.txt    # flattened chain prompt sent
│   └── <task_id>_meta.json     # timing, steps, success/error
└── baseline_executions/<type>/<exec_model>/<benchmark>/    # Stage B: baseline results
    ├── <task_id>.txt           # response from EXECUTION model
    ├── <task_id>_prompt.txt    # prompt sent
    └── <task_id>_meta.json     # timing, tokens, success/error
```

## Benchmarks

| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| hover | 5 | Multi-hop fact verification |
| ifbench | 5 | Instruction following with constraints |
| hotpotqa | 5 | Multi-hop question answering |
| pupa | 3 | Personalized agent tasks |
| livebench_math | 3 | Math reasoning |
| simpleqa | 3 | Factual QA |

## Baseline types

- **zero_shot** — raw task query as a direct prompt
- **cot** — query + "Let's think step by step."
- **plan_and_solve** — query + "Let's first understand the problem and devise a plan..."

## Models

### Generation models (create CARL chains)
- `deepseek/deepseek-v3.2`
- `minimax/minimax-m2.7`
- `z-ai/glm-5.1`
- `moonshotai/kimi-k2.5`

### Execution models (run chains + baselines on benchmarks)
- `qwen/qwen3.5-9b`
- `meta-llama/llama-3.3-8b-instruct`
- `microsoft/phi-4`
- `mistralai/ministral-3b-2512`

## Chain executor (stub v1)

The current chain executor is a stub (v1) that flattens a CARL chain into a single prompt using step aims as a reasoning guide. This gives a reasonable proxy to measure chain quality before the full CARL chain runtime is built.

The flattened prompt looks like:
```
Follow this reasoning plan to answer the question.

QUESTION: <task query>

REASONING PLAN:
1. <step 1 aim>
2. <step 2 aim>
3. <step 3 aim>

Now work through the plan step by step and give the final answer.
```

TODO: Replace with full CARL chain runtime orchestrator.

## LLM-only mode

Generate chains that contain only `llm` step types — no tool calls, MCP, eval, conditional, aggregator, memory, or transform steps.

**Purpose:** When execution models (small, ≤15B parameters) cannot call tools, all steps must be pure LLM reasoning. This mode ensures generated chains are executable by any model that supports text-in/text-out inference.

**How to enable:** Set `allowed_step_types = ["llm"]` in `eval_config.toml`:

```toml
[evaluation]
allowed_step_types = ["llm"]
```

Or pass it programmatically:

```python
from mmar_mage import MAGEConfig

config = MAGEConfig(allowed_step_types=["llm"])
```

**Example use case:** Benchmarks (HotpotQA, HoVer, LiveBench Math) where only multi-step reasoning is needed and execution models lack tool-calling capabilities.
