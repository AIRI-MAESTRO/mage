"""Agent-features example.

Generates a CARL chain that:
  • routes PDF extraction through the official Anthropic PDF skill
    (via SkillDiscoveryAgent surfacing it to StepPlanner)
  • marks a checkpoint after extraction so the runtime can roll back
    (via enable_replan_generation in config)
  • is dry-run cost-estimated before save (via mmar_mage.estimate_chain_cost)

Run with:
    OPENAI_API_KEY=... python examples/agent_features/run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mmar_mage import MAGEConfig, MAGEGenerator, estimate_chain_cost

QUERY = (
    "Extract structured tabular data from an invoice PDF and produce a JSON "
    "record with vendor, total, currency, and line items."
)
CONFIG_PATH = Path(__file__).parent / "config.toml"
RESULT_PATH = Path(__file__).parent / "result.json"

# Rough pricing as of late 2025; tweak for the model you use.
PRICING_USD_PER_1K = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
}


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MAGE_API_KEY"):
        print("Error: Set OPENAI_API_KEY or MAGE_API_KEY environment variable.")
        sys.exit(1)

    config = MAGEConfig.from_toml(CONFIG_PATH)
    generator = MAGEGenerator(config=config)

    print(f"Query: {QUERY}")
    print(f"Mode:  {config.mode}")
    print(f"Model: {config.model}")
    print(f"Skill discovery: {config.enable_skill_discovery}")
    print(f"RE-PLAN policy:  {config.enable_replan_generation}")
    print()

    start = time.perf_counter()
    result = await generator.generate(QUERY)
    elapsed = time.perf_counter() - start

    RESULT_PATH.write_text(result.chain_json)

    print(f"Domain:  {result.metadata.domain}")
    print(f"Steps:   {result.metadata.num_steps}")
    print(f"Time:    {elapsed:.2f}s")

    # Highlight any agent_skill / agent-features step types.
    used_types: dict[str, int] = {}
    checkpoints: list[str] = []
    for step in result.chain_dict.get("steps", []):
        st = step.get("step_type", "llm")
        used_types[st] = used_types.get(st, 0) + 1
        if step.get("checkpoint"):
            checkpoints.append(step.get("checkpoint_name") or f"step_{step['number']}")

    print(f"Step types: {used_types}")
    if checkpoints:
        print(f"Checkpoints: {', '.join(checkpoints)}")
    if result.chain_dict.get("replan_policy"):
        policy_source = result.chain_dict["replan_policy"].get("source", "n/a")
        print(f"RE-PLAN policy attached (source: {policy_source})")

    # Dry-run cost.
    estimate = estimate_chain_cost(result.chain_dict, pricing=PRICING_USD_PER_1K, default_model=config.model)
    print()
    print("=" * 72)
    print("DRY-RUN COST ESTIMATE")
    print("=" * 72)
    print(estimate.format_table())
    print()
    print(f"Result saved to: {RESULT_PATH}")
    print()
    print(json.dumps(result.chain_dict, indent=2)[:800])


if __name__ == "__main__":
    asyncio.run(main())
