"""Full deep mode example — all features enabled (memory + web research)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mmar_mage import MAGEConfig, MAGEGenerator

QUERY = "Create a comprehensive evaluation framework for large language models"
CONFIG_PATH = Path(__file__).parent / "config.toml"
RESULT_PATH = Path(__file__).parent / "result.json"


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MAGE_API_KEY"):
        print("Error: Set OPENAI_API_KEY or MAGE_API_KEY environment variable.")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("MAGE_WEB_SEARCH_API_KEY"):
        print("Error: Set TAVILY_API_KEY or MAGE_WEB_SEARCH_API_KEY environment variable for web research.")
        sys.exit(1)

    config = MAGEConfig.from_toml(CONFIG_PATH)
    generator = MAGEGenerator(config=config)

    print(f"Query: {QUERY}")
    print(f"Mode:  {config.mode}")
    print(f"Model: {config.model}")
    print(f"Web:   {config.web_search_provider}")
    print("Memory research: enabled")
    print(f"Cold start candidates: {config.cold_start_candidates}")
    print()

    start = time.perf_counter()
    result = await generator.generate(QUERY)
    elapsed = time.perf_counter() - start

    RESULT_PATH.write_text(result.chain_json)

    print(f"Domain:      {result.metadata.domain}")
    print(f"Steps:       {result.metadata.num_steps}")
    print(f"Memory hits: {result.metadata.memory_hits_used}")
    print(f"Web results: {result.metadata.web_results_used}")
    print(f"Cold start:  {result.metadata.was_cold_start}")
    print(f"Time:        {elapsed:.2f}s")
    print(f"Result:      {RESULT_PATH}")
    print()
    print(json.dumps(result.chain_dict, indent=2)[:500])


if __name__ == "__main__":
    asyncio.run(main())
