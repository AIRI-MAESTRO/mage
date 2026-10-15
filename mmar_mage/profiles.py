"""Shipped benchmark profiles (plan item P3.5).

Authoring-time presets that bias generation for the four headline tool-use
benchmarks. A profile bundles MAGEConfig field overrides (``allowed_step_types``,
simplicity knobs) plus metadata (a topology preference, a tool catalogue, and a
``notes`` field for known quirks). Selecting a profile fills any config field the
user didn't set explicitly — see ``MAGEConfig`` (the ``benchmark_profile`` field +
``for_benchmark`` factory).

Out of scope: benchmark *execution* — runners stay in ``experiments/baselines/``.
These are purely authoring-time settings (composing P1.10 simplicity, P1.15
topology, P1.17 tools).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkProfile(BaseModel):
    """A named authoring-time preset for a benchmark.

    Parameters
    ----------
    name : str
        Profile identifier.
    config : dict
        MAGEConfig field overrides applied to *unset* fields when selected.
        Keys MUST be real ``MAGEConfig`` field names (validated in tests).
    default_topology : str
        Preferred topology (metadata; the TopologySelectorAgent still chooses,
        but this documents the expected shape).
    tool_catalog : list[str]
        Representative tool/category names for the benchmark (metadata).
    notes : str
        Known quirks worth remembering when authoring for this benchmark.
    """

    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    default_topology: str = "pipeline"
    tool_catalog: list[str] = Field(default_factory=list)
    notes: str = ""
    # Curated few-shot reference plans for this benchmark (plan item P1.7). Each is
    # ``{query, plan: [{number, title, step_type, aim}]}`` — a proven chain SHAPE
    # the StepPlanner can adapt, seeded for free on profiled runs (no runtime
    # bootstrap). Step types stay within the profile's ``allowed_step_types``.
    demos: list[dict[str, Any]] = Field(default_factory=list)


BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "bfcl": BenchmarkProfile(
        name="bfcl",
        config={"allowed_step_types": ["llm", "tool"], "simplicity_bias": 0.6, "simplicity_max_steps": 5},
        default_topology="pipeline",
        tool_catalog=["function_call", "parallel_function", "multiple_function"],
        notes="BFCL is function-calling: short tool-use chains win. `simple_python` rewards ≤3 steps; "
        "avoid ceremony. Strong simplicity bias.",
        demos=[
            {
                "query": "Get the current weather in Paris in Celsius.",
                "plan": [
                    {
                        "number": 1,
                        "title": "Parse the request",
                        "step_type": "llm",
                        "aim": "Extract the function + args (location, unit)",
                    },
                    {
                        "number": 2,
                        "title": "Call get_weather",
                        "step_type": "tool",
                        "aim": "Invoke the weather function with the parsed arguments",
                    },
                ],
            }
        ],
    ),
    "seal_tools": BenchmarkProfile(
        name="seal_tools",
        config={"allowed_step_types": ["llm", "tool", "parallel_sampling"], "simplicity_bias": 0.3},
        default_topology="diamond",
        tool_catalog=["api_call", "nested_tool", "tool_composition"],
        notes="Seal-Tools has hard tool-selection; self-consistency (parallel_sampling) on the "
        "selection step reduces variance. Moderate simplicity.",
        demos=[
            {
                "query": "Choose and call the right API to compose two services.",
                "plan": [
                    {
                        "number": 1,
                        "title": "Select the tool by self-consistency",
                        "step_type": "parallel_sampling",
                        "aim": "Independently pick the API several ways and take the majority",
                    },
                    {
                        "number": 2,
                        "title": "Invoke the selected tool",
                        "step_type": "tool",
                        "aim": "Call the agreed-upon API with mapped arguments",
                    },
                ],
            }
        ],
    ),
    "rotbench": BenchmarkProfile(
        name="rotbench",
        config={"allowed_step_types": ["llm", "tool", "evaluation"], "simplicity_bias": 0.4},
        default_topology="critique_revise_loop",
        tool_catalog=["robust_tool_use", "noisy_environment"],
        notes="RoTBench injects noise/errors; an early `evaluation` gate catches malformed tool "
        "calls before they cascade. Keep chains tight.",
        demos=[
            {
                "query": "Call the tool robustly under a noisy environment.",
                "plan": [
                    {
                        "number": 1,
                        "title": "Call the tool",
                        "step_type": "tool",
                        "aim": "Invoke the tool with validated arguments",
                    },
                    {
                        "number": 2,
                        "title": "Validate the result",
                        "step_type": "evaluation",
                        "aim": "Check the tool output is well-formed before using it",
                    },
                ],
            }
        ],
    ),
    "toolalpaca": BenchmarkProfile(
        name="toolalpaca",
        config={"allowed_step_types": ["llm", "tool", "mcp", "tool_discovery"], "simplicity_bias": 0.3},
        default_topology="pipeline",
        tool_catalog=["long_tail_api", "open_domain_tool"],
        notes="ToolAlpaca's long tail of APIs benefits from `tool_discovery` (runtime registration) "
        "and broader catalogs; don't over-constrain step types.",
        demos=[
            {
                "query": "Answer using an open-domain API the agent hasn't seen before.",
                "plan": [
                    {
                        "number": 1,
                        "title": "Discover the relevant API",
                        "step_type": "tool_discovery",
                        "aim": "Register the API best matching the task from the catalog",
                    },
                    {
                        "number": 2,
                        "title": "Invoke the discovered tool",
                        "step_type": "tool",
                        "aim": "Call the newly registered API and return its result",
                    },
                ],
            }
        ],
    ),
    "custom": BenchmarkProfile(
        name="custom",
        config={},
        notes="No presets — user-configured. Use as an explicit 'profile selected but no overrides'.",
    ),
}


def get_profile(name: str | None) -> BenchmarkProfile | None:
    """Return the :class:`BenchmarkProfile` for ``name`` (or ``None``)."""
    if not name:
        return None
    return BENCHMARK_PROFILES.get(name)


def list_profiles() -> list[str]:
    """All shipped profile names, sorted."""
    return sorted(BENCHMARK_PROFILES)


def format_profile_demos(profile: BenchmarkProfile | None) -> str:
    """Render a profile's curated demos as a planner few-shot block (plan item P1.7).

    Returns an empty string when the profile is ``None`` or has no demos, so callers
    can pass the result through unconditionally. Self-contained (no bootstrap import).
    """
    if profile is None or not profile.demos:
        return ""
    blocks: list[str] = []
    for i, demo in enumerate(profile.demos, 1):
        lines = [
            f"  {s.get('number')}. [{s.get('step_type', 'llm')}] {s.get('title', '')} — {s.get('aim', '')}"
            for s in demo.get("plan", [])
        ]
        steps_txt = "\n".join(lines) if lines else "  (no steps)"
        blocks.append(
            f"### Curated plan {i} for the {profile.name} benchmark\nQuery: {demo.get('query', '')}\nPlan:\n{steps_txt}"
        )
    return (
        "## Curated reference plans (proven structure for this benchmark — adapt the "
        "shape, don't copy verbatim):\n" + "\n\n".join(blocks)
    )
