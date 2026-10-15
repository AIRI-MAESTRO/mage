"""Per-profile curated demo sets — plan item P1.7 (deferred follow-up).

Each benchmark profile ships proven reference plans; selecting a profile seeds
them as StepPlanner few-shots for free (no runtime bootstrap). Step types in a
demo stay within the profile's ``allowed_step_types``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.profiles import BENCHMARK_PROFILES, format_profile_demos, get_profile
from mmar_mage.schemas import MAGEConfig, PlannedStep, StepPlan


# --------------------------------------------------------------------------- #
# Profile demos + consistency
# --------------------------------------------------------------------------- #
def test_headline_profiles_ship_demos():
    for name in ("bfcl", "seal_tools", "rotbench", "toolalpaca"):
        assert get_profile(name).demos, f"{name} has no curated demos"
    assert get_profile("custom").demos == []  # custom ships none


def test_demo_step_types_within_profile_allowed_types():
    for name, profile in BENCHMARK_PROFILES.items():
        allowed = set(profile.config.get("allowed_step_types") or [])
        if not allowed:
            continue
        for demo in profile.demos:
            for step in demo["plan"]:
                assert step["step_type"] in allowed, f"{name}: {step['step_type']} not in allowed {allowed}"


def test_demos_have_query_and_well_formed_plan():
    for profile in BENCHMARK_PROFILES.values():
        for demo in profile.demos:
            assert demo.get("query")
            assert demo.get("plan")
            for step in demo["plan"]:
                assert {"number", "title", "step_type", "aim"} <= set(step)


# --------------------------------------------------------------------------- #
# format_profile_demos
# --------------------------------------------------------------------------- #
def test_format_renders_curated_block():
    text = format_profile_demos(get_profile("bfcl"))
    assert "Curated reference plans" in text
    assert "bfcl benchmark" in text
    assert "[tool]" in text  # a bfcl demo step type


def test_format_none_or_empty_returns_empty():
    assert format_profile_demos(None) == ""
    assert format_profile_demos(get_profile("custom")) == ""


# --------------------------------------------------------------------------- #
# StepPlanner injection
# --------------------------------------------------------------------------- #
async def test_planner_injects_curated_demos():
    from mmar_mage.agents.step_planner import StepPlanner
    from mmar_mage.schemas import DomainAnalysis

    llm = MagicMock()
    llm.generate_structured = AsyncMock(return_value=StepPlan(steps=[], rationale="ok"))
    await StepPlanner(llm).plan(
        "q", DomainAnalysis(domain="general"), curated_demos=format_profile_demos(get_profile("rotbench"))
    )
    prompt = llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Curated reference plans" in prompt
    assert "rotbench benchmark" in prompt
    # NOT mislabelled as synthetic-benchmark-verified (that header is for P1.7 bootstrap)
    assert "verified on a synthetic benchmark" not in prompt


# --------------------------------------------------------------------------- #
# Real-execution scenario — a profiled generate() seeds curated demos
# --------------------------------------------------------------------------- #
def _planning_only_gen(profile: str | None):
    from mmar_mage.generator import MAGEGenerator

    cfg = MAGEConfig.for_benchmark(profile, api_key="t") if profile else MAGEConfig(api_key="t")
    # isolate the planning stage
    for f in (
        "enable_domain_analysis",
        "enable_topology_selection",
        "enable_dag_optimization",
        "enable_step_descriptions",
    ):
        setattr(cfg, f, False)
    cfg.mode = "deep"
    cfg.enable_memory = False
    gen = MAGEGenerator(config=cfg)
    gen._llm = MagicMock()
    gen._llm.resolve_model = AsyncMock()
    plan = StepPlan(steps=[PlannedStep(number=1, title="S", step_type="llm", rough_aim="x")], rationale="r")
    gen._llm.generate_structured = AsyncMock(return_value=plan)
    gen._llm.generate_json = AsyncMock(return_value={})
    return gen


async def test_profiled_generate_seeds_curated_demos_into_planner():
    gen = _planning_only_gen("bfcl")
    await gen.generate("get the weather")
    prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Curated reference plans" in prompt
    assert "bfcl benchmark" in prompt


async def test_no_profile_no_curated_demos():
    gen = _planning_only_gen(None)
    await gen.generate("anything")
    prompt = gen._llm.generate_structured.await_args.kwargs["user_prompt"]
    assert "Curated reference plans" not in prompt
