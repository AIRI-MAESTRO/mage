"""WP-10 — recommended-models preset loads, applies, and prices correctly."""

from __future__ import annotations

from mmar_mage.models_preset import DEFAULT_PRESET, load_models_preset
from mmar_mage.schemas import MAGEConfig


def test_preset_file_exists() -> None:
    assert DEFAULT_PRESET.exists()


def test_load_stage_overrides_and_pricing() -> None:
    preset = load_models_preset()
    so = preset["stage_llm_overrides"]
    assert so["step_planning"]["model"] == "deepseek/deepseek-v4-pro"
    assert so["step_describing"]["model"] == "deepseek/deepseek-v4-flash"
    assert so["tool_implementation"]["model"] == "qwen/qwen3-coder"
    # generator (deepseek) and judge (another family) must differ — P-5
    assert "deepseek" not in so["chain_grading"]["model"]

    pricing = preset["pricing"]  # per-1k tuples for cost.py
    assert pricing["deepseek/deepseek-v4-pro"] == (0.27 / 1000, 0.435 / 1000)
    assert all(len(v) == 2 for v in pricing.values())


def test_overrides_apply_to_mageconfig() -> None:
    preset = load_models_preset()
    cfg = MAGEConfig(api_key="test", stage_llm_overrides=preset["stage_llm_overrides"])
    # the field validator promotes bare dicts into StageLLMOverride
    assert cfg.stage_llm_overrides["step_planning"].model == "deepseek/deepseek-v4-pro"
    assert cfg.stage_llm_overrides["chain_grading"].temperature == 0.0


def test_pricing_shape_drives_cost_estimator() -> None:
    from mmar_mage.cost import estimate_chain_cost

    preset = load_models_preset()
    chain = {"steps": [{"step_type": "llm", "number": 1, "title": "t", "aim": "do a thing"}]}
    est = estimate_chain_cost(chain, pricing=preset["pricing"], default_model="deepseek/deepseek-v4-pro")
    assert est.total_cost_usd >= 0
