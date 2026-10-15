"""Tests for §8 P2 — CARE profile presets.

Verifies the three new TOML configs load cleanly via
``MAGEConfig.from_toml`` and produce the intended flag combinations.
The presets live alongside the existing `fast.toml` / `deep_*.toml`
under `configs/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmar_mage.schemas import MAGEConfig

PRESETS_DIR = Path(__file__).resolve().parent.parent / "configs"


# ---------------------------------------------------------------------------
# Existence + clean load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    ["care_default.toml", "care_local.toml", "care_research.toml"],
)
def test_preset_file_exists(preset_name: str) -> None:
    assert (PRESETS_DIR / preset_name).is_file(), f"missing preset: {preset_name}"


@pytest.mark.parametrize(
    "preset_name",
    ["care_default.toml", "care_local.toml", "care_research.toml"],
)
def test_preset_loads_via_from_toml(preset_name: str) -> None:
    """Every preset must parse cleanly and produce a valid MAGEConfig.

    This catches stale keys after a schema refactor, typos
    (`temprature`-style), and config drift in general.
    """
    cfg = MAGEConfig.from_toml(str(PRESETS_DIR / preset_name))
    # Sanity-check a field that all presets specify
    assert cfg.model
    assert cfg.mode in {"fast", "deep"}


# ---------------------------------------------------------------------------
# care_default — production-recommended preset
# ---------------------------------------------------------------------------


class TestCareDefault:
    def _load(self) -> MAGEConfig:
        return MAGEConfig.from_toml(str(PRESETS_DIR / "care_default.toml"))

    def test_provider_and_mode(self) -> None:
        cfg = self._load()
        assert cfg.provider == "openai"
        assert cfg.mode == "deep"

    def test_streaming_and_naming_enabled(self) -> None:
        cfg = self._load()
        assert cfg.enable_streaming is True
        assert cfg.enable_suggested_naming_llm is True

    def test_capability_surface_on(self) -> None:
        cfg = self._load()
        assert cfg.enable_capability_lookup is True
        assert cfg.enable_skill_discovery is True

    def test_memory_and_research_off(self) -> None:
        cfg = self._load()
        assert cfg.enable_memory is False
        assert cfg.enable_memory_research is False
        assert cfg.enable_web_research is False

    def test_ecosystem_save_off(self) -> None:
        """Audit + evolution-meta saves are opt-in for care_default."""
        cfg = self._load()
        assert cfg.enable_skill_provenance_save is False
        assert cfg.enable_reusable_step_save is False
        assert cfg.enable_failure_card_save is False
        assert cfg.enable_evolution_meta_save is False
        assert cfg.enable_intermediate_artifact_save is False


# ---------------------------------------------------------------------------
# care_local — local provider preset
# ---------------------------------------------------------------------------


class TestCareLocal:
    def _load(self) -> MAGEConfig:
        return MAGEConfig.from_toml(str(PRESETS_DIR / "care_local.toml"))

    def test_provider_is_local(self) -> None:
        cfg = self._load()
        assert cfg.provider == "local"
        assert cfg.base_url is not None

    def test_auto_model_detection(self) -> None:
        cfg = self._load()
        assert cfg.model == "__auto__"

    def test_json_mode_fallback_on(self) -> None:
        cfg = self._load()
        assert cfg.json_mode_fallback is True

    def test_streaming_on(self) -> None:
        cfg = self._load()
        assert cfg.enable_streaming is True

    def test_naming_llm_off_for_local(self) -> None:
        """Local models often don't ace short LLM calls — keep naming heuristic."""
        cfg = self._load()
        assert cfg.enable_suggested_naming_llm is False

    def test_extra_retries(self) -> None:
        """Local servers are flakier — preset bumps max_retries."""
        cfg = self._load()
        assert cfg.max_retries >= 3

    def test_memory_off(self) -> None:
        cfg = self._load()
        assert cfg.enable_memory is False


# ---------------------------------------------------------------------------
# care_research — power-user preset
# ---------------------------------------------------------------------------


class TestCareResearch:
    def _load(self) -> MAGEConfig:
        return MAGEConfig.from_toml(str(PRESETS_DIR / "care_research.toml"))

    def test_memory_and_research_on(self) -> None:
        cfg = self._load()
        assert cfg.enable_memory is True
        assert cfg.enable_memory_research is True
        assert cfg.enable_web_research is True

    def test_full_capability_surface(self) -> None:
        cfg = self._load()
        assert cfg.enable_capability_lookup is True
        assert cfg.enable_skill_discovery is True
        assert cfg.enable_memory_skill_lookup is True
        assert cfg.enable_local_skill_catalog is True
        assert cfg.enable_skill_web_discovery is True

    def test_quality_enhancements_on(self) -> None:
        cfg = self._load()
        assert cfg.enable_step_critique is True
        assert cfg.enable_chain_verification is True
        assert cfg.enable_self_refine is True
        assert cfg.enable_feedback_loop is True

    def test_full_ecosystem_persistence(self) -> None:
        cfg = self._load()
        assert cfg.enable_skill_provenance_save is True
        assert cfg.enable_reusable_step_save is True
        assert cfg.enable_failure_card_save is True
        assert cfg.enable_evolution_meta_save is True
        assert cfg.enable_intermediate_artifact_save is True

    def test_cold_start_multi_sampling(self) -> None:
        cfg = self._load()
        assert cfg.cold_start_candidates >= 2


# ---------------------------------------------------------------------------
# All three presets pass the schema validators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset_name",
    ["care_default.toml", "care_local.toml", "care_research.toml"],
)
def test_preset_round_trip_via_model_dump(preset_name: str) -> None:
    """A loaded preset round-trips through ``model_dump`` → ``model_validate``.

    Catches any subtle Pydantic typing mismatch (e.g. ``json_mode_fallback``
    being ``None`` vs ``bool`` after the §6 P2 validator runs).
    """
    cfg = MAGEConfig.from_toml(str(PRESETS_DIR / preset_name))
    dumped = cfg.model_dump()
    reloaded = MAGEConfig.model_validate(dumped)
    assert reloaded.provider == cfg.provider
    assert reloaded.mode == cfg.mode
    assert reloaded.json_mode_fallback == cfg.json_mode_fallback


def test_all_three_presets_have_distinct_intent() -> None:
    """Smoke test: presets should differ in at least the provider /
    research / ecosystem-save axes, so picking the right one matters."""
    default = MAGEConfig.from_toml(str(PRESETS_DIR / "care_default.toml"))
    local = MAGEConfig.from_toml(str(PRESETS_DIR / "care_local.toml"))
    research = MAGEConfig.from_toml(str(PRESETS_DIR / "care_research.toml"))
    # provider axis
    assert local.provider != default.provider
    # research axis
    assert research.enable_memory_research is True
    assert default.enable_memory_research is False
    # ecosystem-save axis
    assert research.enable_intermediate_artifact_save is True
    assert default.enable_intermediate_artifact_save is False
