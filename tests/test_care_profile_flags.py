"""T38 — care profiles declare the quality / replan flags explicitly + stay consistent."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from mmar_mage.schemas import MAGEConfig

_CARE_PROFILES = ["care_default", "care_local", "care_research"]
# The MAGEConfig flags the audit flagged as invisible — must be declared in every
# care profile so a user can see (and flip) them without reading the source.
_VISIBLE_FLAGS = [
    "use_schema_constrained_output",
    "enable_replan_generation",
    "enable_step_critique",
    "enable_chain_verification",
]


def _mage(name: str) -> dict[str, Any]:
    return tomllib.loads(Path(f"configs/{name}.toml").read_text(encoding="utf-8"))["mage"]


def test_care_profiles_declare_quality_flags_explicitly() -> None:
    for name in _CARE_PROFILES:
        mage = _mage(name)
        for flag in _VISIBLE_FLAGS:
            assert flag in mage, f"{name}.toml does not declare {flag} explicitly"


def test_care_profiles_load() -> None:
    for name in _CARE_PROFILES:
        MAGEConfig.from_toml(f"configs/{name}.toml")  # must not raise


def test_self_refine_implies_verification() -> None:
    # Drift check across EVERY profile: a chain that self-refines must also verify
    # (refine consumes the verifier's verdict) — catches a self_refine/verification
    # contradiction in any config.
    for path in sorted(Path("configs").glob("*.toml")):
        mage = tomllib.loads(path.read_text(encoding="utf-8")).get("mage", {})
        if mage.get("enable_self_refine"):
            assert mage.get("enable_chain_verification"), f"{path.name}: self_refine on but verification off"


def test_local_profile_does_not_claim_schema_constrained() -> None:
    # json_mode_fallback sends no response_format, so schema-guided decoding is a no-op.
    mage = _mage("care_local")
    assert mage.get("json_mode_fallback") is True
    assert mage["use_schema_constrained_output"] is False


def test_profile_tiers_are_coherent() -> None:
    default, research = _mage("care_default"), _mage("care_research")
    # research is the thorough tier: full (unscoped) critique + chain verification
    assert research["enable_step_critique"] and research["enable_chain_verification"]
    assert not research.get("step_critique_step_types")  # critiques every step
    # default is the lightweight tier: a SCOPED critique (risky step types only, T11),
    # no full chain verification
    assert default["enable_step_critique"] and default.get("step_critique_step_types")
    assert not default["enable_chain_verification"]
