"""Tests for §8 P2 — YAML-backed prompt loader.

Covers:
- The 27 prompts shipped in ``mmar_mage/prompts_data/prompts.yaml``
  all load via :func:`load_prompts` and are non-empty
- Every legacy ``X_PROMPT`` constant in ``mmar_mage.prompts`` still
  exists and matches the YAML
- ``load_prompts(path=...)`` overrides the default path
- ``MAGE_PROMPTS_PATH`` env var overrides the default
- The loader rejects malformed YAML (non-mapping, non-string values,
  empty strings)
- Missing file raises ``FileNotFoundError``
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from mmar_mage.prompt_loader import default_prompts_path, load_prompts

EXPECTED_PROMPT_NAMES: list[str] = [
    "BENCHMARK_SYNTHESIZER_PROMPT",
    "CHAIN_EDIT_PROMPT",
    "CHAIN_FLATTEN_EXECUTION_PROMPT",
    "CHAIN_JUDGE_PROMPT",
    "CHAIN_JUDGE_REAL_OUTPUT_PROMPT",
    "CHAIN_VERIFICATION_PROMPT",
    "DAG_BUILDER_PROMPT",
    "DOMAIN_ANALYZER_PROMPT",
    "FAST_MODE_SYSTEM_PROMPT",
    "FEEDBACK_DIGEST_PROMPT",
    "LLM_REPAIR_PROMPT",
    "MCTS_CRITIC_PROMPT",
    "MCTS_PROPOSE_STEP_PROMPT",
    "MCTS_SIMULATE_PROMPT",
    "MEMORY_PURIFIER_PROMPT",
    "MEMORY_SUB_QUERY_GENERATOR_PROMPT",
    "PLAN_SCORER_PROMPT",
    "SELF_REFINE_CRITIQUE_PROMPT",
    "SELF_REFINE_REVISE_PROMPT",
    "STEP_CRITIC_PROMPT",
    "STEP_DESCRIBER_PROMPT",
    "STEP_PLANNER_PROMPT",
    "SUGGESTED_NAMING_PROMPT",
    "TOT_BRANCH_SCORER_PROMPT",
    "VERIFICATION_FIXER_PROMPT",
    "WEB_DIGEST_PROMPT",
    "WEB_SUB_QUERY_GENERATOR_PROMPT",
]


# ---------------------------------------------------------------------------
# Default YAML contents
# ---------------------------------------------------------------------------


class TestDefaultPromptsYaml:
    def test_default_path_exists(self) -> None:
        path = default_prompts_path()
        assert path.is_file(), f"bundled YAML missing: {path}"

    def test_default_yaml_loads(self) -> None:
        prompts = load_prompts()
        assert len(prompts) == len(EXPECTED_PROMPT_NAMES)

    def test_every_expected_prompt_present(self) -> None:
        prompts = load_prompts()
        missing = set(EXPECTED_PROMPT_NAMES) - set(prompts.keys())
        assert not missing, f"missing prompts in YAML: {sorted(missing)}"

    def test_every_value_non_empty_string(self) -> None:
        prompts = load_prompts()
        for name, value in prompts.items():
            assert isinstance(value, str), f"{name} is not a string"
            assert value.strip(), f"{name} is empty"


# ---------------------------------------------------------------------------
# Legacy module-level constants still work (backwards compat)
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_every_constant_exists_on_prompts_module(self) -> None:
        import mmar_mage.prompts as prompts

        for name in EXPECTED_PROMPT_NAMES:
            assert hasattr(prompts, name), f"missing constant: {name}"
            value = getattr(prompts, name)
            assert isinstance(value, str)
            assert value.strip()

    def test_constants_match_yaml_byte_for_byte(self) -> None:
        import mmar_mage.prompts as prompts

        loaded = load_prompts()
        for name in EXPECTED_PROMPT_NAMES:
            assert getattr(prompts, name) == loaded[name], f"{name} drifted between yaml and module-level constant"

    def test_get_step_types_constraint_still_exported(self) -> None:
        """The non-prompt helper survives the refactor."""
        from mmar_mage.prompts import get_step_types_constraint

        out = get_step_types_constraint(["llm", "tool"])
        # The exact wording is asserted by tests/test_llm_only_mode.py;
        # this is just an existence + sanity test.
        assert "llm" in out
        assert "tool" in out


# ---------------------------------------------------------------------------
# Customisation: explicit path + env var
# ---------------------------------------------------------------------------


class TestCustomYaml:
    def test_load_prompts_accepts_explicit_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "my_prompts.yaml"
        custom.write_text(
            yaml.safe_dump(
                {
                    "FAST_MODE_SYSTEM_PROMPT": "CARE-customised fast prompt.",
                    "STEP_PLANNER_PROMPT": "CARE-customised planner prompt.",
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        prompts = load_prompts(custom)
        assert prompts["FAST_MODE_SYSTEM_PROMPT"] == "CARE-customised fast prompt."
        assert prompts["STEP_PLANNER_PROMPT"] == "CARE-customised planner prompt."

    def test_load_prompts_accepts_string_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "p.yaml"
        custom.write_text("X_PROMPT: 'hello'\n", encoding="utf-8")
        assert load_prompts(str(custom)) == {"X_PROMPT": "hello"}

    def test_env_var_overrides_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom = tmp_path / "env.yaml"
        custom.write_text("X: 'y'\n", encoding="utf-8")
        monkeypatch.setenv("MAGE_PROMPTS_PATH", str(custom))
        assert default_prompts_path() == custom
        assert load_prompts() == {"X": "y"}

    def test_module_constants_reflect_env_override_on_reimport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A power user setting `MAGE_PROMPTS_PATH` before import gets their
        prompts in the module-level constants."""
        # Build a full custom YAML carrying every expected key. The
        # values are intentionally trivial — we just verify the load
        # path picks up the override.
        custom = tmp_path / "override.yaml"
        custom.write_text(
            yaml.safe_dump(
                {name: f"override-for-{name}" for name in EXPECTED_PROMPT_NAMES},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        monkeypatch.setenv("MAGE_PROMPTS_PATH", str(custom))
        # Drop the cached module so the re-import picks up the env var.
        monkeypatch.delitem(sys.modules, "mmar_mage.prompts", raising=False)
        reloaded = importlib.import_module("mmar_mage.prompts")
        try:
            assert reloaded.STEP_PLANNER_PROMPT == "override-for-STEP_PLANNER_PROMPT"
            assert reloaded.DAG_BUILDER_PROMPT == "override-for-DAG_BUILDER_PROMPT"
        finally:
            # Reload the real module so subsequent tests see the
            # bundled YAML again.
            monkeypatch.delenv("MAGE_PROMPTS_PATH", raising=False)
            sys.modules.pop("mmar_mage.prompts", None)
            importlib.import_module("mmar_mage.prompts")


# ---------------------------------------------------------------------------
# Loader error paths
# ---------------------------------------------------------------------------


class TestLoaderErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompts(tmp_path / "does-not-exist.yaml")

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level mapping"):
            load_prompts(bad)

    def test_non_string_value_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "int.yaml"
        bad.write_text("X_PROMPT: 42\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a string"):
            load_prompts(bad)

    def test_empty_value_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "empty.yaml"
        bad.write_text("X_PROMPT: ''\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            load_prompts(bad)

    def test_non_string_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "intkey.yaml"
        bad.write_text("123: 'x'\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not a string"):
            load_prompts(bad)


class TestCapabilitiesPreamble:
    """P6.3 — the capabilities system card enumerates the model's full toolbox."""

    def test_enumerates_all_step_types_tools_and_human_input(self) -> None:
        from mmar_mage.prompts import STEP_TYPE_GUIDE, build_capabilities_preamble
        from mmar_mage.schemas import CARL_VALID_STEP_TYPES

        # the guide covers EXACTLY the 16 authoritative CARL step types
        assert set(STEP_TYPE_GUIDE) == set(CARL_VALID_STEP_TYPES)

        card = build_capabilities_preamble()
        for step_type in CARL_VALID_STEP_TYPES:  # all 16 named
            assert step_type in card, step_type
        # bundled tools incl. the file-I/O primitives + the interaction hook
        for needle in ("read_file", "file_outline", "web_search", "run_python",
                       "human_input", "PAUSE"):
            assert needle in card, needle

    def test_appends_skills_and_mcp_blocks(self) -> None:
        from mmar_mage.prompts import build_capabilities_preamble

        card = build_capabilities_preamble(
            skills_block="Relevant packaged skills: pptx (skill=github://x)",
            mcp_block="- gmail (stdio)",
        )
        assert "skill=github://x" in card
        assert "gmail (stdio)" in card


class TestPlanningGuidance:
    """P6.6 — the planner preamble: think-first + every step type + the
    'skills produce files, llm only writes text' hard rule."""

    def test_lists_every_step_type(self):
        from mmar_mage.prompts import STEP_TYPE_GUIDE, build_planning_guidance

        guidance = build_planning_guidance()
        for name in STEP_TYPE_GUIDE:
            assert f"`{name}`" in guidance

    def test_think_first_and_file_rule(self):
        from mmar_mage.prompts import build_planning_guidance

        guidance = build_planning_guidance()
        # think-first about the desired OUTPUT
        assert "think about what the task actually needs" in guidance.lower()
        assert ".pptx" in guidance
        # the hard rule: files need agent_skill, not llm
        assert "agent_skill" in guidance
        assert "HARD RULE" in guidance
        assert "CANNOT create a file" in guidance
        # a concrete PPTX example
        assert "pptx" in guidance.lower()

    def test_wired_into_step_planner_system_prompt(self):
        # the planner must actually lead with the guidance
        import inspect

        from mmar_mage.agents import step_planner

        src = inspect.getsource(step_planner.StepPlanner.plan)
        # P-1: now passes allowed_step_types so the menu filters to it.
        assert "build_planning_guidance(" in src

    def test_wired_into_fast_mode_and_describer(self):
        # every chain-SHAPING stage leads with the same guidance
        import inspect

        from mmar_mage import generator
        from mmar_mage.agents import step_describer

        assert "build_planning_guidance()" in inspect.getsource(
            generator.MAGEGenerator._generate_fast,
        )
        assert "build_planning_guidance()" in inspect.getsource(
            step_describer.StepDescriber,
        )
