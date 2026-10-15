"""Tests for MAGEConfig factory methods: from_toml(), from_env(), for_openrouter(), for_local()."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

from mmar_mage.schemas import MAGEConfig

# ---------------------------------------------------------------------------
# from_toml()
# ---------------------------------------------------------------------------


class TestFromToml:
    """Tests for MAGEConfig.from_toml()."""

    def test_from_toml_fast(self, tmp_path: Any) -> None:
        """Load a fast-mode TOML config."""
        toml_file = tmp_path / "fast.toml"
        toml_file.write_text('[mage]\nmode = "fast"\nmodel = "gpt-4o"\nenable_memory = false\n')
        cfg = MAGEConfig.from_toml(str(toml_file))
        assert cfg.mode == "fast"
        assert cfg.model == "gpt-4o"
        assert cfg.enable_memory is False

    def test_from_toml_deep_openrouter(self, tmp_path: Any) -> None:
        """Load a deep-mode OpenRouter TOML config."""
        toml_file = tmp_path / "openrouter.toml"
        toml_file.write_text(
            "[mage]\n"
            'provider = "openrouter"\n'
            'base_url = "https://openrouter.ai/api/v1"\n'
            'model = "anthropic/claude-3.5-sonnet"\n'
            "temperature = 0.2\n"
            'mode = "deep"\n'
            'openrouter_site_url = "https://example.com"\n'
            'openrouter_app_name = "test-app"\n'
        )
        cfg = MAGEConfig.from_toml(str(toml_file))
        assert cfg.provider == "openrouter"
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.openrouter_site_url == "https://example.com"

    def test_from_toml_with_stage_overrides(self, tmp_path: Any) -> None:
        """Load TOML with per-stage LLM overrides."""
        toml_file = tmp_path / "overrides.toml"
        toml_file.write_text(
            "[mage]\n"
            'model = "gpt-4o"\n\n'
            "[mage.stage_llm_overrides]\n"
            'domain_analysis = { model = "gpt-4o-mini", temperature = 0.0 }\n'
        )
        cfg = MAGEConfig.from_toml(str(toml_file))
        # §6 P2: now a typed dict[str, StageLLMOverride].
        override = cfg.stage_llm_overrides["domain_analysis"]
        assert override.model == "gpt-4o-mini"
        assert override.temperature == 0.0

    def test_from_toml_local_airi(self, tmp_path: Any) -> None:
        """Load a local AIRI TOML config."""
        toml_file = tmp_path / "airi.toml"
        toml_file.write_text(
            "[mage]\n"
            'provider = "local"\n'
            'base_url = "https://inference.airi.net:46783/v1"\n'
            'model = "__auto__"\n'
            "ssl_verify = false\n"
            "http_timeout = 120.0\n"
        )
        cfg = MAGEConfig.from_toml(str(toml_file))
        assert cfg.provider == "local"
        assert cfg.ssl_verify is False
        assert cfg.http_timeout == 120.0
        assert cfg.model == "__auto__"

    def test_from_toml_missing_mage_section(self, tmp_path: Any) -> None:
        """TOML without [mage] section returns default config."""
        toml_file = tmp_path / "empty.toml"
        toml_file.write_text('[other]\nkey = "value"\n')
        cfg = MAGEConfig.from_toml(str(toml_file))
        assert cfg.mode == "deep"  # default
        assert cfg.model == "gpt-4o"

    def test_from_toml_research_options(self, tmp_path: Any) -> None:
        """Load TOML with research options."""
        toml_file = tmp_path / "research.toml"
        toml_file.write_text(
            "[mage]\n"
            "enable_memory_research = true\n"
            "enable_web_research = true\n"
            'web_search_provider = "tavily"\n'
            "memory_recall_top_k = 10\n"
            "cold_start_candidates = 3\n"
        )
        cfg = MAGEConfig.from_toml(str(toml_file))
        assert cfg.enable_memory_research is True
        assert cfg.enable_web_research is True
        assert cfg.memory_recall_top_k == 10
        assert cfg.cold_start_candidates == 3


# ---------------------------------------------------------------------------
# from_env()
# ---------------------------------------------------------------------------


class TestFromEnv:
    """Tests for MAGEConfig.from_env()."""

    def test_from_env_basic(self) -> None:
        """Load config from environment variables."""
        env = {
            "MAGE_API_KEY": "sk-test-123",
            "MAGE_MODEL": "gpt-4o-mini",
            "MAGE_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = MAGEConfig.from_env()
            assert cfg.api_key == "sk-test-123"
            assert cfg.model == "gpt-4o-mini"
            assert cfg.provider == "openai"

    def test_from_env_bool_fields(self) -> None:
        """Boolean environment variables are parsed correctly."""
        env = {
            "MAGE_API_KEY": "test",
            "MAGE_ENABLE_MEMORY": "false",
            "MAGE_ENABLE_MEMORY_RESEARCH": "true",
            "MAGE_SSL_VERIFY": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = MAGEConfig.from_env()
            assert cfg.enable_memory is False
            assert cfg.enable_memory_research is True
            assert cfg.ssl_verify is False

    def test_from_env_numeric_fields(self) -> None:
        """Numeric environment variables are parsed correctly."""
        env = {
            "MAGE_API_KEY": "test",
            "MAGE_MEMORY_RELEVANCE_THRESHOLD": "0.6",
            "MAGE_MEMORY_RECALL_TOP_K": "10",
            "MAGE_COLD_START_CANDIDATES": "3",
            "MAGE_HTTP_TIMEOUT": "120.0",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = MAGEConfig.from_env()
            assert cfg.memory_relevance_threshold == 0.6
            assert cfg.memory_recall_top_k == 10
            assert cfg.cold_start_candidates == 3
            assert cfg.http_timeout == 120.0

    def test_from_env_fallback_openai_key(self) -> None:
        """Falls back to OPENAI_API_KEY if MAGE_API_KEY not set."""
        env = {"OPENAI_API_KEY": "sk-openai-fallback"}
        # Remove MAGE_API_KEY if present
        clean_env = {k: v for k, v in os.environ.items() if k != "MAGE_API_KEY"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            cfg = MAGEConfig.from_env()
            assert cfg.api_key == "sk-openai-fallback"

    def test_from_env_empty_values_ignored(self) -> None:
        """Empty env vars are treated as absent."""
        env = {"MAGE_BASE_URL": "", "MAGE_API_KEY": "test"}
        with patch.dict(os.environ, env, clear=False):
            cfg = MAGEConfig.from_env()
            assert cfg.base_url is None


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------


class TestFactoryMethods:
    """Tests for convenience factory classmethods."""

    def test_for_openrouter(self) -> None:
        cfg = MAGEConfig.for_openrouter(api_key="or-key-123")
        assert cfg.provider == "openrouter"
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.api_key == "or-key-123"
        assert cfg.model == "anthropic/claude-3.5-sonnet"
        assert cfg.openrouter_site_url == "https://github.com/Glazkoff/carl-mage"

    def test_for_openrouter_custom_model(self) -> None:
        cfg = MAGEConfig.for_openrouter(api_key="key", model="openai/gpt-4o")
        assert cfg.model == "openai/gpt-4o"

    def test_for_local(self) -> None:
        cfg = MAGEConfig.for_local(base_url="http://localhost:11434/v1")
        assert cfg.provider == "local"
        assert cfg.base_url == "http://localhost:11434/v1"
        assert cfg.api_key == "EMPTY"
        assert cfg.model == "__auto__"
        assert cfg.ssl_verify is True

    def test_for_local_custom(self) -> None:
        cfg = MAGEConfig.for_local(
            base_url="https://inference.airi.net/v1",
            api_key="airi-key",
            model="llama-3",
            ssl_verify=False,
        )
        assert cfg.api_key == "airi-key"
        assert cfg.model == "llama-3"
        assert cfg.ssl_verify is False

    def test_for_local_with_kwargs(self) -> None:
        cfg = MAGEConfig.for_local(
            base_url="http://localhost:11434/v1",
            enable_memory=False,
            mode="fast",
        )
        assert cfg.enable_memory is False
        assert cfg.mode == "fast"
