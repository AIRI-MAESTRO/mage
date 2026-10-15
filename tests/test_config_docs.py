"""T39 — docs/CONFIG.md stays in sync with MAGEConfig + the profiles (drift check)."""

from __future__ import annotations

from mmar_mage import config_docs
from mmar_mage.schemas import MAGEConfig


def test_config_md_is_up_to_date() -> None:
    committed = config_docs._OUTPUT.read_text(encoding="utf-8")
    assert committed == config_docs.generate_config_table(), (
        "docs/CONFIG.md is stale — run `python -m mmar_mage.config_docs` and commit the result"
    )


def test_table_covers_every_field() -> None:
    table = config_docs.generate_config_table()
    for name in MAGEConfig.model_fields:
        assert f"| `{name}` |" in table, f"{name} missing from CONFIG.md"


def test_env_vars_and_recent_fields_surface() -> None:
    table = config_docs.generate_config_table()
    assert "`MAGE_API_KEY`" in table and "`MAGE_MODEL`" in table  # env column from the shared map
    for name in ("memory_session_id", "json_parse_retries", "use_schema_constrained_output"):
        assert f"| `{name}` |" in table  # fields added across this audit surface automatically
