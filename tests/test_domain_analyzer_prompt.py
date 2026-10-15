"""P-3 — DOMAIN_ANALYZER_PROMPT schema + examples are in sync."""

from __future__ import annotations

from mmar_mage.prompts import DOMAIN_ANALYZER_PROMPT
from mmar_mage.schemas import DomainAnalysis


def test_schema_and_examples_declare_preferred_step_types() -> None:
    # schema header + all 4 examples now emit preferred_step_types (was: only 3 & 4)
    assert DOMAIN_ANALYZER_PROMPT.count('"preferred_step_types"') >= 5


def test_no_deprecated_step_types_in_prompt() -> None:
    # P-1 consistency: eval/aggregator are MAGE-internal, not real CARL types
    assert '"eval"' not in DOMAIN_ANALYZER_PROMPT
    assert '"aggregator"' not in DOMAIN_ANALYZER_PROMPT


def test_domain_analysis_model_accepts_preferred_step_types() -> None:
    da = DomainAnalysis(domain="finance", preferred_step_types=["mcp", "llm", "evaluation"])
    assert da.preferred_step_types == ["mcp", "llm", "evaluation"]
    # the field is optional — defaults to empty
    assert DomainAnalysis(domain="general").preferred_step_types == []
