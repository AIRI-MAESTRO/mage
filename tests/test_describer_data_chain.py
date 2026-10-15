"""T4 — data-chain worked example + whole-value $history rule in the describer prompt."""

from __future__ import annotations

from mmar_mage.prompts import STEP_DESCRIBER_PROMPT


def test_describer_has_whole_value_history_rule() -> None:
    p = STEP_DESCRIBER_PROMPT
    assert "Whole-value references" in p
    assert "NOT spliced inside a longer string" in p
    assert "add a `transform` step" in p


def test_describer_has_data_chain_example() -> None:
    p = STEP_DESCRIBER_PROMPT
    assert "DATA CHAIN" in p
    # IP -> transform(build URL) -> geo -> llm shape (the incident's composite query)
    assert "api.ipify.org" in p
    assert "Build the geolocation URL" in p
    assert '"transform_type": "format"' in p
    # the geo tool consumes the built URL as a WHOLE value, not embedded
    assert '"url": "$history[-1]"' in p
