"""Tests for MAGEGenerator and CARL validation logic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mmar_mage.exceptions import MAGEValidationError
from mmar_mage.generator import MAGEGenerator, validate_carl_json
from mmar_mage.schemas import MAGEConfig

# ---------------------------------------------------------------------------
# validate_carl_json
# ---------------------------------------------------------------------------


class TestValidateCarlJson:
    """Tests for the standalone validation function."""

    def test_valid_chain(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []},
                {"number": 2, "title": "B", "step_type": "llm", "aim": "do B", "dependencies": [1]},
            ]
        }
        validate_carl_json(chain)  # should not raise

    def test_missing_steps(self) -> None:
        with pytest.raises(MAGEValidationError, match="must contain 'steps'"):
            validate_carl_json({"name": "no steps"})

    def test_empty_steps(self) -> None:
        with pytest.raises(MAGEValidationError, match="non-empty"):
            validate_carl_json({"steps": []})

    def test_duplicate_step_numbers(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A"},
                {"number": 1, "title": "B"},
            ]
        }
        with pytest.raises(MAGEValidationError, match="Duplicate"):
            validate_carl_json(chain)

    def test_missing_title(self) -> None:
        chain = {"steps": [{"number": 1}]}
        with pytest.raises(MAGEValidationError, match="missing 'title'"):
            validate_carl_json(chain)

    def test_self_dependency(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "dependencies": [1]}]}
        with pytest.raises(MAGEValidationError, match="depends on itself"):
            validate_carl_json(chain)

    def test_forward_dependency_is_allowed(self) -> None:
        # Forward references are valid in a DAG as long as no cycle exists.
        # CARL relies on dependency edges, not declaration order.
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": [2]},
                {"number": 2, "title": "B", "step_type": "llm", "aim": "do B", "dependencies": []},
            ]
        }
        validate_carl_json(chain)

    def test_dependency_on_non_existent_step(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": [99]},
            ]
        }
        with pytest.raises(MAGEValidationError, match="non-existent"):
            validate_carl_json(chain)

    def test_cycle_detection(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": [3]},
                {"number": 2, "title": "B", "step_type": "llm", "aim": "do B", "dependencies": [1]},
                {"number": 3, "title": "C", "step_type": "llm", "aim": "do C", "dependencies": [2]},
            ]
        }
        with pytest.raises(MAGEValidationError, match="Cyclic"):
            validate_carl_json(chain)

    def test_invalid_step_type(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "bogus"}]}
        with pytest.raises(MAGEValidationError, match="invalid step_type"):
            validate_carl_json(chain)

    def test_llm_requires_aim(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "aim": ""}]}
        with pytest.raises(MAGEValidationError, match="requires non-empty 'aim'"):
            validate_carl_json(chain)

    def test_strict_carl_rejects_eval(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "eval"}]}
        with pytest.raises(MAGEValidationError, match="invalid step_type 'eval'"):
            validate_carl_json(chain, strict_carl=True)

    def test_strict_carl_accepts_structured_output(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "structured_output"}]}
        validate_carl_json(chain, strict_carl=True)

    def test_missing_number(self) -> None:
        chain = {"steps": [{"title": "A"}]}
        with pytest.raises(MAGEValidationError, match="'number'"):
            validate_carl_json(chain)


# ---------------------------------------------------------------------------
# MAGEGenerator integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_mode_selection_fast(
    config: MAGEConfig,
    mock_openai_fast: MagicMock,
) -> None:
    """Generator uses fast mode when config says fast."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")
    assert result.mode == "fast"


@pytest.mark.asyncio
async def test_generator_mode_selection_deep(
    deep_config: MAGEConfig,
    mock_openai_deep: MagicMock,
) -> None:
    """Generator uses deep mode when config says deep."""
    gen = MAGEGenerator(config=deep_config)
    result = await gen.generate("Test")
    assert result.mode == "deep"


@pytest.mark.asyncio
async def test_generator_chain_json_is_valid(
    config: MAGEConfig,
    mock_openai_fast: MagicMock,
) -> None:
    """Generated chain_json is parseable JSON."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")
    parsed = json.loads(result.chain_json)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_generator_metadata_timing(
    config: MAGEConfig,
    mock_openai_fast: MagicMock,
) -> None:
    """Metadata includes positive generation time."""
    gen = MAGEGenerator(config=config)
    result = await gen.generate("Test")
    assert result.metadata.generation_time_seconds >= 0
